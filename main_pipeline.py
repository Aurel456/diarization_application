from __future__ import annotations

import ast
import csv
import logging
import os
import pickle
import re
import shutil
import sys
from dataclasses import dataclass, field, asdict
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

from settings import settings
from src.utils import (
    setup_environment,
    load_or_run,
    adjust_timestamps_for_sequential_audio,
)
from src.audio_splitter import split_audio_into_chunks
from src.diarizer import run_diarization, process_diarization_results
from src.clusterer import cluster_speakers, update_speaker_labels
from src.transcriber import transcribe_all_segments
from src.cleaner import process_all_text
from src.exporter import export_results
from src.simple_transcriber import transcribe_long_audio
from src.summarizer import summarise_text
from src.speaker_identifier import (
    identify_speakers,
    save_speaker_info as save_speaker_identification,
    SpeakerInfo,
)
from src.meeting_minutes import (
    generate_meeting_minutes,
    save_meeting_minutes,
    MEETING_MINUTES_FORMATS,
    DEFAULT_FORMAT,
    MeetingMinutes,
    MeetingMinuteSection,
)


import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchaudio._backend.soundfile_backend"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils")


PIPELINE_STEP_LABELS: Dict[str, str] = {
    "setup": "Environment Setup",
    "split": "Audio Splitting",
    "diarization": "Speaker Diarization",
    "process_diarization": "Diarization Post-processing",
    "clustering": "Speaker Clustering",
    "label_update": "Speaker Label Update",
    "transcription": "Transcription",
    "speaker_identification": "Speaker Identification",
    "cleaning": "Text Cleaning",
    "timestamp_adjust": "Timestamp Adjustments",
    "export": "Export",
    "meeting_minutes": "Meeting Minutes Generation",
    "summary": "Summary & Metrics",
    "simple_transcription": "Simple Transcription",
    "simple_export": "Simple Export",
}

PIPELINE_STEP_SEQUENCE: List[str] = [
    "setup",
    "split",
    "diarization",
    "process_diarization",
    "clustering",
    "label_update",
    "transcription",
    "speaker_identification",
    "cleaning",
    "timestamp_adjust",
    "export",
    "meeting_minutes",
    "summary",
]


from src.errors import (  # noqa: E402
    PipelineError,
    ConfigurationError,
    AudioInputError,
    AudioSplittingError,
    DiarizationError,
    ClusteringError,
    TranscriptionError,
    LLMError,
    ExportError,
    CancelledError,
)


@dataclass
class PipelineConfig:
    root: str = "."
    hf_token: Optional[str] = None
    audio_processing_mode: str = "concurrent"
    segment_duration: int = 1200
    max_workers: int = 2
    duree_min_speaker: float = 0.5
    vad_filter: bool = True
    input_audio_paths: List[str] = field(default_factory=list)
    server_url: Optional[str] = None
    whisper_model: str = "whisper"
    api_key: Optional[str] = None
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    run_id: Optional[str] = None
    reuse_cache: bool = True
    force_split: bool = False
    clear_saved_state: bool = False
    log_level: int = logging.INFO
    log_to_console: bool = True
    chunks_folder: Optional[str] = None
    progress_metadata: Dict[str, Any] = field(default_factory=dict)
    load_dotenv_path: Optional[str] = ".env"
    simple_mode: bool = False
    chunk_size: int = 600
    enable_llm_cleaning: bool = True
    # Summary generation (independent from LLM cleaning since they answer different questions)
    enable_summary: bool = True
    # Speaker count constraints — when set, override pure HDBSCAN with a constrained clustering
    # Use num_speakers when the exact count is known (forces AgglomerativeClustering)
    # Use min_speakers/max_speakers as a range (HDBSCAN result clamped to this range if outside)
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    experiments_root: Optional[str] = None
    auto_delete_outputs: bool = False
    cleanup_grace_seconds: int = 60 * 60 * 2
    # Speaker identification & meeting minutes
    enable_speaker_identification: bool = False
    speaker_identification_model: Optional[str] = None
    enable_meeting_minutes: bool = False
    meeting_minutes_model: Optional[str] = None
    meeting_minutes_format: str = DEFAULT_FORMAT
    meeting_minutes_instructions: Optional[str] = None
    # Stateless mode: no pickle cache, experiments dir under tmp
    stateless: bool = False
    # Cancellation: set the event to request a graceful stop at the next checkpoint
    cancel_event: Any = None


@dataclass
class PipelineResult:
    success: bool
    run_id: str
    experiments_dir: str
    log_file: str
    cleaned_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    exported_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    docx_path: Optional[str] = None
    num_speakers_final: Optional[int] = None
    message: Optional[str] = None
    txt_path: Optional[str] = None
    srt_path: Optional[str] = None
    cleaned_text: Optional[str] = None
    summary_path: Optional[str] = None
    summary_text: Optional[str] = None
    # Speaker identification results
    speaker_identification_path: Optional[str] = None
    speaker_info: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Meeting minutes results
    meeting_minutes_paths: Dict[str, str] = field(default_factory=dict)
    meeting_minutes: Optional[Any] = None  # MeetingMinutes object
    # Paths needed for post-pipeline features (speaker visualization)
    chunks_folder: Optional[str] = None
    segment_duration: int = 1200


def _check_cancel(config: "PipelineConfig", logger: logging.Logger, step: str) -> None:
    """Raise CancelledError if the config's cancel_event has been set."""
    evt = getattr(config, "cancel_event", None)
    if evt is not None and getattr(evt, "is_set", lambda: False)():
        logger.warning("Pipeline cancellation requested at step '%s'.", step)
        raise CancelledError(f"Pipeline cancelled by user at step '{step}'.")


def _sanitize_run_id(run_id: Optional[str]) -> str:
    value = run_id or time.strftime("run_%Y%m%d_%H%M%S")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    if not sanitized:
        sanitized = time.strftime("run_%Y%m%d_%H%M%S")
    return sanitized


def _resolve_audio_paths(paths: List[str], root: str) -> List[str]:
    resolved: List[str] = []
    for path in paths:
        candidate = path if os.path.isabs(path) else os.path.join(root, path)
        if not os.path.exists(candidate):
            raise AudioInputError(f"Audio file not found: {candidate}")
        resolved.append(os.path.abspath(candidate))
    return resolved


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env_value(
    keys: Tuple[str, ...], default: Optional[str] = None
) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value is not None:
            return value
    return default


def configure_logging(
    log_file: str,
    log_level: int,
    log_to_console: bool,
    extra_handlers: Optional[List[logging.Handler]] = None,
    reset: bool = True,
) -> logging.Logger:
    logger = logging.getLogger()
    if reset:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if log_to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if extra_handlers:
        for handler in extra_handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    return logger


def load_config_from_env(dotenv_path: Optional[str] = ".env") -> PipelineConfig:
    if dotenv_path:
        load_dotenv(dotenv_path)

    root = os.path.abspath(_get_env_value(("ROOT", "APP_ROOT"), settings.root))
    experiments_root_env = _get_env_value(
        ("EXPERIMENTS_ROOT", "APP_EXPERIMENTS_ROOT"), settings.experiments_root
    )
    experiments_root = (
        os.path.abspath(experiments_root_env)
        if experiments_root_env
        else settings.experiments_root
    )

    hf_token_env = _get_env_value(("HF_TOKEN", "APP_HF_TOKEN"), settings.hf_token)
    hf_token = (
        hf_token_env
        if hf_token_env and hf_token_env.strip().lower() != "none"
        else None
    )

    audio_processing_mode = _get_env_value(
        ("AUDIO_PROCESSING_MODE", "APP_AUDIO_PROCESSING_MODE"), "concurrent"
    ).lower()
    if audio_processing_mode not in {"concurrent", "sequential"}:
        logging.warning(
            "Invalid AUDIO_PROCESSING_MODE '%s'. Falling back to 'concurrent'.",
            audio_processing_mode,
        )
        audio_processing_mode = "concurrent"

    segment_duration = int(
        _get_env_value(("SEGMENT_DURATION", "APP_SEGMENT_DURATION"), "1200")
    )
    max_workers = int(_get_env_value(("MAX_WORKERS", "APP_MAX_WORKERS"), "2"))
    duree_min_speaker = float(
        _get_env_value(("DUREE_MIN_SPEAKER", "APP_DUREE_MIN_SPEAKER"), "0.5")
    )
    vad_filter = _parse_bool(
        _get_env_value(("VAD_FILTER", "APP_VAD_FILTER"), None), default=True
    )

    input_audio_paths: List[str] = []
    input_audio_list_str = _get_env_value(("INPUT_AUDIO", "APP_INPUT_AUDIO"), None)
    if input_audio_list_str:
        try:
            parsed = ast.literal_eval(input_audio_list_str)
            if not isinstance(parsed, list):
                raise ValueError("INPUT_AUDIO must be a list.")
            input_audio_paths = [str(item).strip() for item in parsed if item]
        except (ValueError, SyntaxError, TypeError) as exc:
            raise ConfigurationError(
                f"Failed to parse INPUT_AUDIO environment variable: {exc}"
            ) from exc
    else:
        single_audio_file = _get_env_value(
            ("INPUT_AUDIO_FILE", "APP_INPUT_AUDIO_FILE"), None
        )
        if single_audio_file:
            input_audio_paths = [single_audio_file.strip()]

    if not input_audio_paths:
        raise ConfigurationError(
            "No input audio paths provided. Set INPUT_AUDIO or INPUT_AUDIO_FILE."
        )

    server_url = (
        _get_env_value(
            ("SERVER_URL", "APP_SERVER_URL", "WHISPER_URL"), settings.server_url
        )
        or ""
    ).strip() or None
    whisper_model = (
        _get_env_value(("WHISPER_MODEL", "APP_WHISPER_MODEL"), settings.whisper_model)
        or "whisper"
    ).strip() or "whisper"
    api_key = (
        _get_env_value(("API_KEY", "APP_API_KEY"), settings.api_key) or ""
    ).strip() or None
    llm_model = (
        _get_env_value(("LLM_MODEL", "APP_LLM_MODEL"), settings.llm_model) or ""
    ).strip() or None
    # LLM base URL is unified with SERVER_URL (single endpoint for Whisper + LLM)
    llm_base_url = (settings.server_url or "").strip() or None

    run_id_env = _get_env_value(("RUN_ID", "APP_RUN_ID"), None)

    reuse_cache_default = settings.default_reuse_cache
    reuse_cache = _parse_bool(
        _get_env_value(("REUSE_CACHE", "APP_REUSE_CACHE"), None),
        default=reuse_cache_default,
    )
    force_split_default = settings.default_force_split
    force_split = _parse_bool(
        _get_env_value(("FORCE_SPLIT", "APP_FORCE_SPLIT"), None),
        default=force_split_default,
    )
    clear_saved_state_default = settings.default_clear_saved_state
    clear_saved_state = _parse_bool(
        _get_env_value(("CLEAR_SAVED_STATE", "APP_CLEAR_SAVED_STATE"), None),
        default=clear_saved_state_default,
    )

    log_level_str = (
        _get_env_value(("LOG_LEVEL", "APP_LOG_LEVEL"), settings.default_log_level)
        or "INFO"
    ).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_to_console_default = settings.log_to_console
    log_to_console = _parse_bool(
        _get_env_value(("LOG_TO_CONSOLE", "APP_LOG_TO_CONSOLE"), None),
        default=log_to_console_default,
    )

    chunks_folder_env = _get_env_value(("CHUNKS_FOLDER", "APP_CHUNKS_FOLDER"), None)
    chunks_folder = (
        os.path.abspath(chunks_folder_env.strip())
        if chunks_folder_env and chunks_folder_env.strip()
        else None
    )

    simple_mode = _parse_bool(
        _get_env_value(("SIMPLE_MODE", "APP_SIMPLE_MODE"), None), default=False
    )
    chunk_size = int(_get_env_value(("CHUNK_SIZE", "APP_CHUNK_SIZE"), "600"))
    enable_llm_cleaning = _parse_bool(
        _get_env_value(("ENABLE_LLM_CLEANING", "APP_ENABLE_LLM_CLEANING"), None),
        default=settings.enable_llm_cleaning,
    )
    enable_summary = _parse_bool(
        _get_env_value(("ENABLE_SUMMARY", "APP_ENABLE_SUMMARY"), None),
        default=settings.enable_summary,
    )

    def _opt_int(key_pair, fallback):
        raw = _get_env_value(key_pair, None)
        if raw is None or str(raw).strip() == "":
            return fallback
        try:
            return int(str(raw).strip())
        except ValueError:
            return fallback

    num_speakers = _opt_int(("NUM_SPEAKERS", "APP_NUM_SPEAKERS"), settings.num_speakers)
    min_speakers = _opt_int(("MIN_SPEAKERS", "APP_MIN_SPEAKERS"), settings.min_speakers)
    max_speakers = _opt_int(("MAX_SPEAKERS", "APP_MAX_SPEAKERS"), settings.max_speakers)

    auto_delete_outputs = _parse_bool(
        _get_env_value(("AUTO_DELETE_OUTPUTS", "APP_AUTO_DELETE_OUTPUTS"), None),
        default=settings.auto_delete_outputs,
    )
    cleanup_grace_seconds_raw = _get_env_value(
        ("CLEANUP_GRACE_SECONDS", "APP_CLEANUP_GRACE_SECONDS"),
        str(settings.cleanup_grace_seconds),
    )
    try:
        cleanup_grace_seconds = max(1, int(cleanup_grace_seconds_raw))
    except ValueError:
        cleanup_grace_seconds = settings.cleanup_grace_seconds

    # Speaker identification settings
    enable_speaker_identification = _parse_bool(
        _get_env_value(
            ("ENABLE_SPEAKER_IDENTIFICATION", "APP_ENABLE_SPEAKER_IDENTIFICATION"), None
        ),
        default=settings.enable_speaker_identification,
    )
    speaker_identification_model = (
        _get_env_value(
            ("SPEAKER_IDENTIFICATION_MODEL", "APP_SPEAKER_IDENTIFICATION_MODEL"),
            settings.speaker_identification_model,
        )
        or ""
    ).strip() or None

    # Meeting minutes settings
    enable_meeting_minutes = _parse_bool(
        _get_env_value(("ENABLE_MEETING_MINUTES", "APP_ENABLE_MEETING_MINUTES"), None),
        default=settings.enable_meeting_minutes,
    )
    meeting_minutes_model = (
        _get_env_value(
            ("MEETING_MINUTES_MODEL", "APP_MEETING_MINUTES_MODEL"),
            settings.meeting_minutes_model,
        )
        or ""
    ).strip() or None

    return PipelineConfig(
        root=root,
        hf_token=hf_token,
        audio_processing_mode=audio_processing_mode,
        segment_duration=segment_duration,
        max_workers=max_workers,
        duree_min_speaker=duree_min_speaker,
        vad_filter=vad_filter,
        input_audio_paths=input_audio_paths,
        server_url=server_url,
        whisper_model=whisper_model,
        api_key=api_key,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        run_id=run_id_env,
        reuse_cache=reuse_cache,
        force_split=force_split,
        clear_saved_state=clear_saved_state,
        log_level=log_level,
        log_to_console=log_to_console,
        chunks_folder=chunks_folder,
        simple_mode=simple_mode,
        chunk_size=chunk_size,
        enable_llm_cleaning=enable_llm_cleaning,
        enable_summary=enable_summary,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        experiments_root=experiments_root,
        auto_delete_outputs=auto_delete_outputs,
        cleanup_grace_seconds=cleanup_grace_seconds,
        enable_speaker_identification=enable_speaker_identification,
        speaker_identification_model=speaker_identification_model,
        enable_meeting_minutes=enable_meeting_minutes,
        meeting_minutes_model=meeting_minutes_model,
    )


def run_simple_transcription_pipeline(
    config: PipelineConfig,
    progress_callback: Optional[
        Callable[[str, str, str, Optional[Dict[str, Any]]], None]
    ] = None,
    extra_log_handlers: Optional[List[logging.Handler]] = None,
) -> PipelineResult:
    """Orchestrator for simple transcription mode (no diarization)."""
    progress_callback = progress_callback or (
        lambda step, status, message, payload=None: None
    )

    def notify(
        step_key: str,
        status: str,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if step_key not in PIPELINE_STEP_LABELS:
            return
        progress_callback(step_key, status, message, payload)

    sanitized_run_id = _sanitize_run_id(config.run_id)
    experiments_root = os.path.abspath(
        config.experiments_root or os.path.join(config.root, "experiments")
    )
    experiments_dir = os.path.join(experiments_root, sanitized_run_id)
    os.makedirs(experiments_dir, exist_ok=True)

    log_file = os.path.join(experiments_dir, "logs.txt")
    logger = configure_logging(
        log_file=log_file,
        log_level=config.log_level,
        log_to_console=config.log_to_console,
        extra_handlers=extra_log_handlers,
        reset=True,
    )

    redacted_config = {
        "root": config.root,
        "whisper_model": config.whisper_model,
        "chunk_size": config.chunk_size,
        "enable_llm_cleaning": config.enable_llm_cleaning,
        "run_id": sanitized_run_id,
    }

    logger.info("--- Starting Simple Transcription Run: %s ---", sanitized_run_id)
    logger.info("Configuration: %s", redacted_config)

    try:
        input_audio_paths = _resolve_audio_paths(config.input_audio_paths, config.root)
        os.makedirs(experiments_root, exist_ok=True)

        if len(input_audio_paths) > 1:
            raise ConfigurationError("Simple mode supports only one audio file.")

        output_dir = os.path.join(experiments_dir, "simple_transcription")
        os.makedirs(output_dir, exist_ok=True)

        if not config.server_url or not config.api_key:
            raise ConfigurationError(
                "SERVER_URL and API_KEY must be provided for transcription."
            )

        notify("setup", "success", "Environment ready for simple transcription")

        logger.info("Input audio file: %s", input_audio_paths[0])
        logger.info("Output directory: %s", output_dir)

        current_step = "simple_transcription"
        notify(current_step, "start", "Transcribing long audio in chunks")
        audio_path = input_audio_paths[0]
        try:
            # Note: We now pass llm params to simple transcriber
            result_dict = transcribe_long_audio(
                audio_path=audio_path,
                chunk_size=config.chunk_size,
                server_url=config.server_url,
                api_key=config.api_key,
                whisper_model=config.whisper_model,
                enable_llm_cleaning=(
                    config.enable_llm_cleaning
                    and bool(config.llm_base_url and config.llm_model)
                ),
                enable_summary=(
                    config.enable_summary
                    and bool(config.llm_base_url and config.llm_model)
                ),
                llm_base_url=config.llm_base_url,
                llm_model=config.llm_model,
                output_dir=output_dir,
            )
            txt_path = Path(result_dict["txt_path"])
            srt_path = Path(result_dict["srt_path"])
            summary_path = result_dict.get("summary_path")  # Extract summary path
            summary_text = result_dict.get("summary_text")  # Extract summary text
            cleaned_text = result_dict["text"]

            notify(
                current_step,
                "success",
                "Simple transcription completed.",
                {"text_length": len(cleaned_text)},
            )
        except Exception as exc:
            notify(
                current_step,
                "error",
                f"Simple transcription failed: {exc}",
            )
            raise TranscriptionError(f"Simple transcription failed: {exc}", cause=exc) from exc

        current_step = "simple_export"
        notify(
            current_step,
            "success",
            f"Text exported to {txt_path}",
            {"txt_path": str(txt_path)},
        )

        current_step = "summary"
        notify(
            current_step,
            "start",
            "Compiling simple summary metrics",
        )
        experiments_csv = os.path.join(experiments_root, "experiments_log.csv")
        write_header = not os.path.exists(experiments_csv)
        base_audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        try:
            with open(experiments_csv, "a", newline="", encoding="utf-8") as file_obj:
                writer = csv.writer(file_obj)
                if write_header:
                    writer.writerow(
                        [
                            "run_id",
                            "timestamp",
                            "input_audio",
                            "chunk_size",
                            "enable_llm_cleaning",
                            "text_length",
                            "status",
                        ]
                    )
                writer.writerow(
                    [
                        sanitized_run_id,
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        base_audio_name,
                        config.chunk_size,
                        config.enable_llm_cleaning,
                        len(cleaned_text),
                        "Completed",
                    ]
                )
            logger.info("Experiment details logged to %s", experiments_csv)
        except Exception as exc:
            logger.error("Failed to log experiment details: %s", exc)

        notify(
            current_step,
            "success",
            "Simple pipeline completed.",
            {
                "text_length": len(cleaned_text),
            },
        )

        # --- Optional meeting minutes generation (works without diarization) ---
        meeting_minutes_obj = None
        saved_minutes: Dict[str, str] = {}
        if (
            config.enable_meeting_minutes
            and config.llm_base_url
            and (config.meeting_minutes_model or config.llm_model)
            and cleaned_text
        ):
            current_step = "meeting_minutes"
            notify(current_step, "start", "Generating meeting minutes")
            try:
                # Build a single-row DataFrame so we can reuse the same minutes generator.
                transcript_df = pd.DataFrame(
                    [{"start": 0.0, "end": 0.0, "transcription": cleaned_text}]
                )
                meeting_minutes_obj = generate_meeting_minutes(
                    transcript_df,
                    config.llm_base_url,
                    config.api_key,
                    config.meeting_minutes_model or config.llm_model,
                    speaker_info=None,
                    meeting_context=None,
                    format_key=config.meeting_minutes_format,
                    user_instructions=config.meeting_minutes_instructions,
                )
                if meeting_minutes_obj:
                    saved_minutes = save_meeting_minutes(
                        meeting_minutes_obj,
                        experiments_dir,
                        f"meeting_minutes_{sanitized_run_id}",
                    )
                    notify(current_step, "success", "Meeting minutes generated.")
                else:
                    notify(current_step, "skipped", "Meeting minutes returned empty.")
            except Exception as exc:
                logger.error("Meeting minutes generation failed: %s", exc)
                notify(current_step, "error", f"Meeting minutes failed: {exc}")

        logger.info("--- Simple Run %s Completed ---", sanitized_run_id)

        return PipelineResult(
            success=True,
            run_id=sanitized_run_id,
            experiments_dir=os.path.abspath(experiments_dir),
            log_file=log_file,
            txt_path=str(txt_path),
            srt_path=str(srt_path),
            cleaned_text=cleaned_text,
            summary_path=summary_path,
            summary_text=summary_text,
            meeting_minutes_paths=saved_minutes,
            meeting_minutes=meeting_minutes_obj,
            message="Simple transcription completed",
        )

    except Exception as exc:
        logger.exception("Simple pipeline failed: %s", exc)
        notify(current_step, "error", str(exc))
        raise


def run_pipeline(
    config: PipelineConfig,
    progress_callback: Optional[
        Callable[[str, str, str, Optional[Dict[str, Any]]], None]
    ] = None,
    extra_log_handlers: Optional[List[logging.Handler]] = None,
) -> PipelineResult:
    progress_callback = progress_callback or (
        lambda step, status, message, payload=None: None
    )

    def notify(
        step_key: str,
        status: str,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if step_key not in PIPELINE_STEP_LABELS:
            return
        progress_callback(step_key, status, message, payload)

    if config.simple_mode:
        return run_simple_transcription_pipeline(
            config, progress_callback, extra_log_handlers
        )

    sanitized_run_id = _sanitize_run_id(config.run_id)
    experiments_root = os.path.abspath(
        config.experiments_root or os.path.join(config.root, "experiments")
    )
    experiments_dir = os.path.join(experiments_root, sanitized_run_id)
    os.makedirs(experiments_dir, exist_ok=True)

    log_file = os.path.join(experiments_dir, "logs.txt")
    logger = configure_logging(
        log_file=log_file,
        log_level=config.log_level,
        log_to_console=config.log_to_console,
        extra_handlers=extra_log_handlers,
        reset=True,
    )

    redacted_config = {
        "root": config.root,
        "audio_processing_mode": config.audio_processing_mode,
        "segment_duration": config.segment_duration,
        "max_workers": config.max_workers,
        "duree_min_speaker": config.duree_min_speaker,
        "vad_filter": config.vad_filter,
        "whisper_model": config.whisper_model,
        "run_id": sanitized_run_id,
        "reuse_cache": config.reuse_cache,
        "force_split": config.force_split,
        "clear_saved_state": config.clear_saved_state,
        "chunks_folder": config.chunks_folder,
    }

    logger.info("--- Starting Run: %s ---", sanitized_run_id)
    logger.info("Configuration: %s", redacted_config)

    current_step = "setup"
    notify(current_step, "start", "Preparing environment")
    try:
        input_audio_paths = _resolve_audio_paths(config.input_audio_paths, config.root)
        os.makedirs(experiments_root, exist_ok=True)

        chunks_folder = (
            config.chunks_folder
            if config.chunks_folder
            else os.path.join(experiments_dir, f"chunks_{config.segment_duration}s")
        )
        diarization_results_run = os.path.join(experiments_dir, "diarization_results")
        output_folder_run = os.path.join(experiments_dir, "output_DOC")
        saved_state_dir_run = os.path.join(experiments_dir, "saved_state")
        plot_folder_run = os.path.join(experiments_dir, "plot")

        for directory in [
            chunks_folder,
            diarization_results_run,
            output_folder_run,
            saved_state_dir_run,
            plot_folder_run,
        ]:
            os.makedirs(directory, exist_ok=True)

        if config.clear_saved_state and os.path.isdir(saved_state_dir_run):
            logger.info("Clearing saved state directory at %s", saved_state_dir_run)
            shutil.rmtree(saved_state_dir_run)
            os.makedirs(saved_state_dir_run, exist_ok=True)

        if not config.reuse_cache and os.path.isdir(saved_state_dir_run):
            for file_name in os.listdir(saved_state_dir_run):
                if file_name.endswith(".pkl"):
                    file_path = os.path.join(saved_state_dir_run, file_name)
                    try:
                        os.remove(file_path)
                        logger.debug("Removed cache file: %s", file_path)
                    except OSError as exc:
                        logger.warning(
                            "Could not remove cache file %s: %s", file_path, exc
                        )

        # Invalidate cache if cache-relevant config fields changed since last run
        from src.config_builder import ensure_cache_consistency
        ensure_cache_consistency(config, saved_state_dir_run)

        # Raise CancelledError if the caller asked for a stop
        _check_cancel(config, logger, "setup")

        if not config.server_url or not config.api_key:
            raise ConfigurationError(
                "SERVER_URL and API_KEY must be provided for transcription."
            )

        notify(current_step, "success", "Environment ready, starting pipeline")

        logger.info("Input audio files: %s", input_audio_paths)
        logger.info("Chunks folder: %s", chunks_folder)
        logger.info("Saved state directory: %s", saved_state_dir_run)
        logger.info("Output folder: %s", output_folder_run)

        current_step = "setup"
        notify(current_step, "start", "Setting up GPU/CPU device and external services")
        device = setup_environment(config.hf_token)
        notify(
            current_step,
            "success",
            f"Environment setup complete. Using device: {device}",
            {"device": str(device)},
        )

        current_step = "split"
        notify(current_step, "start", "Verifying audio chunks")
        chunks_pickle = os.path.join(
            saved_state_dir_run, "step1_split_audio_result.pkl"
        )

        all_chunks_found = not config.force_split and os.path.isdir(chunks_folder)
        if all_chunks_found:
            for audio_path in input_audio_paths:
                base_audio_name = os.path.splitext(os.path.basename(audio_path))[0]
                expected_chunk_subdir = os.path.join(chunks_folder, base_audio_name)
                try:
                    if not os.path.isdir(expected_chunk_subdir) or not any(
                        file_name.endswith(".wav")
                        for file_name in os.listdir(expected_chunk_subdir)
                    ):
                        all_chunks_found = False
                        break
                except FileNotFoundError:
                    all_chunks_found = False
                    break

        if not all_chunks_found:
            if os.path.exists(chunks_pickle):
                os.remove(chunks_pickle)
            split_results = load_or_run(
                split_audio_into_chunks,
                args=(input_audio_paths, config.segment_duration, chunks_folder),
                pickle_path=chunks_pickle,
                description="Audio Splitting",
            )
            if not split_results or any(
                count is None or count == 0 for count in split_results
            ):
                notify(
                    current_step,
                    "error",
                    "Audio splitting failed. Check input files and logs.",
                )
                raise AudioSplittingError("Audio splitting failed.")
            notify(
                current_step,
                "success",
                f"Audio splitting completed with {sum(split_results)} chunks.",
                {"chunks_created": sum(split_results)},
            )
        else:
            notify(
                current_step,
                "skipped",
                "Existing chunks detected. Skipping splitting step.",
            )

        _check_cancel(config, logger, "diarization")
        current_step = "diarization"
        notify(current_step, "start", "Running diarization on chunks")
        diarization_pickle = os.path.join(
            saved_state_dir_run, "step2_diarization_merged_path.pkl"
        )
        if not config.reuse_cache and os.path.exists(diarization_pickle):
            os.remove(diarization_pickle)

        # Bridge run_diarization's chunk-level progress (and pyannote's
        # internal sub-step name) into the existing notify() stream so the
        # Streamlit bar fills smoothly during a long diarization instead of
        # staying at 0% until the step completes.
        def _diarization_progress(
            done: int, total: int, sub_step: Optional[str]
        ) -> None:
            label = f"Diarization · {done}/{total} chunks"
            if sub_step:
                label = f"{label} · {sub_step}"
            notify(
                "diarization",
                "in_progress",
                label,
                {"completed": done, "total": total, "sub_step": sub_step},
            )

        merged_diarization_path = load_or_run(
            run_diarization,
            args=(
                device,
                chunks_folder,
                diarization_results_run,
                config.hf_token,
                config.segment_duration,
                config.max_workers,
            ),
            kwargs={"progress_callback": _diarization_progress},
            pickle_path=diarization_pickle,
            description="Chunk Diarization",
        )
        if (
            not merged_diarization_path
            or not os.path.exists(merged_diarization_path)
            or os.path.getsize(merged_diarization_path) == 0
        ):
            # Remove corrupted cache so next run retries diarization from scratch
            if os.path.exists(diarization_pickle):
                try:
                    os.remove(diarization_pickle)
                    logger.warning("Removed empty/corrupt diarization cache: %s", diarization_pickle)
                except OSError:
                    pass
            notify(
                current_step,
                "error",
                "Diarization failed. Merged RTTM not found or empty.",
            )
            raise DiarizationError("Diarization failed.")
        notify(
            current_step,
            "success",
            "Diarization completed.",
            {"merged_rttm": merged_diarization_path},
        )

        current_step = "process_diarization"
        notify(current_step, "start", "Processing diarization results")
        diarization_df_pickle = os.path.join(
            saved_state_dir_run, "step3_diarization_df.pkl"
        )
        if not config.reuse_cache and os.path.exists(diarization_df_pickle):
            os.remove(diarization_df_pickle)

        diarization_df = load_or_run(
            process_diarization_results,
            args=(
                merged_diarization_path,
                diarization_results_run,
                config.segment_duration,
                config.duree_min_speaker,
            ),
            pickle_path=diarization_df_pickle,
            description="Diarization Results Processing",
        )
        if diarization_df.empty:
            notify(
                current_step,
                "error",
                "Processed diarization DataFrame is empty.",
            )
            raise DiarizationError("Diarization processing yielded empty DataFrame.")
        # Guard against stale caches that pre-date the 'chunks' column
        if "chunks" not in diarization_df.columns:
            logger.warning(
                "Cached diarization DataFrame is missing the 'chunks' column "
                "(stale cache). Recomputing from timestamps."
            )
            diarization_df["chunks"] = (
                diarization_df["start"] // config.segment_duration
            ).astype(int)
        notify(
            current_step,
            "success",
            "Diarization results processed.",
            {"segments": len(diarization_df)},
        )

        _check_cancel(config, logger, "clustering")
        current_step = "clustering"
        notify(current_step, "start", "Clustering speakers across chunks")
        clustering_results_pickle = os.path.join(
            saved_state_dir_run, "step4_clustering_results.pkl"
        )
        if not config.reuse_cache and os.path.exists(clustering_results_pickle):
            os.remove(clustering_results_pickle)

        clustering_results = load_or_run(
            cluster_speakers,
            args=(
                diarization_df,
                device,
                chunks_folder,
                config.segment_duration,
                plot_folder_run,
            ),
            kwargs={
                "num_speakers": config.num_speakers,
                "min_speakers": config.min_speakers,
                "max_speakers": config.max_speakers,
            },
            pickle_path=clustering_results_pickle,
            description="Speaker Clustering",
        )
        if not clustering_results or clustering_results[0] is None:
            notify(
                current_step,
                "error",
                "Speaker clustering failed. No mapping produced.",
            )
            raise ClusteringError("Speaker clustering failed.")
        mapping_hdbscan = clustering_results[0]
        notify(
            current_step,
            "success",
            "Speaker clustering complete.",
            {"clusters": len({cid for cid in mapping_hdbscan.values() if cid != -1})},
        )

        current_step = "label_update"
        notify(current_step, "start", "Updating speaker labels")
        updated_data_pickle = os.path.join(
            saved_state_dir_run, "step5_updated_data.pkl"
        )
        if not config.reuse_cache and os.path.exists(updated_data_pickle):
            os.remove(updated_data_pickle)

        updated_data_results = load_or_run(
            update_speaker_labels,
            args=(diarization_df, mapping_hdbscan, diarization_results_run),
            kwargs={"segment_duration": config.segment_duration},
            pickle_path=updated_data_pickle,
            description="Speaker Label Update",
        )
        if not updated_data_results or updated_data_results[0].empty:
            notify(
                current_step,
                "error",
                "Updated speaker DataFrame is empty.",
            )
            raise ClusteringError("Speaker label update failed.")
        updated_data = updated_data_results[0]
        # Guard against stale caches missing 'chunks'
        if "chunks" not in updated_data.columns:
            logger.warning(
                "Cached updated_data DataFrame is missing the 'chunks' column "
                "(stale cache). Recomputing from timestamps."
            )
            updated_data["chunks"] = (
                updated_data["start"] // config.segment_duration
            ).astype(int)
        notify(
            current_step,
            "success",
            "Speaker labels updated.",
            {"segments": len(updated_data)},
        )

        _check_cancel(config, logger, "transcription")
        current_step = "transcription"
        notify(current_step, "start", "Transcribing segments with Whisper")
        transcription_pickle = os.path.join(
            saved_state_dir_run, "step6_transcribed_data.pkl"
        )
        if not config.reuse_cache and os.path.exists(transcription_pickle):
            os.remove(transcription_pickle)

        def _transcription_progress(done: int, total: int, current: str) -> None:
            # Only emit every 5 segments or at the end to avoid flooding the UI
            if total and (done % 5 == 0 or done == total):
                notify(
                    "transcription",
                    "progress",
                    f"Transcription {done}/{total} — {current[:40]}",
                    {"done": done, "total": total},
                )

        def _transcribe_wrapper(*args):
            return transcribe_all_segments(
                *args,
                progress_callback=_transcription_progress,
                cancel_event=config.cancel_event,
            )

        transcribed_data = load_or_run(
            _transcribe_wrapper,
            args=(
                updated_data,
                chunks_folder,
                config.server_url,
                config.api_key,
                config.whisper_model,
                config.segment_duration,
                config.duree_min_speaker,
                config.vad_filter,
            ),
            pickle_path=transcription_pickle,
            description="Transcription",
        )
        if transcribed_data.empty or "transcription" not in transcribed_data.columns:
            notify(
                current_step,
                "error",
                "Transcription failed or produced empty results.",
            )
            raise TranscriptionError("Transcription failed.")
        notify(
            current_step,
            "success",
            "Transcription completed.",
            {
                "transcribed_segments": int(
                    transcribed_data["transcription"].astype(bool).sum()
                )
            },
        )

        current_step = "speaker_identification"
        notify(current_step, "skipped", "Speaker identification will run after cleaning.")
        speaker_info_dict: Dict[str, SpeakerInfo] = {}
        speaker_id_path: Optional[str] = None

        _check_cancel(config, logger, "cleaning")
        current_step = "cleaning"
        cleaning_pickle = os.path.join(saved_state_dir_run, "step7_cleaned_data.pkl")
        if not config.reuse_cache and os.path.exists(cleaning_pickle):
            os.remove(cleaning_pickle)

        if config.enable_llm_cleaning and config.llm_base_url and config.llm_model:
            notify(current_step, "start", "Cleaning transcriptions with LLM")
            cleaned_data = load_or_run(
                process_all_text,
                args=(
                    transcribed_data,
                    config.api_key,
                    config.llm_base_url,
                    config.llm_model,
                ),
                pickle_path=cleaning_pickle,
                description="Text Cleaning",
            )
            if "cleaned_transcription" not in cleaned_data.columns:
                cleaned_data["cleaned_transcription"] = cleaned_data["transcription"]
                logger.warning(
                    "Text cleaning missing 'cleaned_transcription'. Using raw transcriptions."
                )
            notify(
                current_step,
                "success",
                "Text cleaning finished.",
                {"cleaned_segments": len(cleaned_data)},
            )
        else:
            # LLM cleaning disabled — copy raw transcript into cleaned_transcription column
            # so downstream steps (export, speaker ID, summary) can read it uniformly.
            reason = (
                "LLM cleaning disabled by user."
                if not config.enable_llm_cleaning
                else "LLM cleaning skipped — missing llm_base_url or llm_model."
            )
            logger.info(reason)
            cleaned_data = transcribed_data.copy()
            cleaned_data["cleaned_transcription"] = cleaned_data["transcription"]
            notify(current_step, "skipped", reason)

        # Speaker Identification — runs on cleaned transcript for best quality
        current_step = "speaker_identification"
        if config.enable_speaker_identification:
            notify(current_step, "start", "Identifying speakers via LLM (full diarized transcript)")
            speaker_id_pickle = os.path.join(
                saved_state_dir_run, "step7_5_speaker_identification.pkl"
            )
            if not config.reuse_cache and os.path.exists(speaker_id_pickle):
                os.remove(speaker_id_pickle)

            def _load_speaker_cache(path: str) -> Optional[Dict[str, SpeakerInfo]]:
                """Load speaker cache with backward compat for old pickle format."""
                try:
                    with open(path, "rb") as f:
                        raw = pickle.load(f)
                    if not raw:
                        return {}
                    # Check if it's dict-of-dicts (new format) or dict-of-SpeakerInfo (old format)
                    first_val = next(iter(raw.values()))
                    if isinstance(first_val, dict):
                        return {sid: SpeakerInfo(**data) for sid, data in raw.items()}
                    # Old format: already SpeakerInfo objects, return as-is
                    return raw
                except Exception as e:
                    logging.warning("Failed to load speaker ID cache: %s", e)
                    return None

            def _save_speaker_cache(result: Dict[str, SpeakerInfo], path: str) -> None:
                """Save as plain dicts to avoid dataclass pickle issues."""
                os.makedirs(os.path.dirname(path), exist_ok=True)
                dict_data = {sid: asdict(info) for sid, info in result.items()}
                with open(path, "wb") as f:
                    pickle.dump(dict_data, f)

            speaker_info_dict: Dict[str, SpeakerInfo] = {}
            if os.path.exists(speaker_id_pickle):
                cached = _load_speaker_cache(speaker_id_pickle)
                if cached is not None:
                    speaker_info_dict = cached
                    logging.info("Successfully loaded Speaker Identification from cache.")

            if not speaker_info_dict:
                speaker_info_dict = (
                    identify_speakers(
                        cleaned_data,
                        config.llm_base_url or config.server_url,
                        config.api_key,
                        config.speaker_identification_model or config.llm_model,
                        None,
                        120,
                    )
                    or {}
                )
                if speaker_info_dict:
                    try:
                        _save_speaker_cache(speaker_info_dict, speaker_id_pickle)
                        logging.info(
                            "Successfully saved Speaker Identification results to cache: %s",
                            speaker_id_pickle,
                        )
                    except Exception as e:
                        logging.warning("Failed to save speaker ID cache: %s", e)

            speaker_id_path = os.path.join(output_folder_run, "speaker_identification.json")
            save_speaker_identification(speaker_info_dict, speaker_id_path)

            notify(
                current_step,
                "success",
                f"Identified {len(speaker_info_dict)} speakers.",
                {"identified_speakers": len(speaker_info_dict)},
            )
        else:
            notify(current_step, "skipped", "Speaker identification disabled.")

        current_step = "timestamp_adjust"
        if config.audio_processing_mode == "sequential" and len(input_audio_paths) > 1:
            notify(
                current_step,
                "start",
                "Adjusting timestamps for sequential audio files",
            )
            cleaned_data = adjust_timestamps_for_sequential_audio(
                cleaned_data, input_audio_paths
            )
            adjustment_pickle = os.path.join(
                saved_state_dir_run, "step7_5_final_adjusted_data.pkl"
            )
            cleaned_data.to_pickle(adjustment_pickle)
            notify(
                current_step,
                "success",
                "Timestamps adjusted for sequential audio.",
            )
        else:
            cleaned_data = cleaned_data.sort_values(by="start").reset_index(drop=True)
            notify(
                current_step,
                "skipped",
                "Concurrent mode: timestamps sorted chronologically.",
            )

        _check_cancel(config, logger, "export")
        current_step = "export"
        notify(current_step, "start", "Exporting results to DOCX")
        try:
            exported_df = export_results(
                cleaned_data,
                input_audio_paths,
                output_folder_run,
                speaker_info={sid: asdict(info) for sid, info in speaker_info_dict.items()}
                if config.enable_speaker_identification and speaker_info_dict
                else None,
            )
        except Exception as exc:
            notify(
                current_step,
                "error",
                f"Export failed: {exc}",
            )
            raise ExportError(f"Export failed: {exc}", cause=exc) from exc

        docx_path = os.path.join(output_folder_run, "Transcription_Final.docx")
        if not os.path.exists(docx_path):
            docx_path = None
            logger.warning("DOCX export file not found.")
        notify(
            current_step,
            "success",
            "Export completed.",
            {"docx_path": docx_path},
        )

        # Meeting Minutes Generation
        current_step = "meeting_minutes"
        if config.enable_meeting_minutes:
            notify(current_step, "start", "Generating meeting minutes with LLM")
            minutes_pickle = os.path.join(
                saved_state_dir_run, "step8_meeting_minutes.pkl"
            )
            if not config.reuse_cache and os.path.exists(minutes_pickle):
                os.remove(minutes_pickle)

            def _load_minutes_cache(path: str) -> Optional[MeetingMinutes]:
                """Load meeting minutes cache with backward compat for old pickle format."""
                try:
                    with open(path, "rb") as f:
                        raw = pickle.load(f)
                    if raw is None:
                        return None
                    # New format: plain dict
                    if isinstance(raw, dict):
                        discussions = [
                            MeetingMinuteSection(**d) for d in raw.get("discussions", [])
                        ]
                        return MeetingMinutes(
                            titre=raw.get("titre", "Compte rendu de réunion"),
                            format_used=raw.get("format_used", DEFAULT_FORMAT),
                            date=raw.get("date"),
                            lieux=raw.get("lieux"),
                            participants=raw.get("participants", []),
                            ordre_du_jour=raw.get("ordre_du_jour", []),
                            discussions=discussions,
                            decisions=raw.get("decisions", []),
                            actions=raw.get("actions", []),
                            prochaine_reunion=raw.get("prochaine_reunion"),
                        )
                    # Old format: already MeetingMinutes object, return as-is
                    return raw
                except Exception as e:
                    logging.warning("Failed to load meeting minutes cache: %s", e)
                    return None

            def _save_minutes_cache(mm: MeetingMinutes, path: str) -> None:
                """Save as plain dict to avoid dataclass pickle issues."""
                os.makedirs(os.path.dirname(path), exist_ok=True)
                dict_data = {
                    "titre": mm.titre,
                    "format_used": mm.format_used,
                    "date": mm.date,
                    "lieux": mm.lieux,
                    "participants": mm.participants,
                    "ordre_du_jour": mm.ordre_du_jour,
                    "discussions": [{"title": d.title, "content": d.content} for d in mm.discussions],
                    "decisions": mm.decisions,
                    "actions": mm.actions,
                    "prochaine_reunion": mm.prochaine_reunion,
                }
                with open(path, "wb") as f:
                    pickle.dump(dict_data, f)

            meeting_minutes_obj: Optional[MeetingMinutes] = None
            if os.path.exists(minutes_pickle):
                cached = _load_minutes_cache(minutes_pickle)
                if cached is not None:
                    meeting_minutes_obj = cached
                    logging.info("Successfully loaded Meeting Minutes from cache.")

            if meeting_minutes_obj is None:
                meeting_minutes_obj = generate_meeting_minutes(
                    cleaned_data,
                    config.llm_base_url or config.server_url,
                    config.api_key,
                    config.meeting_minutes_model or config.llm_model,
                    {sid: asdict(info) for sid, info in speaker_info_dict.items()}
                    if speaker_info_dict
                    else None,
                    None,
                    config.meeting_minutes_format,
                    config.meeting_minutes_instructions,
                    180,
                )
                if meeting_minutes_obj:
                    try:
                        _save_minutes_cache(meeting_minutes_obj, minutes_pickle)
                        logging.info(
                            "Successfully saved Meeting Minutes results to cache: %s",
                            minutes_pickle,
                        )
                    except Exception as e:
                        logging.warning("Failed to save meeting minutes cache: %s", e)

            if meeting_minutes_obj:
                saved_minutes = save_meeting_minutes(
                    meeting_minutes_obj,
                    output_folder_run,
                    f"meeting_minutes_{sanitized_run_id}",
                )
                notify(
                    current_step,
                    "success",
                    "Meeting minutes generated.",
                    {"files": list(saved_minutes.keys())},
                )
            else:
                notify(
                    current_step,
                    "warning",
                    "Meeting minutes generation returned no result.",
                )
        else:
            notify(current_step, "skipped", "Meeting minutes generation disabled.")
            meeting_minutes_obj = None

        current_step = "summary"
        notify(current_step, "start", "Compiling summary and metrics")

        # --- GENERATE SUMMARY FOR FULL PIPELINE ---
        summary_text: Optional[str] = None
        summary_path: Optional[str] = None

        try:
            if config.enable_summary and config.llm_model and config.llm_base_url:
                logger.info("Generating summary for full pipeline results...")
                # Gather all text (filter NaNs and convert to string)
                full_text = " ".join(
                    cleaned_data["cleaned_transcription"].dropna().astype(str).tolist()
                )
                if full_text.strip():
                    summary_text = summarise_text(
                        full_text, config.api_key, config.llm_base_url, config.llm_model
                    )
                    summary_path = os.path.join(output_folder_run, "Summary.txt")
                    with open(summary_path, "w", encoding="utf-8") as f:
                        f.write(summary_text)
                    logger.info(f"Summary saved to {summary_path}")
                else:
                    logger.warning("No text available to summarize.")
        except Exception as exc:
            logger.error(f"Failed to generate summary: {exc}")

        num_speakers_final: Optional[int] = None
        try:
            if not cleaned_data.empty and "global_speaker" in cleaned_data.columns:
                num_speakers_final = cleaned_data[
                    cleaned_data["global_speaker"] != "Noise"
                ]["global_speaker"].nunique()
                logger.info(
                    "Detected %s unique non-noise speakers.",
                    num_speakers_final,
                )
            else:
                logger.warning("Unable to determine number of speakers (missing data).")
        except Exception as exc:
            logger.warning("Error computing speaker count: %s", exc)

        experiments_csv = os.path.join(experiments_root, "experiments_log.csv")
        write_header = not os.path.exists(experiments_csv)
        base_audio_name_combined = "+".join(
            [os.path.splitext(os.path.basename(path))[0] for path in input_audio_paths]
        )
        try:
            with open(experiments_csv, "a", newline="", encoding="utf-8") as file_obj:
                writer = csv.writer(file_obj)
                if write_header:
                    writer.writerow(
                        [
                            "run_id",
                            "timestamp",
                            "input_audio_combined",
                            "segment_duration",
                            "min_speaker_duration",
                            "vad_filter",
                            "num_speakers_final",
                            "status",
                        ]
                    )
                writer.writerow(
                    [
                        sanitized_run_id,
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        base_audio_name_combined,
                        config.segment_duration,
                        config.duree_min_speaker,
                        config.vad_filter,
                        num_speakers_final if num_speakers_final is not None else "N/A",
                        "Completed",
                    ]
                )
            logger.info("Experiment details logged to %s", experiments_csv)
        except Exception as exc:
            logger.error("Failed to log experiment details: %s", exc)

        notify(
            current_step,
            "success",
            "Pipeline run completed successfully.",
            {
                "num_speakers_final": num_speakers_final,
                "segments": len(cleaned_data),
            },
        )
        logger.info("--- Run %s Completed ---", sanitized_run_id)

        return PipelineResult(
            success=True,
            run_id=sanitized_run_id,
            experiments_dir=os.path.abspath(experiments_dir),
            log_file=log_file,
            cleaned_data=cleaned_data,
            exported_df=exported_df,
            docx_path=docx_path,
            summary_path=summary_path,
            summary_text=summary_text,
            num_speakers_final=num_speakers_final,
            message="Completed",
            # Speaker identification
            speaker_identification_path=speaker_id_path
            if config.enable_speaker_identification
            else None,
            speaker_info={sid: asdict(info) for sid, info in speaker_info_dict.items()}
            if speaker_info_dict
            else {},
            # Meeting minutes
            meeting_minutes_paths=saved_minutes
            if config.enable_meeting_minutes and meeting_minutes_obj
            else {},
            meeting_minutes=meeting_minutes_obj,
            chunks_folder=chunks_folder,
            segment_duration=config.segment_duration,
        )

    except CancelledError as exc:
        logger.warning("Pipeline cancelled: %s", exc)
        notify(current_step, "cancelled", str(exc))
        raise
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        notify(current_step, "error", str(exc))
        raise


def main() -> None:
    result: Optional[PipelineResult] = None
    config: Optional[PipelineConfig] = None
    try:
        config = load_config_from_env()
        result = run_pipeline(config)
        logging.getLogger().info(
            "Pipeline finished successfully. DOCX: %s", result.docx_path
        )
    except PipelineError as exc:
        logging.getLogger().error("Pipeline failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logging.getLogger().exception("Unexpected error: %s", exc)
        sys.exit(1)
    finally:
        if (
            config
            and config.auto_delete_outputs
            and result
            and result.experiments_dir
            and os.path.isdir(result.experiments_dir)
        ):
            logging.getLogger().info(
                "Auto-deleting experiments directory: %s",
                result.experiments_dir,
            )
            try:
                shutil.rmtree(result.experiments_dir, ignore_errors=True)
            except Exception as exc:
                logging.getLogger().warning(
                    "Failed to auto-delete experiments directory %s: %s",
                    result.experiments_dir,
                    exc,
                )


if __name__ == "__main__":
    main()
