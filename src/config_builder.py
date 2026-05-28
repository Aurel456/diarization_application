"""
Shared PipelineConfig builder used by every entry point (CLI, Streamlit, API).

The previous code duplicated this logic in three places which drifted over time.
One source of truth, one set of defaults, same validation everywhere.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

from settings import settings
from src.errors import ConfigurationError


def _coerce_log_level(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return getattr(logging, value.upper(), logging.INFO)
    return logging.INFO


def build_pipeline_config(
    audio_paths: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> "PipelineConfig":
    """
    Create a PipelineConfig from application settings, overridden by `overrides`.

    Args:
        audio_paths: List of audio file paths to process. Required for execution
                     but accepted empty for config-only construction (e.g. API before upload).
        overrides:   Partial dict of PipelineConfig field names with values that
                     take precedence over settings defaults.

    Returns:
        A PipelineConfig instance.
    """
    # Import lazily to avoid circular import with main_pipeline
    from main_pipeline import PipelineConfig
    from src.meeting_minutes import DEFAULT_FORMAT

    overrides = overrides or {}
    audio_paths = audio_paths or []

    cfg = PipelineConfig(
        root=str(overrides.get("root", settings.root)),
        hf_token=overrides.get("hf_token", settings.hf_token),
        audio_processing_mode=overrides.get(
            "audio_processing_mode", overrides.get("processing_mode", "sequential")
        ),
        segment_duration=int(overrides.get("segment_duration", settings.segment_duration)),
        max_workers=int(overrides.get("max_workers", settings.max_workers)),
        duree_min_speaker=float(
            overrides.get("duree_min_speaker", overrides.get("min_speaker_duration", settings.min_speaker_duration))
        ),
        vad_filter=bool(overrides.get("vad_filter", settings.vad_filter)),
        input_audio_paths=list(audio_paths),
        server_url=overrides.get("server_url", settings.server_url) or None,
        whisper_model=overrides.get("whisper_model", settings.whisper_model) or "whisper",
        api_key=overrides.get("api_key", settings.api_key) or None,
        language=overrides.get("language", settings.language) or "fr",
        llm_model=overrides.get("llm_model", settings.llm_model),
        # Single endpoint: llm_base_url defaults to server_url unless explicitly overridden.
        llm_base_url=overrides.get(
            "llm_base_url",
            overrides.get("server_url", settings.server_url) or None,
        ),
        run_id=overrides.get("run_id"),
        reuse_cache=bool(overrides.get("reuse_cache", settings.default_reuse_cache)),
        force_split=bool(overrides.get("force_split", settings.default_force_split)),
        clear_saved_state=bool(overrides.get("clear_saved_state", settings.default_clear_saved_state)),
        log_level=_coerce_log_level(overrides.get("log_level", settings.default_log_level)),
        log_to_console=bool(overrides.get("log_to_console", settings.log_to_console)),
        chunks_folder=overrides.get("chunks_folder"),
        simple_mode=bool(overrides.get("simple_mode", False)),
        chunk_size=int(overrides.get("chunk_size", settings.chunk_size)),
        enable_llm_cleaning=bool(
            overrides.get("enable_llm_cleaning", settings.enable_llm_cleaning)
        ),
        enable_summary=bool(
            overrides.get("enable_summary", settings.enable_summary)
        ),
        num_speakers=overrides.get("num_speakers", settings.num_speakers),
        min_speakers=overrides.get("min_speakers", settings.min_speakers),
        max_speakers=overrides.get("max_speakers", settings.max_speakers),
        experiments_root=overrides.get("experiments_root", settings.experiments_root),
        auto_delete_outputs=bool(
            overrides.get("auto_delete_outputs", settings.auto_delete_outputs)
        ),
        cleanup_grace_seconds=int(
            overrides.get("cleanup_grace_seconds", settings.cleanup_grace_seconds)
        ),
        enable_speaker_identification=bool(
            overrides.get(
                "enable_speaker_identification", settings.enable_speaker_identification
            )
        ),
        speaker_identification_model=overrides.get(
            "speaker_identification_model", settings.speaker_identification_model
        ),
        enable_meeting_minutes=bool(
            overrides.get("enable_meeting_minutes", settings.enable_meeting_minutes)
        ),
        meeting_minutes_model=overrides.get(
            "meeting_minutes_model", settings.meeting_minutes_model
        ),
        meeting_minutes_format=overrides.get("meeting_minutes_format", DEFAULT_FORMAT),
        meeting_minutes_instructions=overrides.get("meeting_minutes_instructions"),
        stateless=bool(overrides.get("stateless", settings.stateless)),
        cancel_event=overrides.get("cancel_event"),
    )

    if cfg.stateless:
        # Stateless implies: don't reuse cache, auto-delete outputs
        cfg.reuse_cache = False
        cfg.clear_saved_state = True
        cfg.auto_delete_outputs = True

    return cfg


# ---------------------------------------------------------------------------
# Config hashing — invalidate caches when relevant parameters change
# ---------------------------------------------------------------------------

# Fields whose change should invalidate the pickle cache.
# Kept deliberately narrow so purely cosmetic changes (log level, cleanup)
# don't trigger recomputation.
_CACHE_RELEVANT_FIELDS: tuple[str, ...] = (
    "audio_processing_mode",
    "segment_duration",
    "max_workers",
    "duree_min_speaker",
    "vad_filter",
    "input_audio_paths",
    "server_url",
    "whisper_model",
    "language",
    "llm_model",
    "llm_base_url",
    "enable_llm_cleaning",
    "enable_summary",
    "num_speakers",
    "min_speakers",
    "max_speakers",
    "enable_speaker_identification",
    "speaker_identification_model",
    "enable_meeting_minutes",
    "meeting_minutes_model",
    "meeting_minutes_format",
)


def compute_config_hash(config: "PipelineConfig") -> str:
    """Short deterministic hash of cache-relevant PipelineConfig fields."""
    payload: Dict[str, Any] = {}
    cfg_dict = asdict(config)
    for field in _CACHE_RELEVANT_FIELDS:
        value = cfg_dict.get(field)
        # Normalise paths so absolute/relative variants hash the same
        if field == "input_audio_paths" and isinstance(value, list):
            value = [os.path.basename(p) for p in value]
        payload[field] = value
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def ensure_cache_consistency(config: "PipelineConfig", saved_state_dir: str) -> None:
    """
    Ensure the pickle cache in `saved_state_dir` matches the current config.

    If the stored config hash does not match, all *.pkl files are deleted so
    the next run recomputes from scratch. In stateless mode this function is
    a no-op because reuse_cache is already False.
    """
    if config.stateless or not config.reuse_cache:
        return

    os.makedirs(saved_state_dir, exist_ok=True)
    hash_file = os.path.join(saved_state_dir, ".config_hash")
    current = compute_config_hash(config)
    previous = None
    if os.path.exists(hash_file):
        try:
            with open(hash_file, "r", encoding="utf-8") as f:
                previous = f.read().strip()
        except OSError:
            previous = None

    if previous and previous != current:
        logging.info(
            "Config changed (%s -> %s). Invalidating pickle cache in %s.",
            previous, current, saved_state_dir,
        )
        for name in os.listdir(saved_state_dir):
            if name.endswith(".pkl"):
                try:
                    os.remove(os.path.join(saved_state_dir, name))
                except OSError as exc:
                    logging.warning("Could not remove stale cache %s: %s", name, exc)

    try:
        with open(hash_file, "w", encoding="utf-8") as f:
            f.write(current)
    except OSError as exc:
        logging.warning("Could not write config hash file %s: %s", hash_file, exc)


def apply_cli_overrides(
    config: "PipelineConfig",
    overrides: Dict[str, Any],
) -> "PipelineConfig":
    """Apply CLI-style overrides to an existing PipelineConfig (in place)."""
    for name, value in overrides.items():
        if value is None:
            continue
        if not hasattr(config, name):
            continue
        setattr(config, name, value)
    return config
