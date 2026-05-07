#!/usr/bin/env python3
"""
run.py — CLI entry point for the audio diarization/transcription pipeline.

Usage examples:
  # Run from .env (all config in environment)
  python run.py

  # Specify audio files directly
  python run.py --audio meeting.mp4

  # Multiple files (sequential mode)
  python run.py --audio part1.mp4 part2.mp4 --mode sequential

  # Enable speaker identification and meeting minutes
  python run.py --audio meeting.mp4 --speaker-id --meeting-minutes

  # Override server URL (single endpoint for Whisper + LLM)
  python run.py --audio meeting.mp4 --server-url http://gpu-server:8000/v1

  # Simple transcription mode (no diarization)
  python run.py --audio meeting.mp4 --simple

  # Start as FastAPI server
  python run.py --serve
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv(".env")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Audio diarization, transcription and meeting minutes pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input ---
    p.add_argument(
        "--audio",
        nargs="+",
        metavar="FILE",
        help="Audio/video file(s) to process. Overrides INPUT_AUDIO from .env.",
    )
    p.add_argument(
        "--root",
        default=None,
        metavar="DIR",
        help="Root directory for resolving relative audio paths.",
    )
    p.add_argument(
        "--run-id",
        default=None,
        metavar="ID",
        help="Unique run identifier (used as output folder name).",
    )

    # --- Pipeline mode ---
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--simple",
        action="store_true",
        help="Simple transcription mode: no diarization, single file only.",
    )
    mode_group.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI server instead of running the pipeline.",
    )

    p.add_argument(
        "--mode",
        choices=["sequential", "concurrent"],
        default=None,
        help="Audio processing mode when multiple files are given.",
    )

    # --- Diarization ---
    p.add_argument(
        "--segment-duration",
        type=int,
        default=None,
        metavar="SEC",
        help="Chunk duration in seconds for diarization.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel workers for diarization.",
    )
    p.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Voice Activity Detection filter.",
    )

    # --- Transcription server ---
    p.add_argument(
        "--server-url",
        default=None,
        metavar="URL",
        help="Whisper API server URL (e.g. http://gpu:8000/v1).",
    )
    p.add_argument(
        "--whisper-model",
        default=None,
        metavar="NAME",
        help="Whisper model name as deployed on the server.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="API key for the Whisper/LLM server.",
    )

    # --- LLM (uses --server-url) ---
    p.add_argument(
        "--llm-model",
        default=None,
        metavar="NAME",
        help="LLM model name for cleaning / speaker ID / meeting minutes.",
    )
    p.add_argument(
        "--no-cleaning",
        action="store_true",
        help="Disable LLM transcription cleaning step.",
    )

    # --- Optional features ---
    p.add_argument(
        "--speaker-id",
        action="store_true",
        help="Enable LLM-based speaker identification.",
    )
    p.add_argument(
        "--speaker-id-model",
        default=None,
        metavar="NAME",
        help="Specific LLM model for speaker identification (defaults to --llm-model).",
    )
    p.add_argument(
        "--meeting-minutes",
        action="store_true",
        help="Enable LLM-based meeting minutes generation.",
    )
    p.add_argument(
        "--meeting-minutes-model",
        default=None,
        metavar="NAME",
        help="Specific LLM model for meeting minutes (defaults to --llm-model).",
    )
    p.add_argument(
        "--meeting-minutes-format",
        default=None,
        metavar="KEY",
        choices=["standard", "executif", "technique", "projet", "rh_social", "formation"],
        help="Meeting minutes format template.",
    )

    # --- Cache / output ---
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached intermediate results and recompute everything.",
    )
    p.add_argument(
        "--experiments-dir",
        default=None,
        metavar="DIR",
        help="Root directory for experiment outputs.",
    )

    # --- Server mode ---
    p.add_argument("--host", default="0.0.0.0", help="FastAPI server host.")
    p.add_argument("--port", type=int, default=8000, help="FastAPI server port.")

    # --- Logging ---
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity level.",
    )

    return p


def run_server(host: str, port: int) -> None:
    import uvicorn
    uvicorn.run("api:app", host=host, port=port, reload=False)


def run_pipeline_cli(args: argparse.Namespace) -> None:
    from main_pipeline import run_pipeline
    from src.config_builder import build_pipeline_config
    from src.errors import PipelineError

    # CLI overrides — skip None values so settings defaults apply
    overrides = {
        "root": os.path.abspath(args.root) if args.root else None,
        "run_id": args.run_id,
        "audio_processing_mode": args.mode,
        "segment_duration": args.segment_duration,
        "max_workers": args.max_workers,
        "vad_filter": False if args.no_vad else None,
        "server_url": args.server_url,
        "whisper_model": args.whisper_model,
        "api_key": args.api_key,
        "llm_model": args.llm_model,
        "enable_llm_cleaning": False if args.no_cleaning else None,
        "enable_speaker_identification": True if args.speaker_id else None,
        "speaker_identification_model": args.speaker_id_model,
        "enable_meeting_minutes": True if args.meeting_minutes else None,
        "meeting_minutes_model": args.meeting_minutes_model,
        "meeting_minutes_format": getattr(args, "meeting_minutes_format", None),
        "reuse_cache": False if args.no_cache else None,
        "experiments_root": os.path.abspath(args.experiments_dir) if args.experiments_dir else None,
        "simple_mode": True if args.simple else None,
        "log_level": args.log_level,
        "log_to_console": True,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    audio_paths = args.audio or []
    try:
        config = build_pipeline_config(audio_paths=audio_paths, overrides=overrides)
    except PipelineError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not config.input_audio_paths:
        print("ERROR: No audio files specified. Use --audio FILE or set INPUT_AUDIO in .env")
        sys.exit(1)

    result = run_pipeline(config)

    print("\n=== Pipeline completed ===")
    print(f"Run ID       : {result.run_id}")
    print(f"Output dir   : {result.experiments_dir}")
    if result.docx_path:
        print(f"DOCX         : {result.docx_path}")
    if result.txt_path:
        print(f"TXT          : {result.txt_path}")
    if result.srt_path:
        print(f"SRT          : {result.srt_path}")
    if result.speaker_identification_path:
        print(f"Speaker ID   : {result.speaker_identification_path}")
    if result.meeting_minutes_paths:
        for fmt, path in result.meeting_minutes_paths.items():
            print(f"Minutes ({fmt:4s}): {path}")
    if result.num_speakers_final is not None:
        print(f"Speakers     : {result.num_speakers_final}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.serve:
        run_server(args.host, args.port)
    else:
        run_pipeline_cli(args)


if __name__ == "__main__":
    main()
