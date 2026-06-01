"""
Centralised application configuration via Pydantic settings.

All values come from environment variables (or `.env` file) with typed
validation and sensible defaults that vary by environment
(development vs. production).

Usage:
    from settings import settings, AppEnv
    settings.server_url
    settings.is_prod
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Centralised application configuration with environment-aware defaults.

    Every field reads from env with an `alias_priority` list to support legacy
    variable names. Defaults that depend on environment (prod vs dev) are
    applied in the post-init validator.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # --- Core ---
    app_env: AppEnv = Field(default=AppEnv.DEVELOPMENT, alias="APP_ENV")
    root: str = Field(default=".", alias="ROOT")
    experiments_root: Optional[str] = Field(default=None, alias="EXPERIMENTS_ROOT")

    # Stateless mode (for Docker): skip pickle cache, put everything in tmp dirs
    stateless: bool = Field(default=False, alias="APP_STATELESS")

    # --- Logging ---
    default_log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")
    log_to_console: Optional[bool] = Field(default=None, alias="APP_LOG_TO_CONSOLE")
    hide_log_panel: Optional[bool] = Field(default=None, alias="APP_HIDE_LOG_PANEL")

    # --- Transcription server ---
    server_url: str = Field(default="", alias="SERVER_URL")
    api_key: str = Field(default="", alias="API_KEY")
    whisper_model: str = Field(default="whisper", alias="WHISPER_MODEL")
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")
    # Audio language for Whisper (ISO 639-1). Empty / "auto" disables the hint
    # so the server auto-detects — required for endpoints (cohere-transcribe…)
    # that reject the `language` parameter.
    language: str = Field(default="fr", alias="APP_LANGUAGE")

    # --- LLM ---
    # NB: `llm_base_url` was removed as a separate setting — it's now derived from
    # `server_url` via a property below. A single SERVER_URL is used for both
    # Whisper transcription and LLM calls (cleaning / summary / speaker ID / minutes).
    llm_model: Optional[str] = Field(default=None, alias="LLM_MODEL")
    enable_llm_cleaning: Optional[bool] = Field(
        default=None, alias="APP_ENABLE_LLM_CLEANING"
    )
    # Summary is now independent from cleaning (was previously controlled by the same flag)
    enable_summary: Optional[bool] = Field(default=None, alias="APP_ENABLE_SUMMARY")

    # --- Pipeline parameters ---
    chunk_size: int = Field(default=600, alias="APP_CHUNK_SIZE")
    # Smaller default for SRT mode — finer subtitle granularity matters more
    # than long-form context for word-level alignment.
    srt_chunk_size: int = Field(default=35, alias="APP_SRT_CHUNK_SIZE")
    segment_duration: int = Field(default=1200, alias="APP_SEGMENT_DURATION")
    min_speaker_duration: float = Field(default=0.5, alias="APP_MIN_SPEAKER_DURATION")
    max_workers: int = Field(default=2, alias="APP_MAX_WORKERS")
    vad_filter: bool = Field(default=True, alias="APP_VAD_FILTER")

    # --- Speaker count hints (optional — improve clustering when known) ---
    num_speakers: Optional[int] = Field(default=None, alias="APP_NUM_SPEAKERS")
    min_speakers: Optional[int] = Field(default=None, alias="APP_MIN_SPEAKERS")
    max_speakers: Optional[int] = Field(default=None, alias="APP_MAX_SPEAKERS")

    # --- Speaker identification ---
    enable_speaker_identification: bool = Field(
        default=False, alias="APP_ENABLE_SPEAKER_IDENTIFICATION"
    )
    speaker_identification_model: Optional[str] = Field(
        default=None, alias="SPEAKER_IDENTIFICATION_MODEL"
    )

    # --- Meeting minutes ---
    enable_meeting_minutes: bool = Field(default=False, alias="APP_ENABLE_MEETING_MINUTES")
    meeting_minutes_model: Optional[str] = Field(
        default=None, alias="MEETING_MINUTES_MODEL"
    )

    # --- Caching & cleanup ---
    auto_delete_uploads: Optional[bool] = Field(default=None, alias="APP_AUTO_DELETE_UPLOADS")
    auto_delete_outputs: Optional[bool] = Field(default=None, alias="APP_AUTO_DELETE_OUTPUTS")
    cleanup_grace_seconds: int = Field(default=10800, alias="APP_CLEANUP_GRACE_SECONDS")

    default_reuse_cache: Optional[bool] = Field(default=None, alias="APP_REUSE_CACHE")
    default_force_split: Optional[bool] = Field(default=None, alias="APP_FORCE_SPLIT")
    default_clear_saved_state: Optional[bool] = Field(default=None, alias="APP_CLEAR_SAVED_STATE")

    # --- Validators ---

    @field_validator("app_env", mode="before")
    @classmethod
    def _normalize_env(cls, v: object) -> object:
        if isinstance(v, str):
            v = v.strip().lower()
            if v in ("prod", "production"):
                return AppEnv.PRODUCTION
            if v in ("dev", "development"):
                return AppEnv.DEVELOPMENT
        return v

    @field_validator("default_log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, v: object) -> object:
        return v.upper() if isinstance(v, str) else v

    @field_validator("hf_token", "llm_model", "speaker_identification_model", "meeting_minutes_model", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped or stripped.lower() == "none":
                return None
            return stripped
        return v

    @model_validator(mode="after")
    def _apply_env_defaults(self) -> "Settings":
        # Expand root to absolute
        object.__setattr__(self, "root", os.path.abspath(self.root))

        is_prod = self.app_env is AppEnv.PRODUCTION

        # Production always implies stateless — no pickle cache, experiments in /tmp
        if is_prod:
            object.__setattr__(self, "stateless", True)
        stateless = self.stateless

        # Stateless mode forces no persistent experiments dir
        if stateless and not self.experiments_root:
            import tempfile

            object.__setattr__(
                self,
                "experiments_root",
                os.path.abspath(os.path.join(tempfile.gettempdir(), "diarization_experiments")),
            )

        if not self.experiments_root:
            object.__setattr__(
                self, "experiments_root", os.path.abspath(os.path.join(self.root, "experiments"))
            )
        else:
            object.__setattr__(self, "experiments_root", os.path.abspath(self.experiments_root))

        def _default(value: Optional[bool], prod_val: bool, dev_val: bool) -> bool:
            if value is not None:
                return value
            return prod_val if is_prod else dev_val

        defaults_map = {
            "log_to_console": _default(self.log_to_console, False, True),
            "hide_log_panel": _default(self.hide_log_panel, True, False),
            "enable_llm_cleaning": _default(self.enable_llm_cleaning, False, True),
            # Summary defaults to True in dev (mirrors old combined behaviour) and False in prod
            "enable_summary": _default(self.enable_summary, False, True),
            # Stateless forces aggressive cleanup and cache invalidation
            "auto_delete_uploads": stateless or _default(self.auto_delete_uploads, True, False),
            "auto_delete_outputs": stateless or _default(self.auto_delete_outputs, True, False),
            "default_reuse_cache": False if stateless else _default(self.default_reuse_cache, False, True),
            "default_force_split": stateless or _default(self.default_force_split, True, False),
            "default_clear_saved_state": stateless or _default(self.default_clear_saved_state, True, False),
        }
        for name, value in defaults_map.items():
            object.__setattr__(self, name, value)

        if self.cleanup_grace_seconds < 1:
            object.__setattr__(self, "cleanup_grace_seconds", 1)

        return self

    # --- Convenience ---

    @property
    def is_prod(self) -> bool:
        return self.app_env is AppEnv.PRODUCTION

    @property
    def llm_base_url(self) -> Optional[str]:
        """LLM endpoint — unified with the Whisper server URL."""
        return self.server_url or None


settings = Settings()
