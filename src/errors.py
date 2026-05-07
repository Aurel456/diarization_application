"""Types d'erreurs spécialisés pour le pipeline.

Chaque erreur mappe à une étape du pipeline et porte un flag retryable
que les appelants (UI, API) peuvent utiliser pour décider de retenter.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class PipelineStep(str, Enum):
    """Étapes du pipeline pour le traçage des erreurs."""

    SETUP = "setup"
    SPLIT = "split"
    DIARIZATION = "diarization"
    PROCESS_DIARIZATION = "process_diarization"
    CLUSTERING = "clustering"
    LABEL_UPDATE = "label_update"
    TRANSCRIPTION = "transcription"
    SPEAKER_IDENTIFICATION = "speaker_identification"
    CLEANING = "cleaning"
    TIMESTAMP_ADJUST = "timestamp_adjust"
    EXPORT = "export"
    MEETING_MINUTES = "meeting_minutes"
    SUMMARY = "summary"
    SIMPLE_TRANSCRIPTION = "simple_transcription"
    SIMPLE_EXPORT = "simple_export"


class PipelineError(Exception):
    """Classe de base pour toutes les erreurs du pipeline."""

    step: str = "pipeline"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        step: Optional[PipelineStep] = None,
        retryable: bool = False,
        cause: Optional[BaseException] = None,
        details: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.step = step.value if step else "pipeline"
        self.retryable = retryable
        self.cause = cause
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.step}] {self.message}"


class ConfigurationError(PipelineError):
    """Erreur de configuration (invalides, manquantes, incohérentes)."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message,
            step=PipelineStep.SETUP,
            retryable=False,
            details=details,
        )


class AudioInputError(PipelineError):
    """Erreur liée aux fichiers audio (introuvables, corrompus, format non supporté)."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message,
            step=PipelineStep.SETUP,
            retryable=False,
            details=details,
        )


class AudioSplittingError(PipelineError):
    """Erreur lors du découpage audio."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message,
            step=PipelineStep.SPLIT,
            retryable=False,
            details=details,
        )


class DiarizationError(PipelineError):
    """Erreur lors de la diarization Pyannote."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message,
            step=PipelineStep.DIARIZATION,
            retryable=False,
            details=details,
        )


class ClusteringError(PipelineError):
    """Erreur lors du clustering HDBSCAN."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message,
            step=PipelineStep.CLUSTERING,
            retryable=False,
            details=details,
        )


class TranscriptionError(PipelineError):
    """Erreur lors de la transcription Whisper (réseau, timeout, modèle)."""

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
        *,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(
            message,
            step=PipelineStep.TRANSCRIPTION,
            retryable=True,
            details=details,
            cause=cause,
        )


class LLMError(PipelineError):
    """Erreur lors d'un appel LLM (timeout, rate limit, parsing JSON)."""

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
        *,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(
            message,
            step=PipelineStep.CLEANING,
            retryable=True,
            details=details,
            cause=cause,
        )


class ExportError(PipelineError):
    """Erreur lors de l'export DOCX/TXT/SRT."""

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
        *,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(
            message,
            step=PipelineStep.EXPORT,
            retryable=False,
            details=details,
            cause=cause,
        )


class CancelledError(PipelineError):
    """Pipeline annulé par l'utilisateur (cancellation gracieuse)."""

    def __init__(self, message: str = "Pipeline annulé par l'utilisateur"):
        super().__init__(
            message,
            step=None,
            retryable=False,
            details={"cancelled": True},
        )
