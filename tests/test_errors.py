"""Tests for the error hierarchy defined in src/errors.py."""

from __future__ import annotations

import pytest

from src.errors import (
    AudioInputError,
    AudioSplittingError,
    CancelledError,
    ClusteringError,
    ConfigurationError,
    DiarizationError,
    ExportError,
    LLMError,
    PipelineError,
    PipelineStep,
    TranscriptionError,
)


class TestPipelineError:
    def test_defaults(self) -> None:
        err = PipelineError("test message")
        assert err.message == "test message"
        assert err.step == "pipeline"
        assert err.retryable is False
        assert err.cause is None
        assert err.details == {}

    def test_with_cause(self) -> None:
        cause = ValueError("original")
        err = PipelineError("wrapped", cause=cause)
        assert err.cause is cause

    def test_with_step(self) -> None:
        err = PipelineError("msg", step=PipelineStep.EXPORT)
        assert err.step == "export"

    def test_with_details(self) -> None:
        err = PipelineError("msg", details={"key": "val"})
        assert err.details == {"key": "val"}

    def test_str_format(self) -> None:
        err = PipelineError("something failed")
        assert str(err) == "[pipeline] something failed"

    def test_retryable_flag(self) -> None:
        err = PipelineError("msg", retryable=True)
        assert err.retryable is True


class TestErrorSubclasses:
    """Each subclass should auto-set the correct step and retryable flag."""

    @pytest.mark.parametrize(
        "cls,expected_step,expected_retryable",
        [
            (ConfigurationError, "setup", False),
            (AudioInputError, "setup", False),
            (AudioSplittingError, "split", False),
            (DiarizationError, "diarization", False),
            (ClusteringError, "clustering", False),
            (TranscriptionError, "transcription", True),
            (LLMError, "cleaning", True),
            (ExportError, "export", False),
        ],
    )
    def test_step_and_retryable(self, cls, expected_step, expected_retryable) -> None:
        err = cls("test")
        assert err.step == expected_step
        assert err.retryable == expected_retryable

    def test_transcription_error_with_cause(self) -> None:
        cause = ConnectionError("timeout")
        err = TranscriptionError("whisper failed", cause=cause)
        assert err.cause is cause
        assert err.step == "transcription"

    def test_export_error_with_cause(self) -> None:
        cause = OSError("disk full")
        err = ExportError("docx write failed", cause=cause)
        assert err.cause is cause
        assert err.step == "export"

    def test_llm_error_with_cause(self) -> None:
        cause = TimeoutError("request timeout")
        err = LLMError("llm failed", cause=cause)
        assert err.cause is cause
        assert err.step == "cleaning"
        assert err.retryable is True


class TestCancelledError:
    def test_default_message(self) -> None:
        err = CancelledError()
        assert "annulé" in err.message.lower()

    def test_custom_message(self) -> None:
        err = CancelledError("User hit stop")
        assert err.message == "User hit stop"

    def test_has_cancelled_detail(self) -> None:
        err = CancelledError()
        assert err.details.get("cancelled") is True

    def test_not_retryable(self) -> None:
        err = CancelledError()
        assert err.retryable is False
