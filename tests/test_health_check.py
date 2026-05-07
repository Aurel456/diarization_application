"""Tests for src/health_check.py — ServiceStatus, check_all_services dedup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.health_check import (
    ServiceStatus,
    check_all_services,
    check_service,
)


class TestServiceStatus:
    def test_ok_label(self) -> None:
        status = ServiceStatus(
            name="Whisper",
            url="http://whisper:8000/v1",
            ok=True,
            model_found=True,
            available_models=["whisper-large"],
            latency_ms=42,
            error=None,
        )
        label = status.label
        assert "✅" in label
        assert "42 ms" in label

    def test_model_not_found_label(self) -> None:
        status = ServiceStatus(
            name="LLM",
            url="http://llm:8000/v1",
            ok=True,
            model_found=False,
            available_models=["model-a", "model-b"],
            latency_ms=10,
            error=None,
        )
        label = status.label
        assert "⚠️" in label
        assert "modèle introuvable" in label
        assert "model-a" in label

    def test_error_label(self) -> None:
        status = ServiceStatus(
            name="Whisper",
            url="http://bad:8000/v1",
            ok=False,
            model_found=False,
            available_models=[],
            latency_ms=None,
            error="Connection refused",
        )
        label = status.label
        assert "❌" in label
        assert "Connection refused" in label

    def test_available_models_truncation(self) -> None:
        """More than 5 models should only show first 5 in label."""
        models = [f"model-{i}" for i in range(10)]
        status = ServiceStatus(
            name="LLM",
            url="http://llm:8000/v1",
            ok=True,
            model_found=False,
            available_models=models,
            latency_ms=10,
            error=None,
        )
        label = status.label
        # Should mention at least the first model
        assert "model-0" in label
        # The 6th model should not appear
        assert "model-5" not in label


class TestCheckService:
    @patch("src.health_check.OpenAI")
    def test_no_url_returns_error(self, mock_openai) -> None:
        result = check_service(
            name="Test",
            base_url="",
            api_key=None,
            expected_model="m",
        )
        assert result.ok is False
        assert result.model_found is False
        mock_openai.assert_not_called()

    @patch("src.health_check.OpenAI")
    def test_no_url_none(self, mock_openai) -> None:
        result = check_service(
            name="Test",
            base_url=None,
            api_key=None,
            expected_model="m",
        )
        assert result.ok is False
        # base_url is None → ServiceStatus receives "" as url
        mock_openai.assert_not_called()


class TestCheckAllServices:
    @patch("src.health_check.check_service")
    def test_dedup_when_same_url(self, mock_check) -> None:
        """When LLM and Whisper share the same URL, only one check_service call is made."""
        whisper_status = ServiceStatus(
            name="Whisper",
            url="http://shared:8000/v1",
            ok=True,
            model_found=True,
            available_models=["whisper", "llm-model"],
            latency_ms=50,
            error=None,
        )
        mock_check.return_value = whisper_status

        results = check_all_services(
            server_url="http://shared:8000/v1",
            llm_base_url="http://shared:8000/v1",  # same URL
            api_key="sk-test",
            whisper_model="whisper",
            llm_model="llm-model",
        )

        # Should have called check_service only once (dedup)
        assert mock_check.call_count == 1
        assert len(results) == 2
        # Both should be OK
        assert results[0].ok is True  # Whisper
        assert results[1].ok is True  # LLM (reused)

    @patch("src.health_check.check_service")
    def test_separate_calls_when_different_urls(self, mock_check) -> None:
        """When URLs differ, both services are checked independently."""
        def make_status(name, base_url, **kwargs):
            return ServiceStatus(
                name=name,
                url=base_url,
                ok=True,
                model_found=True,
                available_models=["test"],
                latency_ms=10,
                error=None,
            )

        mock_check.side_effect = make_status

        results = check_all_services(
            server_url="http://whisper:8000/v1",
            llm_base_url="http://llm:8000/v1",  # different URL
            api_key="sk-test",
            whisper_model="whisper",
            llm_model="llm",
        )

        assert mock_check.call_count == 2
        assert len(results) == 2

    @patch("src.health_check.check_service")
    def test_llm_fallback_to_server_url(self, mock_check) -> None:
        """When llm_base_url is None, it falls back to server_url (same server, dedup)."""
        whisper_status = ServiceStatus(
            name="Whisper",
            url="http://server:8000/v1",
            ok=True,
            model_found=True,
            available_models=["whisper", "llm"],
            latency_ms=30,
            error=None,
        )
        mock_check.return_value = whisper_status

        results = check_all_services(
            server_url="http://server:8000/v1",
            llm_base_url=None,  # falls back to server_url
            api_key="sk-test",
            whisper_model="whisper",
            llm_model="llm",
        )

        # Dedup because llm_base_url falls back to server_url
        assert mock_check.call_count == 1
        assert len(results) == 2
        assert results[1].ok is True
