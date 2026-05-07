"""Tests for src/llm_client.py — UsageStats, wrap_untrusted, make_llm_client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.errors import LLMError
from src.llm_client import LLMClient, UsageStats, make_llm_client


class TestUsageStats:
    def test_initial_zero(self) -> None:
        stats = UsageStats()
        assert stats.prompt_tokens == 0
        assert stats.completion_tokens == 0
        assert stats.total_tokens == 0
        assert stats.n_calls == 0
        assert stats.n_retries == 0
        assert stats.errors == []

    def test_add_usage(self) -> None:
        stats = UsageStats()
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        stats.add(mock_usage)
        assert stats.prompt_tokens == 100
        assert stats.completion_tokens == 50
        assert stats.total_tokens == 150

    def test_add_none_does_nothing(self) -> None:
        stats = UsageStats()
        stats.add(None)
        assert stats.total_tokens == 0

    def test_add_missing_attrs(self) -> None:
        stats = UsageStats()
        mock_usage = MagicMock()
        del mock_usage.prompt_tokens  # missing attribute → getattr returns 0
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 10
        stats.add(mock_usage)
        assert stats.completion_tokens == 10

    def test_to_dict(self) -> None:
        stats = UsageStats()
        d = stats.to_dict()
        assert "prompt_tokens" in d
        assert "completion_tokens" in d
        assert "total_tokens" in d
        assert "n_calls" in d
        assert "n_retries" in d
        assert "errors" in d

    def test_to_dict_errors_capped(self) -> None:
        stats = UsageStats()
        stats.errors = [f"err{i}" for i in range(10)]
        d = stats.to_dict()
        assert len(d["errors"]) == 5  # last 5


class TestWrapUntrusted:
    def test_basic_wrapping(self) -> None:
        wrapped = LLMClient.wrap_untrusted("hello world", tag="transcript")
        assert "<transcript>" in wrapped
        assert "</transcript>" in wrapped
        assert "hello world" in wrapped

    def test_custom_tag(self) -> None:
        wrapped = LLMClient.wrap_untrusted("data", tag="per_speaker")
        assert "<per_speaker>" in wrapped
        assert "</per_speaker>" in wrapped

    def test_empty_string(self) -> None:
        wrapped = LLMClient.wrap_untrusted("", tag="test")
        assert "<test>" in wrapped
        assert "</test>" in wrapped


class TestMakeLLMClient:
    def test_returns_none_when_missing_base_url(self) -> None:
        client = make_llm_client(base_url=None, api_key="key", model="model")
        assert client is None

    def test_returns_none_when_missing_model(self) -> None:
        client = make_llm_client(base_url="http://x", api_key="key", model=None)
        assert client is None

    def test_returns_none_when_empty_base_url(self) -> None:
        client = make_llm_client(base_url="", api_key="key", model="model")
        assert client is None

    def test_returns_none_when_empty_model(self) -> None:
        client = make_llm_client(base_url="http://x", api_key="key", model="")
        assert client is None

    @patch("src.llm_client.OpenAI")
    def test_returns_client_when_config_complete(self, mock_openai) -> None:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        client = make_llm_client(
            base_url="http://llm:8000/v1",
            api_key="sk-test",
            model="test-model",
        )
        assert client is not None
        assert isinstance(client, LLMClient)
        mock_openai.assert_called_once_with(base_url="http://llm:8000/v1", api_key="sk-test")

    @patch("src.llm_client.OpenAI")
    def test_uses_dummy_api_key_when_none(self, mock_openai) -> None:
        mock_openai.return_value = MagicMock()
        client = make_llm_client(base_url="http://x", api_key=None, model="m")
        assert client is not None
        # LLMClient should use "dummy" as fallback
        assert client.api_key == "dummy"
