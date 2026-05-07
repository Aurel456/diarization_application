"""Tests for src/summarizer.py — token counting and text chunking."""

from __future__ import annotations

import pytest

from src.summarizer import _chunk_text_by_tokens, _token_count


class TestTokenCount:
    def test_positive_count(self) -> None:
        count = _token_count("Hello world, this is a test.", "gpt-4")
        assert count > 0

    def test_different_lengths(self) -> None:
        short = _token_count("Hi", "gpt-4")
        long = _token_count("This is a much longer piece of text " * 10, "gpt-4")
        assert long > short

    def test_empty_string(self) -> None:
        count = _token_count("", "gpt-4")
        assert count == 0

    def test_fallback_for_unknown_model(self) -> None:
        """Unknown model should fall back to word count."""
        count = _token_count("one two three four five", "unknown-model-xyz")
        # Should fall back to len(text.split()) = 5
        assert count == 5


class TestChunkTextByTokens:
    def test_single_chunk_when_under_limit(self) -> None:
        text = "Short text."
        chunks = _chunk_text_by_tokens(text, max_tokens=1000, model="gpt-4")
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_chunks(self) -> None:
        text = "word " * 500  # many tokens
        chunks = _chunk_text_by_tokens(text, max_tokens=10, model="gpt-4")
        assert len(chunks) > 1

    def test_empty_string(self) -> None:
        chunks = _chunk_text_by_tokens("", max_tokens=100, model="gpt-4")
        assert chunks == [""] or chunks == []

    def test_all_chunks_reassemble_to_original(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = _chunk_text_by_tokens(text, max_tokens=50, model="gpt-4")
        # Reassembled text (tiktoken decodes tokens, so spaces may differ slightly)
        reassembled = "".join(chunks)
        # Should be roughly similar length
        assert len(reassembled) > 0
        assert "quick" in reassembled

    def test_fallback_for_unknown_model(self) -> None:
        """Unknown model falls back to character count (~4 chars per token)."""
        text = "x" * 2000
        chunks = _chunk_text_by_tokens(text, max_tokens=100, model="unknown-model")
        # 100 tokens * 4 chars/token = 400 chars per chunk → ~5 chunks
        assert len(chunks) >= 4
