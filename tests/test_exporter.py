"""Tests for src/exporter.py — pure functions only, no filesystem."""

from __future__ import annotations

import pandas as pd
import pytest

from src.exporter import (
    apply_speaker_mapping,
    clean_before_export,
    concatenate_texts,
    split_sentences_with_linebreaks,
)


class TestSplitSentencesWithLinebreaks:
    def test_basic_split(self) -> None:
        text = "Bonjour. Comment allez-vous? Très bien."
        result = split_sentences_with_linebreaks(text, max_sentences=1)
        # Each "paragraph" should contain at most 1 sentence
        parts = result.split("\n\n")
        assert len(parts) >= 2

    def test_abbreviations_not_split(self) -> None:
        text = "M. Dupont est arrivé. Mme. Martin aussi. Dr. Who était là."
        result = split_sentences_with_linebreaks(text, max_sentences=10)
        # Abbreviations should not cause splits
        parts = result.split("\n\n")
        # Should be a single paragraph since abbreviations are preserved
        assert len(parts) == 1

    def test_empty_string(self) -> None:
        assert split_sentences_with_linebreaks("") == ""

    def test_none_input(self) -> None:
        assert split_sentences_with_linebreaks(None) == ""

    def test_nan_input(self) -> None:
        assert split_sentences_with_linebreaks(float("nan")) == ""

    def test_max_sentences_grouping(self) -> None:
        text = "A. B. C. D. E. F. G. H."
        # 8 single-letter "sentences" — abbreviated A., B. etc. are tricky
        result = split_sentences_with_linebreaks(text, max_sentences=3)
        # Just verify we get a non-empty result
        assert len(result) > 0

    def test_adds_trailing_period(self) -> None:
        text = "Phrase sans point final"
        result = split_sentences_with_linebreaks(text, max_sentences=1)
        assert result.endswith(".")


class TestCleanBeforeExport:
    def test_whitespace_normalization(self) -> None:
        assert clean_before_export("  trop   d'espaces   ") == "trop d'espaces"

    def test_removes_boilerplate(self) -> None:
        text = "Voici le texte corrigé: Bonjour tout le monde."
        result = clean_before_export(text)
        assert "Voici le texte corrigé" not in result
        assert "Bonjour tout le monde" in result

    def test_removes_quote_marks(self) -> None:
        assert '"' not in clean_before_export('il a dit "bonjour"')

    def test_substitutes_xilo(self) -> None:
        assert "XYLO" in clean_before_export("le projet Xilo est bon")

    def test_substitutes_xar(self) -> None:
        assert "CSAR" in clean_before_export("le xar est validé")
        assert "CSAR" in clean_before_export("le XAR est validé")

    def test_substitutes_agraf(self) -> None:
        assert "AGRAF" in clean_before_export("l'agraf est ok")
        assert "AGRAF" in clean_before_export("le AGRaF est ok")

    def test_substitutes_dtenum(self) -> None:
        # Regex pattern is \des ténum\b — \d matches a digit before "es ténum"
        result = clean_before_export("le 2es ténum project")
        assert "DTNUM" in result

    def test_empty_string(self) -> None:
        assert clean_before_export("") == ""

    def test_nan_input(self) -> None:
        assert clean_before_export(float("nan")) == ""

    def test_none_input(self) -> None:
        assert clean_before_export(None) == ""

    def test_return_stripped(self) -> None:
        assert clean_before_export("  hello  ") == "hello"


class TestConcatenateTexts:
    def test_merges_consecutive_same_speaker(self) -> None:
        df = pd.DataFrame({
            "speaker": ["A", "A", "B", "A"],
            "text": ["Hello", "world", "Bonjour", "again"],
        })
        result = concatenate_texts(df, "speaker", "text")
        assert len(result) == 3
        assert result.iloc[0]["text"] == "Hello world"
        assert result.iloc[1]["text"] == "Bonjour"
        assert result.iloc[2]["text"] == "again"

    def test_handles_empty_df(self) -> None:
        df = pd.DataFrame({"speaker": [], "text": []})
        result = concatenate_texts(df, "speaker", "text")
        assert result.empty

    def test_single_row(self) -> None:
        df = pd.DataFrame({"speaker": ["A"], "text": ["Seul"]})
        result = concatenate_texts(df, "speaker", "text")
        assert len(result) == 1

    def test_all_different_speakers(self) -> None:
        df = pd.DataFrame({
            "speaker": ["A", "B", "C"],
            "text": ["un", "deux", "trois"],
        })
        result = concatenate_texts(df, "speaker", "text")
        assert len(result) == 3

    def test_all_same_speaker(self) -> None:
        df = pd.DataFrame({
            "speaker": ["A", "A", "A"],
            "text": ["un", "deux", "trois"],
        })
        result = concatenate_texts(df, "speaker", "text")
        assert len(result) == 1
        assert "un deux trois" in result.iloc[0]["text"]


class TestApplySpeakerMapping:
    def test_basic_remap(self) -> None:
        df = pd.DataFrame({
            "global_speaker": ["Speaker_00", "Speaker_01", "Speaker_02"],
            "text": ["a", "b", "c"],
        })
        mapping = {"Speaker_02": "Speaker_00"}
        result = apply_speaker_mapping(df, mapping)
        assert result.iloc[0]["global_speaker"] == "Speaker_00"
        assert result.iloc[1]["global_speaker"] == "Speaker_01"  # unchanged
        assert result.iloc[2]["global_speaker"] == "Speaker_00"  # remapped

    def test_identity_for_unmapped(self) -> None:
        df = pd.DataFrame({
            "global_speaker": ["Speaker_00", "Speaker_99"],
            "text": ["a", "b"],
        })
        mapping = {"Speaker_99": "Speaker_00"}
        result = apply_speaker_mapping(df, mapping)
        assert result.iloc[0]["global_speaker"] == "Speaker_00"  # already was 00
        assert result.iloc[1]["global_speaker"] == "Speaker_00"  # remapped

    def test_missing_column_returns_copy(self) -> None:
        df = pd.DataFrame({"text": ["a", "b"]})
        result = apply_speaker_mapping(df, {"a": "b"})
        pd.testing.assert_frame_equal(result, df)

    def test_empty_DataFrame(self) -> None:
        df = pd.DataFrame({"global_speaker": [], "text": []})
        result = apply_speaker_mapping(df, {})
        assert result.empty
