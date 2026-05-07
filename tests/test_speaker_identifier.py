"""Tests for src/speaker_identifier.py — dataclass, prompt builders, JSON IO."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.speaker_identifier import (
    SpeakerInfo,
    _build_chronological_transcript,
    _build_per_speaker_summary,
    build_speaker_identification_prompt,
    load_speaker_info,
    save_speaker_info,
)


class TestSpeakerInfo:
    def test_basic(self) -> None:
        si = SpeakerInfo(
            speaker_id="Speaker_00",
            nom="MARTIN",
            prenom="Sophie",
            fonction="Directrice",
            confidence=0.95,
        )
        assert si.speaker_id == "Speaker_00"
        assert si.nom == "MARTIN"
        assert si.prenom == "Sophie"
        assert si.fonction == "Directrice"
        assert si.confidence == 0.95

    def test_default_confidence(self) -> None:
        si = SpeakerInfo(speaker_id="S1", nom="?", prenom="?", fonction="Inconnu")
        assert si.confidence is None


class TestBuildChronologicalTranscript:
    def test_basic_output(self, sample_transcript_df) -> None:
        result = _build_chronological_transcript(sample_transcript_df)
        assert "Speaker_00" in result
        assert "Speaker_01" in result
        assert "Bonjour à tous" in result

    def test_timestamps_in_output(self, sample_transcript_df) -> None:
        result = _build_chronological_transcript(sample_transcript_df)
        # First segment starts at 0.0 → [00:00]
        assert "[00:00]" in result

    def test_max_chars_truncation(self, sample_transcript_df) -> None:
        result = _build_chronological_transcript(sample_transcript_df, max_chars=30)
        assert "tronquée" in result.lower()

    def test_uses_cleaned_transcription_when_available(self) -> None:
        df = pd.DataFrame({
            "start": [0.0],
            "speaker": ["Speaker_00"],
            "transcription": ["raw text"],
            "cleaned_transcription": ["cleaned text"],
        })
        result = _build_chronological_transcript(df)
        assert "cleaned text" in result

    def test_uses_global_speaker_when_available(self) -> None:
        df = pd.DataFrame({
            "start": [0.0],
            "global_speaker": ["Speaker_Global"],
            "speaker": ["Speaker_Local"],
            "transcription": ["test"],
        })
        result = _build_chronological_transcript(df)
        assert "Speaker_Global" in result

    def test_empty_transcript(self) -> None:
        df = pd.DataFrame({
            "start": [0.0],
            "speaker": ["Speaker_00"],
            "transcription": [""],
        })
        result = _build_chronological_transcript(df)
        # Should not include empty lines
        assert result == "" or "Speaker_00" not in result

    def test_no_start_column(self) -> None:
        df = pd.DataFrame({
            "speaker": ["Speaker_00"],
            "transcription": ["Hello"],
        })
        result = _build_chronological_transcript(df)
        assert "Hello" in result
        # No timestamps expected
        assert "[" not in result


class TestBuildPerSpeakerSummary:
    def test_groups_by_speaker(self, sample_transcript_df) -> None:
        result = _build_per_speaker_summary(sample_transcript_df)
        assert "Speaker_00" in result
        assert "Speaker_01" in result

    def test_summaries_contain_text(self, sample_transcript_df) -> None:
        result = _build_per_speaker_summary(sample_transcript_df)
        assert "Bonjour" in result["Speaker_00"]

    def test_max_chars_per_speaker(self, sample_transcript_df) -> None:
        result = _build_per_speaker_summary(sample_transcript_df, max_chars_per_speaker=10)
        assert len(result["Speaker_00"]) <= 10

    def test_uses_global_speaker_when_available(self) -> None:
        df = pd.DataFrame({
            "global_speaker": ["GS_00", "GS_01"],
            "speaker": ["S_00", "S_01"],
            "transcription": ["text1", "text2"],
        })
        result = _build_per_speaker_summary(df)
        assert "GS_00" in result
        assert "GS_01" in result


class TestBuildSpeakerIdentificationPrompt:
    def test_includes_all_speakers(self, sample_transcript_df) -> None:
        prompt = build_speaker_identification_prompt(sample_transcript_df)
        assert "Speaker_00" in prompt
        assert "Speaker_01" in prompt

    def test_includes_known_participants(self, sample_transcript_df) -> None:
        known = [{"prenom": "Sophie", "nom": "MARTIN", "fonction": "Directrice"}]
        prompt = build_speaker_identification_prompt(sample_transcript_df, known)
        assert "Sophie" in prompt
        assert "MARTIN" in prompt

    def test_wraps_in_transcript_tags(self, sample_transcript_df) -> None:
        prompt = build_speaker_identification_prompt(sample_transcript_df)
        assert "<transcript>" in prompt
        assert "</transcript>" in prompt

    def test_json_format_instructions(self, sample_transcript_df) -> None:
        prompt = build_speaker_identification_prompt(sample_transcript_df)
        assert "JSON" in prompt
        assert "speaker_id" in prompt


class TestSaveLoadSpeakerInfo:
    def test_roundtrip(self, sample_speaker_info, tmp_path: Path) -> None:
        path = str(tmp_path / "speaker_info.json")
        save_speaker_info(sample_speaker_info, path)
        loaded = load_speaker_info(path)
        assert len(loaded) == len(sample_speaker_info)
        assert loaded["Speaker_00"].nom == "MARTIN"
        assert loaded["Speaker_00"].prenom == "Sophie"
        assert loaded["Speaker_00"].fonction == "Directrice de projet"
        assert loaded["Speaker_00"].confidence == 0.92

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        path = str(tmp_path / "subdir" / "speaker_info.json")
        save_speaker_info({}, path)
        assert Path(path).exists()

    def test_serializes_as_valid_json(self, sample_speaker_info, tmp_path: Path) -> None:
        path = str(tmp_path / "speaker_info.json")
        save_speaker_info(sample_speaker_info, path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "Speaker_00" in data
        assert data["Speaker_00"]["nom"] == "MARTIN"
