"""Tests for src/meeting_minutes.py — dataclasses, serialization, markdown."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.meeting_minutes import (
    DEFAULT_FORMAT,
    MEETING_MINUTES_FORMATS,
    MeetingMinuteSection,
    MeetingMinutes,
    _build_transcript_block,
    minutes_to_markdown,
    save_meeting_minutes,
)


class TestMeetingMinuteSection:
    def test_basic(self) -> None:
        s = MeetingMinuteSection(title="Budget", content="Approuvé.")
        assert s.title == "Budget"
        assert s.content == "Approuvé."


class TestMeetingMinutes:
    def test_to_dict_roundtrip(self, sample_meeting_minutes: MeetingMinutes) -> None:
        d = sample_meeting_minutes.to_dict()
        assert d["titre"] == "Réunion trimestrielle"
        assert d["format_used"] == "standard"
        assert d["date"] == "27/04/2026"
        assert d["lieux"] == "Salle A"
        assert len(d["participants"]) == 2
        assert len(d["ordre_du_jour"]) == 3
        assert len(d["discussions"]) == 2
        assert len(d["decisions"]) == 2
        assert len(d["actions"]) == 2
        assert d["prochaine_reunion"] == "15/05/2026"

    def test_to_json_valid(self, sample_meeting_minutes: MeetingMinutes) -> None:
        js = sample_meeting_minutes.to_json()
        parsed = json.loads(js)
        assert parsed["titre"] == "Réunion trimestrielle"

    def test_default_values(self) -> None:
        mm = MeetingMinutes(titre="Test")
        assert mm.format_used == DEFAULT_FORMAT
        assert mm.date is None
        assert mm.lieux is None
        assert mm.participants == []
        assert mm.ordre_du_jour == []
        assert mm.discussions == []
        assert mm.decisions == []
        assert mm.actions == []
        assert mm.prochaine_reunion is None

    def test_to_dict_nested_discussions(self) -> None:
        mm = MeetingMinutes(
            titre="T",
            discussions=[MeetingMinuteSection(title="S1", content="C1")]
        )
        d = mm.to_dict()
        assert d["discussions"] == [{"title": "S1", "content": "C1"}]


class TestMinutesToMarkdown:
    def test_full_output(self, sample_meeting_minutes: MeetingMinutes) -> None:
        md = minutes_to_markdown(sample_meeting_minutes)
        assert "# Réunion trimestrielle" in md
        assert "## Participants" in md
        assert "Sophie MARTIN" in md
        assert "Directrice de projet" in md
        assert "## Ordre du jour" in md
        assert "Résultats Q1" in md
        assert "## Discussions" in md
        assert "### Résultats Q1" in md
        assert "## Décisions" in md
        assert "Valider le budget Q2" in md
        assert "## Plan d'action" in md
        assert "Préparer budget Q2" in md
        assert "Prochaine réunion" in md
        assert "15/05/2026" in md

    def test_no_optional_fields(self) -> None:
        mm = MeetingMinutes(titre="Minimal")
        md = minutes_to_markdown(mm)
        assert "Date :" not in md
        assert "Lieu :" not in md
        assert "Participants" not in md
        assert "Ordre du jour" not in md
        assert "Discussions" not in md
        assert "Décisions" not in md
        assert "Plan d'action" not in md
        assert "Prochaine réunion" not in md

    def test_no_next_meeting(self, sample_meeting_minutes: MeetingMinutes) -> None:
        mm = MeetingMinutes(
            titre="T",
            participants=[{"prenom": "X", "nom": "Y", "fonction": "Z"}],
        )
        md = minutes_to_markdown(mm)
        assert "Prochaine réunion" not in md

    def test_participant_edge_cases(self) -> None:
        # Participant with only nom, no prenom
        mm = MeetingMinutes(
            titre="T",
            participants=[{"nom": "DUPONT", "prenom": "", "fonction": ""}],
        )
        md = minutes_to_markdown(mm)
        assert "DUPONT" in md


class TestSaveMeetingMinutes:
    def test_saves_json_and_markdown(
        self, sample_meeting_minutes: MeetingMinutes, tmp_path: Path
    ) -> None:
        saved = save_meeting_minutes(sample_meeting_minutes, str(tmp_path), "test")
        assert "json" in saved
        assert "markdown" in saved
        assert Path(saved["json"]).exists()
        assert Path(saved["markdown"]).exists()
        # Verify JSON content
        data = json.loads(Path(saved["json"]).read_text(encoding="utf-8"))
        assert data["titre"] == "Réunion trimestrielle"


class TestFormatTemplates:
    def test_all_six_formats_exist(self) -> None:
        assert len(MEETING_MINUTES_FORMATS) == 6

    @pytest.mark.parametrize("key", ["standard", "executif", "technique", "projet", "rh_social", "formation"])
    def test_format_has_label_and_description(self, key: str) -> None:
        fmt = MEETING_MINUTES_FORMATS[key]
        assert "label" in fmt
        assert "description" in fmt
        assert "instructions" in fmt
        assert len(fmt["label"]) > 0

    def test_default_format_is_standard(self) -> None:
        assert DEFAULT_FORMAT == "standard"


class TestBuildTranscriptBlock:
    def test_with_global_speakers(self, sample_transcript_df) -> None:
        block = _build_transcript_block(sample_transcript_df)
        assert "Speaker_00" in block
        assert "Speaker_01" in block
        assert "Bonjour à tous" in block

    def test_with_speaker_info(self, sample_transcript_df, sample_speaker_info) -> None:
        speaker_info = {
            "Speaker_00": {"prenom": "Sophie", "nom": "MARTIN"},
            "Speaker_01": {"prenom": "Jean", "nom": "DUPONT"},
        }
        block = _build_transcript_block(sample_transcript_df, speaker_info=speaker_info)
        assert "Sophie MARTIN" in block
        assert "Jean DUPONT" in block
        # The original speaker ID should still appear in parentheses
        assert "Speaker_00" in block

    def test_max_chars_truncation(self, sample_transcript_df) -> None:
        block = _build_transcript_block(sample_transcript_df, max_chars=50)
        assert "tronquée" in block.lower()

    def test_no_speaker_fallback(self) -> None:
        df = __import__("pandas").DataFrame({"transcription": ["Texte sans speaker"]})
        block = _build_transcript_block(df)
        assert "Texte sans speaker" in block
