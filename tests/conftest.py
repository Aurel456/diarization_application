"""Shared fixtures for the diarization application test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from src.meeting_minutes import MeetingMinuteSection, MeetingMinutes
from src.speaker_identifier import SpeakerInfo


@pytest.fixture
def sample_transcript_df() -> pd.DataFrame:
    """Minimal DataFrame mimicking diarized transcript output."""
    return pd.DataFrame({
        "start": [0.0, 5.2, 12.8, 20.1, 35.6, 45.0],
        "finish": [4.8, 12.0, 19.5, 34.8, 44.2, 58.0],
        "speaker": ["Speaker_00", "Speaker_01", "Speaker_00", "Speaker_01", "Speaker_00", "Speaker_01"],
        "global_speaker": ["Speaker_00", "Speaker_01", "Speaker_00", "Speaker_01", "Speaker_00", "Speaker_01"],
        "transcription": [
            "Bonjour à tous, merci d'être présents.",
            "Merci Sophie. Je vais présenter les résultats du trimestre.",
            "Les chiffres sont excellents, nous avons dépassé les objectifs.",
            "Effectivement, la croissance est de 15 pourcent sur le secteur.",
            "Je propose de passer aux questions maintenant.",
            "Bonne idée. Qui veut commencer?",
        ],
        "cleaned_transcription": [
            "Bonjour à tous, merci d'être présents.",
            "Merci Sophie. Je vais présenter les résultats du trimestre.",
            "Les chiffres sont excellents, nous avons dépassé les objectifs.",
            "Effectivement, la croissance est de 15 pourcent sur le secteur.",
            "Je propose de passer aux questions maintenant.",
            "Bonne idée. Qui veut commencer?",
        ],
    })


@pytest.fixture
def sample_speaker_info() -> Dict[str, SpeakerInfo]:
    """Sample speaker identification for two speakers."""
    return {
        "Speaker_00": SpeakerInfo(
            speaker_id="Speaker_00",
            nom="MARTIN",
            prenom="Sophie",
            fonction="Directrice de projet",
            confidence=0.92,
        ),
        "Speaker_01": SpeakerInfo(
            speaker_id="Speaker_01",
            nom="DUPONT",
            prenom="Jean",
            fonction="Responsable commercial",
            confidence=0.88,
        ),
    }


@pytest.fixture
def sample_meeting_minutes() -> MeetingMinutes:
    """Sample meeting minutes object for serialization tests."""
    return MeetingMinutes(
        titre="Réunion trimestrielle",
        format_used="standard",
        date="27/04/2026",
        lieux="Salle A",
        participants=[
            {"prenom": "Sophie", "nom": "MARTIN", "fonction": "Directrice de projet"},
            {"prenom": "Jean", "nom": "DUPONT", "fonction": "Responsable commercial"},
        ],
        ordre_du_jour=["Résultats Q1", "Objectifs Q2", "Questions diverses"],
        discussions=[
            MeetingMinuteSection(title="Résultats Q1", content="Les chiffres sont excellents, 15% de croissance."),
            MeetingMinuteSection(title="Objectifs Q2", content="Nouveaux objectifs fixés à 20% de croissance."),
        ],
        decisions=["Valider le budget Q2", "Recruter 2 commerciaux"],
        actions=[
            {"action": "Préparer budget Q2", "responsable": "Sophie MARTIN", "delai": "15/05/2026"},
            {"action": "Lancer recrutement", "responsable": "Jean DUPONT", "delai": "01/05/2026"},
        ],
        prochaine_reunion="15/05/2026",
    )


@pytest.fixture
def tmp_experiments_dir(tmp_path: Path) -> Path:
    """Temporary experiments directory with standard subdirectories."""
    exp_dir = tmp_path / "experiments" / "test_run"
    for sub in ["saved_state", "output_DOC", "diarization_results", "plot", "chunks_1200s"]:
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
    return exp_dir
