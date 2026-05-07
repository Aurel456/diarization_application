"""Tests for src/config_builder.py — config hash and cache consistency."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config_builder import (
    _CACHE_RELEVANT_FIELDS,
    compute_config_hash,
    ensure_cache_consistency,
)
from main_pipeline import PipelineConfig


def _make_config(**overrides) -> PipelineConfig:
    """Helper to build a minimal PipelineConfig for testing."""
    defaults = {
        "root": ".",
        "audio_processing_mode": "concurrent",
        "segment_duration": 1200,
        "max_workers": 2,
        "duree_min_speaker": 0.5,
        "vad_filter": True,
        "input_audio_paths": [],
        "server_url": "http://whisper:8000/v1",
        "whisper_model": "whisper",
        "api_key": "sk-test",
        "llm_model": "test-model",
        "llm_base_url": "http://llm:8000/v1",
        "enable_llm_cleaning": True,
        "enable_summary": True,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


class TestComputeConfigHash:
    def test_same_config_produces_same_hash(self) -> None:
        cfg1 = _make_config()
        cfg2 = _make_config()
        assert compute_config_hash(cfg1) == compute_config_hash(cfg2)

    def test_different_segment_duration_produces_different_hash(self) -> None:
        cfg1 = _make_config(segment_duration=600)
        cfg2 = _make_config(segment_duration=1200)
        assert compute_config_hash(cfg1) != compute_config_hash(cfg2)

    def test_different_model_produces_different_hash(self) -> None:
        cfg1 = _make_config(llm_model="model-a")
        cfg2 = _make_config(llm_model="model-b")
        assert compute_config_hash(cfg1) != compute_config_hash(cfg2)

    def test_hash_is_string_of_length_12(self) -> None:
        h = compute_config_hash(_make_config())
        assert isinstance(h, str)
        assert len(h) == 12

    def test_different_audio_paths_produce_different_hash(self) -> None:
        cfg1 = _make_config(input_audio_paths=["/path/to/a.mp3"])
        cfg2 = _make_config(input_audio_paths=["/path/to/b.mp3"])
        assert compute_config_hash(cfg1) != compute_config_hash(cfg2)

    def test_non_cache_field_change_does_not_affect_hash(self) -> None:
        """Fields like log_level that aren't in _CACHE_RELEVANT_FIELDS should not change hash."""
        cfg1 = _make_config(log_level=10)
        cfg2 = _make_config(log_level=50)
        assert compute_config_hash(cfg1) == compute_config_hash(cfg2)


class TestCacheRelevantFields:
    def test_fields_are_tuple(self) -> None:
        assert isinstance(_CACHE_RELEVANT_FIELDS, tuple)

    def test_input_audio_paths_in_fields(self) -> None:
        assert "input_audio_paths" in _CACHE_RELEVANT_FIELDS

    def test_llm_model_in_fields(self) -> None:
        assert "llm_model" in _CACHE_RELEVANT_FIELDS


class TestEnsureCacheConsistency:
    def test_creates_hash_file_when_missing(self, tmp_path: Path) -> None:
        saved_state = str(tmp_path / "saved_state")
        os.makedirs(saved_state, exist_ok=True)
        cfg = _make_config()
        ensure_cache_consistency(cfg, saved_state)
        hash_file = os.path.join(saved_state, ".config_hash")
        assert os.path.exists(hash_file)
        with open(hash_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        assert len(content) == 12

    def test_removes_stale_pickles(self, tmp_path: Path) -> None:
        saved_state = str(tmp_path / "saved_state")
        os.makedirs(saved_state, exist_ok=True)
        # Create old hash file with a different config
        hash_file = os.path.join(saved_state, ".config_hash")
        with open(hash_file, "w", encoding="utf-8") as f:
            f.write("deadbeef1234")
        # Create a pickle file
        pickle_path = os.path.join(saved_state, "step1.pkl")
        with open(pickle_path, "wb") as f:
            f.write(b"fake pickle data")
        # Also create a non-pickle file that should survive
        other_file = os.path.join(saved_state, "other.txt")
        with open(other_file, "w") as f:
            f.write("keep me")

        cfg = _make_config()
        ensure_cache_consistency(cfg, saved_state)

        # The stale pickle should be removed
        assert not os.path.exists(pickle_path)
        # The non-pickle file should survive
        assert os.path.exists(other_file)
        # New hash file should be written
        with open(hash_file, "r", encoding="utf-8") as f:
            new_hash = f.read().strip()
        assert new_hash != "deadbeef1234"

    def test_same_hash_keeps_pickles(self, tmp_path: Path) -> None:
        saved_state = str(tmp_path / "saved_state")
        os.makedirs(saved_state, exist_ok=True)
        cfg = _make_config()
        # Pre-write correct hash
        hash_file = os.path.join(saved_state, ".config_hash")
        with open(hash_file, "w", encoding="utf-8") as f:
            f.write(compute_config_hash(cfg))
        # Create a pickle
        pickle_path = os.path.join(saved_state, "step1.pkl")
        with open(pickle_path, "wb") as f:
            f.write(b"data")

        ensure_cache_consistency(cfg, saved_state)
        assert os.path.exists(pickle_path)

    def test_stateless_mode_skips(self, tmp_path: Path) -> None:
        saved_state = str(tmp_path / "saved_state")
        os.makedirs(saved_state, exist_ok=True)
        cfg = _make_config()
        cfg.stateless = True
        cfg.reuse_cache = False
        ensure_cache_consistency(cfg, saved_state)
        # No hash file should be created
        hash_file = os.path.join(saved_state, ".config_hash")
        assert not os.path.exists(hash_file)
