"""Tests for src/simple_transcriber.py — format_timestamp_srt, words_to_srt_blocks."""

from __future__ import annotations

from src.simple_transcriber import format_timestamp_srt, save_files, words_to_srt_blocks


class TestFormatTimestampSRT:
    def test_zero_seconds(self) -> None:
        assert format_timestamp_srt(0.0) == "00:00:00,000"

    def test_one_second_exact(self) -> None:
        assert format_timestamp_srt(1.0) == "00:00:01,000"

    def test_one_minute(self) -> None:
        assert format_timestamp_srt(60.0) == "00:01:00,000"

    def test_one_hour(self) -> None:
        assert format_timestamp_srt(3661.5) == "01:01:01,500"

    def test_milliseconds(self) -> None:
        ts = format_timestamp_srt(5.789)
        # 5.789 seconds → int((5.789 - 5) * 1000) = int(789) = 789
        assert ",789" in ts or ",788" in ts or ",790" in ts

    def test_fractional_milliseconds_floor(self) -> None:
        # 0.9999 seconds should floor to 999 ms, not 1000
        ts = format_timestamp_srt(0.9999)
        assert ts == "00:00:00,999"


class TestWordsToSRTBlocks:
    def test_basic_grouping(self) -> None:
        words = [
            {"word": "Bonjour", "start": 0.0, "end": 0.5},
            {"word": "tout", "start": 0.6, "end": 0.8},
            {"word": "le", "start": 0.9, "end": 1.0},
            {"word": "monde", "start": 1.1, "end": 1.5},
        ]
        blocks = words_to_srt_blocks(words, max_chars=50, max_duration=5.0)
        assert len(blocks) >= 1
        assert "Bonjour" in blocks[0]["text"]

    def test_sentence_end_punctuation_triggers_block(self) -> None:
        words = [
            {"word": "Bonjour.", "start": 0.0, "end": 0.5},
            {"word": "Comment", "start": 0.6, "end": 0.8},
            {"word": "allez-vous?", "start": 0.9, "end": 1.2},
        ]
        blocks = words_to_srt_blocks(words, max_chars=50, max_duration=5.0)
        # Bonjour. should end a block, then Comment allez-vous? ends another
        assert len(blocks) == 2

    def test_empty_words_list(self) -> None:
        blocks = words_to_srt_blocks([])
        assert blocks == []

    def test_words_with_empty_text_skipped(self) -> None:
        words = [
            {"word": "", "start": 0.0, "end": 0.5},
            {"word": "valide", "start": 0.6, "end": 1.0},
        ]
        blocks = words_to_srt_blocks(words, max_chars=50, max_duration=5.0)
        assert len(blocks) == 1
        assert "valide" in blocks[0]["text"]

    def test_remnant_words_flushed(self) -> None:
        words = [
            {"word": "Un", "start": 0.0, "end": 0.3},
            {"word": "deux", "start": 0.4, "end": 0.7},
            {"word": "trois", "start": 0.8, "end": 1.1},
        ]
        blocks = words_to_srt_blocks(words, max_chars=50, max_duration=5.0)
        assert len(blocks) == 1
        assert "Un deux trois" in blocks[0]["text"]

    def test_duration_exceeded_triggers_new_block(self) -> None:
        words = [
            {"word": "Mot1", "start": 0.0, "end": 0.5},
            {"word": "Mot2", "start": 5.0, "end": 5.5},
        ]
        blocks = words_to_srt_blocks(words, max_chars=50, max_duration=1.0)
        # max_duration=1.0s, so after 5.0 - 0.0 = 5.0s > 1.0s, new block triggered
        assert len(blocks) == 2

    def test_timestamps_in_output(self) -> None:
        words = [
            {"word": "Test", "start": 1.5, "end": 2.0},
        ]
        blocks = words_to_srt_blocks(words)
        assert len(blocks) == 1
        assert blocks[0]["start"] == 1.5
        assert blocks[0]["end"] == 2.0


class TestSaveFiles:
    def test_saves_txt_and_srt(self, tmp_path) -> None:
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = save_files(
            text="Hello world",
            srt_blocks=[
                {"text": "Hello world", "start": 0.0, "end": 2.0}
            ],
            summary_text=None,
            audio_path=tmp_path / "audio.mp3",
            output_dir=out_dir,
        )
        assert paths["txt"].exists()
        assert paths["srt"].exists()
        content_txt = paths["txt"].read_text(encoding="utf-8")
        assert "Hello world" in content_txt

    def test_empty_srt_blocks_produces_error_subtitle(self, tmp_path) -> None:
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = save_files(
            text="test",
            srt_blocks=[],
            summary_text=None,
            audio_path=tmp_path / "audio.mp3",
            output_dir=out_dir,
        )
        content_srt = paths["srt"].read_text(encoding="utf-8")
        assert "Erreur" in content_srt or "Aucun timestamp" in content_srt

    def test_saves_summary_when_provided(self, tmp_path) -> None:
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = save_files(
            text="test",
            srt_blocks=[{"text": "t", "start": 0.0, "end": 1.0}],
            summary_text="Résumé test",
            audio_path=tmp_path / "audio.mp3",
            output_dir=out_dir,
        )
        assert "summary" in paths
        assert paths["summary"].exists()
