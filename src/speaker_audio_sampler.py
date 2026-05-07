# src/speaker_audio_sampler.py
"""
Extract representative audio clips for each detected speaker.
Used in the Streamlit UI so the user can listen and manually label speakers.

Returns separate clips (not concatenated) so the user can choose which one to play.
"""

import io
import logging
import os
from typing import Dict, List

import pandas as pd
from pydub import AudioSegment


def _load_clip(
    row: pd.Series,
    chunks_folder: str,
    segment_duration: int,
    max_duration_s: float = 15.0,
) -> bytes | None:
    """Extract one audio clip from the chunk file for the given DataFrame row."""
    chunk_idx = int(row.get("chunks", 0))
    base_audio = str(row.get("base_audio_name", ""))
    chunk_file = f"out{chunk_idx:03d}.wav"
    file_path = (
        os.path.join(chunks_folder, base_audio, chunk_file)
        if base_audio
        else os.path.join(chunks_folder, chunk_file)
    )

    if not os.path.exists(file_path):
        logging.debug("Speaker sample: chunk not found at %s", file_path)
        return None

    start_s = max(0.0, float(row["start"]) - chunk_idx * segment_duration)
    end_s = float(row["finish"]) - chunk_idx * segment_duration
    clip_end_s = start_s + min(end_s - start_s, max_duration_s)

    try:
        audio = AudioSegment.from_file(file_path)
        clip = audio[int(start_s * 1000) : int(clip_end_s * 1000)]
        if len(clip) < 300:  # skip clips shorter than 300 ms
            return None
        buf = io.BytesIO()
        clip.export(buf, format="wav")
        return buf.getvalue()
    except Exception as exc:
        logging.warning("Error extracting clip from %s: %s", file_path, exc)
        return None


def _is_segment_clean(
    seg_start: float,
    seg_end: float,
    base_audio: str,
    others: pd.DataFrame,
    margin_s: float,
) -> bool:
    """
    Return True if no other speaker has a segment overlapping
    [seg_start - margin_s, seg_end + margin_s] in the same source audio.

    Used to pick "clean" voice samples for speaker identification — avoids
    clips where another speaker is also audible (which would confuse the user
    listening to identify the voice).
    """
    if others.empty:
        return True
    a = seg_start - margin_s
    b = seg_end + margin_s
    if "base_audio_name" in others.columns and base_audio:
        candidates = others[others["base_audio_name"] == base_audio]
    else:
        candidates = others
    if candidates.empty:
        return True
    # Two intervals [a,b] and [c,d] overlap iff c < b and d > a
    overlap_mask = (candidates["start"] < b) & (candidates["finish"] > a)
    return not overlap_mask.any()


def extract_speaker_samples(
    transcript_df: pd.DataFrame,
    chunks_folder: str,
    segment_duration: int,
    n_samples: int = 2,
    max_clip_duration_s: float = 15.0,
    overlap_margin_s: float = 0.5,
) -> Dict[str, List[bytes]]:
    """
    For each detected speaker, return a list of up to n_samples separate audio clips.

    Clips are NOT concatenated — each is an independent WAV that can be played
    individually in the UI.

    Selection strategy: prefer "clean" segments where no other speaker has a
    segment overlapping `[start - margin, end + margin]` (default margin: 0.5 s)
    so the user hears only the target speaker. Fall back to non-clean segments
    if not enough clean ones are available.

    Args:
        transcript_df: DataFrame with global_speaker/speaker, start, finish,
                       chunks, base_audio_name columns.
        chunks_folder: Root folder containing per-audio-file chunk subdirectories.
        segment_duration: Duration of each chunk in seconds.
        n_samples: Number of clips to extract per speaker (default 2).
        max_clip_duration_s: Max duration of each individual clip in seconds.
        overlap_margin_s: Safety margin (in seconds) on each side of the segment
                          when checking for overlap with other speakers. Set to 0
                          to disable the margin and only check strict overlap.

    Returns:
        Dict mapping speaker_id -> list of WAV bytes (1 to n_samples items).
        Speakers with no extractable audio are omitted.
    """
    speaker_col = (
        "global_speaker" if "global_speaker" in transcript_df.columns else "speaker"
    )

    result: Dict[str, List[bytes]] = {}
    speakers = sorted(transcript_df[speaker_col].dropna().unique())

    for speaker_id in speakers:
        sid = str(speaker_id)
        if sid.lower() in ("noise", ""):
            continue

        rows = transcript_df[transcript_df[speaker_col] == speaker_id].copy()
        rows["_dur"] = rows["finish"] - rows["start"]
        # All segments from OTHER speakers (used for overlap detection)
        others = transcript_df[transcript_df[speaker_col] != speaker_id]

        # Tag each candidate as clean / not-clean (no overlap with margin)
        rows["_clean"] = rows.apply(
            lambda r: _is_segment_clean(
                float(r["start"]),
                float(r["finish"]),
                str(r.get("base_audio_name", "")),
                others,
                overlap_margin_s,
            ),
            axis=1,
        )

        # Prefer clean + long. Pick a generous candidate pool so we have
        # fallbacks if a clip extraction fails (file missing, too short, etc.).
        candidates = rows.sort_values(
            ["_clean", "_dur"], ascending=[False, False]
        ).head(n_samples * 4)

        clips: List[bytes] = []
        n_clean_used = 0
        for _, row in candidates.iterrows():
            if len(clips) >= n_samples:
                break
            wav = _load_clip(row, chunks_folder, segment_duration, max_clip_duration_s)
            if wav is not None:
                clips.append(wav)
                if row.get("_clean", False):
                    n_clean_used += 1

        if clips:
            result[sid] = clips
            if n_clean_used < len(clips):
                logging.info(
                    "Speaker %s: %d/%d clip(s) include other-speaker overlap "
                    "(no fully clean alternative within margin %.1fs).",
                    sid, len(clips) - n_clean_used, len(clips), overlap_margin_s,
                )
        else:
            logging.warning("No audio clips could be extracted for speaker %s", sid)

    return result
