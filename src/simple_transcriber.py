# src/simple_transcriber.py
import json
import logging
import os
import subprocess
import tempfile
import time
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from openai import OpenAI, OpenAIError
from pydub import AudioSegment

from src.exporter import clean_before_export
from src.summarizer import summarise_text

# --- UTILS ---

def format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

def retry(retries: int = 3, delay: int = 5):
    """Retry decorator for API calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except (OpenAIError, Exception) as exc:
                    if attempt < retries - 1:
                        logging.warning("API call failed (%s). Retrying (%d/%d) in %ds...", exc, attempt + 1, retries, delay)
                        time.sleep(delay)
                    else:
                        logging.error("API call failed after %d attempts: %s", retries, exc)
                        raise
        return wrapper
    return decorator

def preprocess_audio(input_path: Path, tmp_dir: Path) -> Tuple[Path, str]:
    """Preprocess audio to 16kHz mono FLAC using FFmpeg."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    clean_stem = re.sub(r'[^\w\-_\.]', '_', input_path.stem)
    output_path = tmp_dir / f"preprocessed_{clean_stem}.flac"
    
    try:
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', str(input_path), '-ar', '16000', '-ac', '1', '-c:a', 'flac', '-y', str(output_path)
        ], check=True, capture_output=True)
        logging.info("Audio preprocessed to 16kHz mono FLAC.")
        return output_path, 'flac'
    except subprocess.CalledProcessError as exc:
        if output_path.exists():
            output_path.unlink()
        error_msg = exc.stderr.decode() if exc.stderr else str(exc)
        raise RuntimeError(f"FFmpeg preprocessing failed: {error_msg}")

# --- GÉNÉRATION SRT ---

def words_to_srt_blocks(words: List[Dict], max_chars: int = 40, max_duration: float = 3.0) -> List[Dict]:
    """Regroupe des mots en blocs de sous-titres (Granularité fine)."""
    blocks = []
    current_block = []
    current_chars = 0
    current_start = None
    
    for word in words:
        if not isinstance(word, dict): continue
        
        w_text = word.get('word', '').strip()
        w_start = word.get('start', 0.0)
        w_end = word.get('end', 0.0)
        
        if not w_text: continue
        if current_start is None: current_start = w_start
        
        duration_exceeded = (w_end - current_start) > max_duration
        chars_exceeded = (current_chars + len(w_text) + 1) > max_chars
        sentence_end = w_text.endswith(('.', '?', '!'))
        
        if (duration_exceeded or chars_exceeded) and current_block:
            blocks.append({
                "text": " ".join([w['word'].strip() for w in current_block]),
                "start": current_block[0]['start'],
                "end": current_block[-1]['end']
            })
            current_block = []
            current_chars = 0
            current_start = w_start
        
        current_block.append(word)
        current_chars += len(w_text) + 1
        
        if sentence_end:
             blocks.append({
                "text": " ".join([w['word'].strip() for w in current_block]),
                "start": current_block[0]['start'],
                "end": current_block[-1]['end']
            })
             current_block = []
             current_chars = 0
             current_start = None

    if current_block:
        blocks.append({
            "text": " ".join([w['word'].strip() for w in current_block]),
            "start": current_block[0]['start'],
            "end": current_block[-1]['end']
        })
        
    return blocks

# --- TRANSCRIPTION ---
#
# This module supports two server capabilities:
#   1. Whisper-style endpoints that return `verbose_json` with word-level
#      timestamps — we use those directly for fine-grained SRT.
#   2. Endpoints that only return plain text (e.g. cohere-transcribe) — we
#      synthesize timestamps client-side via VAD/silence detection over the
#      audio, then transcribe each non-silent region independently.
#
# A single capability probe at startup picks the right path automatically.
# The result is cached per (server_url, model) for the process lifetime so
# repeated runs on the same server don't pay the probe cost.

# Module-level cache: (server_url, model) -> bool
_TIMESTAMP_CAPABILITY: Dict[Tuple[str, str], bool] = {}

# Heuristic patterns that identify a "verbose_json not supported" error
# (kept separate from genuine failures like auth/network errors).
_UNSUPPORTED_HINTS: Tuple[str, ...] = (
    "verbose_json",
    "do not support",
    "not supported",
    "timestamp_granularit",
    "unsupported",
    "invalid response_format",
)


def _looks_like_unsupported_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(hint in msg for hint in _UNSUPPORTED_HINTS)


def supports_verbose_json(
    client: OpenAI,
    whisper_model: str,
    server_url: str,
    sample: AudioSegment,
) -> bool:
    """
    Probe the server once to determine whether `verbose_json` with
    `timestamp_granularities=["word","segment"]` is accepted.

    Result is cached per (server_url, model). On a non-capability error
    (auth, model-not-found, network) the exception is re-raised so the
    caller sees the real problem instead of silently falling back.
    """
    key = (server_url or "", whisper_model)
    if key in _TIMESTAMP_CAPABILITY:
        return _TIMESTAMP_CAPABILITY[key]

    # Use up to 1.5s of real audio for the probe. Real audio probes more
    # reliably than silence on some servers that pre-filter empty input.
    probe_audio = sample[: min(1500, len(sample))] if len(sample) else sample
    if len(probe_audio) == 0:
        # No audio to probe with — assume verbose_json works; we'll fall back
        # at the first real call if not.
        return True

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
            tmp_path = tmp.name
            probe_audio.export(tmp.name, format="flac")
        with open(tmp_path, "rb") as f:
            client.audio.transcriptions.create(
                model=whisper_model,
                file=f,
                language="fr",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
        logging.info(
            "Server supports verbose_json (model=%s) — using fine-grained SRT path.",
            whisper_model,
        )
        _TIMESTAMP_CAPABILITY[key] = True
        return True
    except Exception as exc:
        if _looks_like_unsupported_error(exc):
            logging.warning(
                "Server rejects verbose_json (model=%s). Falling back to "
                "silence-based segmentation. Reason: %s",
                whisper_model, exc,
            )
            _TIMESTAMP_CAPABILITY[key] = False
            return False
        # Genuine failure (auth, model missing, network) — surface it.
        logging.error(
            "Capability probe failed for an unexpected reason (model=%s): %s",
            whisper_model, exc,
        )
        raise
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@retry(retries=2, delay=3)
def _transcribe_chunk(client: OpenAI, chunk: AudioSegment, whisper_model: str, prompt: str = None) -> Dict:
    """Transcribe a single audio chunk requesting verbose_json."""
    tmp_file_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp_file:
            tmp_file_name = tmp_file.name
            chunk.export(tmp_file.name, format="flac")

        with open(tmp_file_name, "rb") as audio_file:
            # On demande explicitement les granularities
            result = client.audio.transcriptions.create(
                model=whisper_model,
                file=audio_file,
                language="fr",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                prompt=prompt or "Transcription précise en français.",
            )

        if hasattr(result, 'to_dict'):
            data = result.to_dict()
        elif hasattr(result, 'model_dump'):
            data = result.model_dump()
        else:
            data = result

        # LOG DE DEBUG CRUCIAL : Voir quelles clés sont renvoyées
        logging.info(f"API Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        if 'words' in data and not data['words']:
            logging.warning("API returned 'words' key but it is empty.")

        return data

    except Exception as exc:
        logging.error("Chunk transcription failed: %s", exc)
        raise
    finally:
        if tmp_file_name and os.path.exists(tmp_file_name):
            os.unlink(tmp_file_name)


@retry(retries=2, delay=3)
def _transcribe_segment_text(
    client: OpenAI,
    segment: AudioSegment,
    whisper_model: str,
    prompt: Optional[str] = None,
) -> str:
    """
    Transcribe an audio segment using plain `json` (no timestamps).
    Used for endpoints like cohere-transcribe that don't support verbose_json.
    """
    tmp_file_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp_file:
            tmp_file_name = tmp_file.name
            segment.export(tmp_file.name, format="flac")

        with open(tmp_file_name, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=whisper_model,
                file=audio_file,
                language="fr",
                prompt=prompt or "Transcription précise en français.",
            )

        if hasattr(result, "text"):
            return (result.text or "").strip()
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()
    except Exception as exc:
        logging.error("Segment text-only transcription failed: %s", exc)
        raise
    finally:
        if tmp_file_name and os.path.exists(tmp_file_name):
            try:
                os.unlink(tmp_file_name)
            except OSError:
                pass


def _detect_speech_regions(
    audio: AudioSegment,
    min_silence_ms: int = 400,
    silence_thresh_db: float = -40.0,
    keep_silence_ms: int = 150,
) -> List[Tuple[int, int]]:
    """
    Return absolute (start_ms, end_ms) tuples for every non-silent region.

    The threshold is anchored to the audio's own dBFS so quiet recordings
    don't end up classified as fully silent. We add a little padding on
    each side to avoid clipping the first/last consonant.
    """
    from pydub.silence import detect_nonsilent

    # Anchor the silence threshold to the average loudness of the file —
    # absolute dBFS is brittle across recordings.
    base_db = audio.dBFS if audio.dBFS != float("-inf") else -40.0
    threshold = max(base_db - 16.0, silence_thresh_db)

    regions = detect_nonsilent(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=threshold,
    )
    if not regions:
        # Nothing detected as speech — treat the whole clip as one region
        # so we still produce *some* output instead of an empty SRT.
        return [(0, len(audio))]

    out: List[Tuple[int, int]] = []
    for s, e in regions:
        s = max(0, s - keep_silence_ms)
        e = min(len(audio), e + keep_silence_ms)
        out.append((int(s), int(e)))
    return out


def _split_long_region(
    start_ms: int, end_ms: int, max_block_ms: int = 7000
) -> List[Tuple[int, int]]:
    """Subdivide a region into <= max_block_ms slices of roughly equal length."""
    duration = end_ms - start_ms
    if duration <= max_block_ms:
        return [(start_ms, end_ms)]
    n = math.ceil(duration / max_block_ms)
    slice_len = duration / n
    return [
        (int(start_ms + i * slice_len), int(start_ms + (i + 1) * slice_len))
        for i in range(n)
    ]


def _transcribe_silence_based(
    audio: AudioSegment,
    client: OpenAI,
    whisper_model: str,
    max_block_ms: int = 7000,
) -> List[Dict[str, Any]]:
    """
    Fallback transcription path for endpoints that don't expose timestamps.
    Returns a list of SRT-ready blocks: ``{"text": str, "start": float, "end": float}``
    where start/end are absolute seconds in the original audio.
    """
    regions = _detect_speech_regions(audio)
    logging.info(
        "Silence-based segmentation: %d region(s) detected over %.1fs of audio.",
        len(regions),
        len(audio) / 1000.0,
    )

    blocks: List[Dict[str, Any]] = []
    last_text_tail: Optional[str] = None

    for region_idx, (r_start, r_end) in enumerate(regions, 1):
        sub_blocks = _split_long_region(r_start, r_end, max_block_ms=max_block_ms)
        for s_ms, e_ms in sub_blocks:
            sub_audio = audio[s_ms:e_ms]
            try:
                text = _transcribe_segment_text(
                    client,
                    sub_audio,
                    whisper_model,
                    prompt=last_text_tail,
                )
            except Exception as exc:
                logging.warning(
                    "Region %d (%.1fs-%.1fs) failed, skipping: %s",
                    region_idx, s_ms / 1000.0, e_ms / 1000.0, exc,
                )
                continue
            if not text:
                continue
            blocks.append({
                "text": text,
                "start": s_ms / 1000.0,
                "end": e_ms / 1000.0,
            })
            last_text_tail = text[-200:]

    return blocks

def transcribe_long_audio(
    audio_path: str,
    server_url: str,
    api_key: str,
    chunk_size: int = 300,
    overlap: int = 10,
    whisper_model: str = "whisper",
    enable_llm_cleaning: bool = True,
    enable_summary: bool = True,
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    output_dir: Optional[str] = None,
    tmp_dir: Optional[str] = None
) -> Dict[str, Any]:
    
    output_dir = Path(output_dir) if output_dir else Path("transcriptions")
    tmp_dir = Path(tmp_dir) if tmp_dir else Path("./tmp")
    audio_path = Path(audio_path)

    output_dir.mkdir(exist_ok=True, parents=True)
    tmp_dir.mkdir(exist_ok=True, parents=True)
    
    client = OpenAI(base_url=server_url, api_key=api_key)
    processed_path = None
    
    try:
        logging.info("Preprocessing audio...")
        processed_path, input_extension = preprocess_audio(audio_path, tmp_dir)
        audio = AudioSegment.from_file(processed_path)
        duration_ms = len(audio)

        chunk_ms = chunk_size * 1000
        overlap_ms = overlap * 1000

        # --- Capability probe — does the server support verbose_json? ---
        # If yes (Whisper-style): use word-level timestamps for fine SRT.
        # If no (cohere-transcribe and similar): switch to silence-based
        # segmentation across the whole audio.
        timestamps_available = supports_verbose_json(
            client, whisper_model, server_url, audio
        )

        raw_results = []
        all_words: List[Dict[str, Any]] = []
        all_segments: List[Dict[str, Any]] = []
        srt_blocks_fallback: List[Dict[str, Any]] = []

        if timestamps_available:
            cursor = 0
            chunk_idx = 0

            if duration_ms <= chunk_ms:
                num_chunks = 1
            else:
                num_chunks = math.ceil(duration_ms / (chunk_ms - overlap_ms))

            while cursor < duration_ms:
                end = min(cursor + chunk_ms, duration_ms)
                chunk = audio[cursor:end]

                logging.info(f"Processing chunk {chunk_idx+1} ({cursor/1000:.1f}s - {end/1000:.1f}s)")
                last_text = raw_results[-1]['text'][-200:] if raw_results and raw_results[-1].get('text') else None

                try:
                    transcription = _transcribe_chunk(client, chunk, whisper_model, prompt=last_text)
                except Exception as exc:
                    # If a chunk fails specifically because of verbose_json (rare —
                    # would mean the server's behaviour changed mid-run), flip the
                    # cache and re-run from scratch in fallback mode.
                    if _looks_like_unsupported_error(exc):
                        logging.warning(
                            "verbose_json rejected mid-run — switching to silence-based fallback."
                        )
                        _TIMESTAMP_CAPABILITY[(server_url or "", whisper_model)] = False
                        timestamps_available = False
                        raw_results = []
                        all_words = []
                        all_segments = []
                        break
                    raise

                offset_sec = cursor / 1000.0

                # --- Ajustement temporel pour Segments ET Mots ---
                segments = transcription.get('segments')
                if segments:
                    for seg in segments:
                        seg['start'] += offset_sec
                        seg['end'] += offset_sec

                words = transcription.get('words')
                if words:
                    for word in words:
                        word['start'] += offset_sec
                        word['end'] += offset_sec

                raw_results.append(transcription)
                cursor += (chunk_ms - overlap_ms)
                chunk_idx += 1

            logging.info("Merging chunks...")
            seen_words_intervals = []
            for res in raw_results:
                words = res.get('words')
                if words:
                    for w in words:
                        is_duplicate = False
                        for (s, e) in seen_words_intervals[-50:]:
                            if abs(w['start'] - s) < 0.05:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            all_words.append(w)
                            seen_words_intervals.append((w['start'], w['end']))

                segments = res.get('segments')
                if segments:
                    all_segments.extend(segments)

            all_words.sort(key=lambda x: x['start'])
            all_segments.sort(key=lambda x: x['start'])

        # If the probe (or a mid-run fallback) determined the server doesn't
        # support timestamps, build the SRT from silence-detected regions.
        if not timestamps_available:
            logging.info(
                "Using silence-based fallback path (model=%s).", whisper_model
            )
            srt_blocks_fallback = _transcribe_silence_based(
                audio, client, whisper_model
            )

        # --- Reconstruction texte complet ---
        if all_words:
            merged_text = " ".join([w['word'] for w in all_words])
        elif all_segments:
            merged_text = " ".join([s['text'].strip() for s in all_segments])
        elif srt_blocks_fallback:
            merged_text = " ".join(b['text'].strip() for b in srt_blocks_fallback)
        else:
            merged_text = " ".join([res.get('text', '') for res in raw_results])

        # Nettoyage
        if enable_llm_cleaning and api_key and merged_text:
            try:
                # Note: This is primarily for text cleaning/punctuation, not summarization
                # We assume clean_before_export logic here for now or specialized LLM call
                cleaned_text = apply_post_transcription_cleaning(merged_text, api_key, None, None)
            except Exception as e:
                logging.warning(f"Cleaning failed: {e}")
                cleaned_text = merged_text
        else:
            cleaned_text = merged_text

        # --- GÉNÉRATION SRT (avec cascade de fallbacks) ---
        logging.info("Generating SRT...")

        if all_words:
            logging.info(f"Using {len(all_words)} words for fine-grained SRT.")
            srt_blocks = words_to_srt_blocks(all_words, max_chars=45, max_duration=4.0)
        elif all_segments:
            logging.warning("⚠️ No word timestamps received. Falling back to SEGMENT timestamps for SRT.")
            srt_blocks = [
                {"text": seg['text'].strip(), "start": seg['start'], "end": seg['end']}
                for seg in all_segments
            ]
        elif srt_blocks_fallback:
            logging.info(
                "Using silence-based SRT (%d block(s)) — model has no timestamps.",
                len(srt_blocks_fallback),
            )
            srt_blocks = srt_blocks_fallback
        else:
            logging.error("❌ No timestamps available. Empty SRT.")
            srt_blocks = []

        # --- GÉNÉRATION DU RÉSUMÉ (indépendant du LLM cleaning) ---
        summary_text = None
        if enable_summary and llm_base_url and llm_model and cleaned_text:
            logging.info("Generating summary...")
            try:
                summary_text = summarise_text(cleaned_text, api_key, llm_base_url, llm_model)
            except Exception as e:
                logging.error(f"Failed to generate summary: {e}")
                summary_text = f"Erreur lors de la génération du résumé: {e}"
        
        paths = save_files(cleaned_text, srt_blocks, summary_text, audio_path, output_dir)
        
        return {
            "text": cleaned_text,
            "words": all_words,
            "txt_path": str(paths['txt']),
            "srt_path": str(paths['srt']),
            "summary_path": str(paths.get('summary')),
            "summary_text": summary_text
        }

    finally:
        if processed_path and processed_path.exists():
            try:
                processed_path.unlink()
            except Exception:
                pass

def apply_post_transcription_cleaning(text, api_key, base_url, model):
    text = clean_before_export(text)
    return text

def save_files(text: str, srt_blocks: List[Dict], summary_text: Optional[str], audio_path: Path, output_dir: Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    audio_path = Path(audio_path)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"{audio_path.stem}_{timestamp}"
    
    txt_path = output_dir / f"{base}.txt"
    txt_path.write_text(text, encoding='utf-8')
    
    srt_path = output_dir / f"{base}.srt"
    with open(srt_path, 'w', encoding='utf-8') as f:
        if srt_blocks:
            for i, block in enumerate(srt_blocks, 1):
                f.write(f"{i}\n")
                f.write(f"{format_timestamp_srt(block['start'])} --> {format_timestamp_srt(block['end'])}\n")
                f.write(f"{block['text'].strip()}\n\n")
        else:
            f.write("1\n00:00:00,000 --> 00:00:05,000\n[Erreur: Aucun timestamp disponible pour ce fichier]\n\n")
    
    paths = {"txt": txt_path, "srt": srt_path}

    if summary_text:
        summary_path = output_dir / f"{base}_summary.txt"
        summary_path.write_text(summary_text, encoding='utf-8')
        paths['summary'] = summary_path
            
    return paths
