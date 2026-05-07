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
        
        raw_results = []
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
            
            transcription = _transcribe_chunk(client, chunk, whisper_model, prompt=last_text)
            
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
        
        all_words = []
        all_segments = [] # NEW: On garde aussi tous les segments pour le fallback
        seen_words_intervals = []
        
        for res in raw_results:
            # Accumuler les mots
            words = res.get('words')
            if words:
                for w in words:
                    is_duplicate = False
                    # Anti-doublon basique pour l'overlap
                    for (s, e) in seen_words_intervals[-50:]: 
                        if abs(w['start'] - s) < 0.05: 
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        all_words.append(w)
                        seen_words_intervals.append((w['start'], w['end']))
            
            # Accumuler les segments (pas de dédoublonnage complexe, on concatène)
            # C'est suffisant pour le mode fallback
            segments = res.get('segments')
            if segments:
                all_segments.extend(segments)
        
        all_words.sort(key=lambda x: x['start'])
        all_segments.sort(key=lambda x: x['start'])
        
        # Reconstruction texte complet
        if all_words:
            merged_text = " ".join([w['word'] for w in all_words])
        elif all_segments:
            merged_text = " ".join([s['text'].strip() for s in all_segments])
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

        # --- GÉNÉRATION SRT (AVEC FALLBACK) ---
        logging.info("Generating SRT...")
        
        if all_words:
            logging.info(f"Using {len(all_words)} words for fine-grained SRT.")
            srt_blocks = words_to_srt_blocks(all_words, max_chars=45, max_duration=4.0)
        elif all_segments:
            # FALLBACK: Si pas de mots, on utilise les segments (phrases)
            logging.warning("⚠️ No word timestamps received. Falling back to SEGMENT timestamps for SRT.")
            srt_blocks = []
            for seg in all_segments:
                srt_blocks.append({
                    "text": seg['text'].strip(),
                    "start": seg['start'],
                    "end": seg['end']
                })
        else:
            logging.error("❌ No timestamps (words or segments) available. Empty SRT.")
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
