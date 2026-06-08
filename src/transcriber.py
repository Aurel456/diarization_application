# src/transcriber.py
import os
import requests
import logging
from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm
import time
import tempfile
from typing import Any, List, Optional
from openai import OpenAI
tqdm.pandas()



def retry(retries=3, delay=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    if i < retries - 1:
                        logging.warning(f"Request failed: {e}. Retrying ({i+1}/{retries}) in {delay}s...")
                        time.sleep(delay)
                    else:
                        logging.error(f"Request failed after {retries} attempts: {e}")
                        raise
                except Exception as e:
                    logging.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                    raise
        return wrapper
    return decorator

@retry(retries=2, delay=3)
def _send_transcription_request(
    audio_path: str,
    server_url: str,
    api_key:str,
    whisper_model: str,
    language: str = "fr",
    vad_filter: bool = True,
    prompt: Optional[str] = None
    ) -> Optional[str]:
    """Sends audio file to Whisper API and returns transcription text."""

 
    # Création d'un client PIA avec la clé API et l'URL de base
    client_PIA = OpenAI(base_url=server_url, api_key=api_key)
    try:
        audio_file = open(audio_path, "rb")

        # Vérification de la connexion au serveur PIA et récupération de la liste des modèles disponibles
        modeles_disponibles = [modele.id for modele in client_PIA.models.list().data]
        assert len(modeles_disponibles) > 0, "Aucun modèle disponible."
        
        if whisper_model not in modeles_disponibles :
            logging.error('check if whisper is correctly deployed and that the model name match')
            return None

        create_kwargs = {
            "model": whisper_model,
            "file": audio_file,
        }
        # cohere-transcribe and similar non-Whisper endpoints reject the
        # `prompt` parameter — only include it for real Whisper-compatible models.
        if "cohere" not in whisper_model.lower():
            create_kwargs["prompt"] = prompt if prompt else "Tu es spécialisé dans la transcription du Français."
        reponse_client = client_PIA.audio.transcriptions.create(**create_kwargs)
        logging.debug(f"Transcription received: '{reponse_client.text[:50]}...'")
        return reponse_client.text
    
    except Exception as e:
        logging.error(f"Erreur de connexion au serveur PIA : {e}")
        return None
    finally:
        if 'audio_file' in locals():
            audio_file.close()


def transcribe_segment(row: pd.Series, chunks_folder: str, server_url: str, api_key:str, whisper_model: str,
                       segment_duration: int, duree_min_speaker: float, vad_filter: bool,
                       prompt_whisper: Optional[str] = None) -> str:
    """
    Extracts an audio segment, sends it for transcription via Whisper API.
    """
    duration = row['finish'] - row['start']
    if duration < duree_min_speaker:
        return ""

    # --- START OF FIX: Reconstruct the correct chunk path ---
    # The row now contains 'base_audio_name' which is the name of the subdirectory.
    # We must use it to construct the correct path to the chunk file.
    if 'base_audio_name' not in row or pd.isna(row['base_audio_name']):
        logging.error(f"Row is missing the 'base_audio_name' column to locate the chunk file. Row data: {row}")
        return ""

    if 'chunks' not in row.index:
        chunk_index = int(row['start'] // segment_duration)
    else:
        chunk_index = row['chunks']
    base_audio_name = row['base_audio_name'] # Get the subdirectory name
    
    # Construct the correct path including the subdirectory from the 'base_audio_name' column.
    chunk_filename = f"out{chunk_index:03d}.wav"
    file_path = os.path.join(chunks_folder, base_audio_name, chunk_filename)
    # --- END OF FIX ---

    # Calculate start/end times relative to the beginning of the chunk file
    start_time_in_chunk = row['start'] - (chunk_index * segment_duration)
    end_time_in_chunk = row['finish'] - (chunk_index * segment_duration)
    
    start_time_in_chunk = max(0, start_time_in_chunk)
    end_time_in_chunk = max(start_time_in_chunk + 0.01, end_time_in_chunk)

    if not os.path.exists(file_path):
        logging.error(f"Chunk file not found for transcription: {file_path}")
        return ""

    transcribed_text = ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_audio_path = tmp_file.name

    try:
        sound = AudioSegment.from_file(file_path)# format="wav" 
        start_ms = int(start_time_in_chunk * 1000)
        end_ms = int(end_time_in_chunk * 1000)
        trimmed_sound = sound[start_ms:end_ms]
        trimmed_sound.export(tmp_audio_path, format="wav")

        transcribed_text = _send_transcription_request(
            tmp_audio_path, server_url, api_key, whisper_model,
            vad_filter=vad_filter, prompt=prompt_whisper
        )
    except FileNotFoundError:
         logging.error(f"Chunk file not found during segment extraction: {file_path}")
         transcribed_text = ""
    except Exception as e:
        logging.error(f"Error processing or transcribing segment from {file_path} ({start_time_in_chunk:.2f}-{end_time_in_chunk:.2f}): {e}", exc_info=True)
        transcribed_text = ""
    finally:
        if os.path.exists(tmp_audio_path):
            try:
                os.unlink(tmp_audio_path)
            except OSError as e:
                logging.warning(f"Could not delete temporary file {tmp_audio_path}: {e}")

    return transcribed_text if transcribed_text is not None else ""


def transcribe_all_segments(
    data: pd.DataFrame,
    chunks_folder: str,
    server_url: str,
    api_key: str,
    whisper_model: str,
    segment_duration: int,
    duree_min_speaker: float,
    vad_filter: bool = True,
    progress_callback: Optional[callable] = None,
    cancel_event: Any = None,
) -> pd.DataFrame:
    """
    Transcribe all segments in parallel using the Whisper API.

    Args:
        progress_callback: Optional callable(done: int, total: int, current: str) -> None
                           called after each segment completes. Used by the UI for
                           fine-grained progress.
        cancel_event: Optional threading.Event; when set, stops submitting new work.
    """
    if data.empty:
        logging.warning("Input DataFrame for transcription is empty. Skipping.")
        data['transcription'] = pd.Series(dtype='str')
        return data

    if 'base_audio_name' not in data.columns:
        logging.error("The required 'base_audio_name' column is missing from the DataFrame. Cannot proceed with transcription.")
        data['transcription'] = ""
        return data

    if not server_url:
        logging.warning("WHISPER_URL not provided. Skipping transcription step.")
        data['transcription'] = ""
        return data

    prompt_whisper = """
    Le contexte est une réunion technique ou une table ronde professionnelle en français.
    Acronymes courants : SSI, DPM, RH, DP1, FAQ, AGRAF, GT, RIE, CFDT, BSI2, DGFIP, DTNUM, CSAR.
    Organisations mentionnées : la DTNUM (prononcé déténum), la DGFIP (prononcé dégéfip ou des géfip).
    """
    total = len(data)
    logging.info("Starting transcription of %d segments using Whisper model '%s'...", total, whisper_model)

    transcripts: List[str] = [""] * total
    done = 0

    for positional_idx, (_, row) in enumerate(data.reset_index(drop=True).iterrows()):
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            logging.warning("Transcription cancelled at %d/%d.", positional_idx, total)
            break

        text = transcribe_segment(
            row, chunks_folder, server_url, api_key, whisper_model,
            segment_duration, duree_min_speaker, vad_filter, prompt_whisper,
        )
        transcripts[positional_idx] = text or ""
        done += 1

        if progress_callback is not None:
            try:
                preview = (text or "")[:50].replace("\n", " ")
                progress_callback(done, total, preview)
            except Exception:
                pass  # progress callback must never break transcription

    data = data.reset_index(drop=True)
    data['transcription'] = transcripts

    logging.info("Transcription finished (%d/%d segments).", done, total)
    return data
