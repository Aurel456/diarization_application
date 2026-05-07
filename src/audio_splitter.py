# src/audio_splitter.py
import os
import logging
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from typing import List, Optional


def split_audio_into_chunks(audio_files: List[str], chunk_duration: int, chunks_folder: str) -> Optional[List[int]]:
    """
    Splits the audio files into chunks of specified duration in seconds.

    Args:
        audio_files: List of paths to the input audio files (e.g., mp3, wav).
        chunk_duration: Duration of each chunk in seconds.
        chunks_folder: Folder to save the audio chunks.

    Returns:
        A list of the number of chunks created for each audio file, or None if an error occurred.
    """
    if not os.path.exists(chunks_folder):
        try:
            os.makedirs(chunks_folder)
            logging.info(f"Created chunks directory: {chunks_folder}")
        except OSError as e:
            logging.error(f"Failed to create directory {chunks_folder}: {e}")
            return None

    all_chunk_counts = []
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            logging.error(f"Audio file not found: {audio_file}")
            all_chunk_counts.append(None)
            continue

        audio_name = os.path.basename(audio_file)
        base_audio_name = os.path.splitext(audio_name)[0]
        audio_chunks_folder = os.path.join(chunks_folder, base_audio_name)

        if not os.path.exists(audio_chunks_folder):
            try:
                os.makedirs(audio_chunks_folder)
                logging.info(f"Created chunks directory for {audio_name}: {audio_chunks_folder}")
            except OSError as e:
                logging.error(f"Failed to create directory {audio_chunks_folder}: {e}")
                all_chunk_counts.append(None)
                continue

        try:
            logging.info(f"Loading audio file: {audio_file}")
            audio = AudioSegment.from_file(audio_file)
            logging.info(f"Audio loaded successfully. Duration: {len(audio) / 1000:.2f} seconds.")
        except CouldntDecodeError:
            logging.error(f"Could not decode audio file: {audio_file}. Check format/integrity.")
            all_chunk_counts.append(None)
            continue
        except Exception as e:
            logging.error(f"Error loading audio file {audio_file}: {e}", exc_info=True)
            all_chunk_counts.append(None)
            continue

        audio_duration_ms = len(audio)
        chunk_duration_ms = chunk_duration * 1000

        num_chunks = 0
        for i, start_ms in enumerate(range(0, audio_duration_ms, chunk_duration_ms)):
            end_ms = start_ms + chunk_duration_ms
            chunk = audio[start_ms:min(end_ms, audio_duration_ms)]  # Handle last chunk correctly
            #  Modified export – 48 kHz, mono, PCM‑S16LE WAV
            chunk = chunk.set_frame_rate(48000).set_channels(1)   # resample + mono
            chunk_filename = f"out{i:03d}.wav"
            chunk_path = os.path.join(audio_chunks_folder, chunk_filename)

            try:
                # Export as WAV with 16‑bit PCM (default for pydub)
                chunk.export(chunk_path, format="wav")
                num_chunks += 1
            except Exception as e:
                logging.error(f"Failed to export chunk {i}: {chunk_path}. Error: {e}")

        logging.info(f"Successfully created {num_chunks} chunks in {audio_chunks_folder}")
        all_chunk_counts.append(num_chunks)

    return all_chunk_counts
