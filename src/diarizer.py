# src/diarizer.py 
import os
import torch
from pyannote.audio import Pipeline
import pyannote.audio
# print("pyannote.audio.__version__ :" , pyannote.audio.__version__)
# from pydub import AudioSegment
import concurrent.futures
from tqdm import tqdm
import glob
import pandas as pd
import logging
import time
# import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

def process_chunk_diarization(file_path: str, pipeline: Pipeline, device: torch.device,
                             segment_duration: int, output_dir: str) -> Optional[str]:
    """
    Processes one audio chunk file for speaker diarization.
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # --- MODIFICATION 1: Create a unique and informative file ID ---
    # This combines the parent directory name (the base audio name) and the chunk name.
    # e.g., "CSAR 070725 - Partie 1 sur 4___out000"
    parent_dir_name = os.path.basename(os.path.dirname(file_path))
    unique_file_id = f"{parent_dir_name}___{base_name}"
    output_file_path = os.path.join(output_dir, f"{unique_file_id}_diarization.txt")

    try:
        segment_index = int(base_name.replace('out',''))
    except ValueError:
        logging.error(f"Could not parse segment index from filename: {base_name}. Skipping.")
        return None

    time_offset = segment_index * segment_duration

    try:
        # Use the unique ID as the URI for the pipeline, so it's written to the RTTM output
        audio_data = {'uri': unique_file_id, 'audio': file_path}
    except Exception as e:
        logging.error(f"Error loading audio chunk {file_path}: {e}", exc_info=True)
        return None

    try:
        logging.debug(f"Diarizing chunk: {file_path}")
        diarization = pipeline(audio_data)

        formatted_output = ""
        for turn, speaker in diarization.speaker_diarization:
        # for turn, _, speaker in diarization.itertracks(yield_label=True): # old code was for pyannote 3.1
            
            abs_start = time_offset + turn.start
            # Write the unique file ID into the RTTM line
            # formatted_output += f"SPEAKER {unique_file_id} 1 {abs_start:.3f} {turn.duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
            formatted_output += f"{unique_file_id},{abs_start:.3f},{turn.duration:.3f},{speaker}\n"
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(formatted_output)
        logging.debug(f"Finished diarizing chunk: {file_path}, output: {output_file_path}")
        return output_file_path
    except Exception as e:
        logging.error(f"Error during diarization pipeline for chunk {file_path}: {e}", exc_info=True)
        return None


def run_diarization(device: torch.device, chunks_folder: str, diarization_results_dir: str,
                    hf_token: str, segment_duration: int, max_workers: int = 2) -> str:
    """
    Runs speaker diarization on all audio chunks in parallel and merges the results.
    """
    if not os.path.exists(diarization_results_dir):
        os.makedirs(diarization_results_dir)

    logging.info("Initializing Pyannote diarization pipeline (pyannote-speaker-diarization-community-1)...")
    try:
        # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        pipeline = Pipeline.from_pretrained(Path("model_storage/pyannote-speaker-diarization-community-1"))
        pipeline.to(device)
        logging.info("Pipeline loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Pyannote pipeline: {e}.", exc_info=True)
        raise

    audio_files = sorted(glob.glob(os.path.join(chunks_folder, "**" ,"out*.wav"), recursive=True))
    if not audio_files:
        logging.error(f"No audio chunks found in {chunks_folder}. Aborting diarization.")
        raise FileNotFoundError(f"No chunks found in {chunks_folder}")

    logging.info(f"Found {len(audio_files)} audio chunks. Starting diarization with {max_workers} workers...")

    start_time = time.time()
    processed_files_paths = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk_diarization, file_path, pipeline, device, segment_duration, diarization_results_dir): file_path
                   for file_path in audio_files}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Diarizing Chunks"):
            file_path = futures[future]
            try:
                result_path = future.result()
                if result_path:
                    processed_files_paths.append(result_path)
                else:
                    logging.warning(f"Chunk diarization failed for: {file_path}")
            except Exception as exc:
                logging.error(f'Chunk {file_path} generated an exception: {exc}', exc_info=True)

    processed_files_paths.sort()

    if not processed_files_paths:
        failed_count = len(audio_files) - len(processed_files_paths)
        raise RuntimeError(
            f"All {failed_count} chunk(s) failed diarization. "
            "Check logs above for individual errors (encoding, CUDA, model, etc.)."
        )

    merged_output_path = os.path.join(diarization_results_dir, "merged_diarization.rttm")
    logging.info(f"Merging {len(processed_files_paths)} individual results into {merged_output_path}")

    try:
        with open(merged_output_path, "w", encoding='utf-8') as outfile:
            for file_path in processed_files_paths:
                try:
                    with open(file_path, "r", encoding='utf-8') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    logging.warning(f"Could not read or append chunk result {file_path}: {e}")
    except Exception as e:
        logging.error(f"Failed to write merged diarization file {merged_output_path}: {e}", exc_info=True)
        raise

    total_time = time.time() - start_time
    logging.info(f"Diarization finished in {total_time:.2f} seconds.")
    return merged_output_path


def process_diarization_results(merged_rttm_path: str, output_dir: str,
                                segment_duration: int, duree_min_speaker: float) -> pd.DataFrame:
    """
    Processes a merged RTTM file into a structured DataFrame, merging consecutive turns
    and filtering short segments.
    """
    logging.info(f"Processing merged diarization file: {merged_rttm_path}")
    output_csv_path = os.path.join(output_dir, "processed_diarization.csv")

    if not os.path.exists(merged_rttm_path):
         logging.error(f"Merged RTTM file not found: {merged_rttm_path}")
         raise FileNotFoundError(f"Merged RTTM file not found: {merged_rttm_path}")

    try:
        col_names = ["file_id", "start", "duration","speaker"]
        data = pd.read_csv(merged_rttm_path, sep=',', header=None, names=col_names,
                           dtype={'start': float, 'duration': float, 'speaker': str})
        # --- MODIFICATION 2: Extract the base audio name from the unique file ID ---
        # 'file_id' is now 'CSAR 070725 - Partie 1 sur 4___out000'.
        # rsplit('___', 1)[0] safely extracts "CSAR 070725 - Partie 1 sur 4".
        data['base_audio_name'] = data['file_id'].apply(lambda x: x.rsplit('___', 1)[0])
        
        data['finish'] = data['start'] + data['duration']
        # Keep the new 'base_audio_name' column
        data = data[['start', 'finish', 'speaker', 'base_audio_name']]
        data = data.sort_values(by='start').reset_index(drop=True)

        data['chunks'] = (data['start'] // segment_duration).astype(int)

        # Merge consecutive segments from the same speaker. Also group by the new 'base_audio_name'
        # column to prevent merging segments from two different original audio files.
        data['group'] = ((data['speaker'] != data['speaker'].shift()) | \
                         (data['chunks'] != data['chunks'].shift()) | \
                         (data['base_audio_name'] != data['base_audio_name'].shift())).cumsum()

        # Aggregate grouped segments
        merged_data = data.groupby('group').agg(
            start=('start', 'min'),
            finish=('finish', 'max'),
            speaker=('speaker', 'first'),
            chunks=('chunks', 'first'),
            # Propagate the 'base_audio_name' through the aggregation
            base_audio_name=('base_audio_name', 'first')
        ).reset_index(drop=True)

        merged_data['segment_duration'] = merged_data['finish'] - merged_data['start']

        initial_count = len(merged_data)
        final_data = merged_data[merged_data['segment_duration'] >= duree_min_speaker].copy()
        filtered_count = initial_count - len(final_data)
        logging.info(f"Filtered out {filtered_count} segments shorter than {duree_min_speaker:.2f}s.")

        final_data.to_csv(output_csv_path, index=False)
        logging.info(f"Processed diarization results saved to: {output_csv_path}")
        return final_data

    except pd.errors.EmptyDataError:
        logging.error(f"Merged RTTM file is empty or unreadable: {merged_rttm_path}")
        # Return an empty DataFrame with the new column
        return pd.DataFrame(columns=['start', 'finish', 'speaker', 'chunks', 'segment_duration', 'base_audio_name'])
    except Exception as e:
        logging.error(f"Error processing diarization results from {merged_rttm_path}: {e}", exc_info=True)
        raise
