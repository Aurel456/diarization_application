# main2.py
import os
import time
import csv
import logging
import pickle
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import ast

# Import functions from src modules
from src.utils import setup_environment, load_or_run, adjust_timestamps_for_sequential_audio
from src.audio_splitter import split_audio_into_chunks
from src.diarizer import run_diarization, process_diarization_results
from src.clusterer import cluster_speakers, update_speaker_labels
from src.transcriber import transcribe_all_segments
from src.cleaner import process_all_text
from src.exporter import export_results

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.soundfile_backend")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils")

from dotenv import load_dotenv
load_dotenv(".env")

# Use a fixed run_id for reproducibility if needed, or keep timestamp
# run_id = time.strftime("%Y-%m-%d %H:%M:%S") # Use current time for unique runs
run_id = "2026-02-20 11:33:06"  # <-- restart run id
# run_id = "run_fix_audio"


def main():
    # --- Configuration Loading ---
    ROOT = os.getenv("ROOT", ".")
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN or HF_TOKEN == "None":
        HF_TOKEN = None
    logging.warning("Hugging Face Token (HF_TOKEN) not set. Recommended for Pyannote.")

    AUDIO_PROCESSING_MODE = os.getenv("AUDIO_PROCESSING_MODE", "concurrent").lower()
    if AUDIO_PROCESSING_MODE not in ["sequential", "concurrent"]:
        logging.warning(f"Invalid AUDIO_PROCESSING_MODE '{AUDIO_PROCESSING_MODE}'. Defaulting to 'concurrent'.")
        AUDIO_PROCESSING_MODE = "concurrent"

    SEGMENT_DURATION = int(os.getenv("SEGMENT_DURATION", 1200))
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 2))
    DUREE_MIN_SPEAKER = float(os.getenv("DUREE_MIN_SPEAKER", 0.5))
    VAD_FILTER = os.getenv("VAD_FILTER", "True").lower() == "true"

    input_audio_list_str = os.getenv("INPUT_AUDIO")
    if not input_audio_list_str:
        single_audio_file = os.getenv("INPUT_AUDIO_FILE")
        if single_audio_file:
            logging.warning("INPUT_AUDIO list variable not set, using single INPUT_AUDIO_FILE.")
            input_audio_paths: List[str] = [os.path.join(ROOT, single_audio_file.strip())]
        else:
            logging.error("Neither INPUT_AUDIO (list) nor INPUT_AUDIO_FILE (single) environment variable is set.")
            return
    else:
        try:
            parsed_list = ast.literal_eval(input_audio_list_str)
            if not isinstance(parsed_list, list):
                raise ValueError("INPUT_AUDIO should be a list of strings in the .env file.")
            input_audio_paths: List[str] = [os.path.join(ROOT, p.strip()) for p in parsed_list]
            if not input_audio_paths:
                raise ValueError("INPUT_AUDIO list cannot be empty.")
            logging.info(f"Parsed Input Audio Paths: {input_audio_paths}")
        except (ValueError, SyntaxError, TypeError) as e:
            logging.error(f"Error parsing INPUT_AUDIO variable: '{input_audio_list_str}'. Expected format like '[\"file1.mov\", \"file2.mov\"]'. Error: {e}", exc_info=True)
            return

    CHUNKS_FOLDER = os.path.join(ROOT, f"chunks_{SEGMENT_DURATION}s")

    SERVER_URL = os.getenv("SERVER_URL")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper")
    LLM_CLEANING_DISABLED = os.getenv("DISABLE_LLM_CLEANING", "False").lower() == "true"

    API_KEY = os.getenv("API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")

    # --- Run Setup ---
    experiments_dir = os.path.join(ROOT, "experiments", run_id)
    os.makedirs(experiments_dir, exist_ok=True)

    # Define run-specific paths
    diarization_results_run = os.path.join(experiments_dir, "diarization_results")
    output_folder_run = os.path.join(experiments_dir, "output_DOC")
    saved_state_dir_run = os.path.join(experiments_dir, "saved_state")
    log_file = os.path.join(experiments_dir, "logs.txt")
    plot_folder_run = os.path.join(experiments_dir, "plot")
    os.makedirs(plot_folder_run, exist_ok=True)

    # Create other necessary directories
    os.makedirs(CHUNKS_FOLDER, exist_ok=True)
    os.makedirs(diarization_results_run, exist_ok=True)
    os.makedirs(output_folder_run, exist_ok=True)
    os.makedirs(saved_state_dir_run, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"--- Starting Run: {run_id} ---")
    logging.info(f"Input Audio Files: {input_audio_paths}")
    logging.info(f"Audio Processing Mode: {AUDIO_PROCESSING_MODE}")
    logging.info(f"Chunk Duration: {SEGMENT_DURATION}s")
    logging.info(f"Chunk Folder: {CHUNKS_FOLDER}")
    logging.info(f"Min Speaker Duration: {DUREE_MIN_SPEAKER}s")
    logging.info(f"Whisper VAD Filter: {VAD_FILTER}")
    logging.info(f"Max Diarization Workers: {MAX_WORKERS}")
    logging.info(f"Whisper Model: {WHISPER_MODEL}, URL: {SERVER_URL}")
    logging.info(f"LLM CLEANING DISABLED: {LLM_CLEANING_DISABLED}")
    logging.info(f"LLM Model: {LLM_MODEL}, URL: {LLM_BASE_URL}")
    logging.info(f"Outputting to: {experiments_dir}")
    logging.info(f"Plot folder: {plot_folder_run}")

    if not SERVER_URL:
        logging.warning("SERVER_URL not set. Transcription will likely fail.")
    if not API_KEY or not LLM_MODEL or not LLM_BASE_URL:
        logging.warning("LLM API Key/Model/URL not set. Text cleaning will likely fail or use original text.")

    try:
        device = setup_environment(HF_TOKEN)
        logging.info(f"Environment setup complete. Using device: {device}")
    except Exception as e:
        logging.error(f"Failed to setup environment: {e}", exc_info=True)
        return

    # --- Pipeline Steps ---

    # Step 1: Split audio into chunks
    logging.info("--- Step 1: Audio Splitting ---")
    chunks_pickle = os.path.join(saved_state_dir_run, "step1_split_audio_result.pkl")

    # --- CORRECTED CHUNK VERIFICATION LOGIC ---
    # This logic now correctly checks for the existence of subdirectories (one for each input audio)
    # inside the main CHUNKS_FOLDER, as created by your audio_splitter.py script.
    all_chunks_found = True
    if not os.path.isdir(CHUNKS_FOLDER):
        all_chunks_found = False
    else:
        for audio_path in input_audio_paths:
            base_audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            expected_chunk_subdir = os.path.join(CHUNKS_FOLDER, base_audio_name)
            
            # Check if the subdirectory exists and contains at least one .wav file.
            try:
                if not os.path.isdir(expected_chunk_subdir) or not any(f.endswith('.wav') for f in os.listdir(expected_chunk_subdir)):
                    logging.info(f"Chunks not found for '{base_audio_name}' in '{expected_chunk_subdir}'.")
                    all_chunks_found = False
                    break # No need to check other files, we must re-run splitting.
            except FileNotFoundError:
                logging.info(f"Chunk directory not found for '{base_audio_name}': {expected_chunk_subdir}")
                all_chunks_found = False
                break


    if not all_chunks_found:
        logging.warning(f"One or more chunk directories are missing or empty in '{CHUNKS_FOLDER}'. Re-running audio splitting.")
        # If we re-run, remove the old cache file to avoid loading incomplete/stale data.
        if os.path.exists(chunks_pickle):
            try:
                os.remove(chunks_pickle)
                logging.info(f"Removed stale cache file: {chunks_pickle}")
            except OSError as e:
                logging.warning(f"Could not remove stale cache file {chunks_pickle}: {e}")

        # Execute the splitting process
        split_results = load_or_run(
            split_audio_into_chunks,
            args=(input_audio_paths, SEGMENT_DURATION, CHUNKS_FOLDER),
            pickle_path=chunks_pickle,
            description="Audio Splitting"
        )
        # Add a check to ensure splitting was successful
        if split_results is None or any(count is None or count == 0 for count in split_results):
            logging.error(f"Audio splitting failed or created 0 chunks in {CHUNKS_FOLDER}. Check input files and logs. Aborting.")
            return # Abort the script if splitting fails
        logging.info("Audio splitting completed successfully.")
    else:
        logging.info(f"All necessary chunks found in {CHUNKS_FOLDER}. Skipping splitting step.")
    # --- END OF CORRECTED LOGIC ---

    # Step 2: Run diarization on chunks
    logging.info("--- Step 2: Diarization ---")
    merged_output_path_pickle = os.path.join(saved_state_dir_run, "step2_diarization_merged_path.pkl")
    merged_diarization_path = load_or_run(
        run_diarization,
        args=(device, CHUNKS_FOLDER, diarization_results_run, HF_TOKEN, SEGMENT_DURATION, MAX_WORKERS),
        pickle_path=merged_output_path_pickle,
        description="Chunk Diarization"
    )
    if not merged_diarization_path or not os.path.exists(merged_diarization_path) or os.path.getsize(merged_diarization_path) == 0:
        logging.error(f"Diarization failed or merged RTTM path is empty/not found: {merged_diarization_path}. Aborting.")
        return

    # Step 3: Process diarization results into DataFrame
    logging.info("--- Step 3: Process Diarization Results ---")
    diarization_df_pickle = os.path.join(saved_state_dir_run, "step3_diarization_df.pkl")
    diarization_df = load_or_run(
        process_diarization_results,
        # args=(merged_diarization_path, diarization_results_run, SEGMENT_DURATION, DUREE_MIN_SPEAKER, AUDIO_PROCESSING_MODE, input_audio_paths),
        args=(merged_diarization_path, diarization_results_run, SEGMENT_DURATION, DUREE_MIN_SPEAKER),
        pickle_path=diarization_df_pickle,
        description="Diarization Results Processing"
    )
    if diarization_df.empty:
        logging.error("Diarization processing resulted in an empty DataFrame. Check RTTM content. Aborting.")
        return

    # Step 4: Cluster speakers across chunks
    logging.info("--- Step 4: Speaker Clustering ---")
    clustering_results_pickle = os.path.join(saved_state_dir_run, "step4_clustering_results.pkl")
    # This step will now work because diarization_df contains the correct `base_audio_name`.
    # Ensure your cluster_speakers function uses this column to build correct audio paths.
    clustering_results = load_or_run(
        cluster_speakers,
        args=(diarization_df, device, CHUNKS_FOLDER, SEGMENT_DURATION, plot_folder_run),
        pickle_path=clustering_results_pickle,
        description="Speaker Clustering"
    )
    if not clustering_results or clustering_results[0] is None:
        logging.error("Speaker clustering failed or produced no mapping. Check clustering logs. Aborting.")
        return
    mapping_hdbscan = clustering_results[0]

    # Step 5: Update speaker labels based on clustering
    logging.info("--- Step 5: Update Speaker Labels ---")
    updated_data_pickle = os.path.join(saved_state_dir_run, "step5_updated_data.pkl")
    updated_data_results = load_or_run(
        update_speaker_labels,
        args=(diarization_df, mapping_hdbscan, diarization_results_run),
        pickle_path=updated_data_pickle,
        description="Speaker Label Update"
    )
    if not updated_data_results or updated_data_results[0].empty:
        logging.error("Updating speaker labels failed or resulted in an empty DataFrame. Aborting.")
        return
    updated_data = updated_data_results[0]

    # Step 6: Transcribe segments using Whisper API
    logging.info("--- Step 6: Transcription ---")
    transcribed_data_pickle = os.path.join(saved_state_dir_run, "step6_transcribed_data.pkl")
    transcribed_data = load_or_run(
        transcribe_all_segments,
        args=(updated_data, CHUNKS_FOLDER, SERVER_URL, API_KEY ,WHISPER_MODEL, SEGMENT_DURATION, DUREE_MIN_SPEAKER, VAD_FILTER),
        pickle_path=transcribed_data_pickle,
        description="Transcription"
    )
    if transcribed_data.empty:
        logging.error("Transcription resulted in an empty DataFrame. Check Whisper API. Aborting.")
        return
    if 'transcription' not in transcribed_data.columns:
        logging.error("Transcription step completed but 'transcription' column is missing. Aborting.")
        return

    # Step 7: Clean transcriptions using LLM API
    logging.info("--- Step 7: Text Cleaning ---")
    if LLM_CLEANING_DISABLED:
        logging.info("LLM cleaning disabled – skipping process_all_text.")
        cleaned_data = transcribed_data.copy()          # on garde les transcriptions brutes
        cleaned_data["cleaned_transcription"] = cleaned_data["transcription"]
    else:
        cleaned_data_pickle = os.path.join(saved_state_dir_run, "step7_cleaned_data.pkl")
        cleaned_data = load_or_run(
            process_all_text,
            args=(transcribed_data, API_KEY, LLM_BASE_URL, LLM_MODEL),
            pickle_path=cleaned_data_pickle,
            description="Text Cleaning"
        )
        if cleaned_data.empty:
            logging.warning("Text cleaning resulted in an empty DataFrame. Export might be empty.")
        if 'cleaned_transcription' not in cleaned_data.columns:
            logging.warning("Text cleaning step ran but 'cleaned_transcription' column is missing. Using original.")
            cleaned_data['cleaned_transcription'] = cleaned_data['transcription'] # Fallback



    # <<< NEW: Step 7.5 - Adjust Timestamps if mode is sequential >>>
    if AUDIO_PROCESSING_MODE == 'sequential' and len(input_audio_paths) > 1:
        logging.info("--- Step 7.5: Adjusting Timestamps for Sequential Audio ---")
        cleaned_data = adjust_timestamps_for_sequential_audio(cleaned_data, input_audio_paths)
        # We can re-save the final dataframe if we want to cache this step too
        final_data_pickle = os.path.join(saved_state_dir_run, "step7_5_final_adjusted_data.pkl")
        with open(final_data_pickle, 'wb') as f:
            pickle.dump(cleaned_data, f)
        logging.info(f"Sequential timestamps adjusted. Final data cached at {final_data_pickle}")
    elif len(input_audio_paths) > 1:
        # If not sequential, still sort by start time to handle concurrent files properly before export
        cleaned_data = cleaned_data.sort_values(by='start').reset_index(drop=True)
        logging.info("Mode is 'concurrent', sorting data by start time before export.")
    # <<< END OF NEW STEP >>>



    # Step 8: Export final results to DOCX
    logging.info("--- Step 8: Export Results ---")
    try:
        exported_df = export_results(cleaned_data, input_audio_paths, output_folder_run)
        if exported_df is None:
            logging.error("Export function returned None, indicating an error during export.")
        elif exported_df.empty and not cleaned_data.empty:
            logging.warning("Export function received data but returned an empty DataFrame.")
        else:
            logging.info(f"Export complete. Results saved in {output_folder_run}")
    except Exception as e:
        logging.error(f"Failed to export results: {e}", exc_info=True)

    # --- Run Summary ---
    final_df_for_summary = cleaned_data
    num_speakers_final = "N/A"
    try:
        if not final_df_for_summary.empty and 'global_speaker' in final_df_for_summary.columns:
            num_speakers_final = final_df_for_summary[final_df_for_summary['global_speaker'] != 'Noise']['global_speaker'].nunique()
            logging.info(f"Number of unique non-noise speakers detected: {num_speakers_final}")
        elif final_df_for_summary.empty:
            logging.warning("Could not determine number of speakers (DataFrame is empty).")
        else:
            logging.warning("Could not determine number of speakers ('global_speaker' column missing?).")
    except Exception as e:
        logging.warning(f"Error calculating number of speakers: {e}")
        num_speakers_final = "Error"
    
    # Record experiment details
    experiments_csv = os.path.join(ROOT, "experiments", "experiments_log.csv")
    write_header = not os.path.exists(experiments_csv)
    base_audio_name_combined = '+'.join([os.path.splitext(os.path.basename(p))[0] for p in input_audio_paths])
    try:
        with open(experiments_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "run_id", "timestamp", "input_audio_combined", "segment_duration",
                    "min_speaker_duration", "vad_filter", "num_speakers_final", "status"
                ])
            writer.writerow([
                run_id, time.strftime("%Y-%m-%d %H:%M:%S"), base_audio_name_combined, SEGMENT_DURATION,
                DUREE_MIN_SPEAKER, VAD_FILTER, num_speakers_final, "Completed"
            ])
        logging.info(f"Experiment details logged to {experiments_csv}")
    except Exception as e:
        logging.error(f"Failed to log experiment details to CSV: {e}")

    logging.info(f"--- Run {run_id} Completed ---")

if __name__ == "__main__":
    main()