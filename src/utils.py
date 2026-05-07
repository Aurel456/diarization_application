# src/utils.py
import os
import torch
import logging
import pickle
from huggingface_hub import login
from typing import Any, Dict, Tuple, Callable, List, Optional
import time
import zipfile
import shutil
import pandas as pd
from pydub import AudioSegment

# Try importing ffmpeg-python
try:
    import ffmpeg
except ImportError:
    ffmpeg = None
    logging.warning("ffmpeg-python not installed. Video processing will be disabled.")


def adjust_timestamps_for_sequential_audio(df: pd.DataFrame, audio_paths: List[str]) -> pd.DataFrame:
    """
    Adjusts the 'start' and 'finish' timestamps in the DataFrame for sequentially recorded audio files.

    This function calculates the duration of each audio file and uses it to create a
    cumulative offset for the next file in the list, ensuring a continuous timeline.

    Args:
        df: The DataFrame containing transcription data with a 'base_audio_name' column.
        audio_paths: The ordered list of input audio file paths.

    Returns:
        A new DataFrame with adjusted timestamps, sorted by the new start time.
    """
    logging.info("Adjusting timestamps for sequential audio input...")
    if 'base_audio_name' not in df.columns:
        logging.error("Cannot adjust timestamps: 'base_audio_name' column is missing.")
        return df

    offsets = {}
    cumulative_offset = 0.0
   
    # Create a mapping from base audio name to its calculated offset
    for i, audio_path in enumerate(audio_paths):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
       
        # The first file has no offset
        if i == 0:
            offsets[base_name] = 0.0
            logging.info(f"File '{base_name}' is the first file, offset = 0.0s.")
        else:
            # The offset for the current file is the cumulative duration of all previous files
            offsets[base_name] = cumulative_offset
            logging.info(f"Calculated offset for '{base_name}': {cumulative_offset:.2f}s.")

        # Add the duration of the *current* file to the cumulative offset for the *next* file
        try:
            audio = AudioSegment.from_file(audio_path)
            duration_seconds = len(audio) / 1000.0
            cumulative_offset += duration_seconds
            logging.info(f"Duration of '{base_name}' is {duration_seconds:.2f}s. New cumulative offset: {cumulative_offset:.2f}s.")
        except Exception as e:
            logging.error(f"Could not read audio file '{audio_path}' to get duration: {e}. Aborting adjustment.")
            # If we can't get a duration, the rest of the offsets will be wrong.
            return df

    # Apply the calculated offsets to the DataFrame
    df['time_offset'] = df['base_audio_name'].map(offsets).fillna(0)
    df['start'] = df['start'] + df['time_offset']
    df['finish'] = df['finish'] + df['time_offset']

    # Drop the temporary offset column
    df = df.drop(columns=['time_offset'])

    # IMPORTANT: Re-sort the entire DataFrame by the new, corrected start times
    df_sorted = df.sort_values(by='start').reset_index(drop=True)
   
    logging.info("Timestamp adjustment complete. DataFrame has been re-sorted chronologically.")
   
    return df_sorted


def setup_environment(hf_token: str) -> torch.device:
    """
    Sets up the environment: Hugging Face login and device selection (GPU/CPU).

    Args:
        hf_token: Hugging Face authentication token.

    Returns:
        The selected torch device.

    Raises:
        ValueError: If hf_token is not provided and login is required.
    """
    # if hf_token:
    #     try:
    #         logging.info("Logging into Hugging Face Hub...")
    #         login(token=hf_token)
    #         logging.info("Hugging Face login successful.")
    #     except Exception as e:
    #         logging.error(f"An unexpected error occurred during Hugging Face login: {e}", exc_info=True)
    #         raise # Re-raise other unexpected errors
    # else:
    #     logging.info("No Hugging Face token provided. Skipping login. Will be using use_auth_token=None")

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available. Using CPU.")

    return device


def load_or_run(
    func: Callable,
    args: Tuple,
    pickle_path: str,
    description: str,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Loads data from a pickle file if it exists, otherwise runs the function,
    saves the result to the pickle file, and returns the result.

    Args:
        func: The function to execute if the pickle file doesn't exist.
        args: A tuple of positional arguments to pass to the function.
        pickle_path: The path to the pickle file for loading/saving.
        description: A short description of the step for logging purposes.
        kwargs: Optional dict of keyword arguments to pass to the function.

    Returns:
        The result from loading the pickle file or executing the function.

    Raises:
        Exception: If the function execution fails or pickling/unpickling fails.
    """
    if os.path.exists(pickle_path):
        logging.info(f"Attempting to load cached {description} from: {pickle_path}")
        try:
            with open(pickle_path, 'rb') as f:
                result = pickle.load(f)
            logging.info(f"Successfully loaded {description} from cache.")
            return result
        except (pickle.UnpicklingError, EOFError, FileNotFoundError, Exception) as e:
            logging.warning(f"Failed to load {description} from cache ({pickle_path}): {e}. Re-running step.")
            # Optionally remove the corrupted file: os.remove(pickle_path)
    else:
        logging.info(f"Cache file not found for {description}: {pickle_path}. Running step.")

    # Run the function if cache doesn't exist or loading failed
    try:
        logging.info(f"Executing function: {func.__name__} for {description}")
        start_time = time.time() # Add timing
        result = func(*args, **(kwargs or {}))
        end_time = time.time()
        logging.info(f"Function {func.__name__} completed in {end_time - start_time:.2f} seconds.")

        # Save the result
        try:
            # Ensure parent directory exists before saving
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
            with open(pickle_path, 'wb') as f:
                pickle.dump(result, f)
            logging.info(f"Successfully saved {description} results to cache: {pickle_path}")
        except (pickle.PicklingError, OSError, Exception) as e:
            logging.error(f"Failed to save {description} results to cache ({pickle_path}): {e}", exc_info=True)
            # Decide if you want to raise the error or just return the result without caching
            # raise # Raise error if caching is critical

        return result

    except Exception as e:
        logging.error(f"Error executing function {func.__name__} for {description}: {e}", exc_info=True)
        raise # Re-raise the exception to indicate the step failed


# --- NEW UTILS FOR VIDEO/ZIP SUPPORT ---

def convert_video_to_audio(video_path: str, output_ext: str = "wav") -> Optional[str]:
    """
    Extracts audio from a video file using ffmpeg-python.
    """
    if ffmpeg is None:
        logging.error("ffmpeg-python is not installed. Cannot process video.")
        return None

    # Replace video extension with audio extension
    base_name = os.path.splitext(video_path)[0]
    output_path = f"{base_name}.{output_ext}"
    
    # Avoid re-converting if it already exists (optional, mostly for safety)
    if os.path.exists(output_path):
        return output_path
    
    logging.info(f"Extracting audio from video: {os.path.basename(video_path)} -> {os.path.basename(output_path)}")
    
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec='libmp3lame', qscale=2, loglevel="error", y=None)
            .run(capture_stdout=True, capture_stderr=True)
        )
        return output_path
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode('utf8') if e.stderr else str(e)
        logging.error(f"FFmpeg error converting {video_path}: {error_msg}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error converting video {video_path}: {e}")
        return None


def extract_zip_content(zip_path: str, extract_to: str) -> List[str]:
    """
    Unzips a file and returns a list of all absolute paths to files extracted.
    """
    logging.info(f"Extracting zip file: {os.path.basename(zip_path)}")
    extracted_files = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
        # Walk through the extraction directory to find all files (handles nested folders)
        for root, _, files in os.walk(extract_to):
            for file in files:
                # Ignore hidden files or macOS metadata
                if not file.startswith(".") and "__MACOSX" not in root:
                    extracted_files.append(os.path.join(root, file))
                    
        return extracted_files
    except zipfile.BadZipFile:
        logging.error(f"Invalid zip file: {zip_path}")
        return []
    except Exception as e:
        logging.error(f"Error extracting zip {zip_path}: {e}")
        return []


def normalize_input_files(file_paths: List[str], temp_dir: str) -> List[str]:
    """
    Takes a list of raw input paths (Audio, Video, Zip) and returns a flat list 
    of processable Audio file paths. 
    Recursively handles Zips and converts Videos.
    """
    # Allowed extensions
    AUDIO_EXTS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
    VIDEO_EXTS = {'.mp4', '.mov', '.qt', '.avi', '.mkv', '.webm', '.wmv', '.mts', '.m2ts'}
    ZIP_EXTS = {'.zip'}
    
    final_audio_paths = []

    for path in file_paths:
        if not os.path.exists(path):
            continue
            
        ext = os.path.splitext(path)[1].lower()
        
        # 1. Handle ZIP
        if ext in ZIP_EXTS:
            # Create a specific folder for this zip content inside the temp dir
            zip_content_dir = os.path.join(temp_dir, f"unzip_{os.path.splitext(os.path.basename(path))[0]}")
            os.makedirs(zip_content_dir, exist_ok=True)
            
            extracted_files = extract_zip_content(path, zip_content_dir)
            # Recursively process extracted files (in case zip contains videos or zips)
            if extracted_files:
                final_audio_paths.extend(normalize_input_files(extracted_files, temp_dir))
            
        # 2. Handle VIDEO
        elif ext in VIDEO_EXTS:
            audio_path = convert_video_to_audio(path)
            if audio_path and os.path.exists(audio_path):
                final_audio_paths.append(audio_path)
            else:
                logging.warning(f"Failed to convert video: {path}")
        
        # 3. Handle AUDIO
        elif ext in AUDIO_EXTS:
            final_audio_paths.append(path)
            
        else:
            logging.debug(f"Skipping unsupported file type: {path}")

    # Remove duplicates and return sorted
    return sorted(list(set(final_audio_paths)))
