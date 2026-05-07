# src/clusterer.py
import os
import logging
import numpy as np
import pandas as pd
from pyannote.core import Segment
from pyannote.audio import Audio, Model, Inference # <<< CHANGED: Added Inference
import soundfile as sf

# <<< REMOVED: SpeechBrain imports >>>
# from speechbrain.inference.speaker import EncoderClassifier
# import torchaudio
# import torchaudio.transforms as T

import hdbscan
from scipy.spatial.distance import cosine
import torch
from typing import Dict, Tuple, List, Optional, Any
from tqdm import tqdm

# Plotting imports
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Initialize tqdm for pandas apply
tqdm.pandas(desc="Computing Embeddings")

def compute_embedding(row: pd.Series, inference: Inference,
                      chunks_folder: str, segment_duration: int) -> Optional[np.ndarray]:
    """
    Single-segment embedding. Kept for backward compatibility — the pipeline
    now uses compute_embeddings_batch which is much faster.
    """
    chunk_index = row['chunks']
    base_audio_name = row['base_audio_name']
    chunk_filename = f"out{chunk_index:03d}.wav"
    file_path = os.path.join(chunks_folder, base_audio_name, chunk_filename)

    start_time_in_chunk = max(0.0, row['start'] - (chunk_index * segment_duration))
    end_time_in_chunk = max(start_time_in_chunk + 0.01, row['finish'] - (chunk_index * segment_duration))
    max_embedding_duration = 20.0
    if (end_time_in_chunk - start_time_in_chunk) > max_embedding_duration:
        end_time_in_chunk = start_time_in_chunk + max_embedding_duration

    if not os.path.exists(file_path):
        logging.error(f"Chunk file not found for embedding: {file_path}")
        return None

    try:
        embedding = inference.crop(file_path, Segment(start_time_in_chunk, end_time_in_chunk))
        if isinstance(embedding, np.ndarray) and embedding.ndim == 2:
            embedding = embedding.squeeze(0)
        return embedding if isinstance(embedding, np.ndarray) else np.array(embedding).squeeze()
    except Exception as e:
        logging.error(f"Error computing embedding for {file_path}: {e}", exc_info=True)
        return None


def compute_embeddings_batch(
    rep_segments: pd.DataFrame,
    inference: Inference,
    chunks_folder: str,
    segment_duration: int,
) -> List[Optional[np.ndarray]]:
    """
    Compute embeddings for all segments grouped by chunk file.

    Loads each chunk WAV only once via soundfile, then calls inference.crop
    with a pre-loaded waveform dict. For N segments sharing a chunk file this
    cuts file-reads from N to 1 — the dominant cost on long audio where each
    chunk has dozens of segments.
    """
    import torch

    results: List[Optional[np.ndarray]] = [None] * len(rep_segments)
    # Map original index position for each row
    positional_indices = list(range(len(rep_segments)))

    grouped = rep_segments.reset_index(drop=True).groupby(['base_audio_name', 'chunks'], sort=False)

    bar = tqdm(total=len(rep_segments), desc="Computing Embeddings (batched)")
    for (base_audio_name, chunk_index), group in grouped:
        chunk_filename = f"out{int(chunk_index):03d}.wav"
        file_path = os.path.join(chunks_folder, str(base_audio_name), chunk_filename)
        if not os.path.exists(file_path):
            logging.error(f"Chunk file not found for embedding: {file_path}")
            bar.update(len(group))
            continue

        try:
            waveform, sample_rate = sf.read(file_path, dtype='float32', always_2d=True)
            # soundfile returns (n_samples, n_channels); pyannote expects (channels, n_samples)
            waveform_t = torch.from_numpy(waveform.T)
            audio_dict = {"waveform": waveform_t, "sample_rate": sample_rate}
        except Exception as exc:
            logging.error(f"Failed to load chunk {file_path}: {exc}")
            bar.update(len(group))
            continue

        for pos_idx, (_, row) in zip(group.index.tolist(), group.iterrows()):
            start = max(0.0, row['start'] - chunk_index * segment_duration)
            end = max(start + 0.01, row['finish'] - chunk_index * segment_duration)
            if (end - start) > 20.0:
                end = start + 20.0
            try:
                emb = inference.crop(audio_dict, Segment(start, end))
                if isinstance(emb, np.ndarray) and emb.ndim == 2:
                    emb = emb.squeeze(0)
                elif not isinstance(emb, np.ndarray):
                    emb = np.array(emb).squeeze()
                results[pos_idx] = emb
            except Exception as exc:
                logging.warning(f"Embedding failed for segment {start:.2f}-{end:.2f} in {file_path}: {exc}")
                results[pos_idx] = None
            bar.update(1)

    bar.close()
    return results


def cluster_speakers(data: pd.DataFrame, device: torch.device, chunks_folder: str,
                     segment_duration: int, plot_folder: Optional[str] = None,
                     rep_segments_top_k: int = 3,
                     eps_factor: float = 0.5,
                     min_cluster_size: int = 2,
                     num_speakers: Optional[int] = None,
                     min_speakers: Optional[int] = None,
                     max_speakers: Optional[int] = None) \
        -> Tuple[Optional[Dict[Tuple[int, str], int]], pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Cluster speakers across chunks. When `num_speakers` is provided, uses
    AgglomerativeClustering with that exact count (overrides HDBSCAN). When
    `min_speakers`/`max_speakers` are provided, runs HDBSCAN first and falls
    back to AgglomerativeClustering with a clamped count if the result is
    outside the requested range.
    """
    
    logging.info(f"Selecting top {rep_segments_top_k} representative segments per speaker-chunk.")
    if 'segment_duration' not in data.columns:
        data['segment_duration'] = data['finish'] - data['start']

    # Tag each segment as "clean" (no overlap with other speakers in the same
    # chunk within a 0.5s margin). Embeddings computed on clean segments are
    # not polluted by background voices, which improves clustering quality.
    overlap_margin_s = 0.5
    try:
        data = data.copy()
        data['_clean'] = True
        for chunk_idx, chunk_group in data.groupby('chunks'):
            for idx, row in chunk_group.iterrows():
                a = float(row['start']) - overlap_margin_s
                b = float(row['finish']) + overlap_margin_s
                others = chunk_group[chunk_group['speaker'] != row['speaker']]
                if not others.empty:
                    overlap = (others['start'] < b) & (others['finish'] > a)
                    if overlap.any():
                        data.at[idx, '_clean'] = False
        n_clean = int(data['_clean'].sum())
        logging.info(
            "Embedding selection: %d/%d segments are clean (no other-speaker "
            "overlap within %.1fs margin).",
            n_clean, len(data), overlap_margin_s,
        )
    except Exception as e:
        logging.warning(f"Clean-segment tagging failed: {e}. Falling back to duration-only ranking.")
        data['_clean'] = True

    try:
        # Per (chunk, speaker): prefer clean + long segments
        def _pick_top(group: pd.DataFrame) -> pd.DataFrame:
            return group.sort_values(
                ['_clean', 'segment_duration'], ascending=[False, False]
            ).head(rep_segments_top_k)

        rep_segments = (
            data.groupby(['chunks', 'speaker'], group_keys=False)
                .apply(_pick_top)
                .copy()
        )
        logging.info(f"Selected {len(rep_segments)} segments for embedding.")
    except Exception as e:
        logging.error(f"Error selecting representative segments: {e}", exc_info=True)
        return None, pd.DataFrame(), None

    if rep_segments.empty:
        logging.error("No representative segments found after selection. Cannot proceed with clustering.")
        return None, pd.DataFrame(), None

    # <<< CHANGED: Load Pyannote WeSpeaker Model >>>
    try:
        # Use the path specified in your snippet
        local_dir = "model_storage/pyannote-wespeaker-voxceleb-resnet34-LM"
        logging.info(f"Loading speaker embedding model from: {local_dir}")
        
        # 1. Load Model
        model = Model.from_pretrained(local_dir)
        
        # 2. Instantiate Inference (window="whole" allows using .crop() flexibly)
        inference = Inference(model, window="whole")
        
        # 3. Move to GPU if available
        inference.to(device)
        logging.info(f"Model loaded and moved to {device}.")

    except Exception as e:
        logging.error(f"Failed to load Pyannote embedding model: {e}", exc_info=True)
        return None, pd.DataFrame(), None

    logging.info("Computing embeddings for representative segments (batched)...")
    rep_segments = rep_segments.reset_index(drop=True)
    rep_segments['embedding'] = compute_embeddings_batch(
        rep_segments, inference, chunks_folder, segment_duration
    )

    num_total_segments = len(rep_segments)
    valid_segments = rep_segments.dropna(subset=['embedding']).copy()
    num_failed_embeddings = num_total_segments - len(valid_segments)
    logging.info(f"Successfully computed {len(valid_segments)} embeddings ({num_failed_embeddings} failed).")

    if len(valid_segments) == 0:
         logging.error("No valid embeddings could be computed. Cannot perform clustering.")
         return None, valid_segments, None

    if len(valid_segments) < min_cluster_size:
        logging.warning(f"Only {len(valid_segments)} valid embeddings found, less than min_cluster_size ({min_cluster_size}).")

    # --- Calculate Epsilon Threshold (Heuristic) ---
    intra_distances = []
    percentile_threshold = None
    for name, group in valid_segments.groupby(['chunks', 'speaker']):
        embeddings = list(group['embedding'])
        if len(embeddings) > 1:
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    emb_i = embeddings[i].flatten()
                    emb_j = embeddings[j].flatten()
                    dist = cosine(emb_i, emb_j)
                    intra_distances.append(dist)

    if intra_distances:
        percentile_value = 90
        percentile_threshold = np.percentile(intra_distances, percentile_value)
        logging.info(f"[INFO ONLY] Intra-speaker distance ({percentile_value}th percentile): {percentile_threshold:.4f}")
    
    embeddings_matrix = np.vstack(valid_segments['embedding'].values)

    # Force Epsilon (as in previous code)
    selected_eps = 0.2
    logging.info(f"Forcing HDBSCAN cluster_selection_epsilon = {selected_eps:.4f}")

    clustering_details = {
        "calculated_percentile_intra_distance": percentile_threshold,
        "selected_epsilon": selected_eps,
        "min_cluster_size": min_cluster_size,
        "num_embeddings_clustered": len(valid_segments)
    }

    # --- Perform Clustering ---
    # HDBSCAN builds an internal k-NN graph (k = min_samples + 1); it fails
    # with "k must be less than or equal to the number of training points"
    # when embeddings are too few. Short audio often yields only 1-3 segments.
    clusters = np.array([-1] * len(valid_segments))
    n_points = embeddings_matrix.shape[0]
    hdbscan_min_points = max(min_cluster_size, 3)

    clustering_details["num_speakers_hint"] = num_speakers
    clustering_details["min_speakers_hint"] = min_speakers
    clustering_details["max_speakers_hint"] = max_speakers

    def _agglomerative(n_clusters: int, reason: str) -> np.ndarray:
        """Force exactly n_clusters via AgglomerativeClustering on cosine distances."""
        n_clusters = max(1, min(n_clusters, n_points))
        if n_clusters == 1:
            logging.info(f"AgglomerativeClustering: forcing single cluster ({reason}).")
            return np.zeros(n_points, dtype=int)
        logging.info(
            f"AgglomerativeClustering with n_clusters={n_clusters} ({reason})."
        )
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        return agg.fit_predict(embeddings_matrix)

    if n_points < hdbscan_min_points:
        logging.warning(
            f"Only {n_points} embeddings available (need >= {hdbscan_min_points} for HDBSCAN). "
            f"Treating all segments as a single speaker."
        )
        clusters = np.zeros(n_points, dtype=int)
        valid_segments['cluster_hdbscan'] = clusters
        clustering_details["num_clusters_found"] = 1
        clustering_details["num_noise_points"] = 0
        clustering_details["fallback_reason"] = f"Too few embeddings ({n_points}) for HDBSCAN"
        clustering_details["clustering_method"] = "single_cluster"
    elif num_speakers is not None and num_speakers >= 1:
        # Exact count known — skip HDBSCAN, use AgglomerativeClustering directly
        try:
            clusters = _agglomerative(num_speakers, f"exact num_speakers={num_speakers}")
            valid_segments['cluster_hdbscan'] = clusters
            clustering_details["num_clusters_found"] = len(set(clusters))
            clustering_details["num_noise_points"] = 0
            clustering_details["clustering_method"] = "agglomerative_exact"
        except Exception as e:
            logging.error(f"Agglomerative (exact) failed: {e}. Falling back to HDBSCAN.", exc_info=True)
            num_speakers = None  # fall through to HDBSCAN

    if (num_speakers is None or num_speakers < 1) and n_points >= hdbscan_min_points:
        # Run HDBSCAN — and if min/max bounds are set, clamp via AgglomerativeClustering when needed
        logging.info(f"Clustering {n_points} embeddings using HDBSCAN...")
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                cluster_selection_epsilon=selected_eps,
            )
            clusters = clusterer.fit_predict(embeddings_matrix)
            valid_segments['cluster_hdbscan'] = clusters
            num_clusters = len(set(clusters) - {-1})
            num_noise = int(np.sum(clusters == -1))
            logging.info(f"HDBSCAN found {num_clusters} speaker clusters and {num_noise} noise points.")
            clustering_details["num_clusters_found"] = num_clusters
            clustering_details["num_noise_points"] = num_noise
            clustering_details["clustering_method"] = "hdbscan"

            # Apply min/max clamp if requested
            target: Optional[int] = None
            if min_speakers is not None and num_clusters < min_speakers:
                target = min_speakers
            elif max_speakers is not None and num_clusters > max_speakers:
                target = max_speakers
            if target is not None:
                logging.info(
                    f"HDBSCAN found {num_clusters} clusters but bounds=[{min_speakers},{max_speakers}]; "
                    f"reclustering to {target} via AgglomerativeClustering."
                )
                clusters = _agglomerative(target, f"clamp to bounds {min_speakers}-{max_speakers}")
                valid_segments['cluster_hdbscan'] = clusters
                clustering_details["num_clusters_found"] = len(set(clusters))
                clustering_details["num_noise_points"] = 0
                clustering_details["clustering_method"] = "agglomerative_clamped"
        except Exception as e:
            logging.error(f"HDBSCAN clustering failed: {e}. Falling back to single-cluster assignment.", exc_info=True)
            clusters = np.zeros(n_points, dtype=int)
            valid_segments['cluster_hdbscan'] = 0
            clustering_details["num_clusters_found"] = 1
            clustering_details["num_noise_points"] = 0
            clustering_details["fallback_reason"] = f"HDBSCAN error: {e}"
            clustering_details["clustering_method"] = "single_cluster_fallback"


    # --- Create Mapping ---
    def get_majority_cluster(x):
        counts = x.value_counts()
        if counts.empty or (len(counts) == 1 and counts.index[0] == -1):
            return -1
        else:
            non_noise_counts = counts.drop(-1, errors='ignore')
            if not non_noise_counts.empty:
                return non_noise_counts.idxmax()
            else:
                return -1

    try:
         mapping_hdbscan = valid_segments.groupby(['chunks', 'speaker'])['cluster_hdbscan'].agg(get_majority_cluster).to_dict()
         logging.info("Speaker clustering and mapping complete.")
    except Exception as e:
         logging.error(f"Failed to create speaker mapping from cluster results: {e}", exc_info=True)
         mapping_hdbscan = None

    # --- Plotting Section (Unchanged logic, just ensure embeddings are ready) ---
    if plot_folder and not valid_segments.empty and 'embedding' in valid_segments.columns and len(valid_segments) > 1 :
        logging.info("Generating speaker embedding plot...")
        try:
            if 'embeddings_matrix' not in locals():
                 embeddings_matrix = np.vstack(valid_segments['embedding'].values)

            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings_matrix)

            # t-SNE internal k-NN requires 3 * perplexity < n_samples.
            # For very small datasets, skip the plot entirely.
            perplexity_value = min(30, len(scaled_embeddings) - 1)
            if len(scaled_embeddings) < 5 or perplexity_value <= 0:
                logging.warning(f"Skipping t-SNE plot: Not enough data points ({len(scaled_embeddings)}) for perplexity.")
            else:
                logging.info(f"Running t-SNE (perplexity={perplexity_value})...")
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, init='random', learning_rate='auto')
                embeddings_2d = tsne.fit_transform(scaled_embeddings) 

                plt.figure(figsize=(12, 10))
                unique_clusters = np.unique(clusters)
                colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))

                for cluster_id, color in zip(unique_clusters, colors):
                    idx = np.where(clusters == cluster_id)[0]
                    if cluster_id == -1:
                        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], c=[color], label=f'Noise', alpha=0.5, marker='x')
                    else:
                        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], c=[color], label=f'Speaker {cluster_id:02d}', alpha=0.8)

                plt.title(f'Speaker Embeddings (WeSpeaker) - eps={selected_eps:.2f}')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                plt.legend(loc='best', fontsize='small')
                plt.grid(True, linestyle='--', alpha=0.5)

                os.makedirs(plot_folder, exist_ok=True)
                plot_path = os.path.join(plot_folder, "speaker_embeddings_clusters.png")
                plt.savefig(plot_path)
                plt.close()
                logging.info(f"Embedding plot saved to: {plot_path}")

        except Exception as e:
            logging.error(f"Failed to generate or save plot: {e}", exc_info=True)
            
    if mapping_hdbscan is None:
        return None, pd.DataFrame(), None

    return mapping_hdbscan, valid_segments, clustering_details


def update_speaker_labels(data: pd.DataFrame, mapping_hdbscan: Dict[Tuple[int, str], int],
                          diarization_output_dir: str) -> Tuple[pd.DataFrame, str]:
    """
    Updates speaker labels in the DataFrame based on clustering results.
    (This function remains unchanged as it operates on the DataFrame results, not the model)
    """
    logging.info("Updating speaker labels based on clustering map.")

    def get_global_speaker(row):
        cluster_id = mapping_hdbscan.get((row['chunks'], row['speaker']), -1)
        if cluster_id == -1:
            return "Noise"
        else:
            return f"Speaker_{cluster_id:02d}"

    data['global_speaker'] = data.apply(get_global_speaker, axis=1)
    
    hdbscan_output_path = os.path.join(diarization_output_dir, "diarization_clustered.rttm")
    logging.info(f"Saving clustered diarization (RTTM format) to: {hdbscan_output_path}")
    try:
        with open(hdbscan_output_path, "w", encoding='utf-8') as outfile:
            for _, row in data.iterrows():
                onset = row['start']
                duration = row['finish'] - row['start']
                speaker_id = row['global_speaker'] 
                line = f"SPEAKER audio 1 {onset:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
                outfile.write(line)
    except Exception as e:
        logging.error(f"Failed to save clustered RTTM file: {e}", exc_info=True)

    return data, hdbscan_output_path
