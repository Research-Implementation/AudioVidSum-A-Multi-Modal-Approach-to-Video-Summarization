import numpy as np
import torch
from scipy.spatial.distance import cdist
from fastdtw import fastdtw


def compute_dtw(visual, audio):
    """Compute DTW cost matrix"""
    visual_np = visual.numpy()
    audio_np = audio.numpy()
    distance_matrix = cdist(visual_np, audio_np, metric="euclidean")
    return distance_matrix


def compute_optimal_path(dtw_matrix):
    """Get optimal warping path using FastDTW"""
    _, path = fastdtw(dtw_matrix, radius=10)
    return np.array(path)


def interpolate_features(features, path, target_length):
    """Interpolate features using alignment path"""
    aligned_indices = path[:, 0]
    unique_indices, counts = np.unique(aligned_indices, return_counts=True)
    weights = counts / counts.sum()

    # Weighted average interpolation
    aligned_features = []
    for idx, weight in zip(unique_indices, weights):
        aligned_features.append(features[idx] * weight)

    return torch.stack(aligned_features)[:target_length]
