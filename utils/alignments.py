import torch


def align_shots_to_annotations(shot_boundaries, annotations, fps):
    """
    shot_boundaries: List of (start_frame, end_frame) tuples
    annotations: 1D array of frame-level scores
    fps: Frames per second of original video
    """
    shot_scores = []
    for start, end in shot_boundaries:
        # Convert shot frames to 2-second annotation intervals
        start_time = start / fps
        end_time = end / fps
        start_idx = int(start_time // 2)
        end_idx = int(end_time // 2) + 1

        # Get corresponding annotation segment
        segment = annotations[start_idx:end_idx]
        shot_scores.append(segment.mean())

    return torch.tensor(shot_scores)
