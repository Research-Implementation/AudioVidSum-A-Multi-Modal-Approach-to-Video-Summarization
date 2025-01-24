def compute_temporal_f1(pred_shots, gt_shots, total_frames):
    overlap = sum(
        max(0, min(p_end, g_end) - max(p_start, g_start))
        for p_start, p_end in pred_shots
        for g_start, g_end in gt_shots
    )
    precision = overlap / sum(p_end - p_start for p_start, p_end in pred_shots)
    recall = overlap / sum(g_end - g_start for g_start, g_end in gt_shots)
    return 2 * (precision * recall) / (precision + recall + 1e-8)
