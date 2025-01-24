import numpy as np


def calculate_overlap(pred_segments, gt_segments):
    overlap = 0
    for p_start, p_end in pred_segments:
        for g_start, g_end in gt_segments:
            overlap += max(0, min(p_end, g_end) - max(p_start, g_start))
    return overlap


def compute_f1(pred_segments, gt_segments, video_length):
    overlap = calculate_overlap(pred_segments, gt_segments)
    precision = overlap / sum(p_end - p_start for p_start, p_end in pred_segments)
    recall = overlap / sum(g_end - g_start for g_start, g_end in gt_segments)
    return 2 * (precision * recall) / (precision + recall + 1e-8)
