import numpy as np
from scipy.stats import spearmanr, kendalltau
import torch


def evaluate(model, dataset):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, scores in dataset:
            visual = features["visual"].unsqueeze(0).cuda()
            audio = features["audio"].unsqueeze(0).cuda()
            preds = model(visual, audio).cpu().squeeze()

            all_preds.append(preds.numpy())
            all_targets.append(scores.numpy())

    # Compute metrics
    f1_scores = []
    spearmans = []
    kendalls = []

    for pred, target in zip(all_preds, all_targets):
        binary_pred = (pred > np.mean(pred)).astype(int)
        binary_target = (target > np.mean(target)).astype(int)

        tp = np.logical_and(binary_pred, binary_target).sum()
        precision = tp / binary_pred.sum()
        recall = tp / binary_target.sum()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores.append(f1)

        spearmans.append(spearmanr(pred, target).correlation)
        kendalls.append(kendalltau(pred, target).correlation)

    return {
        "f1": np.mean(f1_scores),
        "spearman": np.mean(spearmans),
        "kendall": np.mean(kendalls),
    }
