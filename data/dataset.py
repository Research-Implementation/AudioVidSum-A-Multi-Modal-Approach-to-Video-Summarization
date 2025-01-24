import os
import torch
import pandas as pd
from scipy.io import loadmat
import numpy as np


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, feature_dir, annotation_path=None):
        self.feature_dir = feature_dir
        self.video_ids = os.listdir(feature_dir)
        self.annotations = (
            self._load_annotations(annotation_path) if annotation_path else None
        )

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        features = {
            "visual": torch.from_numpy(
                np.load(os.path.join(self.feature_dir, vid, "visual.npy"))
            ),
            "audio": torch.from_numpy(
                np.load(os.path.join(self.feature_dir, vid, "audio.npy"))
            ),
        }
        scores = torch.from_numpy(
            np.load(os.path.join(self.feature_dir, vid, "scores.npy"))
        )
        return features, scores


class TVSumDataset(BaseDataset):
    def _load_annotations(self, annotation_path):
        df = pd.read_csv(annotation_path, sep="\t", header=None)
        annotations = {}
        for _, row in df.iterrows():
            vid, scores = row[0], list(map(float, row[1].split(",")))
            annotations[vid] = scores
        return annotations


class SumMeDataset(BaseDataset):
    def _process_mat(self, mat_path):
        data = loadmat(mat_path)
        return data["gt_score"].squeeze()
