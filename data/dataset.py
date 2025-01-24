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
    def __init__(self, mat_annotations_df, feature_dir):
        self.annotations_df = mat_annotations_df
        self.video_ids = self.annotations_df["Video File Name"].unique()
        self.feature_dir = feature_dir

    def __getitem__(self, idx):
        vid = self.video_ids[idx]

        # Load features
        features = {
            "visual": torch.from_numpy(
                np.load(os.path.join(self.feature_dir, vid, "visual.npy"))
            ),
            "audio": torch.from_numpy(
                np.load(os.path.join(self.feature_dir, vid, "audio.npy"))
            ),
        }

        # Get all annotations for this video and average across users
        video_annos = self.annotations_df[
            self.annotations_df["Video File Name"] == vid
        ]["Annotations"]
        avg_scores = np.mean(
            [anno for anno in video_annos], axis=0
        )  # Shape: [n_frames]

        return features, torch.tensor(avg_scores).float()


class SumMeDataset(BaseDataset):
    def _process_mat(self, mat_path):
        data = loadmat(mat_path)
        return data["gt_score"].squeeze()


class AVSummaryDataset(torch.utils.data.Dataset):
    def __init__(self, feature_dir, annotation_path=None):
        self.feature_dir = feature_dir
        self.video_ids = [f for f in os.listdir(feature_dir) if f.endswith(".npy")]
        self.annotations = (
            self._load_annotations(annotation_path) if annotation_path else None
        )

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        features = np.load(os.path.join(self.feature_dir, vid))

        if self.annotations:
            video_id = os.path.splitext(vid)[0]
            scores = self._get_scores(video_id)
            return torch.FloatTensor(features), torch.FloatTensor(scores)
        return torch.FloatTensor(features)

    def _load_annotations(self, path):
        # Implement based on your annotation format (TSV/MAT)
        # Return dictionary: {video_id: np.array of scores}
        return load_annotations(path)

    def _get_scores(self, video_id):
        # Implement score alignment with shots
        return self.annotations.get(video_id, np.zeros(1))
