import os
import torch
import pandas as pd
from scipy.io import loadmat
import numpy as np
from utils.alignments import align_shots_to_annotations


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


def process_tvsum_annotations(df):
    """Convert raw annotation dataframe to video-centric targets"""
    # Average scores across users and normalize per video
    processed = df.groupby("Video File Name")["Annotations"].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    )

    # Normalize scores to 0-1 range per video
    return processed.apply(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    ).to_dict()


def _calculate_shot_boundaries(self, total_frames):
    """Create shot windows based on actual frame count"""
    return [
        (i, min(i + self.fps * self.shot_window, total_frames))
        for i in range(0, total_frames, self.fps * self.shot_window)
    ]


class TVSumDataset(BaseDataset):
    def __init__(self, mat_annotations_df, feature_dir):
        self.annotations_df = mat_annotations_df
        self.annotations = process_tvsum_annotations(mat_annotations_df)
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

    def _load_annotations(self, vid):
        # Get frame-level scores
        frame_scores = self.annotations.get(vid, np.zeros(1))

        # Convert to shot-based targets using your alignment function
        shot_boundaries = self._calculate_shot_boundaries(len(frame_scores))
        return align_shots_to_annotations(
            shot_boundaries, torch.tensor(frame_scores).float(), self.fps
        )


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
