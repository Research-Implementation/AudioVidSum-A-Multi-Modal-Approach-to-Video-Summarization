# src/scripts/create_splits.py
import numpy as np
import json
import os


def create_splits(feature_dir="data/processed", output_file="splits.json"):
    videos = [v[:-4] for v in os.listdir(feature_dir) if v.endswith(".npy")]
    np.random.shuffle(videos)

    split_point = int(0.8 * len(videos))
    splits = {"train": videos[:split_point], "test": videos[split_point:]}

    with open(output_file, "w") as f:
        json.dump(splits, f)
