import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from pathlib import Path
import cv2
import os


def get_video_metadata(video_dir):
    """
    Extract metadata (fps, total_frames) for all videos in directory

    Args:
        video_dir: Path to directory containing videos

    Returns:
        Dictionary of video_id -> {'fps': fps, 'total_frames': total_frames}
    """
    metadata = {}
    video_extensions = [".mp4", ".avi", ".mkv", ".mov"]

    # Convert to Path object for easier handling
    video_path = Path(video_dir)

    # Get all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_path.glob(f"*{ext}"))

    for video_file in video_files:
        # Open video file
        cap = cv2.VideoCapture(str(video_file))

        if not cap.isOpened():
            print(f"Warning: Could not open video {video_file}")
            continue

        # Get video metadata
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Extract video ID from filename
        video_id = video_file.stem  # Gets filename without extension

        metadata[video_id] = {
            "fps": fps,
            "total_frames": total_frames,
            "path": str(video_file),
        }

        cap.release()

    print(f"Found {len(metadata)} videos")
    return metadata


def select_frames_from_annotations(annotations, fps, total_frames):
    """
    Select annotations corresponding to frames extracted at 2 fps

    Args:
        annotations: List of annotation scores
        fps: Original video fps
        total_frames: Total number of frames in original video

    Returns:
        numpy array of selected annotation scores
    """
    # Calculate frame interval for 2 fps
    frame_interval = int(fps / 2)
    target_fps = 2

    # Calculate total number of frames to extract
    num_frames = int(total_frames / frame_interval)

    # Convert annotations to numpy array if it's not already
    annotations = np.array(annotations)

    # Calculate the scaling factor between annotations length and total frames
    scale_factor = len(annotations) / total_frames

    # Select annotations corresponding to extracted frames
    selected_indices = [
        int(frame_idx * frame_interval * scale_factor)
        for frame_idx in range(num_frames)
    ]
    selected_indices = [
        min(i, len(annotations) - 1) for i in selected_indices
    ]  # Ensure indices don't exceed annotation length

    return annotations[selected_indices]


class TVSumDataset(Dataset):
    def __init__(self, features, annotations_df, video_metadata, num_classes=5):
        """
        Args:
            features: Dictionary of video_id -> feature array
            annotations_df: DataFrame containing annotations
            video_metadata: Dictionary of video_id -> metadata
            num_classes: Number of importance score classes (default 5 for scores 0-4)
        """
        self.features = []
        self.targets = []
        self.lengths = []

        # Find maximum sequence length
        max_seq_len = max(feature_array.shape[0] for feature_array in features.values())

        # Group annotations by video file name
        video_groups = annotations_df.groupby("Video File Name")

        for video_id, feature_array in features.items():
            if video_id not in video_metadata or video_id not in video_groups.groups:
                continue

            fps = video_metadata[video_id]["fps"]
            total_frames = video_metadata[video_id]["total_frames"]

            # Get annotations for this video
            video_annotations = video_groups.get_group(video_id)["Annotations"].tolist()

            # Process annotations
            processed_annotations = []
            for annotation in video_annotations:
                if isinstance(annotation, str):
                    scores = [
                        float(x.strip()) for x in annotation.strip("[]").split(",")
                    ]
                    processed_annotations.append(scores)
                else:
                    processed_annotations.append(annotation)

            # Process and average annotations
            selected_annotations = []
            for annotation in processed_annotations:
                selected_annotation = select_frames_from_annotations(
                    annotation, fps, total_frames
                )
                selected_annotations.append(selected_annotation)

            avg_annotation = np.mean(selected_annotations, axis=0)
            print("AVERAGE ANNOTATION", avg_annotation)

            # Convert continuous scores to discrete classes (0 to num_classes-1)
            # First clip to [0, 1] range to handle any outliers
            max_score = 5.0  # Maximum score in annotation scale
            avg_annotation = avg_annotation / max_score  # Normalize to [0, 1]
            avg_annotation = np.clip(avg_annotation, 0, 1)  # Handle any outliers

            # Scale to class range and round to nearest integer
            discrete_scores = np.round(avg_annotation * (num_classes - 1)).astype(int)
            discrete_scores = np.clip(discrete_scores, 0, num_classes - 1)
            print("DISCRETE ARE", discrete_scores)
            # Print some statistics for debugging
            print(f"Video {video_id} score stats:")
            print(f"Min score: {discrete_scores.min()}")
            print(f"Max score: {discrete_scores.max()}")
            print(f"Unique classes: {np.unique(discrete_scores)}")

            # Ensure lengths match
            min_length = min(len(discrete_scores), len(feature_array))
            feature_array = feature_array[:min_length]
            discrete_scores = discrete_scores[:min_length]

            # Pad sequences
            padded_features = np.zeros((max_seq_len, feature_array.shape[1]))
            padded_features[: len(feature_array)] = feature_array

            padded_targets = np.full(max_seq_len, -1)  # Fill with ignore_index
            padded_targets[: len(discrete_scores)] = discrete_scores

            self.features.append(torch.FloatTensor(padded_features))
            self.targets.append(torch.LongTensor(padded_targets))
            self.lengths.append(len(feature_array))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.lengths[idx]


class ImportancePredictor(nn.Module):
    def __init__(
        self, input_dim=4096, hidden_dim=512, num_layers=2, num_classes=5, dropout=0.5
    ):
        super().__init__()

        self.num_classes = num_classes

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, lengths):
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.bilstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        logits = self.classifier(output)

        return logits


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        """
        Args:
            pred: Predicted values (batch_size x seq_len)
            target: Target values (batch_size x seq_len)
            mask: Padding mask (batch_size x seq_len), 1 for real values, 0 for padding
        """
        # Apply mask to both predictions and targets
        masked_pred = pred * mask
        masked_target = target * mask

        # Calculate MSE only on non-padded values
        loss = (masked_pred - masked_target) ** 2

        # Average loss over non-padded elements only
        num_valid = mask.sum()
        if num_valid > 0:  # Avoid division by zero
            loss = loss.sum() / num_valid
        else:
            loss = loss.sum()  # If everything is padded (shouldn't happen)

        return loss


def train_model(
    features_dict,
    annotations_df,
    video_metadata,
    num_epochs=50,
    batch_size=32,
    lr=0.001,
    num_classes=5,
    patience=5,
):
    # Split data
    video_ids = list(features_dict.keys())
    train_ids, val_ids = train_test_split(video_ids, test_size=0.2, random_state=42)

    train_features = {vid: features_dict[vid] for vid in train_ids}
    val_features = {vid: features_dict[vid] for vid in val_ids}

    train_annotations = annotations_df[
        annotations_df["Video File Name"].isin(train_ids)
    ]
    val_annotations = annotations_df[annotations_df["Video File Name"].isin(val_ids)]

    # Create datasets
    train_dataset = TVSumDataset(
        train_features, train_annotations, video_metadata, num_classes=num_classes
    )
    val_dataset = TVSumDataset(
        val_features, val_annotations, video_metadata, num_classes=num_classes
    )

    def custom_collate(batch):
        # Sort batch by sequence length (descending)
        sorted_batch = sorted(batch, key=lambda x: x[2], reverse=True)

        # Separate features, targets, and lengths
        features = torch.stack([x[0] for x in sorted_batch])
        targets = torch.stack([x[1] for x in sorted_batch])
        lengths = [x[2] for x in sorted_batch]

        # Get the maximum sequence length in this batch
        max_len = max(lengths)

        # Trim features and targets to max length of this batch
        features = features[:, :max_len, :]
        targets = targets[:, :max_len]

        return features, targets, lengths

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=custom_collate
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImportancePredictor(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_model = None
    no_improve_epochs = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for features, targets, lengths in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(features, lengths)

            # Get actual batch size and sequence length
            B, T, C = logits.shape

            # Print shapes for debugging
            print(f"Logits shape: {logits.shape}")
            print(f"Targets shape: {targets.shape}")
            print(f"Reshaped logits shape: {logits.view(-1, C).shape}")
            # print(f"Reshaped targets shape: {targets.view(-1).shape}")

            # Ensure shapes match before computing loss
            loss = criterion(
                logits.contiguous().reshape(
                    -1, num_classes
                ),  # (batch*seq_len, num_classes)
                targets.contiguous().reshape(-1),  # (batch*seq_len)
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets, lengths in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                logits = model(features, lengths)
                B, T, C = logits.shape
                loss = criterion(
                    logits.contiguous().view(-1, C), targets.contiguous().view(-1)
                )
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"No improvement count: {no_improve_epochs}/{patience}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}!")
                break

    model.load_state_dict(best_model)
    return model


def main():
    # Set up paths
    video_dir = "Evaluation/TVSum/videos"
    annotations_path = "Evaluation/TVSum/df_annotations.pkl"
    features_dir = r"data/processed"

    # Get video metadata
    video_metadata = get_video_metadata(video_dir)

    # Print some statistics
    print("\nVideo Statistics:")
    for video_id, meta in video_metadata.items():
        print(f"Video {video_id}:")
        print(f"  FPS: {meta['fps']}")
        print(f"  Total Frames: {meta['total_frames']}")
        print(f"  Path: {meta['path']}")

    # Prepare features dictionary
    features_dict = {}
    for video_id in video_metadata:
        try:
            feature_path = os.path.join(features_dir, f"{video_id}/visual.npy")
            features = np.load(feature_path)
            features_dict[video_id] = features
            print(f"Loaded features for {video_id}: shape {features.shape}")
        except Exception as e:
            print(f"Warning: Could not load features for {video_id}: {e}")

    # Load annotations
    with open(annotations_path, "rb") as f:
        annotations_df = pickle.load(f)

    # Train model
    model = train_model(features_dict, annotations_df, video_metadata)

    # Save model
    torch.save(model.state_dict(), "tvsum_model.pth")


if __name__ == "__main__":
    print("HERE IT IS RUNNING")
    main()
