import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # Import for F.softmax
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from pathlib import Path
import cv2
import json
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


def load_or_generate_metadata(video_dir=None, metadata_path="video_metadata.json"):
    """
    Load metadata from JSON or generate and save if not exists
    Returns metadata dictionary
    """
    metadata_path = Path(metadata_path)

    if metadata_path.exists():
        print(f"Loading cached metadata from {metadata_path}")
        with open(metadata_path, "r") as f:
            return json.load(f)

    print("Generating new video metadata...")
    metadata = get_video_metadata(video_dir)

    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


class TVSumDataset(Dataset):
    def __init__(
        self, features, audio_features, annotations_df, video_metadata, num_classes=5
    ):
        """
        Args:
            features: Dictionary of video_id -> visual feature array
            audio_features: Dictionary of video_id -> audio feature array
            annotations_df: DataFrame containing annotations
            video_metadata: Dictionary of video_id -> metadata
            num_classes: Number of importance score classes (default 5 for scores 0-4)
        """
        self.features = []
        self.audio_features = []  # Store audio features
        self.targets = []
        self.lengths = []
        self.total_target = 0
        # Find maximum sequence length (using visual features for now, assuming audio is same length)
        max_seq_len = max(feature_array.shape[0] for feature_array in features.values())

        # Group annotations by video file name
        video_groups = annotations_df.groupby("Video File Name")

        for video_id, feature_array in features.items():
            if (
                video_id not in video_metadata
                or video_id not in video_groups.groups
                or video_id not in audio_features
            ):  # Check for audio features too
                continue

            fps = video_metadata[video_id]["fps"]
            total_frames = video_metadata[video_id]["total_frames"]
            audio_feature_array = audio_features[video_id]  # Get audio features

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
            # Convert continuous scores to discrete classes (0 to num_classes-1)
            # First clip to [0, 1] range to handle any outliers
            max_score = 5.0  # Maximum score in annotation scale
            # avg_annotation = avg_annotation / max_score  # Normalize to [0, 1]
            # avg_annotation = np.clip(avg_annotation, 0, 1)  # Handle any outliers

            # # Scale to class range and round to nearest integer
            # discrete_scores = np.round(avg_annotation * (num_classes - 1)).astype(int)
            # discrete_scores = np.clip(discrete_scores, 0, num_classes - 1)
            avg_annotation = np.mean(selected_annotations, axis=0)
            # print("AVERAGE ANNOTATION", avg_annotation)

            # Directly discretize average annotations (rounding and clipping)
            discrete_scores = np.round(avg_annotation).astype(int)  # Round directly
            discrete_scores = np.clip(
                discrete_scores, 1, num_classes
            )  # Clip to 1-4 range
            # print("DISCRETE ARE", discrete_scores)
            # Print some statistics for debugging
            print(f"Video {video_id} score stats:")
            # print(f"Min score: {discrete_scores.min()}")
            # print(f"Max score: {discrete_scores.max()}")
            # print(f"Unique classes: {np.unique(discrete_scores)}")

            # Ensure lengths match for visual, audio and scores
            min_length = min(
                len(discrete_scores), len(feature_array), len(audio_feature_array)
            )  # Ensure audio length is also considered
            feature_array = feature_array[:min_length]
            audio_feature_array = audio_feature_array[
                :min_length
            ]  # Trim audio features too
            discrete_scores = discrete_scores[:min_length]

            # Pad sequences
            padded_features = np.zeros((max_seq_len, feature_array.shape[1]))
            padded_features[: len(feature_array)] = feature_array

            padded_audio_features = np.zeros(
                (max_seq_len, audio_feature_array.shape[1])
            )  # Pad audio features
            padded_audio_features[: len(audio_feature_array)] = audio_feature_array

            padded_targets = np.full(max_seq_len, -1)  # Fill with ignore_index
            padded_targets[: len(discrete_scores)] = discrete_scores

            self.features.append(torch.FloatTensor(padded_features))
            self.audio_features.append(
                torch.FloatTensor(padded_audio_features)
            )  # Append audio features
            self.targets.append(torch.LongTensor(padded_targets))
            self.lengths.append(
                len(feature_array)
            )  # Length is still based on visual feature length
            self.total_target += len(discrete_scores)
            print("DISCRETE SCORES ARE", len(discrete_scores))
            print("THE TOTAL NUMBER OF TARGET IS", self.total_target)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.audio_features[idx],
            self.targets[idx],
            self.lengths[idx],
        )  # Return audio features


class CrossModalAttention(nn.Module):
    """Bahdanau-style attention between visual and audio features"""

    def __init__(self, vis_dim: int, aud_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.vis_proj = nn.Linear(vis_dim, hidden_dim)
        self.aud_proj = nn.Linear(aud_dim, hidden_dim)
        self.energy = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.vis_proj.weight)
        nn.init.xavier_uniform_(self.aud_proj.weight)
        nn.init.xavier_uniform_(self.energy.weight)

    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        # Project both modalities to same space
        proj_vis = torch.tanh(self.vis_proj(visual))  # [B, T, H]
        proj_aud = torch.tanh(self.aud_proj(audio))  # [B, T, H]

        # Compute attention scores
        combined = proj_vis + proj_aud.unsqueeze(1)  # [B, T, T, H]
        energy = self.energy(torch.tanh(combined)).squeeze(-1)  # [B, T, T]
        attention = F.softmax(energy, dim=-1)

        return attention


class AVSummarizer(nn.Module):
    def __init__(self, vis_dim: int = 4096, aud_dim: int = 384, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

        # Reduce visual input dimension more aggressively
        self.vis_dim_reduction = nn.Sequential(
            nn.Linear(vis_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 256)
        )

        # Smaller LSTM for visual encoding
        self.visual_encoder = nn.LSTM(
            256, 128, bidirectional=True, batch_first=True, num_layers=1, dropout=0.3
        )

        # Smaller audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(aud_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # More efficient cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            256 + 64, 4, dropout=0.1, batch_first=True
        )

        # Smaller transformer with fewer heads and layers
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=256 + 64,  # Combined feature size
            nhead=4,  # Reduced number of attention heads
            dim_feedforward=512,  # Smaller feedforward dimension
            dropout=0.1,
            batch_first=True,
            norm_first=True,  # More stable training
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=1  # Reduced number of layers
        )

        # Smaller prediction head
        self.head = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)

    def forward(self, visual: torch.Tensor, audio: torch.Tensor):
        # Process in smaller chunks if sequence is too long
        batch_size, seq_len = visual.shape[0], visual.shape[1]
        chunk_size = 128  # Process sequences in chunks of 128 frames

        outputs = []
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)

            # Process chunk
            vis_chunk = visual[:, i:end_idx]
            aud_chunk = audio[:, i:end_idx]

            # Reduce dimensions
            vis_chunk = self.vis_dim_reduction(vis_chunk)
            vis_encoded, _ = self.visual_encoder(vis_chunk)
            aud_encoded = self.audio_encoder(aud_chunk)

            # Concatenate features
            combined = torch.cat([vis_encoded, aud_encoded], dim=-1)

            # Apply attention and transformer
            combined = self.transformer(combined)

            # Get predictions for chunk
            chunk_output = self.head(combined)
            outputs.append(chunk_output)

        # Concatenate chunks back together
        return torch.cat(outputs, dim=1)


def train_model(
    features_dict,
    audio_features_dict,  # Added audio features dict
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
    train_audio_features = {
        vid: audio_features_dict[vid] for vid in train_ids
    }  # Split audio features too
    val_audio_features = {
        vid: audio_features_dict[vid] for vid in val_ids
    }  # Split audio features too

    train_annotations = annotations_df[
        annotations_df["Video File Name"].isin(train_ids)
    ]
    val_annotations = annotations_df[annotations_df["Video File Name"].isin(val_ids)]

    # Create datasets
    train_dataset = TVSumDataset(
        train_features,
        train_audio_features,
        train_annotations,
        video_metadata,
        num_classes=num_classes,  # Pass audio features to dataset
    )
    val_dataset = TVSumDataset(
        val_features,
        val_audio_features,
        val_annotations,
        video_metadata,
        num_classes=num_classes,  # Pass audio features to dataset
    )

    def custom_collate(batch):
        # Sort batch by sequence length (descending)
        sorted_batch = sorted(
            batch, key=lambda x: x[3], reverse=True
        )  # Sort by length (index 3 now)

        # Separate features, audio_features, targets, and lengths
        features = torch.stack([x[0] for x in sorted_batch])
        audio_features = torch.stack(
            [x[1] for x in sorted_batch]
        )  # Stack audio features
        targets = torch.stack([x[2] for x in sorted_batch])
        lengths = [x[3] for x in sorted_batch]

        # Get the maximum sequence length in this batch
        max_len = max(lengths)

        # Trim features, audio_features and targets to max length of this batch
        features = features[:, :max_len, :]
        audio_features = audio_features[:, :max_len, :]  # Trim audio features too
        targets = targets[:, :max_len]

        return features, audio_features, targets, lengths  # Return audio features

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        pin_memory=True,  # Add this for better memory management
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=custom_collate, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AVSummarizer(num_classes=num_classes).to(device)  # Use AVSummarizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_model = None
    no_improve_epochs = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for (
            features,
            audio_features,
            targets,
            lengths,
        ) in train_loader:  # Get audio features from loader
            features = features.to(device)
            audio_features = audio_features.to(device)  # Move audio features to device
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(
                visual=features, audio=audio_features
            )  # Pass both visual and audio to model

            # Get actual batch size and sequence length
            B, T, C = logits.shape

            # Print shapes for debugging - removed as it's verbose, can re-enable if needed

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
            for (
                features,
                audio_features,
                targets,
                lengths,
            ) in val_loader:  # Get audio features from val loader
                features = features.to(device)
                audio_features = audio_features.to(
                    device
                )  # Move audio features to device
                targets = targets.to(device)

                logits = model(
                    visual=features, audio=audio_features
                )  # Pass both visual and audio to model
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
    # video_dir = "data/videos"
    annotations_path = "data/df_annotations.pkl"
    features_dir = r"data/processed"
    audio_features_dir = r"data/processed"  # Assuming audio features are in the same processed dir, adjust if needed
    metadata_path = "data/video_metadata.json"
    # Get video metadata
    video_metadata = load_or_generate_metadata(None, metadata_path)

    # Print some statistics
    print("\nVideo Statistics:")
    for video_id, meta in video_metadata.items():
        print(f"Video {video_id}:")
        print(f"  FPS: {meta['fps']}")
        print(f"  Total Frames: {meta['total_frames']}")
        print(f"  Path: {meta['path']}")

    # Prepare features dictionaries
    features_dict = {}
    audio_features_dict = {}  # Dictionary for audio features
    for video_id in video_metadata:
        try:
            feature_path = os.path.join(features_dir, f"{video_id}/visual.npy")
            features = np.load(feature_path)
            features_dict[video_id] = features
            print(f"Loaded visual features for {video_id}: shape {features.shape}")
        except Exception as e:
            print(f"Warning: Could not load visual features for {video_id}: {e}")

        try:  # Load audio features
            audio_feature_path = os.path.join(
                audio_features_dir, f"{video_id}/audio.npy"
            )  # Assuming audio features are in same dir, with 'audio.npy'
            audio_features = np.load(audio_feature_path)
            audio_features_dict[video_id] = audio_features
            print(f"Loaded audio features for {video_id}: shape {audio_features.shape}")
        except Exception as e:
            print(f"Warning: Could not load audio features for {video_id}: {e}")

    # Load annotations
    with open(annotations_path, "rb") as f:
        annotations_df = pickle.load(f)

    # Train model
    model = train_model(
        features_dict, audio_features_dict, annotations_df, video_metadata
    )  # Pass audio features dict to train_model

    # Save model
    torch.save(
        model.state_dict(), "av_tvsum_model.pth"
    )  # Saved model name changed to av_tvsum_model


if __name__ == "__main__":
    print("HERE IT IS RUNNING")
    main()
