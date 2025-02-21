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
from sklearn.model_selection import KFold
import time
from datetime import datetime, timedelta


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


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
    Process all frames instead of sampling at 2fps

    Args:
        annotations: List of annotation scores
        fps: Original video fps
        total_frames: Total number of frames in original video

    Returns:
        numpy array of annotation scores
    """
    annotations = np.array(annotations)

    # Calculate the scaling factor between annotations length and total frames
    scale_factor = len(annotations) / total_frames

    # Generate indices for all frames
    indices = [int(i * scale_factor) for i in range(total_frames)]
    indices = [min(i, len(annotations) - 1) for i in indices]

    return annotations[indices]


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
        self.features = []
        self.audio_features = []
        self.targets = []
        self.lengths = []

        video_groups = annotations_df.groupby("Video File Name")

        for video_id, feature_array in features.items():
            if (
                video_id not in video_metadata
                or video_id not in video_groups.groups
                or video_id not in audio_features
            ):
                continue

            # Process in chunks to save memory
            chunk_size = 1000
            num_chunks = (len(feature_array) + chunk_size - 1) // chunk_size

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(feature_array))

                feature_chunk = feature_array[start_idx:end_idx]
                audio_chunk = audio_features[video_id][start_idx:end_idx]

                # Process annotations for this chunk
                video_annotations = video_groups.get_group(video_id)[
                    "Annotations"
                ].tolist()
                processed_annotations = []
                for annotation in video_annotations:
                    if isinstance(annotation, str):
                        scores = [
                            float(x.strip()) for x in annotation.strip("[]").split(",")
                        ]
                        processed_annotations.append(scores[start_idx:end_idx])
                    else:
                        processed_annotations.append(annotation[start_idx:end_idx])

                avg_annotation = np.mean(processed_annotations, axis=0)
                discrete_scores = np.clip(
                    np.round(avg_annotation).astype(int), 1, num_classes
                )

                # Ensure target length matches feature chunk length
                if len(discrete_scores) > len(feature_chunk):
                    discrete_scores = discrete_scores[: len(feature_chunk)]
                elif len(discrete_scores) < len(feature_chunk):
                    padding_needed = len(feature_chunk) - len(discrete_scores)
                    discrete_scores = np.concatenate(
                        [discrete_scores, np.zeros(padding_needed)]
                    )  # Pad with 0s - consider if this is appropriate padding

                self.features.append(torch.FloatTensor(feature_chunk))
                self.audio_features.append(torch.FloatTensor(audio_chunk))
                self.targets.append(torch.LongTensor(discrete_scores))
                self.lengths.append(len(feature_chunk))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.audio_features[idx],
            self.targets[idx],
            self.lengths[idx],
        )


# class CrossModalAttention(nn.Module):
#     def __init__(self, vis_dim: int, aud_dim: int, hidden_dim: int = 768):
#         super().__init__()
#         self.vis_proj = nn.Linear(vis_dim, hidden_dim)
#         self.aud_proj = nn.Linear(aud_dim, hidden_dim)
#         self.attention = nn.MultiheadAttention(
#             hidden_dim, num_heads=8, batch_first=True
#         )

#         # Initialize weights
#         nn.init.xavier_uniform_(self.vis_proj.weight)
#         nn.init.xavier_uniform_(self.aud_proj.weight)

#     def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
#         # Project both modalities to same space
#         vis_proj = self.vis_proj(visual)  # [B, T, H]
#         aud_proj = self.aud_proj(audio)  # [B, T, H]

#         # Apply multi-head attention
#         attended_features, _ = self.attention(vis_proj, aud_proj, aud_proj)
#         return attended_features


class CrossModalAttention(nn.Module):
    def __init__(
        self, vis_dim: int, aud_dim: int, hidden_dim: int = 512
    ):  # Changed from 768
        super().__init__()
        self.vis_proj = nn.Linear(vis_dim, hidden_dim)
        self.aud_proj = nn.Linear(aud_dim, hidden_dim)

        # Fixed: Remove num_heads parameter from initialization
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=4,  # Hardcoded number of heads
            batch_first=True,
            dropout=0.5,  # Added dropout
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.vis_proj.weight)
        nn.init.xavier_uniform_(self.aud_proj.weight)

    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        # Project both modalities to same space
        vis_proj = self.vis_proj(visual)  # [B, T, H]
        aud_proj = self.aud_proj(audio)  # [B, T, H]

        # Apply multi-head attention
        attended_features, _ = self.attention(vis_proj, aud_proj, aud_proj)
        return attended_features


class AVSummarizer(nn.Module):
    def __init__(self, vis_dim: int = 4096, aud_dim: int = 384, num_classes=5):
        super().__init__()

        # Reduced hidden dimension to prevent overfitting
        self.hidden_dim = 512  # Reduced from 768

        # Visual feature processing with stronger regularization
        self.vis_dim_reduction = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(1024, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # LSTM with reduced complexity
        self.visual_encoder = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
            num_layers=2,  # Reduced from 3
            dropout=0.5,  # Increased dropout
        )

        # Audio feature processing with stronger regularization
        self.audio_encoder = nn.Sequential(
            nn.LayerNorm(aud_dim),
            nn.Linear(aud_dim, 512),
            nn.GELU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(512, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        # Cross-modal attention with reduced complexity
        self.cross_modal_attention = CrossModalAttention(
            vis_dim=self.hidden_dim,
            aud_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            # num_heads=4,  # Reduced from 8
        )

        # Simplified temporal modeling
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
        )

        # Reduced transformer complexity
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,  # Reduced from 12
            dim_feedforward=1024,  # Reduced from 2048
            dropout=0.5,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=2
        )  # Reduced from 4

        # Simplified prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self.post_conv_norm = nn.LayerNorm(self.hidden_dim)

        # Add L2 regularization
        self.l2_lambda = 0.01

    def forward(self, visual: torch.Tensor, audio: torch.Tensor):
        batch_size, seq_len = visual.shape[0], visual.shape[1]
        chunk_size = 64  # Reduced from 128
        outputs = []

        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)

            vis_chunk = visual[:, i:end_idx]
            aud_chunk = audio[:, i:end_idx]

            # Process features with residual connections and layer normalization
            vis_features = self.vis_dim_reduction(vis_chunk)
            vis_encoded, _ = self.visual_encoder(vis_features)

            aud_encoded = self.audio_encoder(aud_chunk)

            # Add skip connection
            attended_features = self.cross_modal_attention(vis_encoded, aud_encoded)
            combined = attended_features + vis_encoded

            # Temporal processing
            temp_conv = combined.transpose(1, 2)
            temp_conv = self.temporal_conv(temp_conv)
            temp_conv = temp_conv.transpose(1, 2)
            temp_conv = self.post_conv_norm(temp_conv)

            # Transformer with residual
            transformer_out = self.transformer(temp_conv + combined)

            chunk_output = self.head(transformer_out)
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=1)


def custom_collate(batch):
    """Memory-optimized collate function"""
    try:
        # Sort by length to minimize padding
        sorted_batch = sorted(batch, key=lambda x: x[3], reverse=True)

        # Get maximum sequence length for this batch
        max_len = min(sorted_batch[0][3], 512)  # Cap maximum sequence length

        features, audio_features, targets, lengths = [], [], [], []

        for f, a, t, l in sorted_batch:
            if f.shape[0] == 0 or a.shape[0] == 0 or t.shape[0] == 0:
                continue

            # Truncate sequences to max_len
            curr_len = min(l, max_len)
            f = f[:curr_len]
            a = a[:curr_len]
            t = t[:curr_len]

            # Create padded tensors
            padded_f = torch.zeros((max_len, f.shape[1]), dtype=torch.float32)
            padded_a = torch.zeros((max_len, a.shape[1]), dtype=torch.float32)
            padded_t = torch.full((max_len,), -1, dtype=torch.long)

            # Copy data
            padded_f[:curr_len].copy_(f)
            padded_a[:curr_len].copy_(a)
            padded_t[:curr_len].copy_(t)

            features.append(padded_f)
            audio_features.append(padded_a)
            targets.append(padded_t)
            lengths.append(curr_len)

        if not features:
            raise ValueError("No valid samples in batch")

        return (
            torch.stack(features),
            torch.stack(audio_features),
            torch.stack(targets),
            lengths,
        )
    except Exception as e:
        print(f"Error in collate_fn: {str(e)}")
        raise


class DataAugmentation:
    """Data augmentation techniques for time series data"""

    @staticmethod
    def add_noise(x, noise_factor=0.05):
        noise = torch.randn_like(x) * noise_factor
        return x + noise

    @staticmethod
    def time_mask(x, max_mask_size=10):
        B, T, C = x.shape
        mask_size = torch.randint(1, max_mask_size, (1,))
        mask_start = torch.randint(0, T - mask_size, (1,))
        x[:, mask_start : mask_start + mask_size, :] = 0
        return x


def train_model_kfold(
    features_dict,
    audio_features_dict,
    annotations_df,
    video_metadata,
    n_folds=5,
    num_epochs=60,
    batch_size=16,
    lr=0.0005,  # Reduced learning rate
    num_classes=5,
    patience=20,  # Reduced patience
    collate_fn=custom_collate,
):
    all_metrics = []
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    video_ids = list(features_dict.keys())

    for fold, (train_idx, val_idx) in enumerate(kfold.split(video_ids)):
        print(f"\nTraining Fold {fold + 1}/{n_folds}")

        train_ids = [video_ids[i] for i in train_idx]
        val_ids = [video_ids[i] for i in val_idx]

        train_dataset = TVSumDataset(
            {vid: features_dict[vid] for vid in train_ids},
            {vid: audio_features_dict[vid] for vid in train_ids},
            annotations_df[annotations_df["Video File Name"].isin(train_ids)],
            video_metadata,
            num_classes=num_classes,
        )

        val_dataset = TVSumDataset(
            {vid: features_dict[vid] for vid in val_ids},
            {vid: audio_features_dict[vid] for vid in val_ids},
            annotations_df[annotations_df["Video File Name"].isin(val_ids)],
            video_metadata,
            num_classes=num_classes,
        )

        # Data loaders with augmentation
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=2,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AVSummarizer(num_classes=num_classes).to(device)

        # Weighted loss to handle class imbalance
        class_weights = calculate_class_weights(train_dataset)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device), ignore_index=-1
        )

        # AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,  # L2 regularization
            betas=(0.9, 0.999),
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        best_val_loss = float("inf")
        best_model = None
        no_improve_epochs = 0

        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                features, audio_features, targets, lengths = [
                    x.to(device) if isinstance(x, torch.Tensor) else x for x in batch
                ]

                optimizer.zero_grad()

                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = model(features, audio_features)
                    loss = criterion(outputs.view(-1, num_classes), targets.view(-1))

                    # Add L2 regularization
                    l2_reg = 0
                    for param in model.parameters():
                        l2_reg += torch.norm(param)
                    loss += model.l2_lambda * l2_reg

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    features, audio_features, targets, lengths = [
                        x.to(device) if isinstance(x, torch.Tensor) else x
                        for x in batch
                    ]
                    outputs = model(features, audio_features)
                    loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict().copy()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print("Early stopping!")
                    break

        # Save fold results
        all_metrics.append(
            {
                "fold": fold + 1,
                "best_val_loss": best_val_loss,
                "final_train_loss": train_loss,
                "epochs_trained": epoch + 1,
            }
        )

        # Save best model for this fold
        torch.save(best_model, f"best_model_fold_{fold + 1}.pth")

    return all_metrics


def calculate_class_weights(dataset):
    """Calculate class weights to handle class imbalance"""
    targets = torch.cat([t for _, _, t, _ in dataset])
    valid_targets = targets[targets != -1]
    class_counts = torch.bincount(valid_targets)
    total = len(valid_targets)
    weights = total / (len(class_counts) * class_counts.float())
    return weights


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
    best_metrics = train_model_kfold(
        features_dict,
        audio_features_dict,
        annotations_df,
        video_metadata,
        collate_fn=custom_collate,
    )

    # Save model and metrics
    # print("\nSaving model and metrics...")
    # torch.save(model.state_dict(), "av_tvsum_model.pth")

    # Optionally save the metrics too
    with open("training_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    print("Model and metrics saved successfully!")


if __name__ == "__main__":
    print("HERE IT IS RUNNING")
    main()
