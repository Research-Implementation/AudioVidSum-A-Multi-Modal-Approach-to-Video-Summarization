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


# class TVSumDataset(Dataset):
#     def __init__(
#         self, features, audio_features, annotations_df, video_metadata, num_classes=5
#     ):
#         self.features = []
#         self.audio_features = []
#         self.targets = []
#         self.lengths = []
#         self.total_target = 0

#         # Group annotations by video file name
#         video_groups = annotations_df.groupby("Video File Name")

#         for video_id, feature_array in features.items():
#             if (
#                 video_id not in video_metadata
#                 or video_id not in video_groups.groups
#                 or video_id not in audio_features
#             ):
#                 continue

#             fps = video_metadata[video_id]["fps"]
#             total_frames = video_metadata[video_id]["total_frames"]
#             audio_feature_array = audio_features[video_id]

#             # Process annotations
#             video_annotations = video_groups.get_group(video_id)["Annotations"].tolist()
#             processed_annotations = []
#             for annotation in video_annotations:
#                 if isinstance(annotation, str):
#                     scores = [
#                         float(x.strip()) for x in annotation.strip("[]").split(",")
#                     ]
#                     processed_annotations.append(scores)
#                 else:
#                     processed_annotations.append(annotation)

#             # Process annotations for all frames
#             selected_annotations = []
#             for annotation in processed_annotations:
#                 selected_annotation = select_frames_from_annotations(
#                     annotation, fps, total_frames
#                 )
#                 selected_annotations.append(selected_annotation)

#             avg_annotation = np.mean(selected_annotations, axis=0)
#             discrete_scores = np.round(avg_annotation).astype(int)
#             discrete_scores = np.clip(discrete_scores, 1, num_classes)

#             # Skip if shapes don't match
#             if len(discrete_scores) != len(feature_array) or len(
#                 discrete_scores
#             ) != len(audio_feature_array):
#                 print(f"Skipping {video_id} due to shape mismatch:")
#                 print(
#                     f"Scores: {len(discrete_scores)}, Features: {len(feature_array)}, Audio: {len(audio_feature_array)}"
#                 )
#                 continue

#             self.features.append(torch.FloatTensor(feature_array))
#             self.audio_features.append(torch.FloatTensor(audio_feature_array))
#             self.targets.append(torch.LongTensor(discrete_scores))
#             self.lengths.append(len(feature_array))
#             self.total_target += len(discrete_scores)
#             print("DISCRETE SCORES ARE", len(discrete_scores))
#             print("THE TOTAL NUMBER OF TARGET IS", self.total_target)

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return (
#             self.features[idx],
#             self.audio_features[idx],
#             self.targets[idx],
#             self.lengths[idx],
#         )  # Return audio features


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


def train_model(
    features_dict,
    audio_features_dict,
    annotations_df,
    video_metadata,
    num_epochs=50,
    batch_size=32,
    lr=0.001,
    num_classes=5,
    patience=20,
    collate_fn=custom_collate,
):
    start_time = time.time()
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_amp = True
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")
        use_amp = False

    video_ids = list(features_dict.keys())
    train_ids, val_ids = train_test_split(video_ids, test_size=0.2, random_state=42)

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

    if device.type == "cpu":
        batch_size = min(batch_size, 8)
        num_workers = 0
    else:
        num_workers = 2

    print(f"Using batch size: {batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        timeout=120,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        timeout=120,
    )

    model = AVSummarizer(num_classes=num_classes).to(device)
    print(f"Model moved to {device}")

    scaler = torch.amp.GradScaler() if use_amp else None
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    best_val_loss = float("inf")
    best_model = None
    best_metrics = None
    no_improve_epochs = 0

    def calculate_metrics(predictions, targets):
        # Remove ignored indices (-1)
        mask = targets != -1
        predictions = predictions[mask]
        targets = targets[mask]

        if len(predictions) == 0:
            return 0, 0, 0, 0

        # Calculate accuracy
        accuracy = (predictions == targets).float().mean().item()

        # Calculate metrics for each class and average
        precisions = []
        recalls = []
        f1_scores = []

        for class_idx in range(1, num_classes + 1):  # Starting from 1 as per your setup
            true_positives = torch.sum(
                (predictions == class_idx) & (targets == class_idx)
            )
            false_positives = torch.sum(
                (predictions == class_idx) & (targets != class_idx)
            )
            false_negatives = torch.sum(
                (predictions != class_idx) & (targets == class_idx)
            )

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        # Calculate weighted averages
        class_weights = torch.bincount(targets, minlength=num_classes + 1)[1:].float()
        class_weights = class_weights / class_weights.sum()

        weighted_precision = sum(p * w for p, w in zip(precisions, class_weights))
        weighted_recall = sum(r * w for r, w in zip(recalls, class_weights))
        weighted_f1 = sum(f * w for f, w in zip(f1_scores, class_weights))

        return (
            weighted_precision.item(),
            weighted_recall.item(),
            weighted_f1.item(),
            accuracy,
        )

    print("Starting training...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    epoch_times = []
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets = []

        for features, audio_features, targets, lengths in train_loader:
            features = features.to(device)
            audio_features = audio_features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(visual=features, audio=audio_features)
                    loss = criterion(
                        logits.contiguous().view(-1, num_classes),
                        targets.contiguous().view(-1),
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(visual=features, audio=audio_features)
                loss = criterion(
                    logits.contiguous().view(-1, num_classes),
                    targets.contiguous().view(-1),
                )
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            train_predictions.extend(predictions.view(-1).cpu())
            train_targets.extend(targets.view(-1).cpu())

        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for features, audio_features, targets, lengths in val_loader:
                features = features.to(device)
                audio_features = audio_features.to(device)
                targets = targets.to(device)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(visual=features, audio=audio_features)
                        loss = criterion(
                            logits.contiguous().view(-1, num_classes),
                            targets.contiguous().view(-1),
                        )
                else:
                    logits = model(visual=features, audio=audio_features)
                    loss = criterion(
                        logits.contiguous().view(-1, num_classes),
                        targets.contiguous().view(-1),
                    )
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                val_predictions.extend(predictions.view(-1).cpu())
                val_targets.extend(targets.view(-1).cpu())

        # Calculate metrics
        train_predictions = torch.stack(train_predictions)
        train_targets = torch.stack(train_targets)
        val_predictions = torch.stack(val_predictions)
        val_targets = torch.stack(val_targets)

        train_precision, train_recall, train_f1, train_accuracy = calculate_metrics(
            train_predictions, train_targets
        )
        val_precision, val_recall, val_f1, val_accuracy = calculate_metrics(
            val_predictions, val_targets
        )

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)

        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        estimated_time_remaining = avg_epoch_time * (num_epochs - epoch - 1)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Time Statistics:")
        print(f"  Epoch Duration: {format_time(epoch_duration)}")
        print(f"  Average Epoch Time: {format_time(avg_epoch_time)}")
        if epoch < num_epochs - 1:
            print(
                f"  Estimated Time Remaining: {format_time(estimated_time_remaining)}"
            )

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training Metrics:")
        print(f"  Loss: {avg_train_loss:.4f}")
        print(f"  Accuracy: {train_accuracy:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  Recall: {train_recall:.4f}")
        print(f"  F1 Score: {train_f1:.4f}")
        print("Validation Metrics:")
        print(f"  Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {val_accuracy:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        print(f"  F1 Score: {val_f1:.4f}")

        if device.type == "cuda":
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            best_metrics = {
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            }
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}!")
                print("\nBest Validation Metrics:")
                print(f"  Accuracy: {best_metrics['val_accuracy']:.4f}")
                print(f"  Precision: {best_metrics['val_precision']:.4f}")
                print(f"  Recall: {best_metrics['val_recall']:.4f}")
                print(f"  F1 Score: {best_metrics['val_f1']:.4f}")
                break
    end_time = time.time()
    total_training_time = end_time - start_time

    print("\nTraining Complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {format_time(total_training_time)}")
    print(f"Average epoch time: {format_time(sum(epoch_times) / len(epoch_times))}")
    if len(epoch_times) > 1:
        print(f"Fastest epoch: {format_time(min(epoch_times))}")
        print(f"Slowest epoch: {format_time(max(epoch_times))}")

    # Add timing information to best_metrics
    best_metrics.update(
        {
            "total_training_time": total_training_time,
            "average_epoch_time": sum(epoch_times) / len(epoch_times),
            "fastest_epoch_time": min(epoch_times),
            "slowest_epoch_time": max(epoch_times),
            "num_epochs_completed": len(epoch_times),
        }
    )

    model.load_state_dict(best_model)
    return model, best_metrics


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
        features_dict,
        audio_features_dict,
        annotations_df,
        video_metadata,
        collate_fn=custom_collate,
    )

    # Save model
    torch.save(
        model.state_dict(), "av_tvsum_model.pth"
    )  # Saved model name changed to av_tvsum_model


if __name__ == "__main__":
    print("HERE IT IS RUNNING")
    main()
