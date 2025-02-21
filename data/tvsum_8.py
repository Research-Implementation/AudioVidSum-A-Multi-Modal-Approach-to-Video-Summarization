import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # Import for F.softmax
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
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


class TVSumDataset(Dataset):
    def __init__(self, features, audio_features, annotations_df, video_metadata):
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

                # Calculate average annotation scores (continuous values)
                avg_annotation = np.mean(processed_annotations, axis=0)

                # Ensure target length matches feature chunk length
                if len(avg_annotation) > len(feature_chunk):
                    avg_annotation = avg_annotation[: len(feature_chunk)]
                elif len(avg_annotation) < len(feature_chunk):
                    padding_needed = len(feature_chunk) - len(avg_annotation)
                    avg_annotation = np.concatenate(
                        [avg_annotation, np.zeros(padding_needed)]
                    )

                self.features.append(torch.FloatTensor(feature_chunk))
                self.audio_features.append(torch.FloatTensor(audio_chunk))
                self.targets.append(
                    torch.FloatTensor(avg_annotation)
                )  # Changed to FloatTensor
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
    def __init__(self, vis_dim: int, aud_dim: int, hidden_dim: int = 768):
        super().__init__()
        self.vis_proj = nn.Linear(vis_dim, hidden_dim)
        self.aud_proj = nn.Linear(aud_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
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
    def __init__(self, vis_dim: int = 4096, aud_dim: int = 384):
        super().__init__()

        # Dimension constants
        self.hidden_dim = 768

        # Visual feature processing
        self.vis_dim_reduction = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # LSTM for visual features
        self.visual_encoder = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
            num_layers=3,
            dropout=0.4,
        )

        # Audio feature processing
        self.audio_encoder = nn.Sequential(
            nn.LayerNorm(aud_dim),
            nn.Linear(aud_dim, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            vis_dim=self.hidden_dim, aud_dim=self.hidden_dim, hidden_dim=self.hidden_dim
        )

        # Enhanced temporal modeling
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
        )

        # Transformer layers
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=12,
            dim_feedforward=2048,
            dropout=0.4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=4)

        # Prediction head (modified for continuous output)
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),  # Changed to output 1 continuous value
            nn.Sigmoid(),  # Added to bound the output between 0 and 1
        )

        self.post_conv_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, visual: torch.Tensor, audio: torch.Tensor):
        batch_size, seq_len = visual.shape[0], visual.shape[1]
        chunk_size = 128
        outputs = []

        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)

            # Process chunks
            vis_chunk = visual[:, i:end_idx]
            aud_chunk = audio[:, i:end_idx]

            # Process visual features with residual connection
            vis_features = self.vis_dim_reduction(vis_chunk)
            vis_residual = vis_features
            vis_encoded, _ = self.visual_encoder(vis_features)
            vis_encoded = vis_encoded + vis_residual

            # Process audio features
            aud_encoded = self.audio_encoder(aud_chunk)

            # Apply cross-modal attention
            attended_features = self.cross_modal_attention(vis_encoded, aud_encoded)

            # Add residual connection
            combined = attended_features + vis_encoded

            # Apply temporal convolution
            temp_conv = combined.transpose(1, 2)  # [B, T, H] -> [B, H, T]
            temp_conv = self.temporal_conv(temp_conv)
            temp_conv = temp_conv.transpose(1, 2)  # [B, H, T] -> [B, T, H]
            temp_conv = self.post_conv_norm(
                temp_conv
            )  # Apply normalization after reshaping

            # Apply transformer with residual connection
            transformer_out = self.transformer(temp_conv + combined)

            # Get predictions
            chunk_output = self.head(transformer_out)
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=1)


def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculate both regression and classification metrics while handling ignored indices

    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth values
        threshold (float): Threshold for converting continuous values to binary

    Returns:
        tuple: (mse, mae, correlation, precision, recall, f1, accuracy)
    """
    # Remove ignored indices (-1) and padding (0)
    mask = (targets != -1) & (targets != 0)
    predictions = predictions[mask]
    targets = targets[mask]

    if len(predictions) == 0:
        return 0, 0, 0, 0, 0, 0, 0

    # Calculate regression metrics
    mse = F.mse_loss(predictions, targets).item()
    mae = F.l1_loss(predictions, targets).item()

    # Calculate Pearson correlation
    predictions_mean = predictions.mean()
    targets_mean = targets.mean()
    numerator = torch.sum((predictions - predictions_mean) * (targets - targets_mean))
    denominator = torch.sqrt(
        torch.sum((predictions - predictions_mean) ** 2)
        * torch.sum((targets - targets_mean) ** 2)
    )
    correlation = (numerator / denominator).item() if denominator != 0 else 0

    # Convert to binary predictions using threshold for classification metrics
    binary_predictions = (predictions > threshold).float()
    binary_targets = (targets > threshold).float()

    # Calculate classification metrics
    true_positives = torch.sum(
        (binary_predictions == 1) & (binary_targets == 1)
    ).float()
    false_positives = torch.sum(
        (binary_predictions == 1) & (binary_targets == 0)
    ).float()
    false_negatives = torch.sum(
        (binary_predictions == 0) & (binary_targets == 1)
    ).float()
    true_negatives = torch.sum(
        (binary_predictions == 0) & (binary_targets == 0)
    ).float()

    # Calculate precision, recall, F1, and accuracy
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
    accuracy = (true_positives + true_negatives) / len(binary_predictions)

    return (
        mse,
        mae,
        correlation,
        precision.item(),
        recall.item(),
        f1.item(),
        accuracy.item(),
    )


def custom_collate(batch):
    """Memory-optimized collate function that handles ignored indices"""
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
            padded_t = torch.full(
                (max_len,), -1, dtype=torch.float32
            )  # Changed to float32 for continuous values

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
    train_loader,
    val_loader,
    num_epochs=60,
    lr=0.001,
    patience=40,
    threshold=0.5,
):
    """Modified training function to work with k-fold cross validation while maintaining detailed progress reporting"""
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

    model = AVSummarizer().to(device)
    print(f"Model moved to {device}")

    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )
    scaler = torch.amp.GradScaler() if use_amp else None

    best_val_loss = float("inf")
    best_model = None
    best_metrics = None
    no_improve_epochs = 0

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
                    predictions = model(visual=features, audio=audio_features)
                    loss = criterion(predictions.squeeze(), targets)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(visual=features, audio=audio_features)
                loss = criterion(predictions.squeeze(), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            train_predictions.extend(predictions.squeeze().detach().cpu())
            train_targets.extend(targets.cpu())

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
                    with torch.amp.autocast(device_type="cuda"):
                        predictions = model(visual=features, audio=audio_features)
                        loss = criterion(predictions.squeeze(), targets)
                else:
                    predictions = model(visual=features, audio=audio_features)
                    loss = criterion(predictions.squeeze(), targets)

                val_loss += loss.item()
                val_predictions.extend(predictions.squeeze().cpu())
                val_targets.extend(targets.cpu())

        # Calculate metrics
        train_predictions = torch.stack(train_predictions)
        train_targets = torch.stack(train_targets)
        val_predictions = torch.stack(val_predictions)
        val_targets = torch.stack(val_targets)

        # Calculate all metrics
        (
            train_mse,
            train_mae,
            train_corr,
            train_precision,
            train_recall,
            train_f1,
            train_accuracy,
        ) = calculate_metrics(train_predictions, train_targets, threshold)

        (
            val_mse,
            val_mae,
            val_corr,
            val_precision,
            val_recall,
            val_f1,
            val_accuracy,
        ) = calculate_metrics(val_predictions, val_targets, threshold)

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

        print("\nTraining Metrics:")
        print(f"  Loss: {avg_train_loss:.4f}")
        print(f"  MSE: {train_mse:.4f}")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  Correlation: {train_corr:.4f}")
        print(f"  Accuracy: {train_accuracy:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  Recall: {train_recall:.4f}")
        print(f"  F1 Score: {train_f1:.4f}")

        print("\nValidation Metrics:")
        print(f"  Loss: {avg_val_loss:.4f}")
        print(f"  MSE: {val_mse:.4f}")
        print(f"  MAE: {val_mae:.4f}")
        print(f"  Correlation: {val_corr:.4f}")
        print(f"  Accuracy: {val_accuracy:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        print(f"  F1 Score: {val_f1:.4f}")

        if device.type == "cuda":
            print(f"\nGPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

        # Update best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            best_metrics = {
                "val_loss": avg_val_loss,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "val_correlation": val_corr,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "epoch": epoch + 1,
            }
            no_improve_epochs = 0
            print("\nNew best model saved!")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}!")
                print("\nBest Validation Metrics:")
                for metric, value in best_metrics.items():
                    print(f"  {metric}: {value:.4f}")
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


def k_fold_train_model(
    features_dict,
    audio_features_dict,
    annotations_df,
    video_metadata,
    n_folds=5,
    num_epochs=60,
    batch_size=16,
    lr=0.001,
    patience=40,
    threshold=0.5,
    collate_fn=custom_collate,
):
    # Get list of all video IDs
    video_ids = list(features_dict.keys())

    # Initialize K-Fold
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Store metrics for each fold
    fold_metrics = []
    all_models = []

    start_time = time.time()
    print(
        f"\nStarting {n_folds}-fold cross validation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 80)

    # Perform k-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(video_ids)):
        print(f"\nFold {fold + 1}/{n_folds}")
        print("=" * 80)

        # Get train and validation video IDs for this fold
        train_ids = [video_ids[i] for i in train_idx]
        val_ids = [video_ids[i] for i in val_idx]

        print(f"Training videos: {len(train_ids)}")
        print(f"Validation videos: {len(val_ids)}")

        # Create datasets for this fold
        train_dataset = TVSumDataset(
            {vid: features_dict[vid] for vid in train_ids},
            {vid: audio_features_dict[vid] for vid in train_ids},
            annotations_df[annotations_df["Video File Name"].isin(train_ids)],
            video_metadata,
        )

        val_dataset = TVSumDataset(
            {vid: features_dict[vid] for vid in val_ids},
            {vid: audio_features_dict[vid] for vid in val_ids},
            annotations_df[annotations_df["Video File Name"].isin(val_ids)],
            video_metadata,
        )

        # Create data loaders
        if torch.cuda.is_available():
            num_workers = 2
        else:
            batch_size = min(batch_size, 8)
            num_workers = 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=2,
            timeout=120,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=2,
            timeout=120,
        )

        # Train model for this fold
        model, metrics = train_model(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            lr=lr,
            patience=patience,
            threshold=threshold,
        )

        # Store results
        metrics["fold"] = fold + 1
        fold_metrics.append(metrics)
        all_models.append(model)

        print(f"\nFold {fold + 1} Complete!")
        print("=" * 50)
        print("Best Metrics for this fold:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
        print("=" * 50)

    # Calculate and print average metrics across folds
    print("\nAverage Metrics Across All Folds:")
    print("=" * 50)
    avg_metrics = {}
    std_metrics = {}
    for metric in fold_metrics[0].keys():
        if metric != "fold" and isinstance(fold_metrics[0][metric], (int, float)):
            values = [fold[metric] for fold in fold_metrics]
            avg_metrics[metric] = sum(values) / len(values)
            std_metrics[metric] = np.std(values)
            print(f"{metric}:")
            print(f"  Mean: {avg_metrics[metric]:.4f}")
            print(f"  Std:  {std_metrics[metric]:.4f}")

    # Save all results
    results = {
        "fold_metrics": fold_metrics,
        "average_metrics": avg_metrics,
        "std_metrics": std_metrics,
        "training_time": time.time() - start_time,
        "n_folds": n_folds,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
    }

    # Save results to file
    with open("kfold_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save all models
    for i, model in enumerate(all_models):
        torch.save(model.state_dict(), f"av_tvsum_model_fold_{i+1}.pth")

    return all_models, results


def k_fold_train_model(
    features_dict,
    audio_features_dict,
    annotations_df,
    video_metadata,
    n_folds=5,
    num_epochs=60,
    batch_size=16,
    lr=0.001,
    patience=40,
    threshold=0.5,
    collate_fn=custom_collate,
):
    # Get list of all video IDs
    video_ids = list(features_dict.keys())

    # Initialize K-Fold
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Store metrics for each fold
    fold_metrics = []
    all_models = []

    start_time = time.time()
    print(
        f"Starting {n_folds}-fold cross validation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Perform k-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(video_ids)):
        print(f"\nTraining Fold {fold + 1}/{n_folds}")
        print("=" * 50)

        # Get train and validation video IDs for this fold
        train_ids = [video_ids[i] for i in train_idx]
        val_ids = [video_ids[i] for i in val_idx]

        # Create datasets for this fold
        train_dataset = TVSumDataset(
            {vid: features_dict[vid] for vid in train_ids},
            {vid: audio_features_dict[vid] for vid in train_ids},
            annotations_df[annotations_df["Video File Name"].isin(train_ids)],
            video_metadata,
        )

        val_dataset = TVSumDataset(
            {vid: features_dict[vid] for vid in val_ids},
            {vid: audio_features_dict[vid] for vid in val_ids},
            annotations_df[annotations_df["Video File Name"].isin(val_ids)],
            video_metadata,
        )

        # Create data loaders
        if torch.cuda.is_available():
            num_workers = 2
        else:
            batch_size = min(batch_size, 8)
            num_workers = 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=2,
            timeout=120,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=2,
            timeout=120,
        )

        # Train model for this fold
        model, metrics = train_model(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            lr=lr,
            patience=patience,
            threshold=threshold,
        )

        # Store results
        metrics["fold"] = fold + 1
        fold_metrics.append(metrics)
        all_models.append(model)

        print(f"\nFold {fold + 1} Results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    # Calculate and print average metrics across folds
    print("\nAverage Metrics Across All Folds:")
    avg_metrics = {}
    for metric in fold_metrics[0].keys():
        if metric != "fold" and isinstance(fold_metrics[0][metric], (int, float)):
            values = [fold[metric] for fold in fold_metrics]
            avg_metrics[metric] = sum(values) / len(values)
            std_metrics = np.std(values)
            print(f"{metric}:")
            print(f"  Mean: {avg_metrics[metric]:.4f}")
            print(f"  Std:  {std_metrics:.4f}")

    # Save all results
    results = {
        "fold_metrics": fold_metrics,
        "average_metrics": avg_metrics,
        "training_time": time.time() - start_time,
        "n_folds": n_folds,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
    }

    # Save results to file
    with open("kfold_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save all models
    for i, model in enumerate(all_models):
        torch.save(model.state_dict(), f"av_tvsum_model_fold_{i+1}.pth")

    return all_models, results


def main():
    # Set up paths
    # video_dir = "data/videos"
    annotations_path = "data/df_annotations.pkl"
    features_dir = r"data/processed"
    audio_features_dir = r"data/processed"
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
    model, results = k_fold_train_model(
        features_dict,
        audio_features_dict,
        annotations_df,
        video_metadata,
        n_folds=5,
        collate_fn=custom_collate,
    )

    # Save model and metrics
    # print("\nSaving model and metrics...")
    # torch.save(model.state_dict(), "av_tvsum_model.pth")

    print("\nTraining complete! Results saved to 'kfold_training_results.json'")
    print("Models saved as 'av_tvsum_model_fold_N.pth' where N is the fold number")


if __name__ == "__main__":
    print("HERE IT IS RUNNING")
    main()
