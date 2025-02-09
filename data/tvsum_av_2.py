import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import pickle
from pathlib import Path
import cv2
import os


class AudioVisualDataset(Dataset):
    def __init__(self, visual_features, audio_features, annotations_df, video_metadata):
        self.visual_features = []
        self.audio_features = []
        self.targets = []
        self.lengths = []

        # Find maximum sequence length
        max_seq_len = max(
            feat_array.shape[0] for feat_array in visual_features.values()
        )

        # Group annotations by video file name
        video_groups = annotations_df.groupby("Video File Name")

        for video_id in visual_features.keys():
            if (
                video_id not in video_metadata
                or video_id not in video_groups.groups
                or video_id not in audio_features
            ):
                continue

            # Get features
            v_features = visual_features[video_id]
            a_features = audio_features[video_id]

            # Process annotations
            video_annotations = video_groups.get_group(video_id)["Annotations"].tolist()
            processed_annotations = []
            for annotation in video_annotations:
                if isinstance(annotation, str):
                    scores = [
                        float(x.strip()) for x in annotation.strip("[]").split(",")
                    ]
                else:
                    scores = annotation
                processed_annotations.append(scores)

            # Average annotations and normalize to [0, 1]
            avg_annotation = np.mean(processed_annotations, axis=0)
            normalized_scores = avg_annotation / 4.0  # Assuming original scores are 1-4

            # Ensure all features have same length
            min_length = min(len(normalized_scores), len(v_features), len(a_features))
            v_features = v_features[:min_length]
            a_features = a_features[:min_length]
            normalized_scores = normalized_scores[:min_length]

            # Pad sequences
            padded_visual = torch.zeros((max_seq_len, v_features.shape[1]))
            padded_visual[: len(v_features)] = torch.FloatTensor(v_features)

            padded_audio = torch.zeros((max_seq_len, a_features.shape[1]))
            padded_audio[: len(a_features)] = torch.FloatTensor(a_features)

            padded_targets = torch.full((max_seq_len,), -1.0)
            padded_targets[: len(normalized_scores)] = torch.FloatTensor(
                normalized_scores
            )

            self.visual_features.append(padded_visual)
            self.audio_features.append(padded_audio)
            self.targets.append(padded_targets)
            self.lengths.append(min_length)

    def __len__(self):
        return len(self.visual_features)

    def __getitem__(self, idx):
        return (
            self.visual_features[idx],
            self.audio_features[idx],
            self.targets[idx],
            self.lengths[idx],
        )


class AudioVisualPredictor(nn.Module):
    def __init__(self, visual_dim=4096, audio_dim=384):
        super().__init__()

        # Visual processing network - simplified to use less memory
        self.visual_lstm = nn.LSTM(
            input_size=visual_dim,
            hidden_size=256,  # Reduced from 512
            num_layers=1,  # Reduced from 2
            batch_first=True,
            bidirectional=True,
            dropout=0.5,
        )

        self.visual_attention = nn.Sequential(
            nn.Linear(512, 256), nn.Tanh(), nn.Linear(256, 1)  # Adjusted dimensions
        )

        self.visual_output = nn.Sequential(
            nn.Linear(512, 256),  # Adjusted dimensions
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Audio processing network - simplified
        self.audio_network = nn.Sequential(
            nn.Linear(audio_dim, 512),  # Reduced dimensions
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Fusion layer
        self.fusion = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

    def forward(self, visual_features, audio_features, lengths):
        batch_size = visual_features.size(0)

        # Process visual features
        packed_visual = nn.utils.rnn.pack_padded_sequence(
            visual_features, lengths, batch_first=True, enforce_sorted=False
        )
        visual_output, _ = self.visual_lstm(packed_visual)
        visual_output, _ = nn.utils.rnn.pad_packed_sequence(
            visual_output, batch_first=True
        )

        # Apply attention
        attention_weights = self.visual_attention(visual_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        visual_scores = self.visual_output(visual_output)

        # Process audio features
        audio_scores = self.audio_network(audio_features)

        # Combine scores
        combined_features = torch.cat([visual_scores, audio_scores], dim=-1)
        final_scores = self.fusion(combined_features)

        return final_scores.squeeze(-1)  # Remove last dimension to match target shape


def train_model(
    visual_features,
    audio_features,
    annotations_df,
    video_metadata,
    num_epochs=50,
    batch_size=8,  # Reduced batch size
    lr=0.001,
    patience=5,
    checkpoint_path="av_model_checkpoint.pth",
):
    # Split data
    video_ids = list(visual_features.keys())
    train_ids, val_ids = train_test_split(video_ids, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = AudioVisualDataset(
        {k: visual_features[k] for k in train_ids},
        {k: audio_features[k] for k in train_ids},
        annotations_df[annotations_df["Video File Name"].isin(train_ids)],
        video_metadata,
    )

    val_dataset = AudioVisualDataset(
        {k: visual_features[k] for k in val_ids},
        {k: audio_features[k] for k in val_ids},
        annotations_df[annotations_df["Video File Name"].isin(val_ids)],
        video_metadata,
    )

    def collate_fn(batch):
        # Sort batch by length for packed sequence
        batch = sorted(batch, key=lambda x: x[3], reverse=True)

        # Get max length in this batch
        max_len = max(x[3] for x in batch)

        # Trim padding to max length in batch
        visual_features = torch.stack([x[0][:max_len] for x in batch])
        audio_features = torch.stack([x[1][:max_len] for x in batch])
        targets = torch.stack([x[2][:max_len] for x in batch])
        lengths = [x[3] for x in batch]

        return visual_features, audio_features, targets, lengths

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioVisualPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training state
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0

        for visual_feat, audio_feat, targets, lengths in train_loader:
            try:
                visual_feat = visual_feat.to(device)
                audio_feat = audio_feat.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                scores = model(visual_feat, audio_feat, lengths)

                # Create mask for valid positions
                mask = targets != -1

                # Compute loss only on valid positions
                valid_scores = scores[mask]
                valid_targets = targets[mask]

                loss = criterion(valid_scores, valid_targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )  # Add gradient clipping
                optimizer.step()
                train_loss += loss.item()

                # Clear cache
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("WARNING: out of memory, skipping batch")
                    continue
                else:
                    raise e

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for visual_feat, audio_feat, targets, lengths in val_loader:
                try:
                    visual_feat = visual_feat.to(device)
                    audio_feat = audio_feat.to(device)
                    targets = targets.to(device)

                    scores = model(visual_feat, audio_feat, lengths)
                    mask = targets != -1

                    valid_scores = scores[mask]
                    valid_targets = targets[mask]

                    loss = criterion(valid_scores, valid_targets)
                    val_loss += loss.item()

                    # Clear cache
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print("WARNING: out of memory, skipping batch")
                        continue
                    else:
                        raise e

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping!")
                break

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def main():
    # Load data
    print("Loading data...")
    annotations_path = "data/df_annotations.pkl"
    features_dir = "data/processed"
    metadata_path = "data/video_metadata.json"

    # Load metadata
    with open(metadata_path, "r") as f:
        video_metadata = json.load(f)

    # Load features
    visual_features = {}
    audio_features = {}

    for video_id in video_metadata:
        try:
            visual_path = os.path.join(features_dir, f"{video_id}/visual.npy")
            audio_path = os.path.join(features_dir, f"{video_id}/audio.npy")

            visual_features[video_id] = np.load(visual_path)
            audio_features[video_id] = np.load(audio_path)

            print(f"Loaded features for {video_id}")
            print(f"Visual shape: {visual_features[video_id].shape}")
            print(f"Audio shape: {audio_features[video_id].shape}")

        except Exception as e:
            print(f"Error loading features for {video_id}: {e}")

    # Load annotations
    with open(annotations_path, "rb") as f:
        annotations_df = pickle.load(f)

    # Train model
    print("Training model...")
    model = train_model(visual_features, audio_features, annotations_df, video_metadata)

    # Save final model
    torch.save(model.state_dict(), "final_av_model.pth")
    print("Training complete!")


if __name__ == "__main__":
    main()
