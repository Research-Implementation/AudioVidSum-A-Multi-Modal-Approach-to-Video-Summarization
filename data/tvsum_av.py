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


# Create combined dataset
class AudioVisualDataset(Dataset):
    def __init__(self, visual_features, audio_features, annotations_df, video_metadata):
        self.visual_features = []
        self.audio_features = []
        self.targets = []
        self.lengths = []

        max_seq_len = max(
            feature_array.shape[0] for feature_array in visual_features.values()
        )

        video_groups = annotations_df.groupby("Video File Name")

        for video_id, visual_feature_array in visual_features.items():
            if (
                video_id not in video_metadata
                or video_id not in video_groups.groups
                or video_id not in audio_features
            ):
                continue

            audio_feature_array = audio_features[video_id]

            # Process annotations (similar to your existing TVSumDataset)
            # ... (keeping your existing annotation processing code)

            # Ensure all features have same length
            min_length = min(
                len(discrete_scores),
                len(visual_feature_array),
                len(audio_feature_array),
            )
            visual_feature_array = visual_feature_array[:min_length]
            audio_feature_array = audio_feature_array[:min_length]
            discrete_scores = discrete_scores[:min_length]

            # Pad sequences
            padded_visual = torch.zeros((max_seq_len, visual_feature_array.shape[1]))
            padded_visual[: len(visual_feature_array)] = torch.FloatTensor(
                visual_feature_array
            )

            padded_audio = torch.zeros((max_seq_len, audio_feature_array.shape[1]))
            padded_audio[: len(audio_feature_array)] = torch.FloatTensor(
                audio_feature_array
            )

            padded_targets = torch.full((max_seq_len,), -1)
            padded_targets[: len(discrete_scores)] = torch.FloatTensor(discrete_scores)

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


class AudioScorePredictor(nn.Module):
    def __init__(
        self, input_dim=384
    ):  # Using the dimension from your audio.npy example
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid(),  # Normalize scores between 0 and 1
        )

    def forward(self, x):
        return self.model(x)


class AudioVisualPredictor(nn.Module):
    def __init__(self, visual_input_dim=4096, audio_input_dim=384):
        super().__init__()

        # Visual scoring network (modified from your ImportancePredictor)
        self.visual_bilstm = nn.LSTM(
            input_size=visual_input_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5,
        )

        self.visual_classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        # Audio scoring network
        self.audio_predictor = AudioScorePredictor(audio_input_dim)

    def forward(self, visual_features, audio_features, lengths):
        # Process visual features
        packed_input = nn.utils.rnn.pack_padded_sequence(
            visual_features, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.visual_bilstm(packed_input)
        visual_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        visual_scores = self.visual_classifier(visual_output)

        # Process audio features
        audio_scores = self.audio_predictor(audio_features)

        # Average the scores
        combined_scores = (visual_scores + audio_scores) / 2

        return combined_scores, visual_scores, audio_scores


def train_audio_visual_model(
    visual_features_dict,
    audio_features_dict,
    annotations_df,
    video_metadata,
    num_epochs=50,
    batch_size=32,
    lr=0.001,
    patience=5,
    checkpoint_path="av_training_checkpoint.pth",
):

    # Training setup code...
    model = AudioVisualPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Rest of training loop implementation similar to your existing train_model function
    # but adapted for the audio-visual model...

    # Initialize training state
    start_epoch = 0
    best_val_loss = float("inf")
    best_model = None
    no_improve_epochs = 0

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        best_model = checkpoint["best_model"]
        no_improve_epochs = checkpoint["no_improve_epochs"]
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0

        for features, targets, lengths in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(features, lengths)
            B, T, C = logits.shape

            loss = criterion(
                logits.contiguous().reshape(-1, num_classes),
                targets.contiguous().reshape(-1),
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

        # Save checkpoint after each epoch
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "best_model": best_model,
            "no_improve_epochs": no_improve_epochs,
        }
        torch.save(checkpoint, checkpoint_path)

        if no_improve_epochs >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}!")
            break

    model.load_state_dict(best_model)
    return model


def main():
    # Set up paths
    # video_dir = "Evaluation/TVSum/videos"
    annotations_path = "data/df_annotations.pkl"
    features_dir = r"data/processed"

    metadata_path = "data/video_metadata.json"

    # Get video metadata (with JSON caching)
    video_metadata = load_or_generate_metadata(None, metadata_path)

    # Printing some statistics
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
    model = train_model(
        features_dict, annotations_df, video_metadata, num_classes=4
    )  # Changed num_classes to 4

    # Save model
    torch.save(model.state_dict(), "tvsum_model.pth")


if __name__ == "__main__":
    print("HERE IT IS RUNNING")
    main()
