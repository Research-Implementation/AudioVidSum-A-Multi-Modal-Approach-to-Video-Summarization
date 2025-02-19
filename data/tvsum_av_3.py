import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import pickle
import os
from pathlib import Path
from typing import Dict, Tuple
import logging
from torch.cuda.amp import autocast, GradScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LazyAudioVisualDataset(Dataset):
    """Improved dataset class with lazy loading and dynamic padding"""

    def __init__(self, feature_dir: str, metadata: Dict, annotations_df: pd.DataFrame):
        self.feature_dir = Path(feature_dir)
        self.metadata = metadata
        self.annotations_df = annotations_df
        self.video_ids = self._validate_video_ids()

        # Precompute sequence lengths
        self.sequence_lengths = {
            vid: self.metadata[vid]["length"] for vid in self.video_ids
        }

        logger.info(f"Initialized dataset with {len(self)} samples")

    def _validate_video_ids(self):
        valid_ids = []
        for vid in self.metadata.keys():
            vis_path = self.feature_dir / vid / "visual.npy"
            aud_path = self.feature_dir / vid / "audio.npy"
            if vis_path.exists() and aud_path.exists():
                valid_ids.append(vid)
            else:
                logger.warning(f"Missing features for video {vid}")
        return valid_ids

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_id = self.video_ids[idx]

        # Load features
        visual = np.load(self.feature_dir / video_id / "visual.npy")
        audio = np.load(self.feature_dir / video_id / "audio.npy")

        # Process annotations
        annotations = self.annotations_df[
            self.annotations_df["Video File Name"] == video_id
        ]["Annotations"].values

        # Average and normalize annotations
        scores = np.mean([self._parse_annotation(a) for a in annotations], axis=0)
        scores = scores / 4.0  # Normalize to [0, 1]

        return (
            torch.FloatTensor(visual),
            torch.FloatTensor(audio),
            torch.FloatTensor(scores),
            visual.shape[0],  # Actual sequence length
        )

    def _parse_annotation(self, ann):
        if isinstance(ann, str):
            return np.array([float(x) for x in ann.strip("[]").split(",")])
        return ann


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
    """Enhanced audio-visual model with transformer layers"""

    def __init__(self, vis_dim: int = 4096, aud_dim: int = 384):
        super().__init__()

        # Visual encoder
        self.visual_encoder = nn.LSTM(
            vis_dim, 512, bidirectional=True, batch_first=True, dropout=0.3
        )

        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(aud_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Cross-modal attention
        self.cross_attention = CrossModalAttention(1024, 256)

        # Temporal transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=1024, nhead=8, dim_feedforward=2048, dropout=0.1
            ),
            num_layers=4,
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(
        self, visual: torch.Tensor, audio: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        # Visual processing
        packed_vis = nn.utils.rnn.pack_padded_sequence(
            visual, lengths, batch_first=True, enforce_sorted=False
        )
        vis_out, _ = self.visual_encoder(packed_vis)
        vis_out, _ = nn.utils.rnn.pad_packed_sequence(vis_out, batch_first=True)

        # Audio processing
        aud_out = self.audio_encoder(audio)

        # Cross-modal attention
        attn_weights = self.cross_attention(vis_out, aud_out)
        attended_vis = torch.bmm(attn_weights, vis_out)

        # Transformer encoding
        transformer_out = self.transformer(attended_vis)

        # Temporal pooling
        pooled = transformer_out.mean(dim=1)

        return self.head(pooled).squeeze(-1)


class EarlyStopper:
    """Early stopping utility"""

    def __init__(self, patience: int = 5, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
        return self.early_stop


def dynamic_collate(batch: list) -> tuple:
    """Custom collate function for dynamic padding"""
    visuals, audios, scores, lengths = zip(*batch)

    # Sort by length descending
    sorted_idx = np.argsort(lengths)[::-1]
    visuals = [visuals[i] for i in sorted_idx]
    audios = [audios[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]
    lengths = [lengths[i] for i in sorted_idx]

    # Pad sequences
    max_len = max(lengths)
    vis_padded = torch.zeros((len(visuals), max_len, visuals[0].shape[-1]))
    aud_padded = torch.zeros((len(audios), max_len, audios[0].shape[-1]))
    scores_padded = torch.full((len(scores), max_len), -1.0)

    for i, (v, a, s, l) in enumerate(zip(visuals, audios, scores, lengths)):
        vis_padded[i, :l] = v
        aud_padded[i, :l] = a
        scores_padded[i, :l] = s

    return (vis_padded, aud_padded, scores_padded, torch.tensor(lengths))


def train_model(
    train_dataset: Dataset, val_dataset: Dataset, config: dict
) -> nn.Module:
    """Enhanced training loop with mixed precision and gradient accumulation"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # Initialize model and optimizer
    model = AVSummarizer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scaler = GradScaler()
    stopper = EarlyStopper(patience=config["patience"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=dynamic_collate,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        collate_fn=dynamic_collate,
        num_workers=4,
        pin_memory=True,
    )

    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (vis, aud, scores, lengths) in enumerate(train_loader):
            vis = vis.to(device, non_blocking=True)
            aud = aud.to(device, non_blocking=True)
            scores = scores.to(device, non_blocking=True)

            with autocast():
                preds = model(vis, aud, lengths)
                mask = scores != -1
                loss = F.mse_loss(preds[mask], scores[mask])

            scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % config["accum_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vis, aud, scores, lengths in val_loader:
                vis = vis.to(device)
                aud = aud.to(device)
                scores = scores.to(device)

                preds = model(vis, aud, lengths)
                mask = scores != -1
                loss = F.mse_loss(preds[mask], scores[mask])
                val_loss += loss.item()

        # Metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        logger.info(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config["save_path"])
            logger.info(f"Saved new best model with val loss {best_val_loss:.4f}")

        if stopper(avg_val_loss):
            logger.info("Early stopping triggered")
            break

    # Load best model
    model.load_state_dict(torch.load(config["save_path"]))
    return model


def main():
    config = {
        "batch_size": 16,
        "lr": 3e-4,
        "epochs": 50,
        "patience": 7,
        "accum_steps": 2,
        "save_path": "best_av_model.pth",
    }

    # Load data
    with open("data/video_metadata.json") as f:
        metadata = json.load(f)

    annotations_df = pd.read_pickle("data/df_annotations.pkl")

    # Split datasets
    train_vids, val_vids = train_test_split(
        list(metadata.keys()), test_size=0.2, random_state=42
    )

    train_dataset = LazyAudioVisualDataset(
        Path("data/processed"),
        {k: metadata[k] for k in train_vids},
        annotations_df[annotations_df["Video File Name"].isin(train_vids)],
    )

    val_dataset = LazyAudioVisualDataset(
        Path("data/processed"),
        {k: metadata[k] for k in val_vids},
        annotations_df[annotations_df["Video File Name"].isin(val_vids)],
    )

    # Train model
    model = train_model(train_dataset, val_dataset, config)

    # Save final model
    torch.save(model.state_dict(), "final_av_model.pth")
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
