import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from src.models.av_model import AVSummarizer
import os

class FeatureDataset(Dataset):
    def __init__(self, feature_dir, annotation_dir=None):
        self.features = [
            np.load(os.path.join(feature_dir, f)) for f in os.listdir(feature_dir)
        ]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx])


def collate_fn(batch):
    # Pad sequences to same length
    lengths = [len(x) for x in batch]
    max_length = max(lengths)

    padded_batch = torch.zeros(len(batch), max_length, batch[0].shape[-1])
    for i, seq in enumerate(batch):
        padded_batch[i, : lengths[i]] = seq

    return padded_batch


def train():
    # Initialize dataset and model
    train_set = FeatureDataset("data/processed/train")
    train_loader = DataLoader(train_set, batch_size=8, collate_fn=collate_fn)

    model = AVSummarizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(50):
        for batch in train_loader:
            # Forward pass
            outputs = model(batch)

            # Dummy targets - replace with real annotations
            targets = torch.rand(outputs.shape)  # Remove this line when using real data

            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()
