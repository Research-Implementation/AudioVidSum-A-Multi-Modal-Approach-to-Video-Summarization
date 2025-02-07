import torch.nn as nn


class VisualScoreNet(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),  # Equivalent to global max pooling
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Normalize scores to [0,1]
        )

    def forward(self, x):
        return self.net(x)
