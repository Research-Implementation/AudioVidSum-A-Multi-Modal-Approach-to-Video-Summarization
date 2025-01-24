import torch
import torch.nn as nn
from models.attention import MultiHeadSelfAttention


class AVBiLSTMModel(nn.Module):
    def __init__(self, visual_dim=4096, audio_dim=128, hidden_dim=512):
        super().__init__()
        # Feature projection
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Temporal modeling
        self.bilstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
        )

        # Attention mechanism
        self.attention = MultiHeadSelfAttention(hidden_dim * 2, num_heads=4)

        # Scoring layer
        self.scorer = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())

    def forward(self, visual_feats, audio_feats):
        # Project features
        v_proj = self.visual_proj(visual_feats)
        a_proj = self.audio_proj(audio_feats)

        # Temporal fusion
        fused = torch.cat([v_proj, a_proj], dim=-1)

        # Bi-LSTM processing
        temporal, _ = self.bilstm(fused)

        # Self-attention
        attn_output = self.attention(temporal)

        # Importance scoring
        return self.scorer(attn_output)
