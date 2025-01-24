import torch
import torch.nn as nn
from models.attention import MultiHeadSelfAttention


class AVBiLSTMModel(nn.Module):
    def __init__(self, visual_dim=4096, audio_dim=296, hidden_dim=512):
        super().__init__()
        # Feature compression
        self.visual_fc = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3)
        )
        self.audio_fc = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3)
        )

        # Temporal modeling
        self.visual_bilstm = nn.LSTM(
            hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )
        self.audio_bilstm = nn.LSTM(
            hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )

        # Cross-modal attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4)

        # Scoring
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, visual, audio):
        # Feature compression
        v_emb = self.visual_fc(visual)  # [batch, seq_len, hidden]
        a_emb = self.audio_fc(audio)  # [batch, seq_len, hidden]

        # Temporal modeling
        v_out, _ = self.visual_bilstm(v_emb)  # [batch, seq_len, hidden]
        a_out, _ = self.audio_bilstm(a_emb)  # [batch, seq_len, hidden]

        # Attention fusion
        fused = torch.cat([v_out, a_out], dim=-1)
        attn_out, _ = self.attention(fused, fused, fused)  # [batch, seq_len, hidden*2]

        return self.scorer(attn_out).squeeze()
