import torch
import torch.nn as nn
from models.attention import MultiHeadSelfAttention

class VisualScorePredictor(nn.Module):
    def __init__(self, input_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),  # Maxpool
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class AudioScorePredictor(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__():
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class AVSummarizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_scorer = VisualScorePredictor()
        self.audio_scorer = AudioScorePredictor()
        
        # Additional temporal modeling
        self.bilstm = nn.LSTM(
            input_size=2,  # Combined scores
            hidden_size=64,
            bidirectional=True,
            batch_first=True
        )
        self.attention = MultiHeadSelfAttention(embed_dim=128, num_heads=4)
        
    def forward(self, visual_features, audio_features):
        # Frame-level scoring
        v_scores = self.visual_scorer(visual_features)
        a_scores = self.audio_scorer(audio_features)
        
        # Combine scores
        combined = torch.cat([v_scores, a_scores], dim=-1)
        
        # Temporal modeling
        temporal, _ = self.bilstm(combined)
        attended = self.attention(temporal)
        
        # Shot-level scores
        shot_scores = attended.mean(dim=1)
        return shot_scores


class AVScorer(nn.Module):
    def __init__(self, visual_dim=4096, audio_dim=384):
        super().__init__()
        self.visual_net = VisualScorePredictor(visual_dim)
        self.audio_net = AudioScorePredictor(audio_dim)

    def forward(self, visual_input, audio_input):
        # Process frame-level features
        v_scores = self.visual_net(visual_input)  # [batch_size, 1]
        a_scores = self.audio_net(audio_input)  # [batch_size, 1]

        # Combine scores as per paper
        frame_scores = (v_scores + a_scores) / 2  # Average scores

        return frame_scores.squeeze()  # [batch_size]

    def get_shot_scores(self, frame_scores, shot_boundaries):
        """Convert frame scores to shot scores"""
        shot_scores = []
        for start, end in shot_boundaries:
            shot_scores.append(frame_scores[start:end].mean())
        return torch.stack(shot_scores)

    def predict_shots(self, frame_scores, shot_boundaries):
        """Convert frame scores to shot-level averages"""
        return torch.tensor(
            [frame_scores[start:end].mean() for start, end in shot_boundaries]
        )


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
