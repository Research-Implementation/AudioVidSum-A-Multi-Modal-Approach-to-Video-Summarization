import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.dim_head = embed_dim // num_heads

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.dim_head)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.dim_head)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.dim_head)

        scores = torch.einsum("bqhd,bkhd->bhqk", Q, K) / (self.dim_head**0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.einsum("bhqk,bkhd->bqhd", attn, V)
        context = context.reshape(batch_size, seq_len, -1)
        return self.out(context)
