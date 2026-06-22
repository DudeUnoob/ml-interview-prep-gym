from __future__ import annotations

import math
import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must divide num_heads")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.out = nn.Linear(model_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, model_dim = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.ones(tokens, tokens, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, tokens, model_dim)
        return self.out(attended)


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attention = CausalSelfAttention(model_dim, num_heads)
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_ratio * model_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * model_dim, model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))
