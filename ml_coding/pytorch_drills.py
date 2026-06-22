from __future__ import annotations

import torch
from torch import nn


class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def one_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def assert_shape(actual: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(actual.shape) != expected:
        raise ValueError(f"{name} has shape {tuple(actual.shape)}; expected {expected}")
