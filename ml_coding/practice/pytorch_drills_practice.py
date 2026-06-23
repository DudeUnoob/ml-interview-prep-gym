"""Implement each piece yourself. Do not look at ml_coding/pytorch_drills.py
until tests/test_practice_ml_coding.py passes.

Self-check: pytest tests/test_practice_ml_coding.py -v
"""

from __future__ import annotations

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TwoLayerMLP(nn.Module):
    """A two-layer MLP: Linear -> GELU -> Linear.

    __init__ should build the layers (consider nn.Sequential or two
    separate nn.Linear + an activation). forward() should run input
    through them and return the output logits.

    Shapes: input is (batch, input_dim), output is (batch, output_dim).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
        


def one_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Run one full training step and return the scalar loss.

    Order matters: set the model to train mode, zero gradients, forward
    pass, compute cross-entropy loss against targets, backward pass, step
    the optimizer. Return the loss as a plain Python float (detached from
    the graph).
    """
    


def assert_shape(actual: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    """Raise ValueError with a useful message if actual.shape != expected."""
    raise NotImplementedError("TODO: implement assert_shape")


if __name__ == "__main__":
    # Run `python3 ml_coding/practice/pytorch_drills_practice.py` to watch
    # the loss drop over a few manual steps. Add prints inside the functions
    # above (e.g. print(logits.shape) in one_training_step) as you debug.
    torch.manual_seed(0)
    model = TwoLayerMLP(input_dim=4, hidden_dim=8, output_dim=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(5, 4)
    targets = torch.randint(0, 3, (5,))

    for step in range(5):
        loss = one_training_step(model, optimizer, x, targets)
        print(f"step {step}: loss={loss:.4f}")
