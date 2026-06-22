from __future__ import annotations

import numpy as np


def sinusoidal_positional_encoding(
    sequence_length: int, model_dim: int, base: float = 10000.0
) -> np.ndarray:
    positions = np.arange(sequence_length)[:, None]
    even_dims = np.arange(0, model_dim, 2)
    angles = positions / np.power(base, even_dims / model_dim)
    encoding = np.zeros((sequence_length, model_dim), dtype=float)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding


def rope_angles(sequence_length: int, head_dim: int, base: float = 10000.0) -> np.ndarray:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even head dimension")
    positions = np.arange(sequence_length)[:, None]
    frequencies = 1.0 / np.power(base, np.arange(0, head_dim, 2) / head_dim)
    return positions * frequencies
