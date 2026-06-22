from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def causal_mask(sequence_length: int) -> np.ndarray:
    """True where an attention score is allowed."""
    return np.tril(np.ones((sequence_length, sequence_length), dtype=bool))


def scaled_dot_product_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inputs: (..., Tq, D), (..., Tk, D), (..., Tk, Dv)."""
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("query and key dimensions must match")
    scores = queries @ np.swapaxes(keys, -1, -2)
    scores = scores / np.sqrt(queries.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = _softmax(scores)
    return weights @ values, weights
