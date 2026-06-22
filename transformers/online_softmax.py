"""Numerically stable one-pass softmax summaries used to understand FlashAttention."""

from __future__ import annotations

import numpy as np


def online_logsumexp(logits: np.ndarray) -> float:
    """Return log(sum(exp(logits))) without ever exponentiating a large value."""
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("expected a non-empty 1D vector")
    running_max = -np.inf
    running_denom = 0.0
    for logit in logits:
        new_max = max(running_max, float(logit))
        running_denom = running_denom * np.exp(running_max - new_max) + np.exp(logit - new_max)
        running_max = new_max
    return float(running_max + np.log(running_denom))


def online_weighted_softmax(logits: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Compute sum_i softmax(logits)_i * values_i in one stable pass.

    This is the per-query core idea behind tiled attention, not a GPU FlashAttention kernel.
    """
    if logits.ndim != 1 or values.ndim != 2 or len(logits) != len(values):
        raise ValueError("expected logits=(T,) and values=(T, value_dim)")
    running_max = -np.inf
    running_denom = 0.0
    running_numerator = np.zeros(values.shape[1], dtype=float)
    for logit, value in zip(logits, values, strict=True):
        new_max = max(running_max, float(logit))
        rescale = np.exp(running_max - new_max)
        new_weight = np.exp(logit - new_max)
        running_numerator = running_numerator * rescale + new_weight * value
        running_denom = running_denom * rescale + new_weight
        running_max = new_max
    return running_numerator / running_denom


def streaming_attention_for_query(query: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Attention for one query without materializing its probability vector."""
    if query.ndim != 1 or keys.ndim != 2 or values.ndim != 2 or keys.shape != values.shape:
        raise ValueError("expected q=(D,), K=(T,D), V=(T,D)")
    if query.shape[0] != keys.shape[1]:
        raise ValueError("query/key dimensions must match")
    logits = (keys @ query) / np.sqrt(query.shape[0])
    return online_weighted_softmax(logits, values)
