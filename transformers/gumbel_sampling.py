"""Sampling primitives for Gumbel-Max and differentiable Gumbel-Softmax practice."""

from __future__ import annotations

import numpy as np


def sample_gumbel(shape: tuple[int, ...], rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    uniform = rng.uniform(low=1e-12, high=1.0 - 1e-12, size=shape)
    return -np.log(-np.log(uniform))


def gumbel_max(logits: np.ndarray, rng: np.random.Generator | None = None) -> int:
    """Sample exactly from categorical(softmax(logits))."""
    if logits.ndim != 1:
        raise ValueError("expected a 1D logits vector")
    return int(np.argmax(logits + sample_gumbel(logits.shape, rng)))


def gumbel_softmax(
    logits: np.ndarray, temperature: float = 1.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Differentiable, stochastic relaxation of categorical sampling."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    noisy_logits = (logits + sample_gumbel(logits.shape, rng)) / temperature
    shifted = noisy_logits - noisy_logits.max()
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()
