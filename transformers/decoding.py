from __future__ import annotations

import numpy as np


def _probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits / temperature
    scaled = scaled - scaled.max()
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum()


def greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def top_k_sample(logits: np.ndarray, k: int, temperature: float = 1.0, rng=None) -> int:
    if not 1 <= k <= len(logits):
        raise ValueError("k must be in [1, vocabulary size]")
    rng = np.random.default_rng() if rng is None else rng
    candidate_ids = np.argpartition(logits, -k)[-k:]
    probabilities = _probabilities(logits[candidate_ids], temperature)
    return int(rng.choice(candidate_ids, p=probabilities))


def nucleus_sample(logits: np.ndarray, p: float = 0.9, temperature: float = 1.0, rng=None) -> int:
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    rng = np.random.default_rng() if rng is None else rng
    probabilities = _probabilities(logits, temperature)
    order = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[order]
    cutoff = np.searchsorted(np.cumsum(sorted_probs), p, side="left") + 1
    ids = order[:cutoff]
    chosen_probs = probabilities[ids]
    chosen_probs /= chosen_probs.sum()
    return int(rng.choice(ids, p=chosen_probs))
