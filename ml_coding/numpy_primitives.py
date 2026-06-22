from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    positive = x >= 0
    out = np.empty_like(x, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    probabilities = softmax(logits)
    selected = probabilities[np.arange(len(targets)), targets]
    return float(-np.mean(np.log(selected + 1e-12)))


def softmax_cross_entropy_gradient(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    gradient = softmax(logits)
    gradient[np.arange(len(targets)), targets] -= 1.0
    return gradient / len(targets)


def linear_regression_gradients(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[np.ndarray, float]:
    predictions = x @ weights + bias
    error = predictions - y
    count = len(y)
    return (2.0 / count) * (x.T @ error), float((2.0 / count) * error.sum())


def logistic_regression_gradients(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[np.ndarray, float]:
    probabilities = sigmoid(x @ weights + bias)
    error = probabilities - y
    count = len(y)
    return (x.T @ error) / count, float(error.mean())
