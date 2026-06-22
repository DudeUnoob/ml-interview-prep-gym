"""Manual linear layer with the math convention W: (n_in, n_out)."""

from __future__ import annotations

import numpy as np


def linear_forward(inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Compute Z = XW + b.

    X: (batch, n_in), W: (n_in, n_out), b: (n_out,)
    Returns: (batch, n_out)
    """
    if inputs.ndim != 2 or weights.ndim != 2 or bias.ndim != 1:
        raise ValueError("expected X=(B, n_in), W=(n_in, n_out), b=(n_out,)")
    if inputs.shape[1] != weights.shape[0] or weights.shape[1] != bias.shape[0]:
        raise ValueError("incompatible linear-layer shapes")
    return inputs @ weights + bias


def linear_backward(
    inputs: np.ndarray, weights: np.ndarray, upstream: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dX, dW, db for Z = XW + b.

    dX preserves the batch dimension; dW and db sum contributions across it.
    """
    if upstream.shape != (inputs.shape[0], weights.shape[1]):
        raise ValueError("upstream gradient has the wrong shape")
    d_inputs = upstream @ weights.T
    d_weights = inputs.T @ upstream
    d_bias = upstream.sum(axis=0)
    return d_inputs, d_weights, d_bias


def relu(inputs: np.ndarray) -> np.ndarray:
    return np.maximum(inputs, 0.0)


def relu_backward(pre_activation: np.ndarray, upstream: np.ndarray) -> np.ndarray:
    return upstream * (pre_activation > 0.0)
