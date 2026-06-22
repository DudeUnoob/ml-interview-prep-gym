"""Small NumPy AdamW implementation for interview practice."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AdamWState:
    step: int
    first_moment: np.ndarray
    second_moment: np.ndarray


def init_state(parameter: np.ndarray) -> AdamWState:
    return AdamWState(0, np.zeros_like(parameter, dtype=float), np.zeros_like(parameter, dtype=float))


def adamw_step(
    parameter: np.ndarray,
    gradient: np.ndarray,
    state: AdamWState,
    learning_rate: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    epsilon: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[np.ndarray, AdamWState]:
    """Perform decoupled weight decay followed by a bias-corrected Adam update."""
    if parameter.shape != gradient.shape:
        raise ValueError("parameter and gradient shapes must match")
    beta1, beta2 = betas
    if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
        raise ValueError("betas must lie in (0, 1)")

    updated_parameter = parameter.astype(float, copy=True)
    updated_parameter -= learning_rate * weight_decay * updated_parameter

    first = beta1 * state.first_moment + (1.0 - beta1) * gradient
    second = beta2 * state.second_moment + (1.0 - beta2) * (gradient ** 2)
    step = state.step + 1
    first_hat = first / (1.0 - beta1**step)
    second_hat = second / (1.0 - beta2**step)
    updated_parameter -= learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)

    return updated_parameter, AdamWState(step, first, second)
