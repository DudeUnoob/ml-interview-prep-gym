"""Implement each function yourself. Do not look at ml_coding/adamw_numpy.py
until tests/test_practice_ml_coding.py passes.

Self-check: pytest tests/test_practice_ml_coding.py -v
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AdamWState:
    step: int
    first_moment: np.ndarray
    second_moment: np.ndarray


def init_state(parameter: np.ndarray) -> AdamWState:
    """Create a zero-initialized AdamWState matching parameter's shape.

    step starts at 0; first_moment and second_moment are zero arrays the
    same shape as `parameter` (use float dtype).
    """
    raise NotImplementedError("TODO: implement init_state")


def adamw_step(
    parameter: np.ndarray,
    gradient: np.ndarray,
    state: AdamWState,
    learning_rate: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    epsilon: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[np.ndarray, AdamWState]:
    """Perform one AdamW update step.

    AdamW = decoupled weight decay + bias-corrected Adam. In order:
    1. Validate parameter.shape == gradient.shape, and that both betas are
       in (0, 1) — raise ValueError otherwise.
    2. Apply weight decay directly to the parameter (NOT through the
       gradient): param -= lr * weight_decay * param.
    3. Update the first moment (momentum) as an exponential moving average
       of the gradient, using beta1.
    4. Update the second moment (uncentered variance) as an exponential
       moving average of the squared gradient, using beta2.
    5. Increment the step counter, then bias-correct both moments by
       dividing by (1 - beta^step).
    6. Apply the Adam update: param -= lr * first_hat / (sqrt(second_hat) + eps).

    Returns the updated parameter and a new AdamWState (don't mutate the
    one passed in).
    """
    raise NotImplementedError("TODO: implement adamw_step")


if __name__ == "__main__":
    # Run `python3 ml_coding/practice/adamw_numpy_practice.py` to watch the
    # parameter converge toward 0 on a simple quadratic (loss = sum(p**2)).
    parameter = np.array([10.0, -5.0])
    state = init_state(parameter)
    for step in range(10):
        gradient = 2 * parameter
        parameter, state = adamw_step(parameter, gradient, state, learning_rate=0.1)
        print(f"step {step}: parameter={parameter}")
