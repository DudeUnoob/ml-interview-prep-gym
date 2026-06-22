"""Finite-difference checks for hand-written backward passes."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np


def central_difference(
    scalar_function: Callable[[np.ndarray], float], parameter: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    """Numerically approximate d scalar_function / d parameter."""
    gradient = np.zeros_like(parameter, dtype=float)
    iterator = np.nditer(parameter, flags=["multi_index"], op_flags=["readonly"])
    while not iterator.finished:
        index = iterator.multi_index
        plus, minus = parameter.copy(), parameter.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (scalar_function(plus) - scalar_function(minus)) / (2.0 * epsilon)
        iterator.iternext()
    return gradient


def assert_gradients_close(analytic: np.ndarray, numerical: np.ndarray, atol: float = 1e-5) -> None:
    if not np.allclose(analytic, numerical, atol=atol, rtol=atol):
        error = np.max(np.abs(analytic - numerical))
        raise AssertionError(f"gradient check failed; max absolute error={error:.3e}")
