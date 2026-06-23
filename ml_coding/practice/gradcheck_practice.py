"""Implement each function yourself. Do not look at ml_coding/gradcheck.py
until tests/test_practice_ml_coding.py passes.

This is the tool you'll use to verify your own backward-pass implementations
in manual_linear_practice.py, so get it right.

Self-check: pytest tests/test_practice_ml_coding.py -v
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def central_difference(
    scalar_function: Callable[[np.ndarray], float], parameter: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    """Numerically approximate d scalar_function / d parameter via central differences.

    For each entry of `parameter` (use np.nditer with multi_index to iterate
    over arbitrary shapes), perturb that single entry by +epsilon and by
    -epsilon, call scalar_function on each perturbed copy, and estimate the
    partial derivative as (f(x+eps) - f(x-eps)) / (2 * eps).

    Returns an array the same shape as `parameter`, one derivative estimate
    per entry.
    """
    raise NotImplementedError("TODO: implement central_difference")


def assert_gradients_close(analytic: np.ndarray, numerical: np.ndarray, atol: float = 1e-5) -> None:
    """Raise AssertionError if analytic and numerical gradients disagree.

    Use np.allclose with both atol and rtol set to the given tolerance.
    On failure, include the maximum absolute difference in the error message
    so a debugging session has something to go on.
    """
    raise NotImplementedError("TODO: implement assert_gradients_close")


if __name__ == "__main__":
    # Run `python3 ml_coding/practice/gradcheck_practice.py` to check your
    # central_difference against a function with a known derivative.
    square_sum = lambda v: float((v**2).sum())  # d/dv = 2v
    parameter = np.array([1.0, 2.0, 3.0])
    numerical = central_difference(square_sum, parameter)
    print("numerical gradient:", numerical)
    print("expected (2 * parameter):", 2 * parameter)
