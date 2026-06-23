"""Implement each function yourself. Do not look at ml_coding/numpy_primitives.py
until tests/test_practice_ml_coding.py passes.

Self-check: pytest tests/test_practice_ml_coding.py -v
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Elementwise sigmoid: 1 / (1 + exp(-x)).

    Pitfall to think about: exp(-x) overflows for very negative x and
    exp(x) overflows for very large x if you rearrange the formula naively.
    Find a numerically stable way to branch on the sign of x.

    Shapes: x is any shape; output matches x's shape.
    """
    
    return (np.exp(x)) / (1 + np.exp(x))    
    


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis.

    Pitfall: exp() of large logits overflows. Subtract something from the
    logits first that doesn't change the result mathematically.

    Shapes: logits is (..., n_classes); output is the same shape and each
    row along the last axis sums to 1.
    """
    
    
    
    raise NotImplementedError("TODO: implement softmax")


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean cross-entropy loss for a batch of logits against integer class targets.

    Steps: convert logits to probabilities, pick out the probability of the
    correct class for each row, take -log of it, average over the batch.
    Add a small epsilon inside the log to avoid log(0).

    Shapes: logits is (batch, n_classes), targets is (batch,) of ints in
    [0, n_classes). Returns a scalar float.
    """
    raise NotImplementedError("TODO: implement cross_entropy")


def softmax_cross_entropy_gradient(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Gradient of mean softmax-cross-entropy loss with respect to logits.

    Hint: the combined softmax+cross-entropy gradient has a famously clean
    closed form in terms of the softmax probabilities and the one-hot targets.
    Don't forget to average over the batch (divide by batch size).

    Shapes: logits is (batch, n_classes), targets is (batch,) of ints.
    Returns an array the same shape as logits.
    """
    raise NotImplementedError("TODO: implement softmax_cross_entropy_gradient")


def linear_regression_gradients(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[np.ndarray, float]:
    """Gradients of mean-squared-error loss for linear regression.

    Loss = mean((x @ weights + bias - y) ** 2)
    Derive d(loss)/d(weights) and d(loss)/d(bias) by hand first.

    Shapes: x is (n, d), y is (n,), weights is (d,), bias is scalar.
    Returns (grad_weights of shape (d,), grad_bias as a float).
    """
    raise NotImplementedError("TODO: implement linear_regression_gradients")


def logistic_regression_gradients(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[np.ndarray, float]:
    """Gradients of mean binary cross-entropy loss for logistic regression.

    probabilities = sigmoid(x @ weights + bias)
    Derive d(loss)/d(weights) and d(loss)/d(bias) by hand first — the
    sigmoid derivative cancels nicely against the cross-entropy derivative.

    Shapes: x is (n, d), y is (n,) of 0/1, weights is (d,), bias is scalar.
    Returns (grad_weights of shape (d,), grad_bias as a float).
    """
    raise NotImplementedError("TODO: implement logistic_regression_gradients")


if __name__ == "__main__":
    # Run `python3 ml_coding/practice/numpy_primitives_practice.py` to call
    # these directly and add prints wherever you're stuck — no pytest capture.
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("sigmoid:", sigmoid(x))

    logits = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]])
    targets = np.array([2, 0])
    print("softmax:\n", softmax(logits))
    print("cross_entropy:", cross_entropy(logits, targets))
    print("softmax_cross_entropy_gradient:\n", softmax_cross_entropy_gradient(logits, targets))
