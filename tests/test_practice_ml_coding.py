"""Tests for your own implementations in ml_coding/practice/.

Run: pytest tests/test_practice_ml_coding.py -v

Until you implement each function, the matching test fails with
NotImplementedError — that's expected, not a bug.
"""

import numpy as np
import torch

from ml_coding.practice.adamw_numpy_practice import adamw_step, init_state
from ml_coding.practice.gradcheck_practice import assert_gradients_close, central_difference
from ml_coding.practice.manual_linear_practice import (
    linear_backward,
    linear_forward,
    relu,
    relu_backward,
)
from ml_coding.practice.numpy_primitives_practice import (
    cross_entropy,
    linear_regression_gradients,
    logistic_regression_gradients,
    sigmoid,
    softmax,
    softmax_cross_entropy_gradient,
)
from ml_coding.practice.pytorch_drills_practice import TwoLayerMLP, assert_shape, one_training_step


def test_sigmoid_matches_definition_and_is_stable():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    result = sigmoid(x)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))), atol=1e-6)


def test_softmax_sums_to_one_and_is_stable():
    logits = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 999.0]])
    probabilities = softmax(logits)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=-1), 1.0)


def test_cross_entropy_and_gradient():
    logits = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
    targets = np.array([2, 0])
    assert cross_entropy(logits, targets) > 0
    grad = softmax_cross_entropy_gradient(logits, targets)
    assert np.allclose(grad.sum(axis=1), 0.0, atol=1e-6)


def test_linear_regression_gradients_match_finite_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 3))
    y = rng.normal(size=20)
    weights = rng.normal(size=3)
    bias = 0.5

    grad_w, grad_b = linear_regression_gradients(x, y, weights, bias)

    def loss(w):
        predictions = x @ w + bias
        return float(np.mean((predictions - y) ** 2))

    numerical = central_difference(loss, weights)
    assert_gradients_close(grad_w, numerical, atol=1e-4)
    assert isinstance(grad_b, float)


def test_logistic_regression_gradients_match_finite_difference():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 3))
    y = (rng.normal(size=20) > 0).astype(float)
    weights = rng.normal(size=3)
    bias = -0.2

    grad_w, _ = logistic_regression_gradients(x, y, weights, bias)

    def loss(w):
        probabilities = sigmoid(x @ w + bias)
        eps = 1e-12
        return float(-np.mean(y * np.log(probabilities + eps) + (1 - y) * np.log(1 - probabilities + eps)))

    numerical = central_difference(loss, weights)
    assert_gradients_close(grad_w, numerical, atol=1e-4)


def test_linear_forward_and_backward_match_finite_difference():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 2))
    w = rng.normal(size=(2, 4))
    b = rng.normal(size=4)
    upstream = rng.normal(size=(3, 4))

    output = linear_forward(x, w, b)
    assert output.shape == (3, 4)

    d_x, d_w, d_b = linear_backward(x, w, upstream)
    assert d_x.shape == x.shape
    assert d_w.shape == w.shape
    assert d_b.shape == b.shape

    loss_for_w = lambda candidate: float((linear_forward(x, candidate, b) * upstream).sum())
    loss_for_b = lambda candidate: float((linear_forward(x, w, candidate) * upstream).sum())
    assert_gradients_close(d_w, central_difference(loss_for_w, w))
    assert_gradients_close(d_b, central_difference(loss_for_b, b))


def test_relu_and_relu_backward():
    pre_activation = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    activated = relu(pre_activation)
    assert np.array_equal(activated, np.array([0.0, 0.0, 0.0, 0.5, 2.0]))

    upstream = np.ones_like(pre_activation)
    grad = relu_backward(pre_activation, upstream)
    assert np.array_equal(grad, np.array([0.0, 0.0, 0.0, 1.0, 1.0]))


def test_adamw_step_reduces_a_simple_quadratic():
    parameter = np.array([10.0, -5.0])
    state = init_state(parameter)
    for _ in range(200):
        gradient = 2 * parameter  # gradient of sum(parameter ** 2)
        parameter, state = adamw_step(parameter, gradient, state, learning_rate=0.1)
    assert np.all(np.abs(parameter) < 0.5)


def test_two_layer_mlp_training_step_reduces_loss():
    torch.manual_seed(0)
    model = TwoLayerMLP(input_dim=4, hidden_dim=8, output_dim=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(5, 4)
    targets = torch.randint(0, 3, (5,))

    first_loss = one_training_step(model, optimizer, x, targets)
    for _ in range(20):
        last_loss = one_training_step(model, optimizer, x, targets)
    assert last_loss < first_loss


def test_assert_shape_raises_on_mismatch():
    tensor = torch.zeros(2, 3)
    assert_shape(tensor, (2, 3), "tensor")
    try:
        assert_shape(tensor, (3, 2), "tensor")
        raise AssertionError("expected assert_shape to raise on mismatch")
    except ValueError:
        pass
