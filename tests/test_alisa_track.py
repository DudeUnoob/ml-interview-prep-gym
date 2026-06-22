import numpy as np

from ml_coding.gradcheck import assert_gradients_close, central_difference
from ml_coding.manual_linear import linear_backward, linear_forward
from transformers.gumbel_sampling import gumbel_softmax
from transformers.online_softmax import online_logsumexp, online_weighted_softmax


def test_linear_backward_matches_finite_difference():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 2))
    w = rng.normal(size=(2, 4))
    b = rng.normal(size=4)
    upstream = rng.normal(size=(3, 4))

    _, d_w, d_b = linear_backward(x, w, upstream)
    loss_for_w = lambda candidate: float((linear_forward(x, candidate, b) * upstream).sum())
    loss_for_b = lambda candidate: float((linear_forward(x, w, candidate) * upstream).sum())

    assert_gradients_close(d_w, central_difference(loss_for_w, w))
    assert_gradients_close(d_b, central_difference(loss_for_b, b))


def test_online_softmax_summaries_match_dense_calculation():
    logits = np.array([1000.0, 999.0, 1001.0])
    values = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    shifted = logits - logits.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()

    assert np.isclose(online_logsumexp(logits), logits.max() + np.log(np.exp(shifted).sum()))
    assert np.allclose(online_weighted_softmax(logits, values), probabilities @ values)


def test_gumbel_softmax_is_a_distribution():
    result = gumbel_softmax(np.array([1.0, 2.0, -1.0]), temperature=0.7, rng=np.random.default_rng(1))
    assert np.isclose(result.sum(), 1.0)
    assert np.all(result > 0.0)
