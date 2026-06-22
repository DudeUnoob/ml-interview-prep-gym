import numpy as np

from dsa.reference import combination_sum_ii, course_schedule, num_islands, two_sum
from ml_coding.numpy_primitives import cross_entropy, softmax, softmax_cross_entropy_gradient
from transformers.attention_numpy import causal_mask, scaled_dot_product_attention
from transformers.kv_cache import KVCache
from transformers.positional_encoding import sinusoidal_positional_encoding


def test_two_sum():
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]


def test_islands():
    grid = [["1", "1", "0"], ["0", "1", "0"], ["1", "0", "1"]]
    assert num_islands(grid) == 3


def test_course_schedule():
    assert course_schedule(2, [[1, 0]])
    assert not course_schedule(2, [[1, 0], [0, 1]])


def test_combination_sum_ii():
    result = combination_sum_ii([10, 1, 2, 7, 6, 1, 5], 8)
    assert sorted(result) == sorted([[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]])


def test_softmax_and_cross_entropy():
    logits = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
    probabilities = softmax(logits)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert cross_entropy(logits, np.array([2, 0])) > 0
    assert np.allclose(softmax_cross_entropy_gradient(logits, np.array([2, 0])).sum(axis=1), 0.0)


def test_attention_shapes_and_mask():
    q = np.ones((1, 3, 2))
    k = np.ones((1, 3, 2))
    v = np.arange(6, dtype=float).reshape(1, 3, 2)
    output, weights = scaled_dot_product_attention(q, k, v, causal_mask(3))
    assert output.shape == (1, 3, 2)
    assert np.allclose(weights[0, 0, 1:], 0.0)


def test_position_encoding_and_cache():
    assert sinusoidal_positional_encoding(4, 7).shape == (4, 7)
    cache = KVCache()
    keys = np.zeros((2, 4, 1, 8))
    cache.append(keys, keys)
    cache.append(keys, keys)
    assert cache.sequence_length == 2
