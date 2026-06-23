"""Implement each function yourself. Do not look at ml_coding/manual_linear.py
until tests/test_practice_ml_coding.py passes.

Self-check: pytest tests/test_practice_ml_coding.py -v
"""

from __future__ import annotations

import numpy as np

def linear_forward(inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Compute Z = XW + b. 
    Validates shapes before computing: X is (batch, n_in), W is (n_in, n_out), b is (n_out,).
    Raises ValueError on a mismatch rather than letting NumPy broadcast silently.
    
    Returns:
        np.ndarray: (batch, n_out)
    """
    # 1. Validate inputs are at least 2D matrices
    if inputs.ndim != 2:
        raise ValueError(f"Expected inputs to be a 2D array, got shape {inputs.shape} with {inputs.ndim} dims.")
    if weights.ndim != 2:
        raise ValueError(f"Expected weights to be a 2D array, got shape {weights.shape} with {weights.ndim} dims.")
    
    batch, n_in = inputs.shape
    n_in_w, n_out = weights.shape
    
    # 2. Enforce dimension agreement
    if n_in != n_in_w:
        raise ValueError(f"Dimension mismatch: inputs have {n_in} input features, but weights expect {n_in_w}.")
    
    # 3. Validate bias shape
    if bias.shape != (n_out,):
        raise ValueError(f"Dimension mismatch: bias shape is {bias.shape}, but expected ({n_out},).")
    
    # 4. Compute output
    Z = inputs @ weights + bias
    
    # Optional: Verify final output shape just in case
    expected_shape = (batch, n_out)
    if Z.shape != expected_shape:
        raise RuntimeError(f"Unexpected output shape {Z.shape}. Expected {expected_shape}.")
        
    return Z



def linear_backward(
    inputs: np.ndarray, weights: np.ndarray, upstream: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dX, dW, db for Z = XW + b given the upstream gradient dL/dZ.

    Derive each on paper first:
    - dX = ? (should have the same shape as inputs)
    - dW = ? (should have the same shape as weights; sums over the batch)
    - db = ? (should have the same shape as bias; sums over the batch)

    Validate that upstream.shape == (inputs.shape[0], weights.shape[1])
    before computing.
    """
    
    if upstream.shape == (inputs.shape[0], weights.shape[1]):
        
    
        dLdZ = upstream
        X = inputs

        dZdX = weights

        dX = dLdZ @ dZdX.T
        dW = X.T @ dLdZ
        dB = np.sum(dLdZ, axis=0)

        return (dX, dW, dB)
    
    else:
        raise ValueError
    
    
    


def relu(inputs: np.ndarray) -> np.ndarray:
    """Elementwise ReLU: max(x, 0)."""
    
    return np.maximum(inputs, 0)


def relu_backward(pre_activation: np.ndarray, upstream: np.ndarray) -> np.ndarray:
    """Backward pass through ReLU.

    Gradient flows through where the pre-activation was positive, and is
    zeroed out where it wasn't. pre_activation is the input to relu()
    (i.e. the value *before* the max(x, 0) was applied), not its output.
    """
    
    
    z = pre_activation
    
    dLda = upstream 
    
    da_dz = (z > 0).astype(float)
    
    dLdZ = dLda * da_dz
    
    return dLdZ
    


if __name__ == "__main__":
    # Quick sandbox: run `python3 ml_coding/practice/manual_linear_practice.py`
    # to call your functions directly with small, easy-to-reason-about inputs.
    # Add as many print() calls as you want inside the functions above or
    # right here — nothing here runs through pytest's output capture.
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    b = np.array([0.1, 0.1, 0.1])
    print("x:", x.shape, "w:", w.shape, "b:", b.shape)

    out = linear_forward(x, w, b)
    print("linear_forward output:\n", out)

    upstream = np.ones_like(out)
    d_x, d_w, d_b = linear_backward(x, w, upstream)
    print("d_x:\n", d_x)
    print("d_w:\n", d_w)
    print("d_b:\n", d_b)


