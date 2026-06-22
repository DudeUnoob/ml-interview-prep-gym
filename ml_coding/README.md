# ML Coding Track

These drills mirror the common format: implement a component cleanly, reason about shapes, preserve numerical stability, and explain backward flow.

## Required fluency

- NumPy broadcasting and matrix dimensions
- PyTorch `nn.Module`, parameters, `.train()` / `.eval()`, optimizer loop
- Softmax and cross-entropy stability
- Forward and backward passes for simple networks
- Attention masks and causal masks
- Debugging: print shapes, isolate a tiny case, compare to a known implementation

## Practice sequence

1. Linear + logistic regression
2. Softmax cross-entropy
3. Two-layer MLP and manual gradients
4. Normalization and dropout
5. Attention and decoding

First rewrite each reference implementation without looking. Then deliberately introduce one bug and diagnose it with tests.
