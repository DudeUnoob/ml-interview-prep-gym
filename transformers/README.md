# Transformer Interview Track

You should be able to explain and implement:

1. Shapes: batch `B`, sequence `T`, heads `H`, head dimension `D`.
2. Scaled dot-product attention and why the `sqrt(D)` factor matters.
3. Causal and padding masks.
4. Positional encodings: sinusoidal, learned, RoPE, ALiBi at a conceptual level.
5. Autoregressive decoding: greedy, temperature, top-k, nucleus sampling.
6. Prefill versus decode; why KV cache changes compute but consumes memory.
7. How to debug: check tensor shapes, mask direction, softmax axis, and train/eval behavior.

## Rule

Rebuild `attention_numpy.py`, `positional_encoding.py`, and `decoding.py` from a blank file until they are muscle memory.
