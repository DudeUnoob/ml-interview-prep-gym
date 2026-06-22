# Alisa Notes → Interview Drills Map

This file turns the supplied notes into a concrete sequence. Do not try to memorize every page before coding; use each concept to power an implementation, derivation, or interview answer.

## 1. Neural-net mechanics

**Know:** MLP shape conventions, PyTorch's transposed `nn.Linear` weights, batch-wise gradient accumulation, activation derivatives, and the softmax-cross-entropy simplification.

**Do:**
- `ml_coding/manual_linear.py`: derive `dX = dZ W^T`, `dW = X^T dZ`, `db = sum_batch(dZ)`.
- `ml_coding/gradcheck.py`: verify gradients using centered finite differences.
- Explain why a nonlinear activation is required between stacked linear layers.

## 2. Numerical stability and attention

**Know:** max-shifted softmax, `logsumexp`, stable log-softmax, online softmax, and why standard attention is memory-bound.

**Do:**
- `transformers/online_softmax.py`.
- Explain why tiled FlashAttention avoids materializing the full `T x T` attention matrix.
- State the distinction precisely: this repo's streaming exercise demonstrates the algorithmic invariant; production FlashAttention is a GPU kernel/tiled IO implementation.

## 3. Optimization and memory

**Know:** SGD versus Adam/AdamW, moment estimates, bias correction, gradient clipping, warmup, and activation checkpointing.

**Do:**
- `ml_coding/adamw_numpy.py`.
- Explain why AdamW decouples weight decay from the adaptive gradient transform.
- For checkpointing, articulate the compute-for-memory trade-off and identify which activations must be recomputed.

## 4. Probability, statistics, and experimental judgment

**Know:** distributions, expectation/variance, CLT, Bayes/MLE/MAP, hypothesis-test selection, correlation versus causation, and paired classifier comparison.

**Do:**
- `math_probability/statistical_reasoning.md`.
- For an experiment-design question, specify unit of analysis, split, hypothesis, metric, uncertainty interval/test, and failure slices.

## 5. Differentiating through discrete choices

**Know:** categorical samples block ordinary pathwise gradients; Gumbel-Max samples categoricals; Gumbel-Softmax creates a differentiable stochastic relaxation; straight-through estimators use a biased backward approximation.

**Do:**
- `transformers/gumbel_sampling.py`.
- Explain temperature's trade-off: lower temperatures look more discrete but can produce less useful gradients.

## 6. Exam format

For each topic, prepare a one-page sheet: definition, dimensions/equation, 10-line implementation, numerical or systems failure mode, and three follow-up questions.
