# Alisa-Style LLM Interview Track

This checklist translates the topics named in the supplied interview-prep write-up into concrete repo work.

## Foundations

- Language-modeling pipeline: tokenization, embeddings, transformer blocks, logits, loss, sampling.
- Reimplement and debug a transformer until causal masking, dimensions, residuals, normalization, and cross-entropy are automatic.
- Build comfort with PyTorch and NumPy-only implementations.

## Deep dives

- Positional information: sinusoidal encodings, RoPE, ALiBi—trade-offs and extrapolation intuition.
- Inference: prefill/decode, KV cache, batching, quantization, speculative decoding.
- Post-training: preference data, reward models, PPO-style objectives, GRPO-style relative advantages, evaluation.
- Scaling: data/model/pipeline/tensor/expert parallelism, activation checkpointing, mixed precision.

## How to study

For each topic, make one page containing: definition, shapes/equations, failure modes, implementation exercise, systems implication, and three likely interviewer follow-ups.

## Interview simulation

Treat each interview like a different compressed course exam: make a company-specific three-day cram sheet from the role description, recruiter hints, and team domain. Practice without AI during the timed attempt, then use AI only for postmortem/debugging.
