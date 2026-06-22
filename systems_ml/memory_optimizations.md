# Training-Memory Optimization Packet

## Activation checkpointing

Backprop needs forward activations. Checkpointing saves only selected boundaries and recomputes omitted forward segments during backward.

**Trade-off:** less activation memory, more compute. State this before discussing any formula.

**Prompt:** A model OOMs at target sequence length. Rank mitigations by speed of implementation and likely impact: gradient accumulation, mixed precision, activation checkpointing, sequence packing, optimizer-state sharding, attention kernel, smaller microbatch, and model architecture changes.

## Gradient clipping

Global norm clipping rescales every gradient by the same factor only when the total norm exceeds a threshold. Explain why it protects against catastrophic updates without changing direction.

## Mixed precision

Discuss FP16/BF16 trade-offs, loss scaling when relevant, master weights/optimizer state, and the need to watch overflow/underflow.

## Warmup and schedules

Learning-rate schedules depend primarily on training time; optimizers use per-parameter gradient history. Explain why warmup can reduce unstable early updates and why schedules should be evaluated with the rest of the training recipe.

## Interview checklist

For any optimization: name the memory resident being reduced, what is recomputed/communicated instead, expected performance effect, numerical risk, and measurement used to verify improvement.
