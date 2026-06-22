# Distributed Training Drills

## Compare each technique

For data parallelism, FSDP/ZeRO, tensor parallelism, pipeline parallelism, sequence/context parallelism, and expert parallelism, be ready to explain:

- Which tensors are sharded or replicated
- Communication pattern and synchronization point
- Memory effect
- Throughput and latency trade-off
- Failure/bottleneck mode
- When you would choose it

## Core prompts

1. A 70B model does not fit on one GPU. Propose a training plan for 64 GPUs.
2. Training is communication-bound. What profiler signals would you inspect, and what changes would you try?
3. Loss spikes only at scale, not on one node. Build a debugging tree.
4. Explain gradient accumulation, mixed precision, activation checkpointing, and their trade-offs.
5. Design a reproducible training pipeline: data versioning, seed handling, checkpointing, evaluation gates, and rollback.

## NVIDIA/performance lens

Always connect the algorithm to hardware: FLOPs, HBM bandwidth, communication fabric, kernel launch overhead, sequence lengths, and batch shape.
