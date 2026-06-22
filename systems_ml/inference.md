# LLM Inference Drills

## Know cold

- Prefill is throughput/compute heavy; decode is latency and memory-bandwidth sensitive.
- KV cache avoids recomputing prior key/value projections, but cache memory grows with layers, batch size, context length, heads, and head dimension.
- Continuous batching increases utilization but complicates fairness, cancellation, and tail latency.
- Prefix caching helps repeated prompts; quantization trades precision for memory/bandwidth; speculative decoding trades verifier work for fewer expensive decode steps.

## Whiteboard prompts

1. Design an API that serves a 7B chat model at p95 under 500 ms for first token and 50 tokens/sec afterward.
2. Context length suddenly doubles. Diagnose capacity impact and choose mitigations.
3. GPU utilization is low despite a full request queue. Give at least three hypotheses and a measurement for each.
4. Compare tensor-parallel, pipeline-parallel, and data-parallel inference at small versus large model sizes.
5. Design observability: request queue time, time-to-first-token, inter-token latency, tokens/sec, cache hit rate, OOM rate, quality regressions, and cost/token.

## Follow-up questions

- What changes for multimodal inputs?
- What changes if requests are privacy-sensitive?
- What would you ship first in two weeks, and what would be phase two?
