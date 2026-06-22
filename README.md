# ML Interview Prep Gym

A practical prep repository for ML/AI, applied-science, research-engineering, performance/systems, and quant interviews.

## Start here

1. Read `reference/alisa_notes_curriculum.md` to turn the supplied LLM notes into a focused study sequence.
2. Do the first timed attempt with **no AI and no notes**.
3. Run `pytest`.
4. Log one primary mistake in `notes/interview_log.md`.
5. Re-implement the solution and explain it aloud.

## Recommended order

- **Weeks 1–2:** `dsa/` plus `math_probability/`
- **Weeks 3–4:** `ml_coding/` — manual linear/backward, gradient checks, losses, AdamW, PyTorch
- **Weeks 5–6:** `transformers/` — attention, positions, decoding, KV cache, online softmax, Gumbel sampling
- **Weeks 7–8:** `systems_ml/` — inference, memory, distributed training, evaluation, multimodal/robotics
- **Weeks 9–10:** `research/`, `behavioral/`, and `mocks/`
- **Weeks 11–12:** `quant/` and company-specific mock loops

## Core folders

- `dsa/` — general coding patterns plus ML-flavored variants
- `ml_coding/` — NumPy/PyTorch implementations, manual gradients, gradient checking, AdamW
- `transformers/` — attention, positional encodings, decoding, KV cache, online softmax, mini Transformer
- `systems_ml/` — inference, memory, distributed training, evaluation/RL, multimodal system design
- `math_probability/` — probability, statistics, derivations, and test selection
- `research/` — research-discussion preparation for your public-safe project narratives
- `behavioral/` — tailored STAR-story mapping
- `quant/` — quant research/developer validation and systems drills
- `mocks/` — 60-minute interview simulation and scorecard
- `reference/` — source-to-drill maps, including the Alisa-notes curriculum
- `tests/` — correctness checks for reference implementations

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pytest
```

## Rules that make this work

- Use AI **after** the timed attempt, not during it.
- Derive shapes before coding tensor operations.
- Pair every implementation with a test or numerical check.
- For systems questions: state constraints, baseline, bottleneck, measurement, trade-off, and rollback.
- For research questions: state problem, hypothesis, method, result, limitation, and next experiment.
