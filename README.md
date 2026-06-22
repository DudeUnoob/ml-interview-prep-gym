# ML Interview Prep Gym

A practical prep repository for ML/AI, coding, systems, math, research, and behavioral interviews.

## How to use this repo

1. Do a first timed attempt with **no AI and no notes**.
2. Run the tests.
3. Debug, then write down the failure mode in `notes/interview_log.md`.
4. Re-implement and explain the solution aloud.

## 8-week roadmap

- **Weeks 1–2:** DSA, recursion/backtracking, probability, linear algebra.
- **Weeks 3–4:** NumPy ML primitives, gradients, backprop, PyTorch fluency.
- **Weeks 5–6:** Attention, positional encoding, decoding, mini-transformer concepts.
- **Week 7:** ML inference, KV cache, distributed-training design.
- **Week 8:** Mock interviews, behavioral stories, research pitch, company-specific cramming.

## Core folders

- `dsa/` — general coding foundations
- `ml_from_scratch/` — linear models, optimization, losses
- `deep_learning/` — NumPy and PyTorch exercises
- `transformers/` — attention, positional encoding, decoding
- `systems_ml/` — inference and distributed-systems discussion prompts
- `math_probability/` — derivation and fundamentals drills
- `behavioral/` — STAR story bank
- `research_pitch/` — project-pitch templates

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pytest
```
