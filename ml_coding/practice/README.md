# Your Implementation Workspace

Each file here mirrors one of the reference modules in `ml_coding/` but with
the bodies stripped out. Fill them in yourself — **do not open the
corresponding reference file until you've passed your own tests.**

## Workflow

1. Pick one file below (start with `numpy_primitives_practice.py`).
2. Read the docstring on each function — it has the math/shape spec, not the answer.
3. Implement the function body. Delete the `raise NotImplementedError(...)` line.
4. Self-check with the dedicated test file:
   ```bash
   pytest tests/test_practice_ml_coding.py -v
   ```
5. Once green, open the matching reference file (e.g. `ml_coding/numpy_primitives.py`)
   and diff your approach against it. Note anything you did differently —
   especially numerical-stability tricks you missed.
6. Log one mistake/insight per drill in `notes/interview_log.md`.

## Files, in recommended order

1. `numpy_primitives_practice.py` — sigmoid, softmax, cross-entropy, regression gradients
2. `manual_linear_practice.py` — linear layer forward/backward, ReLU
3. `gradcheck_practice.py` — finite-difference gradient checking (you'll use this to verify #2 yourself)
4. `adamw_numpy_practice.py` — AdamW optimizer step
5. `pytorch_drills_practice.py` — two-layer MLP + training step in PyTorch

## Rules

- No peeking at the reference file or test_core.py/test_alisa_track.py assertions before your own tests pass.
- Derive shapes on paper before writing the matrix ops.
- If `pytest` fails, read the failure message and fix it yourself before asking for help.
