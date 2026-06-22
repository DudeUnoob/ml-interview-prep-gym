# Evaluation, Alignment, PPO, and GRPO

## Rapid-fire questions

- What is the difference between a reward model, a preference model, and a policy?
- Why can offline reward improve while user satisfaction degrades?
- What are KL penalties doing in PPO-style training?
- What problem does a group-relative advantage method such as GRPO aim to address conceptually?
- How would you evaluate a coding, retrieval, or multimodal model beyond a single benchmark?
- What causes benchmark contamination and how would you detect it?

## Experiment prompt

A new post-training method claims better helpfulness with no safety regression. Design the evaluation.

Cover: pre-registration or success criteria; held-out data; human evaluation protocol; preference/model-based metrics; slice analysis; confidence intervals; adversarial/safety cases; cost; and a launch gate.

## Answer standard

Do not just list metrics. Explain what decision each metric supports and what failure mode it can miss.
