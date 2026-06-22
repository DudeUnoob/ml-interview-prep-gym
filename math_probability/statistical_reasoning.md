# Statistical Reasoning for ML and Quant Interviews

## Distribution recognition

| Situation | Default model / tool |
|---|---|
| One binary event | Bernoulli |
| Number of successes in fixed trials | Binomial |
| Count of independent events in a time window | Poisson |
| Trials until first success | Geometric |
| Time until first event | Exponential |
| Mean of many iid observations | CLT / approximately Gaussian |
| Update belief after evidence | Bayes rule |
| Estimate a parameter from data | MLE / MAP |
| Expected count | Indicators + linearity of expectation |
| Hitting time / random walk | First-step analysis |

## Test-selection drill

- Compare means of two independent continuous groups: two-sample t-test, subject to assumptions.
- Compare paired measurements on the same items: paired t-test.
- Compare two classifiers on the same examples with binary correctness: McNemar-style paired comparison.
- Compare continuous distributions: KS test.
- Compare categorical distributions: chi-squared test.
- Test a linear association: Pearson; monotonic/rank association: Spearman.

## Interview answer template

1. Name the population, sampling unit, and randomization/independence assumptions.
2. State null and alternative hypotheses.
3. Select metric and test/interval based on the decision.
4. Address multiple comparisons, distribution shift, and practical—not only statistical—significance.
5. Report an effect size and uncertainty, not only a p-value.

## Derivation drills

- Derive MLE for Bernoulli success probability and exponential rate.
- Use first-step analysis to derive the expected geometric waiting time.
- Prove the variance of a Binomial as the sum of independent Bernoulli variances.
- Derive a normal-approximation confidence interval and explain why standard error shrinks as `1/sqrt(n)`.
- Solve one Bayes-rule puzzle and one indicator-variable expectation problem every week.
