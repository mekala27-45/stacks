# Open Bandit benchmark

These are demonstration recommendations on a public dataset; no real reader's
identity is present and no recommendation is personalized to a real person.

The benchmark reads random and Bernoulli Thompson sampling logs for the `all`,
`men`, and `women` campaigns. Each source file is pinned to the upstream commit
in `data/obd/provenance.json`; the exported parquet keeps actions, positions,
clicks, timestamps, and the actual logging propensities.

## Target policy

The [official benchmark](https://github.com/st-tech/zr-obp/blob/8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4/examples/obd/evaluate_off_policy_estimators.py)
creates `BernoulliTS` with `is_zozotown_prior=True` and the campaign name. It
estimates action probabilities by Monte Carlo simulation, then compares policy
value with observed BTS rewards. Stacks reproduces this construction using the
[published beta parameters](https://github.com/st-tech/zr-obp/blob/8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4/obp/policy/conf/prior_bts.yaml)
and [policy selection rule](https://github.com/st-tech/zr-obp/blob/8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4/obp/policy/contextfree.py).
The prior file is checked against a fixed SHA-256 digest before use.

For each draw, an independent beta reward is sampled for each action; the three
largest draws determine the ordered slate. Frequency at each position estimates
that position's marginal action probability. The implementation uses the
upstream NumPy random generator, and a test compares its batched computation
with the original single-draw rule. These are generated policy probabilities,
not estimates reconstructed from observed action frequencies.

## Evaluation and uncertainty

The first chronological half of the random sample fits a smoothed action reward
model. The latter half provides OPE. Its first timestamp is also the absolute
cutoff used for the BTS reference sample, so the two policies' observation
periods align. IPS, SNIPS, DM and DR use the same logged-action contracts as the
simulator. Uniform-target IPS and SNIPS supply an on-policy identity check.

The BTS benchmark is an empirical reward estimate with a Wilson interval.
The estimator-minus-benchmark interval resamples the random and BTS records
independently. The generated probability artifact also reports the Monte Carlo
standard error and a second independent seed's maximum discrepancy.

## Why the comparison is not exact ground truth

The actual logged BTS propensities vary within a given item and position. A
frozen prior therefore cannot be assumed to reconstruct the actual policy at
every logged impression. The returned evidence quantifies that variation.
The official prior-based policy is a reproducible benchmark approximation;
differences from observed BTS reward can reflect target mismatch as well as
sampling and estimator error. It would be misleading to label them pure
estimator bias against exact truth.

Intervals condition on the fitted reward model and sampled target probabilities.
They omit model-fit and Monte Carlo uncertainty and repeated-reader dependence.
Action-only DM can consequently have a degenerate conditional interval. The
[dataset paper](https://arxiv.org/abs/2008.07146) also discusses the assumption
that an item's reward at a position does not depend on other displayed items.
Position probabilities are never multiplied into a claimed joint-slate
probability here.

The repository sample's Apache notice is retained. The dataset paper separately
specifies CC BY 4.0; the provenance records both statements and the authors'
attribution instead of silently treating original application licensing as
dataset licensing.
