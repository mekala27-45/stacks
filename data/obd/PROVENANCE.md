# Open Bandit samples

Source: https://github.com/st-tech/zr-obp. Pinned commit `8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4`. All six random/BTS campaign files (all, men, women) are exported with selected logged-action columns. Exact URLs, transformations and SHA-256 digests are in `provenance.json`.

The source repository's Apache-2.0 notice is retained in `LICENSE`. The dataset paper also specifies CC BY 4.0; both provenance statements are retained. Attribution: Saito, Aihara, Matsutani, and Narita (2020), Open Bandit Dataset and Pipeline.

Published campaign beta priors and the official BernoulliTS rule define the Monte Carlo target in `bts-probabilities.json`. Target probabilities are never inferred from action frequencies. The reward model fits the first chronological half of random logs; OPE uses the second half. BTS rewards use records after that same absolute timestamp cutoff. Actual BTS propensities vary within item and slot, so the frozen-prior target is an approximation of the logged policy. Empirical BTS reward is an uncertain on-policy reference, not exact simulator truth.

Slots preserve the source's one-based position. Per-position propensities are not joint-slate probabilities. See [benchmark construction](../../docs/ope-benchmark.md) for target-mismatch, bootstrap, reward-fit and repeated-reader limitations.
