# Definitions

These are demonstration recommendations on a public dataset; no real reader's
identity is present and no recommendation is personalized to a real person.

| Term | Meaning in this repository |
| --- | --- |
| Evaluated catalog | The bounded set of books selected by the recorded pipeline protocol |
| Full ranking | Ranking all eligible books in that evaluated catalog after training-history exclusions |
| Source-order split | Partitioning interactions by source CSV order; a proxy with no verifiable event dates |
| Sampled negatives | A diagnostic that compares held-out positives against a smaller sampled candidate set |
| Popularity | A baseline whose signal is derived from training interactions |
| Item cosine | Similarity from item interaction vectors, with training-derived histories |
| ALS | Alternating least squares factorization with implicit-feedback confidence weighting |
| Blend | An explicit combination of component scores; not a trained learning-to-rank model |
| Retrieval recall | Fraction of relevant held-out items included in the stated candidate cutoff |
| NDCG | Discounted ranking gain divided by the ideal gain for that user's held-out relevance |
| Catalog coverage | Share of the evaluated catalog appearing in the measured recommendation lists |
| Novelty | A training-popularity-derived diagnostic; not evidence of reader satisfaction |
| Long tail | Books outside the protocol's training-popularity head, using its stated cutoff |
| Paired bootstrap | Resampling the same users for model comparisons to preserve pairing |
| BH adjustment | Benjamini-Hochberg adjustment within the declared comparison family |
| Propensity | Probability of the logged action under the actual logging policy and action unit |
| IPS | Mean reward weighted by target-policy probability divided by logging probability |
| SNIPS | Reward-weighted importance-weight sum divided by the importance-weight sum |
| Direct method | Target-policy value predicted by a reward model |
| Doubly robust | Direct prediction plus an importance-weighted observed reward residual |
| Effective sample size | Squared sum of weights divided by sum of squared weights |
| Simulator truth | Target-policy value computed directly from the simulation's known reward mechanism |
| Open Bandit sample | Real fashion recommendation logs; they do not measure the bookstore's user response |
| Browser mode | Exploration using static assets and local browser state |
| API mode | Requests handled by a running service with persistent relational interaction records |
| Promotion eligibility | Evaluation of all required evidence gates, separate from serving activation |

Exact cutoffs, aggregation choices, intervals, and data counts are recorded in
the [evaluation protocol](evaluation.md) and generated [results](../RESULTS.md).
Missing evidence is not a zero metric, and a missing outcome is not automatically
a negative outcome before the documented observation window closes.
