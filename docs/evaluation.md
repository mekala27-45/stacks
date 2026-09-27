# Evaluation protocol

Goodbooks-10k has no interaction timestamps. Its publisher describes the file as time sorted, but individual dates cannot be audited. Source order is an ordering proxy, not a verified calendar-time split. Book metadata and tags are an undated snapshot, including when used by the content model.

The current run reads all 5,976,479 ratings and all 10,000 books. Ratings of at least four are binary positives. All observed training ratings, including dislikes, are excluded from recommendations. The first 80% of source rows form training and the final 20% form the holdout. Source revision, exact cutoffs and counts are in the manifest. The earlier 1,000-book, 300,000-row run remains under `results/history/bounded-v0.1.0.json`; its numbers are not evidence for the current population.

## Populations and candidates

Headline evaluation samples at most 5,000 eligible readers with at least five positive training items and at least one unseen positive holdout item, using seed 20260926. Every eligible unseen catalog item is ranked, with item ID breaking score ties. Recall@200 measures retrieval; NDCG@10, Recall@10 and hit rate@10 measure ordering. The source catalog is complete, but reader-level metrics concern a sampled eligible cohort rather than all readers.

Cold-reader evaluation separately samples at most 1,000 readers with zero to four positive training items. No-history readers use popularity where the model has no applicable profile. Cold-item recall concerns holdout positives with no positive training count. The report supplies training-history size, holdout size, target popularity and holdout-size strata for warm and cold cohorts. A larger cold-reader NDCG is not evidence that removing history helps: the cohorts have different relevance sets and candidate distributions.

## Models and ranker isolation

Popularity counts positive training interactions. Item cosine uses binary user-item columns, retains each item's strongest 100 neighbors, and sums similarity over positive training history. Sparsifying the model does not change the full-catalog evaluation universe. Implicit ALS uses confidence `1 + 20r`, 16 factors, regularization 0.1, six alternating iterations, and seed 17. The fixed blend is 0.55 normalized cosine, 0.35 normalized ALS and 0.10 normalized popularity.

The content model fits TF-IDF to title, author, tags and genre with unigram/bigram features and at most 40,000 terms. Its reader representation is a normalized mean of positive-history item vectors. These text features are a snapshot; historical availability cannot be verified.

LambdaMART is a trained ranker. Retrieval models fit the first 60% of source rows. Positives from the next 20% provide ranker labels. Training candidates are the union of the top 200 results from popularity, cosine, ALS and blend. Features are raw and normalized retrieval scores, log popularity, log history size and a cold-item flag; there are no metadata features. A seeded sample of at most 3,000 readers with nontrivial relevance groups trains 70 LightGBM trees with 15 leaves, maximum depth 5, learning rate 0.055, minimum child size 100, lambda regularization 1 and seed 861. Final retrieval features are refit on the first 80% using the same settings. The final 20% never supplies ranker training labels. Full-catalog ranker scoring extends beyond its training candidate distribution, an explicit extrapolation limitation.

Additional model definitions and training diagnostics are recorded in manifest modules and model cards. Learned two-tower and recurrent models remain distinct from the simpler session cosine diagnostic. Model complexity alone is not promotion evidence.

## Uncertainty and comparisons

Metric intervals are percentile 95% intervals from 1,000 reader bootstrap resamples. Paired NDCG differences use identical sampled reader indices. All pairwise model comparisons share one Benjamini-Hochberg family at false discovery rate 0.05, using two-sided paired random-sign permutation p-values with 4,999 permutations. The family expands when models are added. A corrected winner must beat every other model with positive paired difference and corrected p below 0.05. Otherwise only an observed leader is reported.

Observed catalog coverage is the deterministic share of catalog IDs in the measured top-ten lists, stored as `coverage.mean` and `observed_coverage`. Its adjacent low/high values are a central 95% reader-resampling range, not a confidence interval for the observed point or for unseen readers. Resampling existing lists can omit rare IDs and cannot invent new IDs, so the observed union may exceed that range. `expected_resampled_coverage` separately stores the mean resampled value.

Long-tail items are the lower 80% of catalog items by training popularity. Novelty is negative log2 of add-one-smoothed popularity. Main-table diversity is one minus the mean symmetric pair similarity from the pruned item-cosine matrix: each pair averages both directed edges, treating missing pruned edges as zero. This is sparse collaborative dissimilarity, not complete unpruned cosine diversity. Calibration is Jensen-Shannon divergence between smoothed history and recommendation genre distributions. Exported histories preserve source-row order among the last twelve positive training interactions; this is not verified reading chronology.

## Shortcut and reranking diagnostics

A seeded random 80/20 interaction split retrains the same model definitions and evaluates its own eligible readers. Ranker fitting nests a 75/25 split within that outer training partition, keeping the outer test unseen. Differences from the ordered split mix split and cohort effects; they are not a causal estimate of leakage. Sampled ranking instead uses every holdout positive plus 100 sampled unseen negatives for the same ordered-split readers. Seed 61 and candidate sets are shared across models. Full ranking remains the headline protocol.

The reranking diagnostic uses a seeded 100-reader subset, top 200 blend candidates, author cap two, MMR weight 0.25 and calibration weight 0.2. Exact serving functions generate each stage, and popularity is measured on the same subset. Its diversity is tag Jaccard diversity; its calibration is unsmoothed natural-log Jensen-Shannon divergence. Those definitions differ from the main comparison. The table does not measure exploration or an online outcome effect.

## Off-policy evidence

The simulator has twelve equally likely contexts, six actions, a known reward table, a full-support logging policy and a fixed softmax target. Exact policy value enumerates the mechanism. Sample sizes 250 and 1,000 each run 200 independent seeds. Local IPS, SNIPS, DM and DR use 200 bootstrap draws per sample. Reward predictions are either the oracle table or a deliberately constant 0.2 table; oracle DM is not a learned reward model. Bias, variance, RMSE and interval coverage include Monte Carlo uncertainty. Coverage uncertainty uses Wilson binomial intervals, including zero observed coverage.

The real Open Bandit benchmark reads all six random/BTS campaign files. Published campaign beta priors and the official BernoulliTS rule define a Monte Carlo target. The first chronological half of random logs fits an action reward model; the second half supplies OPE. BTS logs after the same timestamp cutoff supply empirical on-policy reward and Wilson intervals. Logged BTS propensities vary within item and position, so the frozen prior cannot be asserted identical to the actual per-impression logging policy. OPE-minus-BTS differences include target mismatch, sampling and estimator error. The target is never inferred from marginal action frequencies. Bootstrap intervals condition on the fitted reward model and estimated target, ignore repeated-reader dependence, and concern individual positions rather than joint slates. See [the benchmark construction](ope-benchmark.md) for exact sources and diagnostics.

Production interaction evidence is separate from both datasets. A live logger needs identified action probabilities, an eligible target with support and matured outcomes before off-policy results are available. Missing outcomes are not immediately negatives.

These are demonstration recommendations on a public dataset; no real reader's identity is present and no recommendation is personalized to a real person.
