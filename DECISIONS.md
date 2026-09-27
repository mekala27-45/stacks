# Decisions

These records describe choices made while implementing and correcting the repository on 2026-09-27. They are not a reconstructed multi-day development history. The supplied brief is a scope reference; delivered modules and remaining limitations are recorded in the generated README and results. Commit history follows actual changes without a manufactured commit quota.

## 01. Preserve source order without inventing dates

**Date: 2026-09-27.** Goodbooks supplies no interaction timestamps. Retain its source-row ordering and label the split a proxy. This permits reproducible exclusion boundaries but cannot establish dated historical feature availability. See [the protocol](docs/evaluation.md).

## 02. Expand the bounded release to the complete source

**Date: 2026-09-27. Reversal.** The initial implementation selected 1,000 books and 300,000 rows for short CPU runs. The audit established that this did not answer the full-source scope. The final pipeline reads every source interaction and all 10,000 books while sampling eligible evaluation readers explicitly. The original evidence is preserved in [bounded history](results/history/bounded-v0.1.0.json), not silently replaced with a broader claim.

## 03. Train the ranker behind a separate label boundary

**Date: 2026-09-27.** The fixed score blend remains a baseline, and LambdaMART is a separate trained model. Prefix retrieval features and the next partition's labels train the ranker before the final holdout. Training candidate selection introduces a distribution limit when scoring the full catalog; this is recorded rather than hidden by evaluating only easy negatives.

## 04. Require paired evidence before naming a winner

**Date: 2026-09-27.** Preserve popularity in every comparison, pair reader resamples and adjust the complete pairwise family. The earlier ALS/blend difference did not establish a corrected winner. A point estimate or a richer architecture is insufficient to justify promotion. Offline eligibility does not activate a served policy.

## 05. Correct the catalog-coverage interval interpretation

**Date: 2026-09-27. Reversal.** The initial generic interval display made aggregate coverage look like an ordinary confidence interval. Distinct-item coverage is the observed union of recommendation lists; resampling those lists usually loses rare items. The published point now retains that observed value and the adjacent bracket explicitly says reader-resampling range, not confidence interval. The average resampled value is a separate field.

## 06. Replace the narrow Open Bandit target with the official benchmark

**Date: 2026-09-27. Reversal.** The first real-log check used only the all campaign and a custom CTR policy. That did not reproduce the requested BTS comparison. The replacement reads all three campaigns under both policies and reconstructs the official prior-based BernoulliTS target. Within-item propensity variation rejects an exact-policy interpretation. Empirical BTS reward and difference intervals are therefore accompanied by a target-mismatch limitation. [Construction and sources](docs/ope-benchmark.md).

## 07. Keep an executable independent arithmetic check

**Date: 2026-09-27.** The pinned OBP package requires an older Python stack. Rather than claim installed-package compatibility on Python 3.12, extract hash-verified, unmodified estimator arithmetic ASTs and compare the same seeded fixtures. This verifies point estimates only, not upstream validation, dependencies or bootstrap behavior. The scope is retained in the result artifact.

## 08. Use free hosting with an explicit serving contract

**Date: 2026-09-27.** The user's free-hosting constraint rules out depending on paid API infrastructure. The architecture provides a portable Worker and D1 store alongside a locally runnable FastAPI reference. GitHub Pages remains a static deployment option. Successful publication and independent persistence checks determine public status; provider configuration alone is not deployment evidence. Both runtimes serve versioned evaluated artifacts, not a substitute scoring formula.

## 09. Keep licenses attached to their actual material

**Date: 2026-09-27.** Original application code deliberately uses MIT. Goodbooks and derived data retain CC BY-SA 4.0 and attribution. Open Bandit retains its source notice and the dataset paper's licensing statement. MIT does not override these rights. Typographic book covers avoid importing remote cover images. This choice differs from a blanket Apache preference in the brief and is explicit.

## 10. Make evidence and quality gates executable

**Date: 2026-09-27.** Reports and cards render from a canonical manifest with complete-file drift checks. Four executed notebooks retain failed assumptions and publication checks. CI applies strict typing, lint, branch coverage, authored-text gates, semantic text contrast, API persistence, edge parity and browser flows. The coverage floor applies to Python application packages; browser and Worker behavior have separate tests. Missing evidence fails a gate rather than becoming a zero or a success.

These are demonstration recommendations on a public dataset; no real reader's identity is present and no recommendation is personalized to a real person.
