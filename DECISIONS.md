# Decisions and scope

These are demonstration recommendations on a public dataset; no real reader's
identity is present and no recommendation is personalized to a real person.

## A completed demonstration within the supplied brief

The supplied Day 10 brief describes a larger research and infrastructure program.
This repository implements a bounded, inspectable platform. It does not claim
every instruction embedded in that document was fulfilled. The repository
history records meaningful changes; no artificial commit-count quota is used.

| Area | Delivered design | Limit or deferred work |
| --- | --- | --- |
| Recommendation models | Popularity, item cosine, implicit ALS, score blend | No neural two-tower or LambdaMART claim |
| Evaluation population | Documented bounded catalog and source-order split | No full-dataset or dated global temporal claim |
| Ranking protocols | Full evaluated-catalog ranking with comparison diagnostics | Full means the evaluated catalog, not all books in the source |
| Off-policy estimation | Repository estimators, simulation and recorded reference checks | Scope and exact measured coverage come from the manifest |
| Bookstore | Static Next.js deployment and typographic covers | Browser state is not a server-side experiment log |
| Durable interactions | Independently runnable relational API | Public API and managed PostgreSQL hosting unprovisioned |
| Promotion | Explicit evidence eligibility checks | Eligibility is separate from policy activation |
| Retrieval infrastructure | Local exact scoring | pgvector and approximate-index benchmarking deferred |
| Documents | Manifest-driven Jinja rendering and whole-file drift checks | Only supported, recorded claims are generated |

## Source-order proxy

The Goodbooks maintainer describes ratings as time-sorted but supplies no event
timestamps. Fabricating dates would conceal that constraint. We preserve row
order and label the split as a proxy. Metadata collected over the full dataset
is not evidence that a feature was available at a particular historical date.

## Bounded compute

A bounded catalog allows repeatable CPU evaluation and comparison of full
candidate ranking with sampled-negative diagnostics. Selection rules and cohort
counts belong in the manifest. Conclusions apply to that selection and do not
establish the same ordering of models on all source interactions.

## Two execution modes

GitHub Pages provides a public, inspectable application without provider secrets.
The FastAPI service makes persistence and request boundaries independently
testable. No browser bundle may contain a database connection string or a
server write credential. Public API status must change only after an actual
deployment and end-to-end verification.

## Source and output licenses

Original code uses the MIT license. Goodbooks data and derived data retain
CC BY-SA 4.0 and attribution. Open Bandit reference code and its repository
sample retain upstream notices. Third-party rights are not replaced by the
application's top-level license. Book-cover URLs are not used as image assets.

## Generated evidence

Templates are source code; rendered reports are reviewable artifacts. Rebuild
the artifacts after changing evidence, then use `--check` to compare exact
content. A failed or empty evidence run cannot be repaired by editing a number
in a generated file.
