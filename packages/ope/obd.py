"""Real Open Bandit sample check with a fully specified, held-out target policy.

This is not an attempt to reconstruct the contextual BTS policy from action
frequencies. Per-position propensities are never treated as joint slate weights.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from packages.evaluation.metrics import bootstrap
from packages.ope.estimators import estimate_intervals

COMMIT = "8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4"
BASE = f"https://raw.githubusercontent.com/st-tech/zr-obp/{COMMIT}/"


def run_obd(root: Path) -> dict[str, object]:
    cache = root / ".cache" / "obd"
    output = root / "data" / "obd"
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, source in {
        "random.csv": "obd/random/all/all.csv",
        "bts.csv": "obd/bts/all/all.csv",
        "LICENSE": "LICENSE",
    }.items():
        path = cache / name
        if not path.exists():
            request = urllib.request.Request(BASE + source, headers={"User-Agent": "stacks-public-data-demo"})
            with urllib.request.urlopen(request, timeout=90) as response:
                path.write_bytes(response.read())
        paths[name] = (path, BASE + source)
    random = pd.read_csv(paths["random.csv"][0]).sort_values("timestamp", kind="stable")
    bts = pd.read_csv(paths["bts.csv"][0]).sort_values("timestamp", kind="stable")
    required = ["timestamp", "item_id", "position", "click", "propensity_score"]
    for name, frame in (("random", random), ("bts", bts)):
        if (
            not frame.propensity_score.between(0, 1, inclusive="right").all()
            or not frame.click.isin([0, 1]).all()
        ):
            raise ValueError("OBD sample violates reward/propensity contract")
        frame[required].to_parquet(output / f"{name}-all.parquet", index=False)
    (output / "LICENSE").write_bytes(paths["LICENSE"][0].read_bytes())
    provenance = {
        "source": "https://github.com/st-tech/zr-obp",
        "source_commit": COMMIT,
        "license": "Apache-2.0",
        "citation": "Saito, Aihara, Matsutani, and Narita (2020), Open Bandit Dataset and Pipeline",
        "files": [
            {"name": name, "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, (path, url) in paths.items()
        ],
        "transformation": "Sorted by timestamp, selected timestamp/item_id/position/click/propensity_score; all 10000 rows retained per policy in campaign all.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (output / "PROVENANCE.md").write_text(
        "# Open Bandit sample\n\nSource: https://github.com/st-tech/zr-obp. "
        f"Pinned commit `{COMMIT}`. Apache-2.0; original license retained beside the data. "
        "Citation: Saito, Aihara, Matsutani, and Narita (2020), Open Bandit Dataset and Pipeline. "
        "The all campaign random and BTS samples each contain 10000 observations. Full source hashes and transformations are in provenance.json. "
        "Position is preserved as the source's 1-based slot. Propensities are per-position marginals, not joint-slate probabilities. "
        "Target policy is uniform or a fixed 20% exploration/80% best smoothed training CTR action policy per position. "
        "Target and reward model fit the first chronological half of random logs; estimates use only the second half. "
        "BTS observed reward is a descriptive reference from the second half of BTS logs, not truth for either explicit target policy.\n",
        encoding="utf-8",
    )
    actions = np.sort(random.item_id.unique())
    action_map = {int(action): index for index, action in enumerate(actions)}
    train, test = random.iloc[: len(random) // 2], random.iloc[len(random) // 2 :]
    bts_test = bts.iloc[len(bts) // 2 :]
    rows = []
    for position in sorted(random.position.unique()):
        fit, heldout = train[train.position == position], test[test.position == position]
        action_indices = heldout.item_id.map(action_map).to_numpy(dtype=int)
        prior = float(train.click.mean())
        summaries = fit.groupby("item_id").click.agg(["sum", "count"]).reindex(actions, fill_value=0)
        prediction = ((summaries["sum"] + 20 * prior) / (summaries["count"] + 20)).to_numpy()
        uniform = np.full(actions.size, 1 / actions.size)
        greedy = 0.2 * uniform
        greedy[int(np.argmax(prediction))] += 0.8
        reward_model = np.tile(prediction, (len(heldout), 1))
        reference = bts_test[bts_test.position == position]
        for name, probabilities in (("Uniform sanity check", uniform), ("Fixed training CTR policy", greedy)):
            target = np.tile(probabilities, (len(heldout), 1))
            estimates = estimate_intervals(
                heldout.click.to_numpy(dtype=float),
                action_indices,
                heldout.propensity_score.to_numpy(),
                target,
                reward_model,
                seed=731 + int(position),
                draws=1000,
            )
            weights = target[np.arange(len(heldout)), action_indices] / heldout.propensity_score.to_numpy()
            ess = float(weights.sum() ** 2 / np.square(weights).sum())
            for estimator, estimate in estimates.items():
                rows.append(
                    {
                        "campaign": "all",
                        "policy": name,
                        "position": int(position),
                        "sample_size": len(heldout),
                        "estimator": estimator,
                        "estimate": estimate,
                        "effective_sample_size": ess,
                        "observed_logging": bootstrap(heldout.click.to_numpy(dtype=float)),
                        "bts_reference": bootstrap(reference.click.to_numpy(dtype=float)),
                        "bts_reference_n": len(reference),
                    }
                )
    return {
        "status": "measured_alternative_target",
        "source_commit": COMMIT,
        "campaign": "all",
        "logging_rows": len(random),
        "training_rows": len(train),
        "evaluation_rows": len(test),
        "actions": len(actions),
        "target": "Uniform sanity check and fixed 20%-uniform/80%-training-CTR policy, separately per slot",
        "detail": "Real random-policy logs, held-out chronological second half. Target and smoothed action reward model fit only the first half. BTS observed reward is descriptive, not target-policy ground truth. Row bootstrap does not model repeated-reader dependence or model-fit uncertainty; action-only DM therefore has a conditional zero-width interval. The learned target has low effective sample size, so no improvement is established. Per-position propensities must not be multiplied into a claimed joint-slate propensity.",
        "rows": rows,
    }
