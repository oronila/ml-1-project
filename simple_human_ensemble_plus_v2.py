#!/usr/bin/env python3
"""Simple ensemble with larger weights for cat/human binary votes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"

PROBA_MODELS = {
    "hgb": "hgb_target_rate_submission",
    "ada": "adaboost_v2_submission",
    "rf": "rf_target_rate_submission",
    "et": "et_target_rate_submission",
}

BINARY_VOTES = {
    "cat": "cat_submission.csv",
    "human": "human_submission.csv",
}

WEIGHTS = {
    "hgb": 0.25,
    "ada": 0.20,
    "rf": 0.12,
    "et": 0.08,
    "cat": 0.20,
    "human": 0.15,
}


def threshold_for_target_rate(scores: np.ndarray, target_rate: float) -> float:
    k = int(round(target_rate * len(scores)))
    k = max(1, min(len(scores) - 1, k))
    return float(np.sort(scores)[::-1][k - 1])


def run(args: argparse.Namespace) -> None:
    test_ids = pd.read_csv(args.test, usecols=[ID_COL]).sort_values(ID_COL).reset_index(drop=True)
    scores = np.zeros(len(test_ids), dtype=float)
    total_weight = sum(WEIGHTS.values())

    for name, base in PROBA_MODELS.items():
        scores += (WEIGHTS[name] / total_weight) * np.load(base + ".proba.npy")

    for name, path in BINARY_VOTES.items():
        vote = pd.read_csv(path, usecols=[ID_COL, TARGET]).sort_values(ID_COL).reset_index(drop=True)
        if not vote[ID_COL].equals(test_ids[ID_COL]):
            raise ValueError(f"{path} INDEX_NR ordering does not match test.csv")
        scores += (WEIGHTS[name] / total_weight) * vote[TARGET].astype(float).to_numpy()

    threshold = threshold_for_target_rate(scores, args.target_rate)
    pred = (scores >= threshold).astype(int)
    submission = pd.DataFrame({ID_COL: test_ids[ID_COL].values, TARGET: pred})
    submission.to_csv(args.out, index=False)
    np.save(args.out.with_suffix(".proba.npy"), scores)
    print(
        f"[submit] wrote {args.out} rows={len(submission)} "
        f"positives={int(pred.sum())} pos_rate={pred.mean():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--out", type=Path, default=Path("simple_human_ensemble_plus_v2_submission.csv"))
    parser.add_argument("--target-rate", type=float, default=0.18)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
