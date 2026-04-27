#!/usr/bin/env python3
"""Simple human-readable ensemble of four tree models.

This deliberately avoids stacking, calibration, seed averaging, and complex
meta-models. It averages saved probabilities from four ordinary classifiers:

- HistGradientBoosting
- AdaBoost
- Random Forest
- ExtraTrees

The weights are simple judgment weights: the two stronger boosting models get
more influence, while Random Forest and ExtraTrees add diversity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"

MODEL_PATHS = {
    "hgb": "hgb_target_rate_submission",
    "ada": "adaboost_v2_submission",
    "rf": "rf_target_rate_submission",
    "et": "et_target_rate_submission",
}

WEIGHTS = {
    "hgb": 0.35,
    "ada": 0.30,
    "rf": 0.20,
    "et": 0.15,
}


def threshold_for_target_rate(proba: np.ndarray, target_rate: float) -> float:
    k = int(round(target_rate * len(proba)))
    k = max(1, min(len(proba) - 1, k))
    return float(np.sort(proba)[::-1][k - 1])


def load_probability_files(suffix: str) -> dict[str, np.ndarray]:
    arrays = {}
    for name, base in MODEL_PATHS.items():
        path = Path(base + suffix)
        if not path.exists():
            raise FileNotFoundError(path)
        arrays[name] = np.load(path)
    return arrays


def weighted_average(arrays: dict[str, np.ndarray]) -> np.ndarray:
    total = sum(WEIGHTS.values())
    return sum((WEIGHTS[name] / total) * arrays[name] for name in WEIGHTS)


def run(args: argparse.Namespace) -> None:
    y = pd.read_csv(args.train, usecols=[TARGET])[TARGET].astype(int).to_numpy()
    test_ids = pd.read_csv(args.test, usecols=[ID_COL])[ID_COL]

    oofs = load_probability_files(".oof.npy")
    tests = load_probability_files(".proba.npy")

    print("[base models]")
    for name in WEIGHTS:
        print(
            f"{name:4s} weight={WEIGHTS[name]:.2f} "
            f"OOF AP={average_precision_score(y, oofs[name]):.5f} "
            f"OOF AUC={roc_auc_score(y, oofs[name]):.5f}"
        )

    oof = weighted_average(oofs)
    test_proba = weighted_average(tests)
    print(f"[ensemble] OOF AP={average_precision_score(y, oof):.6f}")
    print(f"[ensemble] OOF AUC={roc_auc_score(y, oof):.6f}")

    for rate in [0.175, 0.18, 0.185, 0.19, 0.20]:
        threshold = threshold_for_target_rate(oof, rate)
        pred = (oof >= threshold).astype(int)
        print(f"[oof] target_rate={rate:.3f} threshold={threshold:.6f} f1={f1_score(y, pred):.6f}")

    threshold = threshold_for_target_rate(test_proba, args.target_rate)
    pred = (test_proba >= threshold).astype(int)
    submission = pd.DataFrame({ID_COL: test_ids.values, TARGET: pred})
    submission.to_csv(args.out, index=False)
    np.save(args.out.with_suffix(".oof.npy"), oof)
    np.save(args.out.with_suffix(".proba.npy"), test_proba)
    print(
        f"[submit] wrote {args.out} rows={len(submission)} "
        f"positives={int(pred.sum())} pos_rate={pred.mean():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--out", type=Path, default=Path("simple_human_ensemble_submission.csv"))
    parser.add_argument("--target-rate", type=float, default=0.18)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
