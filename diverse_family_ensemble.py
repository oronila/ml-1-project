#!/usr/bin/env python3
"""Build a diverse weighted ensemble from saved base-model probabilities.

This intentionally avoids using near-duplicate variants from the same model
family. The selected models all have reasonable OOF AP and at least several
percentage points of binary disagreement with the CatBoost seed-average model.
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
    "cat_seed": "cat_seedavg_submission",
    "xgb_seed": "xgb_seedavg_submission",
    "hgb": "hgb_target_rate_submission",
    "lgbm_v2": "lgbm_v2_submission",
    "ada": "adaboost_v2_submission",
    "rf": "rf_target_rate_submission",
    "et": "et_target_rate_submission",
}

WEIGHTS = {
    "cat_seed": 0.32,
    "xgb_seed": 0.23,
    "hgb": 0.14,
    "lgbm_v2": 0.12,
    "ada": 0.08,
    "rf": 0.06,
    "et": 0.05,
}


def threshold_for_target_rate(proba: np.ndarray, target_rate: float) -> float:
    if target_rate <= 0:
        return 1.0
    if target_rate >= 1:
        return 0.0
    k = int(round(target_rate * len(proba)))
    k = max(1, min(len(proba) - 1, k))
    return float(np.sort(proba)[::-1][k - 1])


def load_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    oofs = {}
    tests = {}
    for name, base in MODEL_PATHS.items():
        oof_path = Path(base + ".oof.npy")
        test_path = Path(base + ".proba.npy")
        if not oof_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Missing saved probabilities for {name}: {base}")
        oofs[name] = np.load(oof_path)
        tests[name] = np.load(test_path)
    return oofs, tests


def weighted_average(arrays: dict[str, np.ndarray]) -> np.ndarray:
    total_weight = sum(WEIGHTS.values())
    return sum((WEIGHTS[name] / total_weight) * arrays[name] for name in WEIGHTS)


def run(args: argparse.Namespace) -> None:
    train = pd.read_csv(args.train, usecols=[TARGET])
    test_ids = pd.read_csv(args.test, usecols=[ID_COL])[ID_COL]
    y = train[TARGET].astype(int).to_numpy()
    oofs, tests = load_arrays()

    print("[selected]")
    for name in WEIGHTS:
        print(
            f"{name:8s} weight={WEIGHTS[name]:.2f} "
            f"AP={average_precision_score(y, oofs[name]):.5f} "
            f"AUC={roc_auc_score(y, oofs[name]):.5f}"
        )

    oof = weighted_average(oofs)
    test_proba = weighted_average(tests)
    print(f"[blend] OOF AP={average_precision_score(y, oof):.6f}")
    print(f"[blend] OOF AUC={roc_auc_score(y, oof):.6f}")

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
    parser.add_argument("--out", type=Path, default=Path("diverse_family_ensemble_submission.csv"))
    parser.add_argument("--target-rate", type=float, default=0.18)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
