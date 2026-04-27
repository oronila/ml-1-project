#!/usr/bin/env python3
"""Simple ensemble plus two existing human-readable binary submissions.

This is intentionally simple: average four saved model probability files, then
add small voting weights from cat_submission.csv and human_submission.csv.
The final threshold is chosen to keep the target positive rate near 18%.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


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
    "hgb": 0.30,
    "ada": 0.25,
    "rf": 0.15,
    "et": 0.10,
    "cat": 0.10,
    "human": 0.10,
}


def threshold_for_target_rate(scores: np.ndarray, target_rate: float) -> float:
    k = int(round(target_rate * len(scores)))
    k = max(1, min(len(scores) - 1, k))
    return float(np.sort(scores)[::-1][k - 1])


def read_binary_vote(path: Path) -> pd.DataFrame:
    vote = pd.read_csv(path, usecols=[ID_COL, TARGET])
    vote[TARGET] = vote[TARGET].astype(float)
    return vote.sort_values(ID_COL).reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    train_y = pd.read_csv(args.train, usecols=[TARGET])[TARGET].astype(int).to_numpy()
    test_ids = pd.read_csv(args.test, usecols=[ID_COL]).sort_values(ID_COL).reset_index(drop=True)
    total_weight = sum(WEIGHTS.values())

    test_score = np.zeros(len(test_ids), dtype=float)
    oof_score = np.zeros(len(train_y), dtype=float)
    oof_weight = 0.0

    print("[probability models]")
    for name, base in PROBA_MODELS.items():
        weight = WEIGHTS[name] / total_weight
        oof = np.load(base + ".oof.npy")
        proba = np.load(base + ".proba.npy")
        test_score += weight * proba
        oof_score += weight * oof
        oof_weight += weight
        print(
            f"{name:5s} weight={WEIGHTS[name]:.2f} "
            f"OOF AP={average_precision_score(train_y, oof):.5f} "
            f"OOF AUC={roc_auc_score(train_y, oof):.5f}"
        )

    # OOF diagnostics only cover the probability models because the existing
    # binary submissions do not have matching OOF predictions.
    oof_score = oof_score / oof_weight
    print(f"[probability-only OOF] AP={average_precision_score(train_y, oof_score):.6f}")
    print(f"[probability-only OOF] AUC={roc_auc_score(train_y, oof_score):.6f}")
    for rate in [0.175, 0.18, 0.185, 0.19, 0.20]:
        threshold = threshold_for_target_rate(oof_score, rate)
        pred = (oof_score >= threshold).astype(int)
        print(f"[probability-only OOF] target_rate={rate:.3f} f1={f1_score(train_y, pred):.6f}")

    print("[binary votes]")
    for name, path in BINARY_VOTES.items():
        vote = read_binary_vote(Path(path))
        if not vote[ID_COL].equals(test_ids[ID_COL]):
            raise ValueError(f"{path} INDEX_NR ordering does not match test.csv")
        weight = WEIGHTS[name] / total_weight
        test_score += weight * vote[TARGET].to_numpy()
        print(f"{name:5s} weight={WEIGHTS[name]:.2f} pos_rate={vote[TARGET].mean():.6f}")

    threshold = threshold_for_target_rate(test_score, args.target_rate)
    pred = (test_score >= threshold).astype(int)
    submission = pd.DataFrame({ID_COL: test_ids[ID_COL].values, TARGET: pred})
    submission.to_csv(args.out, index=False)
    np.save(args.out.with_suffix(".proba.npy"), test_score)
    print(
        f"[submit] wrote {args.out} rows={len(submission)} "
        f"positives={int(pred.sum())} pos_rate={pred.mean():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--out", type=Path, default=Path("simple_human_ensemble_plus_submission.csv"))
    parser.add_argument("--target-rate", type=float, default=0.18)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
