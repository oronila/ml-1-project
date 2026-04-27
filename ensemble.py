"""Rank-average ensemble across LGBM, XGB, RF, ET, CatBoost.

Pick test threshold by target positive rate.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import feat_pipeline as fp


def to_rank(p: np.ndarray) -> np.ndarray:
    """Convert probabilities to ranks normalized to [0,1]."""
    order = p.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(p))
    return ranks / (len(p) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rate", type=float, default=0.175)
    ap.add_argument("--out", default="ensemble_submission.csv")
    args = ap.parse_args()

    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    y = train[fp.TARGET].astype(int).values

    models = {
        "lgbm": "lgbm_target_rate_submission",
        "xgb": "xgb_target_rate_submission",
        "rf": "rf_target_rate_submission",
        "et": "et_target_rate_submission",
        "cat": "cat_target_rate_submission",
    }

    oofs, probas, weights = {}, {}, {}
    for name, base in models.items():
        oof_path = Path(base + ".oof.npy")
        proba_path = Path(base + ".proba.npy")
        if not oof_path.exists() or not proba_path.exists():
            print(f"  skip {name} (missing files)")
            continue
        oofs[name] = np.load(oof_path)
        probas[name] = np.load(proba_path)
        # weight by OOF ROC-AUC squared (so better models dominate)
        auc = roc_auc_score(y, oofs[name])
        ap_ = average_precision_score(y, oofs[name])
        weights[name] = (auc - 0.5) ** 2  # roughly squared lift over random
        print(f"  {name}: AUC={auc:.5f} AP={ap_:.5f} weight={weights[name]:.5f}")

    # Rank-average ensemble
    ens_oof = np.zeros(len(y))
    ens_proba = np.zeros(len(test))
    total_w = sum(weights.values())
    for name in oofs:
        ens_oof += to_rank(oofs[name]) * weights[name] / total_w
        ens_proba += to_rank(probas[name]) * weights[name] / total_w

    print(f"\nEnsemble OOF ROC-AUC = {roc_auc_score(y, ens_oof):.5f}")
    print(f"Ensemble OOF PR-AUC  = {average_precision_score(y, ens_oof):.5f}")
    for tr in [0.06, 0.10, 0.13, 0.15, 0.175, 0.20]:
        thr = fp.threshold_for_target_rate(ens_oof, tr)
        pred = (ens_oof >= thr).astype(int)
        print(f"  target={tr:.3f} thr={thr:.4f} f1={f1_score(y, pred):.4f}")

    # Also unweighted (simple) rank average
    simple = np.zeros(len(y))
    simple_test = np.zeros(len(test))
    for name in oofs:
        simple += to_rank(oofs[name]) / len(oofs)
        simple_test += to_rank(probas[name]) / len(oofs)
    print(f"\nSimple rank-avg OOF ROC-AUC = {roc_auc_score(y, simple):.5f}")
    print(f"Simple rank-avg OOF PR-AUC  = {average_precision_score(y, simple):.5f}")
    for tr in [0.06, 0.10, 0.13, 0.15, 0.175, 0.20]:
        thr = fp.threshold_for_target_rate(simple, tr)
        pred = (simple >= thr).astype(int)
        print(f"  target={tr:.3f} thr={thr:.4f} f1={f1_score(y, pred):.4f}")

    # Pick best of weighted vs simple based on OOF ROC-AUC
    if roc_auc_score(y, simple) > roc_auc_score(y, ens_oof):
        ens_oof = simple
        ens_proba = simple_test
        print("[choose] using simple rank-avg")
    else:
        print("[choose] using weighted rank-avg")

    thr = fp.threshold_for_target_rate(ens_proba, args.target_rate)
    preds = (ens_proba >= thr).astype(int)
    print(f"[submit] thr={thr:.4f}")
    fp.write_submission(test[fp.ID_COL], preds, Path(args.out))
    np.save(Path(args.out).with_suffix(".proba.npy"), ens_proba)
    np.save(Path(args.out).with_suffix(".oof.npy"), ens_oof)


if __name__ == "__main__":
    main()
