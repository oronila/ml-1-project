"""Stacking: logistic regression meta-learner over base model OOF predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.special import logit, expit

import feat_pipeline as fp

EPS = 1e-7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rate", type=float, default=0.175)
    ap.add_argument("--out", default="stack_submission.csv")
    args = ap.parse_args()

    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    y = train[fp.TARGET].astype(int).values

    base_paths = {
        "lgbm": "lgbm_target_rate_submission",
        "lgbm_v2": "lgbm_v2_submission",
        "lgbm_int": "lgbm_interactions_submission",
        "xgb": "xgb_target_rate_submission",
        "rf": "rf_target_rate_submission",
        "et": "et_target_rate_submission",
        "hgb": "hgb_target_rate_submission",
        "cat": "cat_target_rate_submission",
        "cat_seed": "cat_seedavg_submission",
    }
    base_paths = {k: v for k, v in base_paths.items()
                  if Path(v + ".oof.npy").exists() and Path(v + ".proba.npy").exists()}

    oof_mat, proba_mat, names = [], [], []
    for name, base in base_paths.items():
        op, pp = Path(base + ".oof.npy"), Path(base + ".proba.npy")
        if not op.exists() or not pp.exists():
            continue
        names.append(name)
        oof_mat.append(np.load(op))
        proba_mat.append(np.load(pp))
    oof_mat = np.array(oof_mat).T  # (n_train, n_models)
    proba_mat = np.array(proba_mat).T

    # logit transform + clamp to avoid inf
    def to_logit(p):
        p = np.clip(p, EPS, 1 - EPS)
        return logit(p)

    Z_oof = to_logit(oof_mat)
    Z_test = to_logit(proba_mat)

    print("Base model OOF AUC:")
    for i, n in enumerate(names):
        print(f"  {n:15s} AUC={roc_auc_score(y, oof_mat[:, i]):.5f}  AP={average_precision_score(y, oof_mat[:, i]):.5f}")

    # CV the meta-learner
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(Z_oof, y):
        meta = LogisticRegression(C=1.0, max_iter=200, n_jobs=-1)
        meta.fit(Z_oof[tr_idx], y[tr_idx])
        meta_oof[va_idx] = meta.predict_proba(Z_oof[va_idx])[:, 1]
    print(f"\nStacked OOF AUC = {roc_auc_score(y, meta_oof):.5f}")
    print(f"Stacked OOF AP  = {average_precision_score(y, meta_oof):.5f}")
    for tr in [0.06, 0.13, 0.15, 0.175, 0.20]:
        thr = fp.threshold_for_target_rate(meta_oof, tr)
        pred = (meta_oof >= thr).astype(int)
        print(f"  target={tr:.3f} thr={thr:.4f} f1={f1_score(y, pred):.4f}")

    # final fit on all OOFs
    meta = LogisticRegression(C=1.0, max_iter=200, n_jobs=-1)
    meta.fit(Z_oof, y)
    print("\nFinal meta coefs:")
    for n, c in zip(names, meta.coef_[0]):
        print(f"  {n:15s} = {c:+.4f}")
    print(f"  intercept = {meta.intercept_[0]:+.4f}")

    proba_test = meta.predict_proba(Z_test)[:, 1]
    thr = fp.threshold_for_target_rate(proba_test, args.target_rate)
    preds = (proba_test >= thr).astype(int)
    print(f"[submit] thr={thr:.4f}")
    fp.write_submission(test[fp.ID_COL], preds, Path(args.out))
    np.save(Path(args.out).with_suffix(".proba.npy"), proba_test)
    np.save(Path(args.out).with_suffix(".oof.npy"), meta_oof)


if __name__ == "__main__":
    main()
