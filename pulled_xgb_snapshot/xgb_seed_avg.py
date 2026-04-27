"""XGBoost multi-seed averaging."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import feat_pipeline as fp


def run_seed(X_train, y, X_test, seed):
    pos_w = (1 - y.mean()) / y.mean()
    params = dict(
        objective="binary:logistic",
        eval_metric=["aucpr", "auc"],
        eta=0.04,
        max_depth=8,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=pos_w,
        tree_method="hist",
        seed=seed,
        nthread=-1,
        verbosity=0,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    proba_test = np.zeros(len(X_test))
    iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        dtr = xgb.DMatrix(X_train.iloc[tr_idx], label=y.iloc[tr_idx])
        dva = xgb.DMatrix(X_train.iloc[va_idx], label=y.iloc[va_idx])
        bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "va")],
                        early_stopping_rounds=120, verbose_eval=0)
        iters.append(bst.best_iteration)
        oof[va_idx] = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        proba_test += bst.predict(xgb.DMatrix(X_test), iteration_range=(0, bst.best_iteration + 1)) / 5.0
        print(f"    seed{seed} fold{fold}: iter={bst.best_iteration} auc={roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f}")
    return oof, proba_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="xgb_seedavg_submission.csv")
    ap.add_argument("--target-rate", type=float, default=0.18)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 1337])
    args = ap.parse_args()

    t0 = time.perf_counter()
    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    X_train_raw, X_test_raw = fp.make_features(train, test)
    X_train, X_test, _ = fp.encode_for_lgb(X_train_raw, X_test_raw)
    y = train[fp.TARGET].astype(int)
    print(f"X_train={X_train.shape}")

    oof_avg = np.zeros(len(y))
    proba_avg = np.zeros(len(test))
    for s in args.seeds:
        print(f"\n[seed {s}]")
        oof, proba = run_seed(X_train, y, X_test, s)
        oof_avg += oof / len(args.seeds)
        proba_avg += proba / len(args.seeds)
        print(f"  seed{s} AUC={roc_auc_score(y, oof):.5f} AP={average_precision_score(y, oof):.5f}")

    print(f"\n[multiseed] OOF AUC={roc_auc_score(y, oof_avg):.5f}  AP={average_precision_score(y, oof_avg):.5f}")
    for tr in [0.13, 0.15, 0.175, 0.18, 0.20]:
        thr = fp.threshold_for_target_rate(oof_avg, tr)
        pred = (oof_avg >= thr).astype(int)
        print(f"  target={tr:.3f} thr={thr:.4f} f1={f1_score(y, pred):.4f}")

    thr = fp.threshold_for_target_rate(proba_avg, args.target_rate)
    preds = (proba_avg >= thr).astype(int)
    fp.write_submission(test[fp.ID_COL], preds, Path(args.out))
    np.save(Path(args.out).with_suffix(".proba.npy"), proba_avg)
    np.save(Path(args.out).with_suffix(".oof.npy"), oof_avg)
    print(f"[done] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
