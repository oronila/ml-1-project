"""XGBoost with scale_pos_weight + target-rate threshold. Tree-based."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import feat_pipeline as fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--out", default="xgb_target_rate_submission.csv")
    ap.add_argument("--target-rate", type=float, default=0.175)
    args = ap.parse_args()

    t0 = time.perf_counter()
    train = pd.read_csv(args.train, low_memory=False)
    test = pd.read_csv(args.test, low_memory=False)

    X_train_raw, X_test_raw = fp.make_features(train, test)
    # XGBoost handles categorical natively in newer versions, but let's keep it simple
    # by using integer-coded categoricals as numeric (since they're tree-based, ordering doesn't matter much)
    X_train, X_test, cat_cols = fp.encode_for_lgb(X_train_raw, X_test_raw)
    y = train[fp.TARGET].astype(int)
    print(f"X_train={X_train.shape} pos_rate={y.mean():.4f}")

    pos_weight = (1 - y.mean()) / y.mean()

    params = dict(
        objective="binary:logistic",
        eval_metric=["aucpr", "auc"],
        eta=0.05,
        max_depth=8,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.0,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        tree_method="hist",
        seed=42,
        nthread=-1,
        verbosity=0,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        Xt, Xv = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        yt, yv = y.iloc[tr_idx], y.iloc[va_idx]
        dtr = xgb.DMatrix(Xt, label=yt)
        dva = xgb.DMatrix(Xv, label=yv)
        bst = xgb.train(
            params,
            dtr,
            num_boost_round=2500,
            evals=[(dva, "va")],
            early_stopping_rounds=80,
            verbose_eval=0,
        )
        best_iters.append(bst.best_iteration)
        oof[va_idx] = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        print(f"  fold{fold}: best_iter={bst.best_iteration}  val_auc={roc_auc_score(yv, oof[va_idx]):.5f}")

    print(f"OOF ROC-AUC={roc_auc_score(y, oof):.5f}  PR-AUC={average_precision_score(y, oof):.5f}")
    for tr in [0.06, 0.10, 0.13, 0.15, 0.175, 0.20]:
        thr = fp.threshold_for_target_rate(oof, tr)
        pred = (oof >= thr).astype(int)
        print(f"  target={tr:.3f} thr={thr:.4f} f1={f1_score(y, pred):.4f}")

    final_iters = int(np.mean(best_iters) * 1.05)
    print(f"[final] num_round={final_iters}")
    dtr = xgb.DMatrix(X_train, label=y)
    dte = xgb.DMatrix(X_test)
    bst = xgb.train(params, dtr, num_boost_round=final_iters)
    proba_test = bst.predict(dte)

    thr = fp.threshold_for_target_rate(proba_test, args.target_rate)
    print(f"[submit] target_rate={args.target_rate} thr={thr:.4f}")
    preds = (proba_test >= thr).astype(int)
    fp.write_submission(test[fp.ID_COL], preds, Path(args.out))

    np.save(Path(args.out).with_suffix(".proba.npy"), proba_test)
    np.save(Path(args.out).with_suffix(".oof.npy"), oof)
    print(f"[done] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
