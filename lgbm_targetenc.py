"""LightGBM with K-fold target encoding for high-cardinality cats.

Target encoding gives high-card cats ordered values (mean target by group),
which trees can split better. Uses K-fold to prevent leakage.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import feat_pipeline as fp


def kfold_target_encode(train_X, train_y, test_X, col, n_splits=5, smoothing=20.0, seed=42):
    """Returns (te_train, te_test) — leak-free target encoded values."""
    global_mean = train_y.mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    te_train = np.full(len(train_X), global_mean, dtype="float32")
    for tr_idx, va_idx in skf.split(train_X, train_y):
        df = pd.DataFrame({"c": train_X[col].iloc[tr_idx].values, "y": train_y.iloc[tr_idx].values})
        agg = df.groupby("c")["y"].agg(["sum", "count"])
        # smoothed: (sum + smoothing*global_mean) / (count + smoothing)
        smoothed = (agg["sum"] + smoothing * global_mean) / (agg["count"] + smoothing)
        te_train[va_idx] = train_X[col].iloc[va_idx].map(smoothed).fillna(global_mean).astype("float32").values

    # full-train encoding for test
    df_all = pd.DataFrame({"c": train_X[col].values, "y": train_y.values})
    agg_all = df_all.groupby("c")["y"].agg(["sum", "count"])
    smoothed_all = (agg_all["sum"] + smoothing * global_mean) / (agg_all["count"] + smoothing)
    te_test = test_X[col].map(smoothed_all).fillna(global_mean).astype("float32").values
    return te_train, te_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lgbm_targetenc_submission.csv")
    ap.add_argument("--target-rate", type=float, default=0.18)
    args = ap.parse_args()

    t0 = time.perf_counter()
    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    X_train_raw, X_test_raw = fp.make_features(train, test)
    y = train[fp.TARGET].astype(int)

    # target encode high-card cats (do BEFORE encode_for_lgb)
    te_cols = ["AIRPORT_ID", "OPID", "SPECIES_ID", "RUNWAY", "AMA", "AMO"]
    for c in te_cols:
        if c in X_train_raw.columns:
            te_tr, te_te = kfold_target_encode(X_train_raw, y, X_test_raw, c, smoothing=20.0)
            X_train_raw["TE_" + c] = te_tr
            X_test_raw["TE_" + c] = te_te
            print(f"  TE_{c}: train_mean={te_tr.mean():.4f} test_mean={te_te.mean():.4f}")

    X_train, X_test, cat_cols = fp.encode_for_lgb(X_train_raw, X_test_raw)
    print(f"X_train={X_train.shape} cats={len(cat_cols)}")

    pos_w = (1 - y.mean()) / y.mean()
    params = dict(
        objective="binary",
        metric="average_precision",
        learning_rate=0.035,
        num_leaves=127,
        min_data_in_leaf=80,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        lambda_l2=1.0,
        scale_pos_weight=pos_w,
        verbose=-1,
        seed=2026,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof = np.zeros(len(y))
    iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        ds_t = lgb.Dataset(X_train.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=cat_cols)
        ds_v = lgb.Dataset(X_train.iloc[va_idx], y.iloc[va_idx], categorical_feature=cat_cols, reference=ds_t)
        b = lgb.train(params, ds_t, num_boost_round=3000, valid_sets=[ds_v],
                      callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)])
        iters.append(b.best_iteration)
        oof[va_idx] = b.predict(X_train.iloc[va_idx], num_iteration=b.best_iteration)
        print(f"  fold{fold}: iter={b.best_iteration} auc={roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f}")

    print(f"OOF AUC={roc_auc_score(y, oof):.5f} AP={average_precision_score(y, oof):.5f}")
    for tr in [0.06, 0.13, 0.15, 0.175, 0.18, 0.20]:
        thr = fp.threshold_for_target_rate(oof, tr)
        pred = (oof >= thr).astype(int)
        print(f"  target={tr:.3f} thr={thr:.4f} f1={f1_score(y, pred):.4f}")

    fi = int(np.mean(iters) * 1.05)
    print(f"[final] iter={fi}")
    ds_all = lgb.Dataset(X_train, y, categorical_feature=cat_cols)
    bst = lgb.train(params, ds_all, num_boost_round=fi)
    proba_test = bst.predict(X_test)
    thr = fp.threshold_for_target_rate(proba_test, args.target_rate)
    preds = (proba_test >= thr).astype(int)
    fp.write_submission(test[fp.ID_COL], preds, Path(args.out))
    np.save(Path(args.out).with_suffix(".proba.npy"), proba_test)
    np.save(Path(args.out).with_suffix(".oof.npy"), oof)
    print(f"[done] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
