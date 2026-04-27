"""LightGBM with engineered interaction features (no text).

Domain logic: damage probability scales with kinetic energy (mass × speed^2),
phase of flight (climb/descent are faster + complex maneuvers), bird size,
TYPE_ENG (jet vs prop). Build cross-features that capture these.
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


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered interaction features in-place."""
    out = df.copy()

    # combined risk strings (low-card)
    if "SIZE" in out.columns and "PHASE_OF_FLIGHT" in out.columns:
        out["SIZE_X_PHASE"] = out["SIZE"].astype(str) + "_" + out["PHASE_OF_FLIGHT"].astype(str)
    if "SIZE" in out.columns and "TYPE_ENG" in out.columns:
        out["SIZE_X_ENG"] = out["SIZE"].astype(str) + "_" + out["TYPE_ENG"].astype(str)
    if "PHASE_OF_FLIGHT" in out.columns and "TYPE_ENG" in out.columns:
        out["PHASE_X_ENG"] = out["PHASE_OF_FLIGHT"].astype(str) + "_" + out["TYPE_ENG"].astype(str)
    if "SIZE" in out.columns and "AC_CLASS" in out.columns:
        out["SIZE_X_CLASS"] = out["SIZE"].astype(str) + "_" + out["AC_CLASS"].astype(str)

    # ratios + numeric interactions
    if "HEIGHT" in out.columns and "SPEED" in out.columns:
        h = pd.to_numeric(out["HEIGHT"], errors="coerce").fillna(0).clip(lower=0)
        s = pd.to_numeric(out["SPEED"], errors="coerce").fillna(0).clip(lower=0)
        out["HEIGHT_X_SPEED"] = (h * s).astype("float32")
        out["LOG_HEIGHT_X_SPEED"] = np.log1p(h * s).astype("float32")
    if "AC_MASS" in out.columns and "NUM_STRUCK" in out.columns:
        m = pd.to_numeric(out["AC_MASS"], errors="coerce").fillna(0).clip(lower=0)
        ns = pd.to_numeric(out["NUM_STRUCK"], errors="coerce").fillna(0).clip(lower=0)
        out["MASS_X_STRUCK"] = (m * ns).astype("float32")

    # missing-pattern bitmap (low-info records vs high-info records)
    miss_cols = ["SPEED", "HEIGHT", "DISTANCE", "AC_MASS", "PHASE_OF_FLIGHT", "SKY", "AMA"]
    miss_cols = [c for c in miss_cols if c in out.columns]
    n_miss = pd.DataFrame({c: pd.to_numeric(out[c], errors="coerce").isna() if not isinstance(out[c].dtype, pd.CategoricalDtype) else (out[c].astype(str) == "Unknown") for c in miss_cols})
    out["NUM_MISSING_KEY"] = n_miss.sum(axis=1).astype("float32")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lgbm_interactions_submission.csv")
    ap.add_argument("--target-rate", type=float, default=0.175)
    args = ap.parse_args()

    t0 = time.perf_counter()
    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    X_train_raw, X_test_raw = fp.make_features(train, test)
    X_train_raw = add_interactions(X_train_raw)
    X_test_raw = add_interactions(X_test_raw)

    # extend cat lists for the new string features
    extra_cats = [c for c in ["SIZE_X_PHASE", "SIZE_X_ENG", "PHASE_X_ENG", "SIZE_X_CLASS"] if c in X_train_raw.columns]
    saved_cats = list(fp.CATEGORICAL_LOWCARD)
    fp.CATEGORICAL_LOWCARD = saved_cats + extra_cats
    try:
        X_train, X_test, cat_cols = fp.encode_for_lgb(X_train_raw, X_test_raw)
    finally:
        fp.CATEGORICAL_LOWCARD = saved_cats

    y = train[fp.TARGET].astype(int)
    print(f"X_train={X_train.shape} cat_cols={len(cat_cols)} interactions={extra_cats}")

    pos_w = (1 - y.mean()) / y.mean()
    params = dict(
        objective="binary",
        metric="average_precision",
        learning_rate=0.035,
        num_leaves=159,
        min_data_in_leaf=60,
        feature_fraction=0.8,
        bagging_fraction=0.85,
        bagging_freq=5,
        lambda_l2=1.0,
        scale_pos_weight=pos_w,
        verbose=-1,
        seed=2024,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
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
    for tr in [0.06, 0.13, 0.15, 0.175, 0.20]:
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
