"""Extreme-tuned CatBoost: longer iterations, finer lr, more bootstrap diversity."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import feat_pipeline as fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cat_extreme_submission.csv")
    ap.add_argument("--target-rate", type=float, default=0.18)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 13, 97])
    args = ap.parse_args()

    t0 = time.perf_counter()
    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    X_train_raw, X_test_raw = fp.make_features(train, test)
    cat_cols = [c for c in fp.CATEGORICAL_LOWCARD + fp.CATEGORICAL_HIGHCARD if c in X_train_raw.columns]
    for c in cat_cols:
        X_train_raw[c] = X_train_raw[c].fillna("Unknown").astype(str)
        X_test_raw[c] = X_test_raw[c].fillna("Unknown").astype(str)
    for c in [c for c in X_train_raw.columns if c not in cat_cols]:
        X_train_raw[c] = pd.to_numeric(X_train_raw[c], errors="coerce").astype("float32")
        X_test_raw[c] = pd.to_numeric(X_test_raw[c], errors="coerce").astype("float32")
    y = train[fp.TARGET].astype(int)

    oof_avg = np.zeros(len(y))
    proba_avg = np.zeros(len(test))
    for seed in args.seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y))
        proba_test = np.zeros(len(test))
        for fold, (tr, va) in enumerate(skf.split(X_train_raw, y), 1):
            Xt = X_train_raw.iloc[tr]; Xv = X_train_raw.iloc[va]
            yt = y.iloc[tr]; yv = y.iloc[va]
            pool_t = Pool(Xt, yt, cat_features=cat_cols)
            pool_v = Pool(Xv, yv, cat_features=cat_cols)
            cb = CatBoostClassifier(
                iterations=4000,
                learning_rate=0.025,
                depth=8,
                l2_leaf_reg=2.5,
                random_strength=2.5,
                bootstrap_type="Bernoulli",
                subsample=0.85,
                border_count=192,
                grow_policy="SymmetricTree",
                loss_function="Logloss",
                eval_metric="AUC",
                auto_class_weights="Balanced",
                random_seed=seed,
                verbose=0,
                early_stopping_rounds=120,
                allow_writing_files=False,
            )
            cb.fit(pool_t, eval_set=pool_v)
            oof[va] = cb.predict_proba(pool_v)[:, 1]
            proba_test += cb.predict_proba(Pool(X_test_raw, cat_features=cat_cols))[:, 1] / 5
            print(f"  seed{seed} fold{fold}: iter={cb.tree_count_} auc={roc_auc_score(yv, oof[va]):.5f}")
        print(f"  seed{seed} OOF AUC={roc_auc_score(y, oof):.5f}  AP={average_precision_score(y, oof):.5f}")
        oof_avg += oof / len(args.seeds)
        proba_avg += proba_test / len(args.seeds)

    print(f"\n[cat_extreme] OOF AUC={roc_auc_score(y, oof_avg):.5f} AP={average_precision_score(y, oof_avg):.5f}")
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
