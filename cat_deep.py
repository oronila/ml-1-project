"""CatBoost with deeper trees + slower learning rate. Diversity for ensemble."""
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
    ap.add_argument("--out", default="cat_deep_submission.csv")
    ap.add_argument("--target-rate", type=float, default=0.18)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.025)
    args = ap.parse_args()

    t0 = time.perf_counter()
    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    X_train_raw, X_test_raw = fp.make_features(train, test)
    cat_cols = [c for c in fp.CATEGORICAL_LOWCARD + fp.CATEGORICAL_HIGHCARD if c in X_train_raw.columns]
    for c in cat_cols:
        X_train_raw[c] = X_train_raw[c].fillna("Unknown").astype(str)
        X_test_raw[c] = X_test_raw[c].fillna("Unknown").astype(str)
    num_cols = [c for c in X_train_raw.columns if c not in cat_cols]
    for c in num_cols:
        X_train_raw[c] = pd.to_numeric(X_train_raw[c], errors="coerce").astype("float32")
        X_test_raw[c] = pd.to_numeric(X_test_raw[c], errors="coerce").astype("float32")
    y = train[fp.TARGET].astype(int)
    print(f"depth={args.depth} lr={args.lr} X_train={X_train_raw.shape}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
    oof = np.zeros(len(y))
    proba_test = np.zeros(len(test))
    iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_raw, y), 1):
        Xt = X_train_raw.iloc[tr_idx]; Xv = X_train_raw.iloc[va_idx]
        yt = y.iloc[tr_idx]; yv = y.iloc[va_idx]
        pool_t = Pool(Xt, yt, cat_features=cat_cols)
        pool_v = Pool(Xv, yv, cat_features=cat_cols)
        cb = CatBoostClassifier(
            iterations=3500,
            learning_rate=args.lr,
            depth=args.depth,
            l2_leaf_reg=4.0,
            random_strength=2.0,
            bagging_temperature=0.5,
            border_count=128,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=11,
            verbose=0,
            early_stopping_rounds=120,
            allow_writing_files=False,
        )
        cb.fit(pool_t, eval_set=pool_v)
        iters.append(cb.tree_count_)
        oof[va_idx] = cb.predict_proba(pool_v)[:, 1]
        proba_test += cb.predict_proba(Pool(X_test_raw, cat_features=cat_cols))[:, 1] / 5.0
        print(f"  fold{fold}: iter={cb.tree_count_} auc={roc_auc_score(yv, oof[va_idx]):.5f}")

    print(f"OOF AUC={roc_auc_score(y, oof):.5f} AP={average_precision_score(y, oof):.5f}")
    for tr in [0.13, 0.15, 0.175, 0.18, 0.20]:
        thr = fp.threshold_for_target_rate(oof, tr)
        pred = (oof >= thr).astype(int)
        print(f"  target={tr:.3f} thr={thr:.4f} f1={f1_score(y, pred):.4f}")

    thr = fp.threshold_for_target_rate(proba_test, args.target_rate)
    preds = (proba_test >= thr).astype(int)
    fp.write_submission(test[fp.ID_COL], preds, Path(args.out))
    np.save(Path(args.out).with_suffix(".proba.npy"), proba_test)
    np.save(Path(args.out).with_suffix(".oof.npy"), oof)
    print(f"[done] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
