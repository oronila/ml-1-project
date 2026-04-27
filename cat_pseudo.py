"""Pseudo-labeled CatBoost: use stacker high-confidence test preds as extra training.

Steps:
1. Load best stacker test probas (ensemble_lab2_submission.proba.npy)
2. Take top P_pos% as pseudo-positives, bottom P_neg% as pseudo-negatives
3. Build augmented training set: real_train (weight=1.0) + pseudo_test (weight=0.5)
4. Train CatBoost
5. Use proper CV with the original train rows only to compute OOF (so we can stack later)
"""
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
    ap.add_argument("--out", default="cat_pseudo_submission.csv")
    ap.add_argument("--target-rate", type=float, default=0.18)
    ap.add_argument("--pseudo-pos-q", type=float, default=0.85,
                    help="quantile threshold above which test is pseudo-positive")
    ap.add_argument("--pseudo-neg-q", type=float, default=0.40,
                    help="quantile threshold below which test is pseudo-negative")
    ap.add_argument("--pseudo-weight", type=float, default=0.5,
                    help="sample weight for pseudo-labeled rows")
    ap.add_argument("--stacker-proba", default="ensemble_lab2_submission.proba.npy",
                    help="path to .npy of stacker test probabilities")
    args = ap.parse_args()

    t0 = time.perf_counter()
    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)

    # Load stacker test probas — the more confident the better
    stack_proba = np.load(args.stacker_proba)

    # Threshold pseudo-labels by quantile
    q_pos = np.quantile(stack_proba, args.pseudo_pos_q)
    q_neg = np.quantile(stack_proba, args.pseudo_neg_q)
    pseudo_pos_mask = stack_proba >= q_pos
    pseudo_neg_mask = stack_proba <= q_neg
    n_pos = int(pseudo_pos_mask.sum())
    n_neg = int(pseudo_neg_mask.sum())
    print(f"pseudo-positives: {n_pos}  pseudo-negatives: {n_neg}  unlabeled: {len(test) - n_pos - n_neg}")

    # Build features
    X_train_raw, X_test_raw = fp.make_features(train, test)
    cat_cols = [c for c in fp.CATEGORICAL_LOWCARD + fp.CATEGORICAL_HIGHCARD if c in X_train_raw.columns]
    for c in cat_cols:
        X_train_raw[c] = X_train_raw[c].fillna("Unknown").astype(str)
        X_test_raw[c] = X_test_raw[c].fillna("Unknown").astype(str)
    for c in [c for c in X_train_raw.columns if c not in cat_cols]:
        X_train_raw[c] = pd.to_numeric(X_train_raw[c], errors="coerce").astype("float32")
        X_test_raw[c] = pd.to_numeric(X_test_raw[c], errors="coerce").astype("float32")
    y = train[fp.TARGET].astype(int)
    print(f"X_train={X_train_raw.shape} pos_rate={y.mean():.4f}")

    # Pseudo-rows: take from test, label them, weight 0.5
    pseudo_idx = np.where(pseudo_pos_mask | pseudo_neg_mask)[0]
    X_pseudo = X_test_raw.iloc[pseudo_idx].reset_index(drop=True)
    y_pseudo = pseudo_pos_mask[pseudo_idx].astype(int)
    w_pseudo = np.full(len(pseudo_idx), args.pseudo_weight)
    w_train = np.ones(len(y))
    print(f"pseudo aug: pos={int(y_pseudo.sum())} neg={int((y_pseudo==0).sum())} weight={args.pseudo_weight}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2027)
    oof = np.zeros(len(y))
    proba_test = np.zeros(len(X_test_raw))
    iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_raw, y), 1):
        # Train on real_train[tr] + pseudo
        Xt_real = X_train_raw.iloc[tr_idx]; yt_real = y.iloc[tr_idx]; wt_real = w_train[tr_idx]
        Xt = pd.concat([Xt_real, X_pseudo], ignore_index=True)
        yt = np.concatenate([yt_real.values, y_pseudo])
        wt = np.concatenate([wt_real, w_pseudo])
        Xv = X_train_raw.iloc[va_idx]; yv = y.iloc[va_idx]

        pool_t = Pool(Xt, yt, weight=wt, cat_features=cat_cols)
        pool_v = Pool(Xv, yv, cat_features=cat_cols)
        cb = CatBoostClassifier(
            iterations=2500, learning_rate=0.04, depth=8, l2_leaf_reg=3.0,
            random_strength=1.5, bagging_temperature=0.4, border_count=128,
            loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
            random_seed=2027, verbose=0, early_stopping_rounds=80, allow_writing_files=False,
        )
        cb.fit(pool_t, eval_set=pool_v)
        iters.append(cb.tree_count_)
        oof[va_idx] = cb.predict_proba(pool_v)[:, 1]
        proba_test += cb.predict_proba(Pool(X_test_raw, cat_features=cat_cols))[:, 1] / 5
        print(f"  fold{fold}: iter={cb.tree_count_} auc={roc_auc_score(yv, oof[va_idx]):.5f}")

    print(f"\n[cat_pseudo] OOF AUC={roc_auc_score(y, oof):.5f} AP={average_precision_score(y, oof):.5f}")
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
