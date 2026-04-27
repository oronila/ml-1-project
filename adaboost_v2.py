"""AdaBoost over depth-15 balanced decision tree — replicate original_notebook winner.

Use OUR feature pipeline (no text) and OUR positive-rate calibration.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import feat_pipeline as fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="adaboost_v2_submission.csv")
    ap.add_argument("--target-rate", type=float, default=0.18)
    args = ap.parse_args()

    t0 = time.perf_counter()
    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    X_train_raw, X_test_raw = fp.make_features(train, test)
    X_train, X_test, _ = fp.encode_for_lgb(X_train_raw, X_test_raw)
    y = train[fp.TARGET].astype(int)
    # AdaBoost cannot handle NaN — fill
    X_train = X_train.fillna(-1).astype("float32")
    X_test = X_test.fillna(-1).astype("float32")
    print(f"X_train={X_train.shape} pos_rate={y.mean():.4f}")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold for speed
    oof = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        base = DecisionTreeClassifier(max_depth=15, min_samples_leaf=100, class_weight="balanced", random_state=42)
        ada = AdaBoostClassifier(estimator=base, n_estimators=200, learning_rate=0.1, random_state=42)
        ada.fit(X_train.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = ada.predict_proba(X_train.iloc[va_idx])[:, 1]
        print(f"  fold{fold}: auc={roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f}")

    print(f"OOF AUC={roc_auc_score(y, oof):.5f} AP={average_precision_score(y, oof):.5f}")
    for tr in [0.13, 0.15, 0.175, 0.18, 0.20]:
        thr = fp.threshold_for_target_rate(oof, tr)
        pred = (oof >= thr).astype(int)
        print(f"  target={tr:.3f} thr={thr:.4f} f1={f1_score(y, pred):.4f}")

    base = DecisionTreeClassifier(max_depth=15, min_samples_leaf=100, class_weight="balanced", random_state=42)
    ada = AdaBoostClassifier(estimator=base, n_estimators=300, learning_rate=0.1, random_state=42)
    ada.fit(X_train, y)
    proba_test = ada.predict_proba(X_test)[:, 1]
    thr = fp.threshold_for_target_rate(proba_test, args.target_rate)
    preds = (proba_test >= thr).astype(int)
    fp.write_submission(test[fp.ID_COL], preds, Path(args.out))
    np.save(Path(args.out).with_suffix(".proba.npy"), proba_test)
    np.save(Path(args.out).with_suffix(".oof.npy"), oof)
    print(f"[done] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
