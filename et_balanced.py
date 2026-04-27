"""ExtraTrees with class_weight=balanced + target-rate threshold."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import feat_pipeline as fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--out", default="et_target_rate_submission.csv")
    ap.add_argument("--target-rate", type=float, default=0.175)
    args = ap.parse_args()

    t0 = time.perf_counter()
    train = pd.read_csv(args.train, low_memory=False)
    test = pd.read_csv(args.test, low_memory=False)

    X_train_raw, X_test_raw = fp.make_features(train, test)
    X_train, X_test = fp.encode_for_dense(X_train_raw, X_test_raw, top_k=40)
    y = train[fp.TARGET].astype(int)
    print(f"X_train={X_train.shape} pos_rate={y.mean():.4f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        et = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=24,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42 + fold,
        )
        et.fit(X_train.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = et.predict_proba(X_train.iloc[va_idx])[:, 1]
        print(f"  fold{fold}: val_auc={roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f}")

    print(f"OOF ROC-AUC={roc_auc_score(y, oof):.5f}  PR-AUC={average_precision_score(y, oof):.5f}")
    for tr in [0.06, 0.10, 0.13, 0.15, 0.175, 0.20]:
        thr = fp.threshold_for_target_rate(oof, tr)
        pred = (oof >= thr).astype(int)
        print(f"  target={tr:.3f} thr={thr:.4f} f1={f1_score(y, pred):.4f}")

    et = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=24,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    et.fit(X_train, y)
    proba_test = et.predict_proba(X_test)[:, 1]
    thr = fp.threshold_for_target_rate(proba_test, args.target_rate)
    preds = (proba_test >= thr).astype(int)
    print(f"[submit] thr={thr:.4f}")
    fp.write_submission(test[fp.ID_COL], preds, Path(args.out))
    np.save(Path(args.out).with_suffix(".proba.npy"), proba_test)
    np.save(Path(args.out).with_suffix(".oof.npy"), oof)
    print(f"[done] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
