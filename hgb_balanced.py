"""HistGradientBoosting (sklearn) with native categorical + sample weights."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import feat_pipeline as fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hgb_target_rate_submission.csv")
    ap.add_argument("--target-rate", type=float, default=0.175)
    args = ap.parse_args()

    t0 = time.perf_counter()
    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    X_train_raw, X_test_raw = fp.make_features(train, test)
    X_train, X_test, cat_cols = fp.encode_for_lgb(X_train_raw, X_test_raw)
    y = train[fp.TARGET].astype(int)

    # HGB caps cardinality at 255; only mark low-card cats as categorical
    native_cats = []
    for c in cat_cols:
        n_unique = max(X_train[c].max(), X_test[c].max()) + 1
        if n_unique <= 255 and c in fp.CATEGORICAL_LOWCARD:
            native_cats.append(c)
    cat_mask = [c in native_cats for c in X_train.columns]
    print(f"X_train={X_train.shape} native_cats={sum(cat_mask)} (of {len(cat_cols)} encoded)")

    pos_w = (1 - y.mean()) / y.mean()
    sw = np.where(y == 1, pos_w, 1.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        h = HistGradientBoostingClassifier(
            max_iter=1500,
            learning_rate=0.04,
            max_leaf_nodes=63,
            min_samples_leaf=40,
            l2_regularization=1.0,
            categorical_features=cat_mask,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=80,
            random_state=42 + fold,
        )
        h.fit(X_train.iloc[tr_idx], y.iloc[tr_idx], sample_weight=sw[tr_idx])
        oof[va_idx] = h.predict_proba(X_train.iloc[va_idx])[:, 1]
        iters.append(h.n_iter_)
        print(f"  fold{fold}: n_iter={h.n_iter_} auc={roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f}")

    print(f"OOF AUC={roc_auc_score(y, oof):.5f} AP={average_precision_score(y, oof):.5f}")
    for tr in [0.06, 0.13, 0.15, 0.175, 0.20]:
        thr = fp.threshold_for_target_rate(oof, tr)
        pred = (oof >= thr).astype(int)
        print(f"  target={tr:.3f} thr={thr:.4f} f1={f1_score(y, pred):.4f}")

    final_iters = int(np.mean(iters) * 1.05)
    print(f"[final] iters={final_iters}")
    h = HistGradientBoostingClassifier(
        max_iter=final_iters,
        learning_rate=0.04,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=cat_mask,
        random_state=42,
    )
    h.fit(X_train, y, sample_weight=sw)
    proba_test = h.predict_proba(X_test)[:, 1]
    thr = fp.threshold_for_target_rate(proba_test, args.target_rate)
    preds = (proba_test >= thr).astype(int)
    fp.write_submission(test[fp.ID_COL], preds, Path(args.out))
    np.save(Path(args.out).with_suffix(".proba.npy"), proba_test)
    np.save(Path(args.out).with_suffix(".oof.npy"), oof)
    print(f"[done] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
