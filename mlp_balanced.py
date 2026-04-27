"""Small MLP on dense features. Fundamentally different from trees → diversity."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import feat_pipeline as fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mlp_submission.csv")
    ap.add_argument("--target-rate", type=float, default=0.18)
    args = ap.parse_args()

    t0 = time.perf_counter()
    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    X_train_raw, X_test_raw = fp.make_features(train, test)
    X_train, X_test = fp.encode_for_dense(X_train_raw, X_test_raw, top_k=30)
    y = train[fp.TARGET].astype(int)
    print(f"X_train={X_train.shape}")

    # Standardize for MLP
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # MLP doesn't natively support class_weight; oversample positives
    from sklearn.utils import resample

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    proba_test = np.zeros(len(X_test))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_s, y), 1):
        Xt = X_train_s[tr_idx]; yt = y.iloc[tr_idx].values
        # oversample positives to ~50:50 (lower than balanced to limit overfit)
        pos_idx = np.where(yt == 1)[0]
        neg_idx = np.where(yt == 0)[0]
        pos_resampled = resample(pos_idx, n_samples=int(len(neg_idx) * 0.5), random_state=42 + fold)
        all_idx = np.concatenate([neg_idx, pos_resampled])
        np.random.RandomState(fold).shuffle(all_idx)
        Xt2 = Xt[all_idx]; yt2 = yt[all_idx]

        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=40,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=42 + fold,
            verbose=False,
        )
        mlp.fit(Xt2, yt2)
        oof[va_idx] = mlp.predict_proba(X_train_s[va_idx])[:, 1]
        proba_test += mlp.predict_proba(X_test_s)[:, 1] / 5.0
        print(f"  fold{fold}: n_iter={mlp.n_iter_} auc={roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f}")

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
