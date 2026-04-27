"""Load saved models and produce predictions on a new test CSV — no retraining.

Usage:
    python predict_with_saved.py --test test.csv --target-rate 0.18 --out my_sub.csv

Reads from models/ directory. Each saved model produces test predictions, then
we apply the same isotonic+logistic stacker used in ensemble_lab.py to blend.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import Pool
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import feat_pipeline as fp

MODEL_DIR = Path("models")
EPS = 1e-7


def to_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def predict_one(name, X_test_lgb, X_test_raw, cat_cols):
    estimators = joblib.load(MODEL_DIR / f"{name}.joblib")
    meta = json.loads((MODEL_DIR / f"{name}.meta.json").read_text())
    mt = meta["model_type"]
    pred = np.zeros(len(X_test_lgb))
    if mt == "lightgbm":
        for b in estimators:
            pred += b.predict(X_test_lgb, num_iteration=b.best_iteration) / len(estimators)
    elif mt == "xgboost":
        d = xgb.DMatrix(X_test_lgb)
        for b in estimators:
            pred += b.predict(d, iteration_range=(0, b.best_iteration + 1)) / len(estimators)
    elif mt == "catboost":
        pool = Pool(X_test_raw, cat_features=cat_cols)
        for cb in estimators:
            pred += cb.predict_proba(pool)[:, 1] / len(estimators)
    elif mt == "hgb":
        for h in estimators:
            pred += h.predict_proba(X_test_lgb)[:, 1] / len(estimators)
    elif mt == "adaboost":
        Xa = X_test_lgb.fillna(-1).astype("float32")
        for a in estimators:
            pred += a.predict_proba(Xa)[:, 1] / len(estimators)
    elif mt == "rf":
        # rf used dense encoding — we'd need to recompute it; user can call save_models.py to refresh
        raise RuntimeError("rf needs dense re-encoding; not implemented in predict_with_saved")
    return pred, np.load(MODEL_DIR / f"{name}.oof.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--target-rate", type=float, default=0.18)
    ap.add_argument("--out", default="predict_with_saved_submission.csv")
    args = ap.parse_args()

    train = pd.read_csv(args.train, low_memory=False)
    test = pd.read_csv(args.test, low_memory=False)
    X_train_raw, X_test_raw = fp.make_features(train, test)
    X_train_lgb, X_test_lgb, cat_cols_lgb = fp.encode_for_lgb(X_train_raw, X_test_raw)
    cat_cols_raw = [c for c in fp.CATEGORICAL_LOWCARD + fp.CATEGORICAL_HIGHCARD if c in X_train_raw.columns]
    # match catboost's expectation
    Xs_raw_for_cat = X_test_raw.copy()
    for c in cat_cols_raw:
        Xs_raw_for_cat[c] = Xs_raw_for_cat[c].fillna("Unknown").astype(str)
    for c in [c for c in Xs_raw_for_cat.columns if c not in cat_cols_raw]:
        Xs_raw_for_cat[c] = pd.to_numeric(Xs_raw_for_cat[c], errors="coerce").astype("float32")
    y = train[fp.TARGET].values

    available = sorted(p.stem for p in MODEL_DIR.glob("*.meta.json"))
    print(f"available saved models: {available}")

    test_preds, oofs, names = [], [], []
    for name in available:
        try:
            t, o = predict_one(name, X_test_lgb, Xs_raw_for_cat, cat_cols_raw)
        except RuntimeError as e:
            print(f"  skip {name}: {e}")
            continue
        test_preds.append(t)
        oofs.append(o)
        names.append(name)
        print(f"  {name}: test_mean={t.mean():.4f}")

    oof_mat = np.array(oofs).T
    test_mat = np.array(test_preds).T

    # apply same isotonic+logistic blend used in ensemble_lab.py (Variant E)
    iso_oof_mat = np.zeros_like(oof_mat)
    iso_test_mat = np.zeros_like(test_mat)
    for i in range(len(names)):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof_mat[:, i], y)
        iso_oof_mat[:, i] = iso.transform(oof_mat[:, i])
        iso_test_mat[:, i] = iso.transform(test_mat[:, i])
    Z_oof = to_logit(iso_oof_mat)
    Z_test = to_logit(iso_test_mat)
    meta = LogisticRegression(C=1.0, max_iter=300).fit(Z_oof, y)
    proba_test = meta.predict_proba(Z_test)[:, 1]
    print(f"meta coefs: {dict(zip(names, np.round(meta.coef_[0], 3)))}")

    thr = fp.threshold_for_target_rate(proba_test, args.target_rate)
    preds = (proba_test >= thr).astype(int)
    fp.write_submission(test[fp.ID_COL], preds, Path(args.out))


if __name__ == "__main__":
    main()
