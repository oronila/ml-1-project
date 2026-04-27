"""Non-linear stacker (LightGBM meta) + top-3 blend variants."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.special import logit

import feat_pipeline as fp

EPS = 1e-7


def to_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def to_rank(p):
    order = p.argsort()
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(p))
    return r / (len(p) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rate", type=float, default=0.18)
    ap.add_argument("--out", default="stack_v2_submission.csv")
    args = ap.parse_args()

    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    y = train[fp.TARGET].astype(int).values

    base_paths = {
        "lgbm": "lgbm_target_rate_submission",
        "lgbm_v2": "lgbm_v2_submission",
        "lgbm_int": "lgbm_interactions_submission",
        "lgbm_te": "lgbm_targetenc_submission",
        "xgb": "xgb_target_rate_submission",
        "rf": "rf_target_rate_submission",
        "et": "et_target_rate_submission",
        "hgb": "hgb_target_rate_submission",
        "cat": "cat_target_rate_submission",
        "cat_seed": "cat_seedavg_submission",
        "cat_deep": "cat_deep_submission",
        "cat_grand": "cat_grand_submission",
        "xgb_seed": "xgb_seedavg_submission",
        "ada": "adaboost_v2_submission",
    }
    base_paths = {k: v for k, v in base_paths.items() if Path(v + ".oof.npy").exists()}

    oofs, probas, names = [], [], []
    for name, base in base_paths.items():
        oofs.append(np.load(base + ".oof.npy"))
        probas.append(np.load(base + ".proba.npy"))
        names.append(name)
    oof_mat = np.array(oofs).T
    proba_mat = np.array(probas).T

    print("Per-base AUC:")
    for i, n in enumerate(names):
        a = roc_auc_score(y, oof_mat[:, i])
        ap_ = average_precision_score(y, oof_mat[:, i])
        print(f"  {n:10s} AUC={a:.5f} AP={ap_:.5f}")

    # ---- Variant 1: Top-3 rank average (cat_seed, xgb, cat) ----
    top3 = [names.index(n) for n in ["cat_seed", "xgb", "cat"] if n in names]
    rank_oof = np.mean([to_rank(oof_mat[:, i]) for i in top3], axis=0)
    rank_test = np.mean([to_rank(proba_mat[:, i]) for i in top3], axis=0)
    print(f"\n[top3-rank] OOF AUC={roc_auc_score(y, rank_oof):.5f}  AP={average_precision_score(y, rank_oof):.5f}")

    # ---- Variant 2: weighted blend by AP score ----
    aps = np.array([average_precision_score(y, oof_mat[:, i]) for i in range(len(names))])
    w = (aps - aps.min() + 0.001) ** 3
    w = w / w.sum()
    print("[weighted] weights:", dict(zip(names, np.round(w, 3))))
    wblend_oof = (oof_mat * w).sum(axis=1)
    wblend_test = (proba_mat * w).sum(axis=1)
    print(f"[weighted] OOF AUC={roc_auc_score(y, wblend_oof):.5f}  AP={average_precision_score(y, wblend_oof):.5f}")

    # ---- Variant 3: LightGBM meta stacker on logit features ----
    Z_oof = to_logit(oof_mat)
    Z_test = to_logit(proba_mat)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(y))
    iters = []
    params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=15,
                  min_data_in_leaf=200, feature_fraction=0.9, lambda_l2=2.0, verbose=-1, seed=42)
    for tr_idx, va_idx in skf.split(Z_oof, y):
        ds_t = lgb.Dataset(Z_oof[tr_idx], y[tr_idx])
        ds_v = lgb.Dataset(Z_oof[va_idx], y[va_idx], reference=ds_t)
        b = lgb.train(params, ds_t, num_boost_round=600, valid_sets=[ds_v],
                      callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
        iters.append(b.best_iteration)
        meta_oof[va_idx] = b.predict(Z_oof[va_idx], num_iteration=b.best_iteration)
    print(f"\n[lgb-meta] OOF AUC={roc_auc_score(y, meta_oof):.5f}  AP={average_precision_score(y, meta_oof):.5f}")
    fi = int(np.mean(iters) * 1.05)
    final = lgb.train(params, lgb.Dataset(Z_oof, y), num_boost_round=fi)
    meta_test = final.predict(Z_test)

    # ---- Variant 4: Logistic regression meta (we already have this in stack.py) ----
    log_oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(Z_oof, y):
        m = LogisticRegression(C=1.0, max_iter=300)
        m.fit(Z_oof[tr_idx], y[tr_idx])
        log_oof[va_idx] = m.predict_proba(Z_oof[va_idx])[:, 1]
    print(f"[log-meta] OOF AUC={roc_auc_score(y, log_oof):.5f}  AP={average_precision_score(y, log_oof):.5f}")
    final_lr = LogisticRegression(C=1.0, max_iter=300)
    final_lr.fit(Z_oof, y)
    log_test = final_lr.predict_proba(Z_test)[:, 1]

    # Pick the best by OOF AUC
    candidates = [
        ("top3-rank", rank_oof, rank_test),
        ("weighted", wblend_oof, wblend_test),
        ("lgb-meta", meta_oof, meta_test),
        ("log-meta", log_oof, log_test),
    ]
    best_name, best_oof, best_test = max(candidates, key=lambda x: roc_auc_score(y, x[1]))
    print(f"\n[choose] {best_name}: AUC={roc_auc_score(y, best_oof):.5f}  AP={average_precision_score(y, best_oof):.5f}")
    for tr in [0.13, 0.15, 0.175, 0.18, 0.19, 0.20]:
        thr = fp.threshold_for_target_rate(best_oof, tr)
        pred = (best_oof >= thr).astype(int)
        print(f"  target={tr:.3f} thr={thr:.4f} f1={f1_score(y, pred):.4f}")

    thr = fp.threshold_for_target_rate(best_test, args.target_rate)
    preds = (best_test >= thr).astype(int)
    print(f"[submit] thr={thr:.4f}")
    fp.write_submission(test[fp.ID_COL], preds, Path(args.out))
    np.save(Path(args.out).with_suffix(".proba.npy"), best_test)
    np.save(Path(args.out).with_suffix(".oof.npy"), best_oof)


if __name__ == "__main__":
    main()
