"""More aggressive ensemble exploration: calibration variants, meta variants, blends.

Picks the best by OOF PR-AUC.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import feat_pipeline as fp

EPS = 1e-7


def to_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def to_rank(p):
    order = p.argsort()
    r = np.empty_like(order, dtype=np.float64); r[order] = np.arange(len(p))
    return r / (len(p) - 1)


def fit_ap_direct(oof_mat, y, n_starts=4, seed=42):
    M = oof_mat.shape[1]

    def neg_ap(w):
        w_norm = np.maximum(w, 0)
        s = w_norm.sum()
        if s <= 0:
            return 1.0
        w_norm = w_norm / s
        return -average_precision_score(y, (oof_mat * w_norm).sum(axis=1))

    rng = np.random.default_rng(seed)
    best, bv = None, 1.0
    inits = [np.ones(M) / M]
    for _ in range(n_starts - 1):
        inits.append(rng.dirichlet(np.ones(M)))
    for x0 in inits:
        res = minimize(neg_ap, x0, method="SLSQP", bounds=[(0, 1)] * M,
                       constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                       options={"maxiter": 200, "ftol": 1e-7})
        if res.fun < bv:
            bv = res.fun; best = res.x
    w = np.maximum(best, 0); w = w / w.sum()
    return w, -bv


def isotonic_cv_calibrate(oof, y, test_proba, n_splits=5, seed=42):
    """Cross-validated isotonic calibration to avoid overfit."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_cal = np.zeros_like(oof)
    test_cals = []
    for tr, va in skf.split(oof, y):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[tr], y[tr])
        oof_cal[va] = iso.transform(oof[va])
        test_cals.append(iso.transform(test_proba))
    test_cal = np.mean(test_cals, axis=0)
    return oof_cal, test_cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rate", type=float, default=0.18)
    ap.add_argument("--out", default="ensemble_lab2_submission.csv")
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
        "xgb_seed": "xgb_seedavg_submission",
        "rf": "rf_target_rate_submission",
        "et": "et_target_rate_submission",
        "hgb": "hgb_target_rate_submission",
        "cat": "cat_target_rate_submission",
        "cat_seed": "cat_seedavg_submission",
        "cat_deep": "cat_deep_submission",
        "cat_grand": "cat_grand_submission",
        "cat_pseudo": "cat_pseudo_submission",
        "cat_pseudo2": "cat_pseudo2_submission",
        "ada": "adaboost_v2_submission",
    }
    base_paths = {k: v for k, v in base_paths.items() if Path(v + ".oof.npy").exists()}
    oofs, probas, names = [], [], []
    for n, b in base_paths.items():
        oofs.append(np.load(b + ".oof.npy"))
        probas.append(np.load(b + ".proba.npy"))
        names.append(n)
    oof_mat = np.array(oofs).T
    proba_mat = np.array(probas).T
    print(f"== {len(names)} base models ==")

    # CV-isotonic calibrate every base
    iso_oof = np.zeros_like(oof_mat)
    iso_test = np.zeros_like(proba_mat)
    for i in range(len(names)):
        iso_oof[:, i], iso_test[:, i] = isotonic_cv_calibrate(oof_mat[:, i], y, proba_mat[:, i])
        ap_pre = average_precision_score(y, oof_mat[:, i])
        ap_post = average_precision_score(y, iso_oof[:, i])
        print(f"  {names[i]:10s} AP_pre={ap_pre:.5f} AP_post={ap_post:.5f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Variant E: iso + logit-LR (default C=1)
    Z = to_logit(iso_oof); Zt = to_logit(iso_test)
    e_oof = np.zeros(len(y))
    for tr, va in skf.split(Z, y):
        m = LogisticRegression(C=1.0, max_iter=300).fit(Z[tr], y[tr])
        e_oof[va] = m.predict_proba(Z[va])[:, 1]
    e_lr = LogisticRegression(C=1.0, max_iter=300).fit(Z, y)
    e_test = e_lr.predict_proba(Zt)[:, 1]
    print(f"\n[E iso+logit-LR(C=1.0) ] AUC={roc_auc_score(y, e_oof):.5f} AP={average_precision_score(y, e_oof):.5f}")

    # Variant E2: stronger reg
    e2_oof = np.zeros(len(y))
    for tr, va in skf.split(Z, y):
        m = LogisticRegression(C=0.1, max_iter=400).fit(Z[tr], y[tr])
        e2_oof[va] = m.predict_proba(Z[va])[:, 1]
    e2_lr = LogisticRegression(C=0.1, max_iter=400).fit(Z, y)
    e2_test = e2_lr.predict_proba(Zt)[:, 1]
    print(f"[E2 iso+logit-LR(C=.1)  ] AUC={roc_auc_score(y, e2_oof):.5f} AP={average_precision_score(y, e2_oof):.5f}")

    # Variant E3: balanced class weight
    e3_oof = np.zeros(len(y))
    for tr, va in skf.split(Z, y):
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=400).fit(Z[tr], y[tr])
        e3_oof[va] = m.predict_proba(Z[va])[:, 1]
    e3_lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=400).fit(Z, y)
    e3_test = e3_lr.predict_proba(Zt)[:, 1]
    print(f"[E3 iso+logit-LR(bal)   ] AUC={roc_auc_score(y, e3_oof):.5f} AP={average_precision_score(y, e3_oof):.5f}")

    # Variant E4: AP-direct on iso probas
    w_e4, _ = fit_ap_direct(iso_oof, y, n_starts=8)
    e4_oof = (iso_oof * w_e4).sum(axis=1)
    e4_test = (iso_test * w_e4).sum(axis=1)
    print(f"[E4 iso+AP-direct       ] AUC={roc_auc_score(y, e4_oof):.5f} AP={average_precision_score(y, e4_oof):.5f}")

    # Variant E5: AP-direct on iso logits (rank-aware)
    w_e5, _ = fit_ap_direct(Z, y, n_starts=8)
    e5_oof = (Z * w_e5).sum(axis=1)
    e5_test = (Zt * w_e5).sum(axis=1)
    print(f"[E5 iso+AP-on-logits    ] AUC={roc_auc_score(y, e5_oof):.5f} AP={average_precision_score(y, e5_oof):.5f}")

    # Variant E6: LightGBM meta on iso logits
    e6_oof = np.zeros(len(y)); iters = []
    params = dict(objective="binary", metric="average_precision", learning_rate=0.04,
                  num_leaves=15, min_data_in_leaf=200, feature_fraction=0.9, lambda_l2=2.0,
                  verbose=-1, seed=42)
    for tr, va in skf.split(Z, y):
        b = lgb.train(params, lgb.Dataset(Z[tr], y[tr]), num_boost_round=800,
                      valid_sets=[lgb.Dataset(Z[va], y[va])],
                      callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
        iters.append(b.best_iteration)
        e6_oof[va] = b.predict(Z[va], num_iteration=b.best_iteration)
    fi = int(np.mean(iters) * 1.05)
    e6_lgb = lgb.train(params, lgb.Dataset(Z, y), num_boost_round=fi)
    e6_test = e6_lgb.predict(Zt)
    print(f"[E6 iso+lgb-meta        ] AUC={roc_auc_score(y, e6_oof):.5f} AP={average_precision_score(y, e6_oof):.5f}")

    # Variant J: blend E + E4 + E6
    j_oof = (e_oof + e4_oof + e6_oof) / 3
    j_test = (e_test + e4_test + e6_test) / 3
    print(f"[J 3-way meta blend     ] AUC={roc_auc_score(y, j_oof):.5f} AP={average_precision_score(y, j_oof):.5f}")

    # Variant K: AP-optimal of metas
    meta_oof = np.column_stack([e_oof, e2_oof, e3_oof, e4_oof, e5_oof, e6_oof])
    meta_test = np.column_stack([e_test, e2_test, e3_test, e4_test, e5_test, e6_test])
    w_k, _ = fit_ap_direct(meta_oof, y, n_starts=10)
    k_oof = (meta_oof * w_k).sum(axis=1)
    k_test = (meta_test * w_k).sum(axis=1)
    print(f"[K AP-opt over metas    ] AUC={roc_auc_score(y, k_oof):.5f} AP={average_precision_score(y, k_oof):.5f}")
    print(f"  K weights: E={w_k[0]:.3f} E2={w_k[1]:.3f} E3={w_k[2]:.3f} E4={w_k[3]:.3f} E5={w_k[4]:.3f} E6={w_k[5]:.3f}")

    candidates = [
        ("E iso+lr(C=1)", e_oof, e_test),
        ("E2 iso+lr(C=.1)", e2_oof, e2_test),
        ("E3 iso+lr(bal)", e3_oof, e3_test),
        ("E4 iso+ap-prob", e4_oof, e4_test),
        ("E5 iso+ap-logit", e5_oof, e5_test),
        ("E6 iso+lgb", e6_oof, e6_test),
        ("J 3way", j_oof, j_test),
        ("K ap-of-metas", k_oof, k_test),
    ]
    print("\n== Final ranking by OOF AP ==")
    sorted_c = sorted(candidates, key=lambda x: average_precision_score(y, x[1]), reverse=True)
    for name, o, t in sorted_c:
        print(f"  {name:18s} AUC={roc_auc_score(y, o):.5f} AP={average_precision_score(y, o):.5f}")

    best_name, best_oof, best_test = sorted_c[0]
    print(f"\n[winner] {best_name}")
    for tr in [0.13, 0.15, 0.175, 0.18, 0.185, 0.19, 0.20]:
        thr = fp.threshold_for_target_rate(best_oof, tr)
        pred = (best_oof >= thr).astype(int)
        print(f"  target={tr:.3f} thr={thr:.4f} f1={f1_score(y, pred):.4f}")

    thr = fp.threshold_for_target_rate(best_test, args.target_rate)
    preds = (best_test >= thr).astype(int)
    fp.write_submission(test[fp.ID_COL], preds, Path(args.out))
    np.save(Path(args.out).with_suffix(".proba.npy"), best_test)
    np.save(Path(args.out).with_suffix(".oof.npy"), best_oof)


if __name__ == "__main__":
    main()
