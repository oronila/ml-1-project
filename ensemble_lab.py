"""Ensemble laboratory: many blender variants, picks the best by OOF PR-AUC.

Reads cached .oof.npy and .proba.npy — does NOT retrain any base models.
Compares: rank-avg, weighted, AP-direct optimizer, isotonic-calibrated,
LR meta, LGBM meta, isotonic-then-LR.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import logit, expit
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
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(p))
    return r / (len(p) - 1)


def fit_ap_direct(oof_mat, y, init=None, n_starts=4, seed=42):
    """Find blend weights w (sum=1, w>=0) that MAXIMIZE OOF PR-AUC."""
    M = oof_mat.shape[1]

    def neg_ap(w):
        # softmax-style positive weights summing to 1
        w_norm = np.maximum(w, 0)
        s = w_norm.sum()
        if s <= 0:
            return 1.0
        w_norm = w_norm / s
        blend = (oof_mat * w_norm).sum(axis=1)
        return -average_precision_score(y, blend)

    rng = np.random.default_rng(seed)
    best = None
    best_val = 1.0
    inits = []
    if init is not None:
        inits.append(init)
    inits.append(np.ones(M) / M)
    for _ in range(n_starts - len(inits)):
        inits.append(rng.dirichlet(np.ones(M)))

    for x0 in inits:
        res = minimize(
            neg_ap, x0, method="SLSQP",
            bounds=[(0.0, 1.0)] * M,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            options={"maxiter": 200, "ftol": 1e-7},
        )
        if res.fun < best_val:
            best_val = res.fun
            best = res.x
    w = np.maximum(best, 0)
    w = w / w.sum()
    return w, -best_val


def isotonic_calibrate(oof, y, test_proba):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(oof, y)
    return iso.transform(oof), iso.transform(test_proba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rate", type=float, default=0.18)
    ap.add_argument("--out", default="ensemble_lab_submission.csv")
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

    print(f"== Base models ({len(names)}) ==")
    for i, n in enumerate(names):
        a = roc_auc_score(y, oof_mat[:, i])
        ap_ = average_precision_score(y, oof_mat[:, i])
        print(f"  {n:10s} AUC={a:.5f}  AP={ap_:.5f}")

    # ---- Variant A: simple rank-average of all ----
    rank_oof = np.mean([to_rank(oof_mat[:, i]) for i in range(len(names))], axis=0)
    rank_test = np.mean([to_rank(proba_mat[:, i]) for i in range(len(names))], axis=0)
    print(f"\n[A rank-avg-all  ] AUC={roc_auc_score(y, rank_oof):.5f}  AP={average_precision_score(y, rank_oof):.5f}")

    # ---- Variant B: top-3 rank avg ----
    top3 = sorted(range(len(names)), key=lambda i: average_precision_score(y, oof_mat[:, i]), reverse=True)[:3]
    print(f"  top3: {[names[i] for i in top3]}")
    top3_oof = np.mean([to_rank(oof_mat[:, i]) for i in top3], axis=0)
    top3_test = np.mean([to_rank(proba_mat[:, i]) for i in top3], axis=0)
    print(f"[B top3-rank     ] AUC={roc_auc_score(y, top3_oof):.5f}  AP={average_precision_score(y, top3_oof):.5f}")

    # ---- Variant C: AP-direct weight optimizer (on raw probas) ----
    w_raw, ap_raw = fit_ap_direct(oof_mat, y)
    apdir_oof = (oof_mat * w_raw).sum(axis=1)
    apdir_test = (proba_mat * w_raw).sum(axis=1)
    print(f"[C AP-direct/raw ] AUC={roc_auc_score(y, apdir_oof):.5f}  AP={average_precision_score(y, apdir_oof):.5f}")
    print("  weights:", dict(zip(names, np.round(w_raw, 3))))

    # ---- Variant D: AP-direct optimizer (on RANK features) ----
    rank_oof_mat = np.column_stack([to_rank(oof_mat[:, i]) for i in range(len(names))])
    rank_test_mat = np.column_stack([to_rank(proba_mat[:, i]) for i in range(len(names))])
    w_rank, ap_rank = fit_ap_direct(rank_oof_mat, y)
    apdir_rank_oof = (rank_oof_mat * w_rank).sum(axis=1)
    apdir_rank_test = (rank_test_mat * w_rank).sum(axis=1)
    print(f"[D AP-direct/rank] AUC={roc_auc_score(y, apdir_rank_oof):.5f}  AP={average_precision_score(y, apdir_rank_oof):.5f}")
    print("  rank weights:", dict(zip(names, np.round(w_rank, 3))))

    # ---- Variant E: isotonic-calibrate each base then logistic stack ----
    iso_oof_mat = np.zeros_like(oof_mat)
    iso_test_mat = np.zeros_like(proba_mat)
    for i in range(len(names)):
        iso_oof_mat[:, i], iso_test_mat[:, i] = isotonic_calibrate(oof_mat[:, i], y, proba_mat[:, i])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    iso_log_oof = np.zeros(len(y))
    Z = to_logit(iso_oof_mat)
    Zt = to_logit(iso_test_mat)
    for tr_idx, va_idx in skf.split(Z, y):
        m = LogisticRegression(C=1.0, max_iter=300)
        m.fit(Z[tr_idx], y[tr_idx])
        iso_log_oof[va_idx] = m.predict_proba(Z[va_idx])[:, 1]
    final_lr = LogisticRegression(C=1.0, max_iter=300)
    final_lr.fit(Z, y)
    iso_log_test = final_lr.predict_proba(Zt)[:, 1]
    print(f"[E iso+logit-LR  ] AUC={roc_auc_score(y, iso_log_oof):.5f}  AP={average_precision_score(y, iso_log_oof):.5f}")

    # ---- Variant F: logistic stacker on logit features (matches stack_v2) ----
    Z = to_logit(oof_mat)
    Zt = to_logit(proba_mat)
    log_oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(Z, y):
        m = LogisticRegression(C=1.0, max_iter=300)
        m.fit(Z[tr_idx], y[tr_idx])
        log_oof[va_idx] = m.predict_proba(Z[va_idx])[:, 1]
    final_lr = LogisticRegression(C=1.0, max_iter=300)
    final_lr.fit(Z, y)
    log_test = final_lr.predict_proba(Zt)[:, 1]
    print(f"[F log-meta      ] AUC={roc_auc_score(y, log_oof):.5f}  AP={average_precision_score(y, log_oof):.5f}")

    # ---- Variant G: LGBM meta on logits ----
    lgb_oof = np.zeros(len(y))
    iters = []
    params = dict(objective="binary", metric="average_precision", learning_rate=0.04, num_leaves=15,
                  min_data_in_leaf=200, feature_fraction=0.9, lambda_l2=2.0, verbose=-1, seed=42)
    for tr_idx, va_idx in skf.split(Z, y):
        b = lgb.train(params, lgb.Dataset(Z[tr_idx], y[tr_idx]), num_boost_round=800,
                      valid_sets=[lgb.Dataset(Z[va_idx], y[va_idx])],
                      callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
        iters.append(b.best_iteration)
        lgb_oof[va_idx] = b.predict(Z[va_idx], num_iteration=b.best_iteration)
    fi = int(np.mean(iters) * 1.05)
    final_lgb = lgb.train(params, lgb.Dataset(Z, y), num_boost_round=fi)
    lgb_test = final_lgb.predict(Zt)
    print(f"[G lgb-meta      ] AUC={roc_auc_score(y, lgb_oof):.5f}  AP={average_precision_score(y, lgb_oof):.5f}")

    # ---- Variant H: stacked-of-stackers: blend log-meta + lgb-meta + AP-direct rank ----
    stk_oof = (log_oof + lgb_oof + apdir_rank_oof) / 3.0
    stk_test = (log_test + lgb_test + apdir_rank_test) / 3.0
    print(f"[H stack-of-stack] AUC={roc_auc_score(y, stk_oof):.5f}  AP={average_precision_score(y, stk_oof):.5f}")

    # ---- Variant I: AP-optimal blend of (log_meta, lgb_meta, apdir_rank, iso_log) ----
    meta_mat = np.column_stack([log_oof, lgb_oof, apdir_rank_oof, iso_log_oof])
    meta_test_mat = np.column_stack([log_test, lgb_test, apdir_rank_test, iso_log_test])
    w_meta, ap_meta = fit_ap_direct(meta_mat, y, n_starts=8)
    metablend_oof = (meta_mat * w_meta).sum(axis=1)
    metablend_test = (meta_test_mat * w_meta).sum(axis=1)
    print(f"[I AP-meta-blend ] AUC={roc_auc_score(y, metablend_oof):.5f}  AP={average_precision_score(y, metablend_oof):.5f}")
    print(f"  meta weights: log={w_meta[0]:.3f} lgb={w_meta[1]:.3f} apdir={w_meta[2]:.3f} iso={w_meta[3]:.3f}")

    # ---- Pick the best by OOF PR-AUC ----
    candidates = [
        ("A rank-avg-all", rank_oof, rank_test),
        ("B top3-rank", top3_oof, top3_test),
        ("C ap-raw", apdir_oof, apdir_test),
        ("D ap-rank", apdir_rank_oof, apdir_rank_test),
        ("E iso+log", iso_log_oof, iso_log_test),
        ("F log-meta", log_oof, log_test),
        ("G lgb-meta", lgb_oof, lgb_test),
        ("H stk-of-stk", stk_oof, stk_test),
        ("I ap-meta-blend", metablend_oof, metablend_test),
    ]
    print("\n== Final ranking by OOF AP ==")
    sorted_c = sorted(candidates, key=lambda x: average_precision_score(y, x[1]), reverse=True)
    for name, o, t in sorted_c:
        print(f"  {name:18s} AUC={roc_auc_score(y, o):.5f}  AP={average_precision_score(y, o):.5f}")

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
