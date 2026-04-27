"""Run a LightGBM experiment with 5-fold OOF validation.

Tracks: val ROC-AUC, val PR-AUC, train AUC, train-val gap (overfit).
Saves: experiments/results/<name>.json with metrics and OOF probs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import (HIGH_CARD_CAT, LOW_CARD_CAT, TARGET, ID_COL,
                      get_airport_map, kfold_target_encode, make_base)

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def best_threshold(y_true, p):
    pr, rc, th = precision_recall_curve(y_true, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    i = int(np.argmax(f1[:-1])) if len(th) else 0
    return float(th[i]), float(f1[i])


def threshold_for_rate(p, rate):
    """Threshold such that fraction of (p>=t) equals rate."""
    rate = float(np.clip(rate, 1e-4, 1 - 1e-4))
    return float(np.quantile(p, 1 - rate))


def build_features(train_df, test_df, te_cols, drop_high_card=False):
    am = get_airport_map(pd.concat([train_df, test_df], axis=0))
    Xtr = make_base(train_df, am)
    Xte = make_base(test_df, am)

    cat_cols = list(LOW_CARD_CAT)
    # frequency-encode high-card; optionally also target encode
    keep_high = [c for c in HIGH_CARD_CAT if c in Xtr.columns]
    for col in keep_high:
        # frequency encoding
        vc = Xtr[col].value_counts()
        Xtr[col + "_FREQ"] = Xtr[col].map(vc).fillna(0).astype("float32")
        Xte[col + "_FREQ"] = Xte[col].map(vc).fillna(0).astype("float32")

    y = train_df[TARGET].astype(int).reset_index(drop=True)

    # Out-of-fold target encoding
    for col in te_cols:
        if col not in Xtr.columns:
            continue
        oof, te_enc = kfold_target_encode(Xtr[col].reset_index(drop=True), y, Xte[col].reset_index(drop=True))
        Xtr[col + "_TE"] = oof
        Xte[col + "_TE"] = te_enc

    if drop_high_card:
        for col in keep_high:
            if col in Xtr.columns:
                Xtr = Xtr.drop(columns=[col])
                Xte = Xte.drop(columns=[col])
    else:
        # convert remaining string high-card categoricals to category dtype for LGBM native handling
        for col in keep_high:
            cat_cols.append(col)

    for col in cat_cols:
        if col in Xtr.columns:
            Xtr[col] = Xtr[col].astype("category")
            # align categories
            Xte[col] = pd.Categorical(Xte[col], categories=Xtr[col].cat.categories)

    return Xtr, Xte, y, [c for c in cat_cols if c in Xtr.columns]


def run(name, params, te_cols, drop_high_card=False, n_splits=5, seed=42, num_boost_round=2000, early_stop=100):
    t0 = time.time()
    train_df = pd.read_csv(ROOT / "data" / "train.csv", low_memory=False)
    test_df = pd.read_csv(ROOT / "data" / "test.csv", low_memory=False)

    Xtr, Xte, y, cat_cols = build_features(train_df, test_df, te_cols, drop_high_card)
    print(f"[{name}] features: {Xtr.shape[1]}, cats: {len(cat_cols)}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype="float32")
    test_pred = np.zeros(len(Xte), dtype="float32")
    train_aucs, val_aucs, val_aps = [], [], []

    base_params = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        verbosity=-1,
        seed=seed,
        feature_pre_filter=False,
    )
    base_params.update(params)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xtr, y)):
        Xtr_f = Xtr.iloc[tr_idx]
        Xv_f = Xtr.iloc[val_idx]
        ytr_f = y.iloc[tr_idx]
        yv_f = y.iloc[val_idx]
        dtr = lgb.Dataset(Xtr_f, label=ytr_f, categorical_feature=cat_cols)
        dv = lgb.Dataset(Xv_f, label=yv_f, categorical_feature=cat_cols, reference=dtr)
        model = lgb.train(
            base_params, dtr,
            num_boost_round=num_boost_round,
            valid_sets=[dtr, dv],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(early_stop, verbose=False),
                       lgb.log_evaluation(0)],
        )
        p_tr = model.predict(Xtr_f, num_iteration=model.best_iteration)
        p_v = model.predict(Xv_f, num_iteration=model.best_iteration)
        train_aucs.append(roc_auc_score(ytr_f, p_tr))
        val_aucs.append(roc_auc_score(yv_f, p_v))
        val_aps.append(average_precision_score(yv_f, p_v))
        oof[val_idx] = p_v
        test_pred += model.predict(Xte, num_iteration=model.best_iteration) / n_splits
        print(f"  fold{fold}: train_auc={train_aucs[-1]:.4f} val_auc={val_aucs[-1]:.4f} val_ap={val_aps[-1]:.4f} best_it={model.best_iteration}")

    val_auc = float(np.mean(val_aucs))
    train_auc = float(np.mean(train_aucs))
    val_ap = float(np.mean(val_aps))
    overfit_gap = train_auc - val_auc

    # Threshold selection on OOF
    base_rate = float(y.mean())
    thr_f1, f1 = best_threshold(y.values, oof)
    # robust threshold: assume real test rate ~7-8% (slightly higher than 6.36%)
    thr_rate7 = threshold_for_rate(oof, 0.07)
    thr_rate8 = threshold_for_rate(oof, 0.08)
    f1_at_base = f1_score(y.values, (oof >= thr_rate7).astype(int))

    summary = dict(
        name=name,
        params=params,
        te_cols=te_cols,
        drop_high_card=drop_high_card,
        n_features=int(Xtr.shape[1]),
        train_auc=train_auc,
        val_auc=val_auc,
        val_ap=val_ap,
        overfit_gap=overfit_gap,
        oof_f1=f1,
        oof_thr_f1=thr_f1,
        thr_rate7=thr_rate7,
        thr_rate8=thr_rate8,
        f1_at_rate7=float(f1_at_base),
        base_rate=base_rate,
        elapsed_s=time.time() - t0,
    )
    (RESULTS / f"{name}.json").write_text(json.dumps(summary, indent=2))
    np.save(RESULTS / f"{name}_oof.npy", oof)
    np.save(RESULTS / f"{name}_test.npy", test_pred)
    pd.DataFrame({ID_COL: test_df[ID_COL], "PROBA": test_pred}).to_csv(
        RESULTS / f"{name}_test_proba.csv", index=False)
    print(f"[{name}] DONE val_auc={val_auc:.4f} ap={val_ap:.4f} gap={overfit_gap:.4f} ({summary['elapsed_s']:.1f}s)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True, help="JSON config string")
    args = ap.parse_args()
    cfg = json.loads(args.config)
    run(args.name, cfg.get("params", {}), cfg.get("te_cols", []),
        cfg.get("drop_high_card", False),
        n_splits=cfg.get("n_splits", 5),
        num_boost_round=cfg.get("num_boost_round", 2000),
        early_stop=cfg.get("early_stop", 100))
