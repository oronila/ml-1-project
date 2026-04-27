"""XGBoost runner with same interface as run_lgbm."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import (HIGH_CARD_CAT, LOW_CARD_CAT, TARGET, ID_COL,
                      get_airport_map, kfold_target_encode, make_base)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def threshold_for_rate(p, rate):
    rate = float(np.clip(rate, 1e-4, 1 - 1e-4))
    return float(np.quantile(p, 1 - rate))


def best_threshold(y_true, p):
    pr, rc, th = precision_recall_curve(y_true, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    i = int(np.argmax(f1[:-1])) if len(th) else 0
    return float(th[i]), float(f1[i])


def build_features(train_df, test_df, te_cols):
    am = get_airport_map(pd.concat([train_df, test_df], axis=0))
    Xtr = make_base(train_df, am)
    Xte = make_base(test_df, am)
    keep_high = [c for c in HIGH_CARD_CAT if c in Xtr.columns]
    for col in keep_high:
        vc = Xtr[col].value_counts()
        Xtr[col + "_FREQ"] = Xtr[col].map(vc).fillna(0).astype("float32")
        Xte[col + "_FREQ"] = Xte[col].map(vc).fillna(0).astype("float32")
    y = train_df[TARGET].astype(int).reset_index(drop=True)
    for col in te_cols:
        if col not in Xtr.columns:
            continue
        oof, te_enc = kfold_target_encode(Xtr[col].reset_index(drop=True), y, Xte[col].reset_index(drop=True))
        Xtr[col + "_TE"] = oof
        Xte[col + "_TE"] = te_enc
    # XGBoost: use enable_categorical via category dtype for low-card; drop high-card raw
    for col in keep_high:
        if col in Xtr.columns:
            Xtr = Xtr.drop(columns=[col])
            Xte = Xte.drop(columns=[col])
    for col in LOW_CARD_CAT:
        if col in Xtr.columns:
            Xtr[col] = Xtr[col].astype("category")
            Xte[col] = pd.Categorical(Xte[col], categories=Xtr[col].cat.categories)
    return Xtr, Xte, y


def run(name, params, te_cols, n_splits=5, seed=42, num_boost_round=2000, early_stop=100):
    t0 = time.time()
    train_df = pd.read_csv(ROOT / "data" / "train.csv", low_memory=False)
    test_df = pd.read_csv(ROOT / "data" / "test.csv", low_memory=False)
    Xtr, Xte, y = build_features(train_df, test_df, te_cols)
    print(f"[{name}] features: {Xtr.shape[1]}", flush=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype="float32")
    test_pred = np.zeros(len(Xte), dtype="float32")
    train_aucs, val_aucs, val_aps = [], [], []

    base = dict(
        objective="binary:logistic",
        eval_metric=["auc", "aucpr"],
        tree_method="hist",
        device="cuda",
        enable_categorical=True,
        verbosity=0,
        seed=seed,
    )
    base.update(params)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xtr, y)):
        Xtr_f, Xv_f = Xtr.iloc[tr_idx], Xtr.iloc[val_idx]
        ytr_f, yv_f = y.iloc[tr_idx], y.iloc[val_idx]
        dtr = xgb.DMatrix(Xtr_f, label=ytr_f, enable_categorical=True)
        dv = xgb.DMatrix(Xv_f, label=yv_f, enable_categorical=True)
        model = xgb.train(base, dtr, num_boost_round=num_boost_round,
                          evals=[(dtr, "train"), (dv, "val")],
                          early_stopping_rounds=early_stop, verbose_eval=False)
        p_tr = model.predict(dtr)
        p_v = model.predict(dv)
        train_aucs.append(roc_auc_score(ytr_f, p_tr))
        val_aucs.append(roc_auc_score(yv_f, p_v))
        val_aps.append(average_precision_score(yv_f, p_v))
        oof[val_idx] = p_v
        dte = xgb.DMatrix(Xte, enable_categorical=True)
        test_pred += model.predict(dte) / n_splits
        print(f"  fold{fold}: train_auc={train_aucs[-1]:.4f} val_auc={val_aucs[-1]:.4f} val_ap={val_aps[-1]:.4f} best_it={model.best_iteration}", flush=True)

    val_auc = float(np.mean(val_aucs))
    train_auc = float(np.mean(train_aucs))
    val_ap = float(np.mean(val_aps))
    overfit_gap = train_auc - val_auc
    base_rate = float(y.mean())
    thr_f1, f1 = best_threshold(y.values, oof)

    summary = dict(
        name=name, params=params, te_cols=te_cols, n_features=int(Xtr.shape[1]),
        train_auc=train_auc, val_auc=val_auc, val_ap=val_ap, overfit_gap=overfit_gap,
        oof_f1=f1, oof_thr_f1=thr_f1,
        thr_rate7=threshold_for_rate(oof, 0.07),
        thr_rate8=threshold_for_rate(oof, 0.08),
        base_rate=base_rate, elapsed_s=time.time() - t0,
    )
    (RESULTS / f"{name}.json").write_text(json.dumps(summary, indent=2))
    np.save(RESULTS / f"{name}_oof.npy", oof)
    np.save(RESULTS / f"{name}_test.npy", test_pred)
    pd.DataFrame({ID_COL: test_df[ID_COL], "PROBA": test_pred}).to_csv(
        RESULTS / f"{name}_test_proba.csv", index=False)
    print(f"[{name}] DONE val_auc={val_auc:.4f} ap={val_ap:.4f} gap={overfit_gap:.4f}", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config)
    run(args.name, cfg.get("params", {}), cfg.get("te_cols", []),
        n_splits=cfg.get("n_splits", 5),
        num_boost_round=cfg.get("num_boost_round", 2000),
        early_stop=cfg.get("early_stop", 100))
