"""CatBoost runner."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from features import (HIGH_CARD_CAT, LOW_CARD_CAT, TARGET, ID_COL,
                      get_airport_map, kfold_target_encode, make_base)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def best_threshold(y_true, p):
    pr, rc, th = precision_recall_curve(y_true, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    i = int(np.argmax(f1[:-1])) if len(th) else 0
    return float(th[i]), float(f1[i])


def threshold_for_rate(p, rate):
    return float(np.quantile(p, 1 - float(np.clip(rate, 1e-4, 1 - 1e-4))))


def build_features(train_df, test_df, te_cols, drop_high_card=False):
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
    cat_cols = list(LOW_CARD_CAT)
    if drop_high_card:
        for col in keep_high:
            if col in Xtr.columns:
                Xtr = Xtr.drop(columns=[col]); Xte = Xte.drop(columns=[col])
    else:
        cat_cols.extend(keep_high)
    cat_cols = [c for c in cat_cols if c in Xtr.columns]
    # CatBoost needs strings, fill numeric NaN with -1 sentinel for stability
    for col in cat_cols:
        Xtr[col] = Xtr[col].astype(str).fillna("Unknown")
        Xte[col] = Xte[col].astype(str).fillna("Unknown")
    num_cols = [c for c in Xtr.columns if c not in cat_cols]
    for c in num_cols:
        Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce").astype("float32")
        Xte[c] = pd.to_numeric(Xte[c], errors="coerce").astype("float32")
    return Xtr, Xte, y, cat_cols


def run(name, params, te_cols, drop_high_card=False, n_splits=5, seed=42, iterations=2500, early_stop=100):
    t0 = time.time()
    train_df = pd.read_csv(ROOT / "data" / "train.csv", low_memory=False)
    test_df = pd.read_csv(ROOT / "data" / "test.csv", low_memory=False)
    Xtr, Xte, y, cat_cols = build_features(train_df, test_df, te_cols, drop_high_card)
    print(f"[{name}] features: {Xtr.shape[1]}, cats: {len(cat_cols)}", flush=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype="float32")
    test_pred = np.zeros(len(Xte), dtype="float32")
    train_aucs, val_aucs, val_aps = [], [], []

    base = dict(
        loss_function="Logloss", eval_metric="AUC", iterations=iterations,
        learning_rate=0.05, depth=6, l2_leaf_reg=3.0, random_seed=seed,
        task_type="GPU", devices="0", verbose=False,
        od_type="Iter", od_wait=early_stop,
    )
    base.update(params)

    cat_idx = [Xtr.columns.get_loc(c) for c in cat_cols]
    for fold, (tr_idx, val_idx) in enumerate(skf.split(Xtr, y)):
        Xtr_f, Xv_f = Xtr.iloc[tr_idx], Xtr.iloc[val_idx]
        ytr_f, yv_f = y.iloc[tr_idx], y.iloc[val_idx]
        ptr = Pool(Xtr_f, ytr_f, cat_features=cat_idx)
        pv = Pool(Xv_f, yv_f, cat_features=cat_idx)
        m = CatBoostClassifier(**base)
        m.fit(ptr, eval_set=pv, use_best_model=True, verbose=False)
        p_tr = m.predict_proba(ptr)[:, 1]
        p_v = m.predict_proba(pv)[:, 1]
        train_aucs.append(roc_auc_score(ytr_f, p_tr))
        val_aucs.append(roc_auc_score(yv_f, p_v))
        val_aps.append(average_precision_score(yv_f, p_v))
        oof[val_idx] = p_v
        pte = Pool(Xte, cat_features=cat_idx)
        test_pred += m.predict_proba(pte)[:, 1] / n_splits
        print(f"  fold{fold}: train_auc={train_aucs[-1]:.4f} val_auc={val_aucs[-1]:.4f} val_ap={val_aps[-1]:.4f} best_it={m.tree_count_}", flush=True)

    val_auc = float(np.mean(val_aucs)); train_auc = float(np.mean(train_aucs)); val_ap = float(np.mean(val_aps))
    overfit_gap = train_auc - val_auc
    thr_f1, f1 = best_threshold(y.values, oof)
    summary = dict(
        name=name, params=params, te_cols=te_cols, n_features=int(Xtr.shape[1]),
        train_auc=train_auc, val_auc=val_auc, val_ap=val_ap, overfit_gap=overfit_gap,
        oof_f1=f1, oof_thr_f1=thr_f1,
        thr_rate7=threshold_for_rate(oof, 0.07),
        thr_rate8=threshold_for_rate(oof, 0.08),
        base_rate=float(y.mean()), elapsed_s=time.time() - t0,
    )
    (RESULTS / f"{name}.json").write_text(json.dumps(summary, indent=2))
    np.save(RESULTS / f"{name}_oof.npy", oof)
    np.save(RESULTS / f"{name}_test.npy", test_pred)
    pd.DataFrame({ID_COL: test_df[ID_COL], "PROBA": test_pred}).to_csv(
        RESULTS / f"{name}_test_proba.csv", index=False)
    print(f"[{name}] DONE val_auc={val_auc:.4f} ap={val_ap:.4f} gap={overfit_gap:.4f}", flush=True)
    return summary


if __name__ == "__main__":
    import sys
    cfg = json.loads(sys.argv[2])
    run(sys.argv[1], cfg.get("params", {}), cfg.get("te_cols", []),
        drop_high_card=cfg.get("drop_high_card", False),
        n_splits=cfg.get("n_splits", 5),
        iterations=cfg.get("iterations", 2500),
        early_stop=cfg.get("early_stop", 100))
