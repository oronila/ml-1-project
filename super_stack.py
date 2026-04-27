"""H100 super stack: LGBM + XGBoost(GPU) + CatBoost(GPU), 5-fold OOF,
seed averaging, logistic-regression stacker. No NLP/text features.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score, log_loss)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"
SEED = 42
N_FOLDS = 5
SEEDS = [42, 7, 123]  # seed averaging

DROP_COLS = [ID_COL, TARGET, "LOCATION", "BIRD_BAND_NUMBER", "REG", "FLT",
             "AIRCRAFT", "LUPDATE", "REMARKS", "COMMENTS"]
NUMBER_COLS = ["LATITUDE", "LONGITUDE", "HEIGHT", "SPEED", "DISTANCE",
               "NUM_SEEN", "NUM_STRUCK", "OUT_OF_RANGE_SPECIES",
               "AC_MASS", "NUM_ENGS",
               "ENG_1_POS", "ENG_2_POS", "ENG_3_POS", "ENG_4_POS",
               "REMAINS_COLLECTED", "REMAINS_SENT", "TRANSFER"]
LOW_CARD_CAT = ["TIME_OF_DAY", "FAAREGION", "AC_CLASS", "TYPE_ENG",
                "PHASE_OF_FLIGHT", "SKY", "WARNED", "SIZE", "SOURCE",
                "PRECIPITATION"]
HIGH_CARD_CAT = ["STATE", "OPID", "AIRPORT_ID", "SPECIES_ID",
                 "AMA", "AMO", "EMA", "EMO", "ENROUTE_STATE", "PERSON",
                 "RUNWAY", "OPERATOR", "AIRPORT", "SPECIES"]
TE_COLS = ["AIRPORT_ID", "OPID", "SPECIES_ID", "AMA", "AMO", "EMA", "EMO",
           "STATE", "RUNWAY", "OPERATOR", "AIRPORT", "SPECIES", "PERSON"]


def get_airport_map(df):
    df = df.copy()
    for col in ["LATITUDE", "LONGITUDE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = df[(df["AIRPORT_ID"].notna()) & (df["AIRPORT_ID"] != "ZZZZ")
              & (df["LATITUDE"].notna()) & (df["LONGITUDE"].notna())]
    coords = valid.groupby("AIRPORT_ID")[["LATITUDE", "LONGITUDE"]].median()
    return coords.to_dict("index")


def time_to_minutes(series):
    s = series.fillna("").astype(str).str.replace(":", "", regex=False)
    s = pd.to_numeric(s, errors="coerce")
    h = (s // 100).clip(0, 23)
    m = (s % 100).clip(0, 59)
    return h * 60 + m


def make_base(df, airport_map):
    df = df.copy()
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    for coord in ["LATITUDE", "LONGITUDE"]:
        m = {k: v[coord] for k, v in airport_map.items()}
        df[coord] = df[coord].fillna(df["AIRPORT_ID"].map(m))

    out = {}
    d = pd.to_datetime(df["INCIDENT_DATE"], format="mixed", errors="coerce")
    year = pd.to_numeric(df.get("INCIDENT_YEAR"), errors="coerce")
    year = year.fillna(d.dt.year).fillna(2015)
    out["YEAR"] = year.astype("float32")
    out["YEAR_REL"] = (year - 2000).astype("float32")
    month = pd.to_numeric(df.get("INCIDENT_MONTH"), errors="coerce").fillna(d.dt.month).fillna(6)
    out["MONTH"] = month.astype("float32")
    out["MONTH_SIN"] = np.sin(2 * np.pi * month / 12.0).astype("float32")
    out["MONTH_COS"] = np.cos(2 * np.pi * month / 12.0).astype("float32")
    out["DAY_OF_YEAR"] = d.dt.dayofyear.fillna(180).astype("float32")
    out["DAY_OF_WEEK"] = d.dt.dayofweek.fillna(3).astype("float32")

    if "TIME" in df.columns:
        mins = time_to_minutes(df["TIME"])
        out["MINUTES"] = mins.astype("float32")
        out["HOUR_SIN"] = np.sin(2 * np.pi * mins / 1440.0).astype("float32")
        out["HOUR_COS"] = np.cos(2 * np.pi * mins / 1440.0).astype("float32")
        out["MISS_TIME"] = mins.isna().astype("int8")

    for col in NUMBER_COLS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    for col in ["HEIGHT", "SPEED", "DISTANCE"]:
        if col in out:
            v = out[col].fillna(0).clip(lower=0)
            out["LOG_" + col] = np.log1p(v).astype("float32")

    if "SPEED" in out and "AC_MASS" in out:
        sp = out["SPEED"].fillna(out["SPEED"].median())
        ms = out["AC_MASS"].fillna(out["AC_MASS"].median())
        out["KE_PROXY"] = (ms * (sp ** 2)).astype("float32")
        out["LOG_KE"] = np.log1p(out["KE_PROXY"].clip(lower=0)).astype("float32")
    if "SPEED" in out and "HEIGHT" in out:
        out["SPEED_HEIGHT"] = (out["SPEED"].fillna(0) * out["HEIGHT"].fillna(0)).astype("float32")

    if "LATITUDE" in out and "LONGITUDE" in out:
        out["LAT_BIN"] = ((out["LATITUDE"] / 2).round() * 2).astype("float32")
        out["LON_BIN"] = ((out["LONGITUDE"] / 2).round() * 2).astype("float32")
        out["LAT_BIN_FINE"] = ((out["LATITUDE"] * 2).round() / 2).astype("float32")
        out["LON_BIN_FINE"] = ((out["LONGITUDE"] * 2).round() / 2).astype("float32")

    for col in ["SPEED", "HEIGHT", "DISTANCE", "AC_MASS", "NUM_STRUCK"]:
        if col in df.columns:
            out["MISS_" + col] = df[col].isna().astype("int8")

    for col in LOW_CARD_CAT + HIGH_CARD_CAT:
        if col in df.columns:
            out[col] = df[col].fillna("Unknown").astype(str)

    return pd.DataFrame(out, index=df.index)


def kfold_target_encode(train_series, target, test_series, n_splits=5, smooth=20, seed=42):
    prior = float(target.mean())
    oof = np.full(len(train_series), prior, dtype="float32")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in skf.split(train_series, target):
        tr_s = train_series.iloc[tr_idx]
        tr_y = target.iloc[tr_idx]
        agg = tr_y.groupby(tr_s).agg(["sum", "count"])
        enc = (agg["sum"] + smooth * prior) / (agg["count"] + smooth)
        oof[val_idx] = train_series.iloc[val_idx].map(enc).fillna(prior).values
    agg = target.groupby(train_series).agg(["sum", "count"])
    full_enc = (agg["sum"] + smooth * prior) / (agg["count"] + smooth)
    test_enc = test_series.map(full_enc).fillna(prior).values.astype("float32")
    return oof, test_enc


def build_features(train_df, test_df):
    am = get_airport_map(pd.concat([train_df, test_df], axis=0, ignore_index=True))
    Xtr = make_base(train_df, am)
    Xte = make_base(test_df, am)
    y = train_df[TARGET].astype(int).reset_index(drop=True)

    # Frequency encode all high-card cats (computed on train+test combined for stability)
    for col in HIGH_CARD_CAT:
        if col in Xtr.columns:
            combined = pd.concat([Xtr[col], Xte[col]], axis=0)
            vc = combined.value_counts()
            Xtr[col + "_FREQ"] = Xtr[col].map(vc).fillna(0).astype("float32")
            Xte[col + "_FREQ"] = Xte[col].map(vc).fillna(0).astype("float32")

    # OOF target encode
    for col in TE_COLS:
        if col in Xtr.columns:
            oof, te = kfold_target_encode(Xtr[col].reset_index(drop=True), y,
                                          Xte[col].reset_index(drop=True))
            Xtr[col + "_TE"] = oof
            Xte[col + "_TE"] = te

    return Xtr, Xte, y


def best_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    if len(th) == 0:
        return 0.5, 0.0
    i = int(np.argmax(f1[:-1]))
    return float(th[i]), float(f1[i])


# --- LGBM ---
def train_lgbm(Xtr, y, Xte, cat_cols, seed):
    params = dict(
        objective="binary", metric="auc",
        learning_rate=0.02, num_leaves=127, max_depth=-1,
        min_data_in_leaf=40, feature_fraction=0.85, bagging_fraction=0.85,
        bagging_freq=5, lambda_l2=1.0, verbosity=-1, seed=seed,
        feature_pre_filter=False,
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype="float32")
    pred = np.zeros(len(Xte), dtype="float32")
    for fold, (tr, va) in enumerate(skf.split(Xtr, y)):
        Xt, Xv = Xtr.iloc[tr], Xtr.iloc[va]
        yt, yv = y.iloc[tr], y.iloc[va]
        dtr = lgb.Dataset(Xt, label=yt, categorical_feature=cat_cols)
        dv = lgb.Dataset(Xv, label=yv, categorical_feature=cat_cols, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=4000,
                      valid_sets=[dv], valid_names=["val"],
                      callbacks=[lgb.early_stopping(150, verbose=False),
                                 lgb.log_evaluation(0)])
        oof[va] = m.predict(Xv, num_iteration=m.best_iteration)
        pred += m.predict(Xte, num_iteration=m.best_iteration) / N_FOLDS
        print(f"  lgbm seed={seed} fold{fold}: auc={roc_auc_score(yv,oof[va]):.4f} ap={average_precision_score(yv,oof[va]):.4f} it={m.best_iteration}")
    return oof, pred


# --- XGBoost GPU ---
def train_xgb(Xtr, y, Xte, seed):
    # XGBoost wants numeric only; encode categoricals as integer codes
    Xtr_n = Xtr.copy()
    Xte_n = Xte.copy()
    for c in Xtr_n.columns:
        if Xtr_n[c].dtype == "object" or str(Xtr_n[c].dtype) == "category":
            combined = pd.concat([Xtr_n[c].astype(str), Xte_n[c].astype(str)], axis=0)
            cats = pd.Categorical(combined).categories
            Xtr_n[c] = pd.Categorical(Xtr_n[c].astype(str), categories=cats).codes.astype("int32")
            Xte_n[c] = pd.Categorical(Xte_n[c].astype(str), categories=cats).codes.astype("int32")

    params = dict(
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", device="cuda",
        learning_rate=0.03, max_depth=8,
        subsample=0.85, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=1.0,
        seed=seed,
    )
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype="float32")
    pred = np.zeros(len(Xte_n), dtype="float32")
    for fold, (tr, va) in enumerate(skf.split(Xtr_n, y)):
        dtr = xgb.DMatrix(Xtr_n.iloc[tr], label=y.iloc[tr])
        dv = xgb.DMatrix(Xtr_n.iloc[va], label=y.iloc[va])
        dte = xgb.DMatrix(Xte_n)
        m = xgb.train(params, dtr, num_boost_round=4000,
                      evals=[(dv, "val")], early_stopping_rounds=150, verbose_eval=False)
        oof[va] = m.predict(dv, iteration_range=(0, m.best_iteration + 1))
        pred += m.predict(dte, iteration_range=(0, m.best_iteration + 1)) / N_FOLDS
        print(f"  xgb seed={seed} fold{fold}: auc={roc_auc_score(y.iloc[va],oof[va]):.4f} ap={average_precision_score(y.iloc[va],oof[va]):.4f} it={m.best_iteration}")
    return oof, pred


# --- CatBoost GPU ---
def train_cat(Xtr, y, Xte, cat_cols, seed):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype="float32")
    pred = np.zeros(len(Xte), dtype="float32")
    cat_idx = [Xtr.columns.get_loc(c) for c in cat_cols if c in Xtr.columns]
    for fold, (tr, va) in enumerate(skf.split(Xtr, y)):
        m = CatBoostClassifier(
            iterations=4000, learning_rate=0.03, depth=8,
            l2_leaf_reg=3.0, random_seed=seed, eval_metric="AUC",
            task_type="GPU", devices="0", verbose=0,
            early_stopping_rounds=150, bootstrap_type="Bernoulli", subsample=0.85,
        )
        Xt = Xtr.iloc[tr]; Xv = Xtr.iloc[va]
        m.fit(Pool(Xt, y.iloc[tr], cat_features=cat_idx),
              eval_set=Pool(Xv, y.iloc[va], cat_features=cat_idx))
        oof[va] = m.predict_proba(Xv)[:, 1]
        pred += m.predict_proba(Xte)[:, 1] / N_FOLDS
        print(f"  cat seed={seed} fold{fold}: auc={roc_auc_score(y.iloc[va],oof[va]):.4f} ap={average_precision_score(y.iloc[va],oof[va]):.4f} it={m.tree_count_}")
    return oof, pred


def report(name, y, oof):
    auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    thr, f1 = best_threshold(y.values, oof)
    print(f"=== {name}: ROC-AUC={auc:.5f}  PR-AUC={ap:.5f}  best-F1={f1:.5f} (thr={thr:.4f})")
    return dict(roc_auc=auc, pr_auc=ap, f1=f1, thr=thr)


def main():
    t0 = time.time()
    train_df = pd.read_csv(ROOT / "data" / "train.csv", low_memory=False)
    test_df = pd.read_csv(ROOT / "data" / "test.csv", low_memory=False)
    print(f"train={len(train_df)}, test={len(test_df)}, base_rate={train_df[TARGET].mean():.4f}")

    Xtr, Xte, y = build_features(train_df, test_df)
    print(f"features={Xtr.shape[1]}")

    cat_cols = [c for c in (LOW_CARD_CAT + HIGH_CARD_CAT) if c in Xtr.columns]

    # LGBM expects category dtype
    Xtr_lgb = Xtr.copy(); Xte_lgb = Xte.copy()
    for c in cat_cols:
        Xtr_lgb[c] = Xtr_lgb[c].astype("category")
        Xte_lgb[c] = pd.Categorical(Xte_lgb[c], categories=Xtr_lgb[c].cat.categories)

    oof_models = {}
    pred_models = {}

    # LGBM seed avg
    print("\n--- LightGBM ---")
    oof_l = np.zeros(len(y), dtype="float32"); pred_l = np.zeros(len(Xte), dtype="float32")
    for s in SEEDS:
        o, p = train_lgbm(Xtr_lgb, y, Xte_lgb, cat_cols, s)
        oof_l += o / len(SEEDS); pred_l += p / len(SEEDS)
    report("LGBM (seed-avg)", y, oof_l)
    oof_models["lgbm"] = oof_l; pred_models["lgbm"] = pred_l

    # XGB seed avg
    print("\n--- XGBoost (GPU) ---")
    oof_x = np.zeros(len(y), dtype="float32"); pred_x = np.zeros(len(Xte), dtype="float32")
    for s in SEEDS:
        o, p = train_xgb(Xtr, y, Xte, s)
        oof_x += o / len(SEEDS); pred_x += p / len(SEEDS)
    report("XGB (seed-avg)", y, oof_x)
    oof_models["xgb"] = oof_x; pred_models["xgb"] = pred_x

    # CatBoost seed avg
    print("\n--- CatBoost (GPU) ---")
    Xtr_c = Xtr.copy(); Xte_c = Xte.copy()
    for c in cat_cols:
        Xtr_c[c] = Xtr_c[c].astype(str)
        Xte_c[c] = Xte_c[c].astype(str)
    oof_c = np.zeros(len(y), dtype="float32"); pred_c = np.zeros(len(Xte), dtype="float32")
    for s in SEEDS:
        o, p = train_cat(Xtr_c, y, Xte_c, cat_cols, s)
        oof_c += o / len(SEEDS); pred_c += p / len(SEEDS)
    report("CAT (seed-avg)", y, oof_c)
    oof_models["cat"] = oof_c; pred_models["cat"] = pred_c

    # --- Stack: simple mean & logistic regression on OOF ---
    print("\n--- Stacking ---")
    OOF = np.column_stack([oof_models[k] for k in ["lgbm", "xgb", "cat"]])
    PRED = np.column_stack([pred_models[k] for k in ["lgbm", "xgb", "cat"]])
    mean_oof = OOF.mean(axis=1)
    mean_pred = PRED.mean(axis=1)
    report("MEAN", y, mean_oof)

    # rank average
    def to_rank(a): return pd.Series(a).rank().values / len(a)
    rank_oof = np.column_stack([to_rank(c) for c in OOF.T]).mean(axis=1)
    rank_pred = np.column_stack([to_rank(c) for c in PRED.T]).mean(axis=1)
    report("RANK-AVG", y, rank_oof)

    # Logistic stacker (with logit-transformed inputs to be linear-friendly)
    eps = 1e-6
    Z_oof = np.log(np.clip(OOF, eps, 1 - eps) / (1 - np.clip(OOF, eps, 1 - eps)))
    Z_pred = np.log(np.clip(PRED, eps, 1 - eps) / (1 - np.clip(PRED, eps, 1 - eps)))
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(Z_oof, y)
    stack_oof = lr.predict_proba(Z_oof)[:, 1]
    stack_pred = lr.predict_proba(Z_pred)[:, 1]
    print("LR stack coefs:", lr.coef_, "intercept:", lr.intercept_)
    stack_metrics = report("LR-STACK", y, stack_oof)

    # Save
    summary = {
        "elapsed_s": time.time() - t0,
        "n_features": int(Xtr.shape[1]),
        "models": {
            "lgbm": dict(roc_auc=float(roc_auc_score(y, oof_l)),
                        pr_auc=float(average_precision_score(y, oof_l))),
            "xgb": dict(roc_auc=float(roc_auc_score(y, oof_x)),
                        pr_auc=float(average_precision_score(y, oof_x))),
            "cat": dict(roc_auc=float(roc_auc_score(y, oof_c)),
                        pr_auc=float(average_precision_score(y, oof_c))),
            "mean": dict(roc_auc=float(roc_auc_score(y, mean_oof)),
                         pr_auc=float(average_precision_score(y, mean_oof))),
            "rank": dict(roc_auc=float(roc_auc_score(y, rank_oof)),
                         pr_auc=float(average_precision_score(y, rank_oof))),
            "lr_stack": stack_metrics,
        },
    }
    (RESULTS / "super_stack_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    np.save(RESULTS / "super_stack_oof.npy", stack_oof)
    np.save(RESULTS / "super_stack_test.npy", stack_pred)
    pd.DataFrame({ID_COL: test_df[ID_COL], "PROBA": stack_pred}).to_csv(
        RESULTS / "super_stack_test_proba.csv", index=False)
    # Threshold submission at OOF best-F1 threshold
    thr = stack_metrics["thr"]
    sub = pd.DataFrame({ID_COL: test_df[ID_COL],
                        "INDICATED_DAMAGE": (stack_pred >= thr).astype(int)})
    sub.to_csv(RESULTS / "super_stack_submission.csv", index=False)
    print(f"\nDONE in {time.time()-t0:.1f}s")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
