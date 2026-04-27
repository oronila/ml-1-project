#!/usr/bin/env python3
"""
Robust ensemble model for wildlife-strike damage prediction.

Design principles:
- NO text features (REMARKS/COMMENTS cause overfitting)
- Drop post-strike proxy columns (SOURCE, PERSON, NUM_STRUCK, etc.)
- Out-of-fold target encoding for categorical features (no leakage)
- Strong regularization to prevent overfitting
- 5-fold LightGBM + 5-fold CatBoost ensemble for stable predictions
- Outputs probabilities for optimal PR-AUC

Run:
    python robust_lgbm_model.py
    python robust_lgbm_model.py --binary   # output 0/1 instead of probabilities
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"
RANDOM_STATE = 42
N_FOLDS = 5

# Post-strike / leaky / ID columns — excluded from features entirely.
# SOURCE is extremely leaky (NTSB=99% damage rate). PERSON is similarly leaky.
# REMARKS/COMMENTS are free text written post-strike describing damage directly.
DROP_COLS = {
    ID_COL, TARGET,
    "REMARKS", "COMMENTS",
    "SOURCE", "PERSON",
    "REMAINS_SENT",
    "TRANSFER", "BIRD_BAND_NUMBER",
    "LOCATION", "REG", "FLT", "AIRCRAFT", "AIRPORT", "SPECIES", "LUPDATE",
    "INCIDENT_DATE", "TIME",
}

# Categorical columns for CatBoost native handling
CATBOOST_CAT_COLS = [
    "TIME_OF_DAY", "STATE", "FAAREGION", "OPID",
    "AC_CLASS", "TYPE_ENG", "PHASE_OF_FLIGHT",
    "SKY", "PRECIPITATION", "WARNED", "SIZE",
    "AMA", "AMO", "EMA", "EMO",
    "ENROUTE_STATE", "RUNWAY",
    "AIRPORT_ID", "SPECIES_ID", "OPERATOR",
    "NUM_STRUCK",
]

# For LightGBM: target-encode these + use remaining as native categorical
LGBM_TARGET_ENCODE_COLS = [
    "AIRPORT_ID", "SPECIES_ID", "OPERATOR", "RUNWAY",
    "OPID", "STATE", "FAAREGION",
    "AC_CLASS", "TYPE_ENG", "PHASE_OF_FLIGHT",
    "SKY", "PRECIPITATION", "WARNED", "SIZE",
    "NUM_STRUCK",
]

LGBM_NATIVE_CAT_COLS = [
    "TIME_OF_DAY", "AMA", "AMO", "EMA", "EMO", "ENROUTE_STATE",
]

NUMERIC_COLS = [
    "LATITUDE", "LONGITUDE", "HEIGHT", "SPEED", "DISTANCE",
    "NUM_SEEN", "OUT_OF_RANGE_SPECIES",
    "AC_MASS", "NUM_ENGS",
    "ENG_1_POS", "ENG_2_POS", "ENG_3_POS", "ENG_4_POS",
    "REMAINS_COLLECTED",
]

# NUM_STRUCK has Excel date corruption ("10-Feb" means "2-10").
NUM_STRUCK_MAP = {"1": 1, "10-Feb": 5, "11-100": 50, "More than 100": 150}

MISSING_FLAG_COLS = [
    "HEIGHT", "SPEED", "DISTANCE", "TIME",
    "PHASE_OF_FLIGHT", "SKY", "SIZE", "LATITUDE",
]


def get_airport_map(df):
    tmp = df.copy()
    for col in ["LATITUDE", "LONGITUDE"]:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    valid = tmp[
        tmp["AIRPORT_ID"].notna()
        & (tmp["AIRPORT_ID"] != "ZZZZ")
        & tmp["LATITUDE"].notna()
        & tmp["LONGITUDE"].notna()
    ]
    return valid.groupby("AIRPORT_ID")[["LATITUDE", "LONGITUDE"]].median().to_dict("index")


def impute_coords(df, airport_map):
    df = df.copy()
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    for coord in ["LATITUDE", "LONGITUDE"]:
        lookup = {k: v[coord] for k, v in airport_map.items()}
        df[coord] = df[coord].fillna(df["AIRPORT_ID"].map(lookup))
    return df


def time_to_minutes(series):
    parts = series.fillna("").astype(str).str.split(":", n=1, expand=True)
    hour = pd.to_numeric(parts[0], errors="coerce")
    minute = pd.to_numeric(parts[1], errors="coerce").fillna(0) if 1 in parts.columns else 0
    return hour * 60 + minute


def make_features(df, airport_map):
    """Build feature DataFrame — shared by both LightGBM and CatBoost."""
    df = impute_coords(df, airport_map)
    out = pd.DataFrame(index=df.index)

    # Temporal
    dates = pd.to_datetime(df["INCIDENT_DATE"], format="mixed", errors="coerce")
    out["YEAR"] = dates.dt.year
    out["INCIDENT_MONTH"] = pd.to_numeric(df["INCIDENT_MONTH"], errors="coerce")
    out["INCIDENT_YEAR"] = pd.to_numeric(df["INCIDENT_YEAR"], errors="coerce")
    month = dates.dt.month.fillna(6)
    out["MONTH_SIN"] = np.sin(2 * np.pi * month / 12.0)
    out["MONTH_COS"] = np.cos(2 * np.pi * month / 12.0)
    doy = dates.dt.dayofyear.fillna(180)
    out["DOY_SIN"] = np.sin(2 * np.pi * doy / 366.0)
    out["DOY_COS"] = np.cos(2 * np.pi * doy / 366.0)

    if "TIME" in df.columns:
        minutes = time_to_minutes(df["TIME"])
        out["MINUTES"] = minutes
        filled = minutes.fillna(720)
        out["TIME_SIN"] = np.sin(2 * np.pi * filled / 1440.0)
        out["TIME_COS"] = np.cos(2 * np.pi * filled / 1440.0)

    # Numeric
    for col in NUMERIC_COLS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["HEIGHT", "SPEED", "DISTANCE", "NUM_SEEN"]:
        if col in out.columns:
            out[f"LOG_{col}"] = np.log1p(out[col].clip(lower=0))

    # Geographic
    if "LATITUDE" in out.columns and "LONGITUDE" in out.columns:
        lat, lon = out["LATITUDE"], out["LONGITUDE"]
        out["ABS_LATITUDE"] = lat.abs()
        out["DIST_FROM_US_CENTER"] = np.sqrt((lat - 39.5) ** 2 + (lon + 98.35) ** 2)
        out["LAT_X_LON"] = lat * lon

    # Physics-inspired interactions
    speed = out.get("SPEED", pd.Series(np.nan, index=df.index))
    mass = out.get("AC_MASS", pd.Series(np.nan, index=df.index))
    height = out.get("HEIGHT", pd.Series(np.nan, index=df.index))
    size_num = df.get("SIZE", pd.Series("", index=df.index)).map(
        {"Small": 1, "Medium": 2, "Large": 3}
    )
    out["ENERGY_PROXY"] = mass * speed ** 2
    out["MOMENTUM_PROXY"] = mass * speed
    out["SIZE_SPEED"] = size_num * speed
    out["SIZE_HEIGHT"] = size_num * height

    # NUM_STRUCK: ordinal numeric for LightGBM, categorical for CatBoost
    if "NUM_STRUCK" in df.columns:
        out["NUM_STRUCK_ORD"] = df["NUM_STRUCK"].map(NUM_STRUCK_MAP)
        out["NUM_STRUCK_MISSING"] = df["NUM_STRUCK"].isna().astype("int8")

    # REMAINS_COLLECTED: inverse signal (0 = bird destroyed = more damage)
    if "REMAINS_COLLECTED" in df.columns:
        out["REMAINS_COLLECTED"] = pd.to_numeric(df["REMAINS_COLLECTED"], errors="coerce")

    # Missing flags
    for col in MISSING_FLAG_COLS:
        if col in df.columns:
            out[f"{col}_MISSING"] = df[col].isna().astype("int8")

    # Categorical (kept as string, encoded differently per model)
    all_cat = set(CATBOOST_CAT_COLS) | set(LGBM_TARGET_ENCODE_COLS) | set(LGBM_NATIVE_CAT_COLS)
    for col in all_cat:
        if col in df.columns:
            out[col] = df[col].fillna("_missing_").astype(str)

    return out


def target_encode_oof(X_train, y_train, X_test, cols, smooth=50):
    """Out-of-fold target encoding. Returns copies with TE_ and FREQ_ columns added, original cols dropped."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    global_mean = float(y_train.mean())
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_arr = y_train.values

    for col in cols:
        if col not in X_train.columns:
            continue

        train_keys = X_train[col].values
        oof = np.full(len(X_train), global_mean, dtype="float64")

        for fit_idx, hold_idx in skf.split(X_train, y_train):
            s = pd.DataFrame({"k": train_keys[fit_idx], "y": y_arr[fit_idx]})
            stats = s.groupby("k")["y"].agg(["sum", "count"])
            rates = (stats["sum"] + smooth * global_mean) / (stats["count"] + smooth)
            oof[hold_idx] = pd.Series(train_keys[hold_idx]).map(rates).fillna(global_mean).values

        s = pd.DataFrame({"k": train_keys, "y": y_arr})
        stats = s.groupby("k")["y"].agg(["sum", "count"])
        rates = (stats["sum"] + smooth * global_mean) / (stats["count"] + smooth)
        test_enc = pd.Series(X_test[col].values).map(rates).fillna(global_mean).values

        X_train[f"TE_{col}"] = oof.astype("float32")
        X_test[f"TE_{col}"] = test_enc.astype("float32")

        counts = pd.Series(train_keys).value_counts()
        X_train[f"FREQ_{col}"] = pd.Series(train_keys).map(counts).fillna(0).astype("float32").values
        X_test[f"FREQ_{col}"] = pd.Series(X_test[col].values).map(counts).fillna(0).astype("float32").values

        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

    return X_train, X_test


def prepare_lgbm_data(X_all, y, X_test):
    """Prepare features for LightGBM: target-encode high-card, keep native cat for low-card."""
    X_tr = X_all.copy()
    X_te = X_test.copy()

    X_tr, X_te = target_encode_oof(X_tr, y, X_te, LGBM_TARGET_ENCODE_COLS)

    for col in LGBM_NATIVE_CAT_COLS:
        if col in X_tr.columns:
            X_tr[col] = X_tr[col].astype("category")
            X_te[col] = X_te[col].astype("category")

    drop = [c for c in X_tr.columns if c in DROP_COLS]
    X_tr = X_tr.drop(columns=drop, errors="ignore")
    X_te = X_te.drop(columns=drop, errors="ignore")

    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    X_te = X_te.replace([np.inf, -np.inf], np.nan)

    cat_features = [c for c in LGBM_NATIVE_CAT_COLS if c in X_tr.columns]
    return X_tr, X_te, cat_features


def prepare_catboost_data(X_all, X_test):
    """Prepare features for CatBoost: native categorical handling for all cat columns."""
    X_tr = X_all.copy()
    X_te = X_test.copy()

    drop = [c for c in X_tr.columns if c in DROP_COLS]
    X_tr = X_tr.drop(columns=drop, errors="ignore")
    X_te = X_te.drop(columns=drop, errors="ignore")

    cat_cols = [c for c in CATBOOST_CAT_COLS if c in X_tr.columns]
    for col in cat_cols:
        X_tr[col] = X_tr[col].fillna("_missing_").astype(str)
        X_te[col] = X_te[col].fillna("_missing_").astype(str)

    numeric_cols = [c for c in X_tr.columns if c not in cat_cols]
    X_tr[numeric_cols] = X_tr[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X_te[numeric_cols] = X_te[numeric_cols].replace([np.inf, -np.inf], np.nan)

    X_te = X_te.reindex(columns=X_tr.columns)
    return X_tr, X_te, cat_cols


def run_lgbm(X_all, y, X_test, cat_features):
    """5-fold LightGBM, returns OOF probas and averaged test probas."""
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l1": 0.05,
        "lambda_l2": 1.0,
        "min_gain_to_split": 0.01,
        "verbose": -1,
        "random_state": RANDOM_STATE,
        "n_estimators": 3000,
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X_all))
    test_preds = np.zeros(len(X_test))

    print("\n--- LightGBM ---")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y)):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_all.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X_all.iloc[va_idx], y.iloc[va_idx])],
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
            categorical_feature=cat_features,
        )
        p = model.predict_proba(X_all.iloc[va_idx])[:, 1]
        oof[va_idx] = p
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"  Fold {fold}: PR-AUC={average_precision_score(y.iloc[va_idx], p):.5f}  iter={model.best_iteration_}")

    roc = roc_auc_score(y, oof)
    pr = average_precision_score(y, oof)
    print(f"  OOF: ROC-AUC={roc:.5f}  PR-AUC={pr:.5f}")
    return oof, test_preds


def run_catboost(X_all, y, X_test, cat_cols):
    """5-fold CatBoost, returns OOF probas and averaged test probas."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X_all))
    test_preds = np.zeros(len(X_test))

    print("\n--- CatBoost ---")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y)):
        train_pool = Pool(X_all.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols)
        valid_pool = Pool(X_all.iloc[va_idx], y.iloc[va_idx], cat_features=cat_cols)
        test_pool = Pool(X_test, cat_features=cat_cols)

        model = CatBoostClassifier(
            iterations=3000,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="SqrtBalanced",
            random_seed=RANDOM_STATE + fold,
            od_type="Iter",
            od_wait=150,
            verbose=0,
            allow_writing_files=False,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        p = model.predict_proba(valid_pool)[:, 1]
        oof[va_idx] = p
        test_preds += model.predict_proba(test_pool)[:, 1] / N_FOLDS
        print(f"  Fold {fold}: PR-AUC={average_precision_score(y.iloc[va_idx], p):.5f}  iter={model.best_iteration_}")

    roc = roc_auc_score(y, oof)
    pr = average_precision_score(y, oof)
    print(f"  OOF: ROC-AUC={roc:.5f}  PR-AUC={pr:.5f}")
    return oof, test_preds


def run(args):
    start = time.perf_counter()

    train_df = pd.read_csv(args.train, low_memory=False)
    test_df = pd.read_csv(args.test, low_memory=False)
    y = train_df[TARGET].astype(int)
    print(f"[load] train={train_df.shape} test={test_df.shape} pos_rate={y.mean():.4f}")

    airport_map = get_airport_map(pd.concat([train_df, test_df], axis=0))
    X_all = make_features(train_df, airport_map)
    X_test = make_features(test_df, airport_map)
    print(f"[features] raw feature columns: {X_all.shape[1]}")

    # LightGBM branch
    X_lgb_train, X_lgb_test, lgb_cats = prepare_lgbm_data(X_all, y, X_test)
    print(f"[lgbm] features={X_lgb_train.shape[1]}  native_cat={len(lgb_cats)}")
    lgb_oof, lgb_test = run_lgbm(X_lgb_train, y, X_lgb_test, lgb_cats)

    # CatBoost branch
    X_cb_train, X_cb_test, cb_cats = prepare_catboost_data(X_all, X_test)
    print(f"[catboost] features={X_cb_train.shape[1]}  cat_features={len(cb_cats)}")
    cb_oof, cb_test = run_catboost(X_cb_train, y, X_cb_test, cb_cats)

    # Ensemble: simple average
    ens_oof = 0.5 * lgb_oof + 0.5 * cb_oof
    ens_test = 0.5 * lgb_test + 0.5 * cb_test

    print("\n--- Ensemble (0.5 LGB + 0.5 CB) ---")
    ens_roc = roc_auc_score(y, ens_oof)
    ens_pr = average_precision_score(y, ens_oof)
    print(f"  OOF: ROC-AUC={ens_roc:.5f}  PR-AUC={ens_pr:.5f}")

    # Optimal F1 threshold
    precision, recall, thresholds = precision_recall_curve(y, ens_oof)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1[:-1]))
    best_thresh = float(thresholds[best_idx])
    print(f"  Best F1={f1[best_idx]:.5f}  threshold={best_thresh:.4f}")
    print(f"  precision={precision[best_idx]:.3f}  recall={recall[best_idx]:.3f}")

    oof_pred = (ens_oof >= best_thresh).astype(int)
    print(f"\n{confusion_matrix(y, oof_pred)}")
    print(classification_report(y, oof_pred, digits=4))

    # Write submission
    if args.binary:
        preds = (ens_test >= best_thresh).astype(int)
        submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: preds})
        print(f"[submit] pos_rate={preds.mean():.4f}")
    else:
        submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: ens_test})
        print(f"[submit] prob_mean={ens_test.mean():.4f}  min={ens_test.min():.4f}  max={ens_test.max():.4f}")

    submission.to_csv(args.out, index=False)
    print(f"\n[done] wrote {args.out}  time={time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--out", type=Path, default=Path("lgbm_submission.csv"))
    parser.add_argument("--binary", action="store_true", help="output 0/1 instead of probabilities")
    args = parser.parse_args()
    run(args)
