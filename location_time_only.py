#!/usr/bin/env python3
"""
Location + DateTime only model.

Tests whether geographic and temporal features alone predict damage.
Uses ONLY: AIRPORT_ID, LATITUDE, LONGITUDE, INCIDENT_DATE, TIME, TIME_OF_DAY.
Everything else is excluded.

Run:
    python -u location_time_only.py
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"
RANDOM_STATE = 42
N_FOLDS = 5


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
    df = impute_coords(df, airport_map)
    out = pd.DataFrame(index=df.index)

    # --- Date features ---
    dates = pd.to_datetime(df["INCIDENT_DATE"], format="mixed", errors="coerce")
    out["YEAR"] = dates.dt.year
    out["MONTH"] = dates.dt.month
    month = dates.dt.month.fillna(6)
    out["MONTH_SIN"] = np.sin(2 * np.pi * month / 12.0)
    out["MONTH_COS"] = np.cos(2 * np.pi * month / 12.0)
    doy = dates.dt.dayofyear.fillna(180)
    out["DOY_SIN"] = np.sin(2 * np.pi * doy / 366.0)
    out["DOY_COS"] = np.cos(2 * np.pi * doy / 366.0)
    out["DAY_OF_WEEK"] = dates.dt.dayofweek

    # --- Time features ---
    if "TIME" in df.columns:
        minutes = time_to_minutes(df["TIME"])
        out["MINUTES"] = minutes
        filled = minutes.fillna(720)
        out["TIME_SIN"] = np.sin(2 * np.pi * filled / 1440.0)
        out["TIME_COS"] = np.cos(2 * np.pi * filled / 1440.0)
        out["TIME_MISSING"] = df["TIME"].isna().astype("int8")

    # TIME_OF_DAY as categorical
    if "TIME_OF_DAY" in df.columns:
        out["TIME_OF_DAY"] = df["TIME_OF_DAY"].fillna("_missing_").astype(str)

    # --- Location features ---
    lat = df["LATITUDE"]
    lon = df["LONGITUDE"]

    # Raw
    out["LATITUDE"] = lat
    out["LONGITUDE"] = lon
    out["LAT_MISSING"] = lat.isna().astype("int8")

    # Sin/cos encoding (treats lat/lon as angles)
    out["LAT_SIN"] = np.sin(np.deg2rad(lat))
    out["LAT_COS"] = np.cos(np.deg2rad(lat))
    out["LON_SIN"] = np.sin(np.deg2rad(lon))
    out["LON_COS"] = np.cos(np.deg2rad(lon))

    # Binned (1-degree bins)
    out["LAT_BIN"] = lat.round(0).fillna(-999).astype(int).astype(str)
    out["LON_BIN"] = lon.round(0).fillna(-999).astype(int).astype(str)

    # Coarser bins (5-degree)
    out["LAT_BIN5"] = (lat / 5).round(0).fillna(-999).astype(int).astype(str)
    out["LON_BIN5"] = (lon / 5).round(0).fillna(-999).astype(int).astype(str)

    # Distance from US center
    out["DIST_FROM_US_CENTER"] = np.sqrt((lat - 39.5) ** 2 + (lon + 98.35) ** 2)

    # Interaction
    out["LAT_X_LON"] = lat * lon

    # AIRPORT_ID as categorical
    if "AIRPORT_ID" in df.columns:
        out["AIRPORT_ID"] = df["AIRPORT_ID"].fillna("_missing_").astype(str)

    return out


def run():
    start = time.perf_counter()

    train_df = pd.read_csv("train.csv", low_memory=False)
    test_df = pd.read_csv("test.csv", low_memory=False)
    y = train_df[TARGET].astype(int)
    print(f"[load] train={train_df.shape} test={test_df.shape} pos_rate={y.mean():.4f}")

    airport_map = get_airport_map(pd.concat([train_df, test_df], axis=0))
    X_all = make_features(train_df, airport_map)
    X_test = make_features(test_df, airport_map)

    # CatBoost categorical columns
    cat_cols = ["TIME_OF_DAY", "AIRPORT_ID", "LAT_BIN", "LON_BIN", "LAT_BIN5", "LON_BIN5"]
    cat_cols = [c for c in cat_cols if c in X_all.columns]

    # Numeric columns (everything except categoricals)
    num_cols = [c for c in X_all.columns if c not in cat_cols]
    X_all[num_cols] = X_all[num_cols].replace([np.inf, -np.inf], np.nan)
    X_test[num_cols] = X_test[num_cols].replace([np.inf, -np.inf], np.nan)
    X_test = X_test.reindex(columns=X_all.columns)

    print(f"[features] total={X_all.shape[1]} categorical={len(cat_cols)} numeric={len(num_cols)}")
    print(f"[features] columns: {list(X_all.columns)}")

    # 5-fold CatBoost (handles categoricals natively)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X_all))
    test_preds = np.zeros(len(X_test))

    print("\n--- CatBoost (location+time only) ---")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y)):
        train_pool = Pool(X_all.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols)
        valid_pool = Pool(X_all.iloc[va_idx], y.iloc[va_idx], cat_features=cat_cols)
        test_pool = Pool(X_test, cat_features=cat_cols)

        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
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

        roc = roc_auc_score(y.iloc[va_idx], p)
        pr = average_precision_score(y.iloc[va_idx], p)
        print(f"  Fold {fold}: ROC-AUC={roc:.5f}  PR-AUC={pr:.5f}  iter={model.best_iteration_}")

    # OOF metrics
    oof_roc = roc_auc_score(y, oof)
    oof_pr = average_precision_score(y, oof)
    print(f"\n[OOF] ROC-AUC = {oof_roc:.5f}")
    print(f"[OOF] PR-AUC  = {oof_pr:.5f}")

    precision, recall, thresholds = precision_recall_curve(y, oof)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1[:-1]))
    print(f"[OOF] Best F1 = {f1[best_idx]:.5f}  threshold={thresholds[best_idx]:.4f}")

    # Feature importance
    imp = model.get_feature_importance()
    feat_names = X_all.columns.tolist()
    imp_df = pd.Series(imp, index=feat_names).sort_values(ascending=False)
    print("\nFeature importances:")
    print(imp_df.to_string())

    # Write submission
    preds = (test_preds >= thresholds[best_idx]).astype(int)
    pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: preds}).to_csv("location_time_submission.csv", index=False)
    print(f"\n[done] wrote location_time_submission.csv  pos_rate={preds.mean():.4f}  time={time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    run()
