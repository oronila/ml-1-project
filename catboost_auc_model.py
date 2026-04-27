#!/usr/bin/env python3
"""CatBoost probability model for the wildlife-strike leaderboard.

The earlier scripts write hard 0/1 labels after threshold tuning. If the
leaderboard metric is ROC-AUC, that caps the score because AUC needs ranked
probabilities. This script writes probabilities by default.

Run:
    python catboost_auc_model.py

Quick smoke test:
    python catboost_auc_model.py --sample-rows 10000 --iterations 200
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"
RANDOM_STATE = 42

# Text and direct post-strike/reporting proxies made local validation look much
# better than the leaderboard. Keep the default model focused on structured,
# pre/submission-stable fields.
DROP_COLS = {
    ID_COL,
    TARGET,
    "REMARKS",
    "COMMENTS",
    "LOCATION",
    "BIRD_BAND_NUMBER",
    "REG",
    "FLT",
    "AIRCRAFT",
    "AIRPORT",
    "SPECIES",
    "LUPDATE",
    "NUM_STRUCK",
    "REMAINS_COLLECTED",
    "REMAINS_SENT",
    "SOURCE",
    "PERSON",
    "TRANSFER",
}

CAT_COLS = [
    "TIME_OF_DAY",
    "AIRPORT_ID",
    "RUNWAY",
    "STATE",
    "FAAREGION",
    "OPID",
    "OPERATOR",
    "AMA",
    "AMO",
    "EMA",
    "EMO",
    "AC_CLASS",
    "TYPE_ENG",
    "PHASE_OF_FLIGHT",
    "SKY",
    "PRECIPITATION",
    "SPECIES_ID",
    "WARNED",
    "SIZE",
    "ENROUTE_STATE",
    "LAT_BIN",
    "LON_BIN",
]

NUM_COLS = [
    "INCIDENT_MONTH",
    "INCIDENT_YEAR",
    "LATITUDE",
    "LONGITUDE",
    "AC_MASS",
    "NUM_ENGS",
    "ENG_1_POS",
    "ENG_2_POS",
    "ENG_3_POS",
    "ENG_4_POS",
    "HEIGHT",
    "SPEED",
    "DISTANCE",
    "OUT_OF_RANGE_SPECIES",
    "NUM_SEEN",
]

MISSING_FLAG_COLS = [
    "LATITUDE",
    "LONGITUDE",
    "HEIGHT",
    "SPEED",
    "DISTANCE",
    "TIME",
    "PHASE_OF_FLIGHT",
    "SKY",
    "PRECIPITATION",
    "SIZE",
]


def read_data(train_path, test_path, sample_rows=None):
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)

    if sample_rows and len(train) > sample_rows:
        train, _ = train_test_split(
            train,
            train_size=sample_rows,
            stratify=train[TARGET],
            random_state=RANDOM_STATE,
        )
        train = train.reset_index(drop=True)

    return train, test


def get_airport_map(df):
    df = df.copy()
    for col in ["LATITUDE", "LONGITUDE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df[
        df["AIRPORT_ID"].notna()
        & (df["AIRPORT_ID"] != "ZZZZ")
        & df["LATITUDE"].notna()
        & df["LONGITUDE"].notna()
    ]
    return valid.groupby("AIRPORT_ID")[["LATITUDE", "LONGITUDE"]].median().to_dict("index")


def recover_airport_id(df, airport_map):
    """Recover K-style airport ids from LOCATION only for coordinate imputation."""
    if "LOCATION" not in df.columns or "AIRPORT_ID" not in df.columns:
        return df

    df = df.copy()
    extracted = df["LOCATION"].fillna("").astype(str).str.extract(r"\b(K[A-Z]{3})\b", expand=False)
    recover = (df["AIRPORT_ID"] == "ZZZZ") & extracted.notna() & extracted.isin(airport_map.keys())
    df.loc[recover, "AIRPORT_ID"] = extracted[recover]
    return df


def time_to_minutes(series):
    parts = series.fillna("").astype(str).str.split(":", n=1, expand=True)
    hour = pd.to_numeric(parts[0], errors="coerce")
    minute = pd.to_numeric(parts[1], errors="coerce").fillna(0) if 1 in parts.columns else 0
    return hour * 60 + minute


def add_date_time_features(out, df):
    dates = pd.to_datetime(df["INCIDENT_DATE"], format="mixed", errors="coerce")
    year = dates.dt.year
    month = dates.dt.month
    day_of_year = dates.dt.dayofyear

    out["YEAR"] = year
    out["MONTH"] = month
    out["DAY_OF_YEAR"] = day_of_year
    out["YEAR_SINCE_1990"] = year - 1990
    out["MONTH_SIN"] = np.sin(2 * np.pi * month.fillna(6) / 12.0)
    out["MONTH_COS"] = np.cos(2 * np.pi * month.fillna(6) / 12.0)
    out["DAY_OF_YEAR_SIN"] = np.sin(2 * np.pi * day_of_year.fillna(180) / 366.0)
    out["DAY_OF_YEAR_COS"] = np.cos(2 * np.pi * day_of_year.fillna(180) / 366.0)

    if "TIME" in df.columns:
        minutes = time_to_minutes(df["TIME"])
        out["MINUTES"] = minutes
        out["TIME_SIN"] = np.sin(2 * np.pi * minutes.fillna(720) / 1440.0)
        out["TIME_COS"] = np.cos(2 * np.pi * minutes.fillna(720) / 1440.0)


def add_numeric_features(out, df):
    for col in NUM_COLS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["HEIGHT", "SPEED", "DISTANCE", "NUM_SEEN"]:
        if col in out.columns:
            out["LOG_" + col] = np.log1p(out[col].clip(lower=0))

    if "LATITUDE" in out.columns and "LONGITUDE" in out.columns:
        lat = out["LATITUDE"]
        lon = out["LONGITUDE"]
        out["ABS_LATITUDE"] = lat.abs()
        out["ABS_LONGITUDE"] = lon.abs()
        out["LAT_X_LON"] = lat * lon
        out["DIST_FROM_US_CENTER"] = np.sqrt((lat - 39.5) ** 2 + (lon + 98.35) ** 2)

    speed = out.get("SPEED", pd.Series(np.nan, index=df.index))
    mass = out.get("AC_MASS", pd.Series(np.nan, index=df.index))
    size = df.get("SIZE", pd.Series("", index=df.index)).map({"Small": 1, "Medium": 2, "Large": 3})
    out["ENERGY_PROXY"] = mass * speed.pow(2)
    out["MOMENTUM_PROXY"] = mass * speed
    out["SIZE_SPEED"] = size.fillna(1) * speed


def add_categorical_features(out, df):
    if "LATITUDE" in df.columns:
        out["LAT_BIN"] = pd.to_numeric(df["LATITUDE"], errors="coerce").round().astype("Int64").astype(str)
    if "LONGITUDE" in df.columns:
        out["LON_BIN"] = pd.to_numeric(df["LONGITUDE"], errors="coerce").round().astype("Int64").astype(str)

    for col in CAT_COLS:
        if col in df.columns and col not in out.columns:
            out[col] = df[col].fillna("Unknown").astype(str)


def add_missing_flags(out, df):
    for col in MISSING_FLAG_COLS:
        if col in df.columns:
            out[col + "_MISSING"] = df[col].isna().astype("int8")


def make_features(df, airport_map):
    df = recover_airport_id(df, airport_map)
    df = df.copy()

    for coord in ["LATITUDE", "LONGITUDE"]:
        if coord in df.columns:
            lookup = {k: v[coord] for k, v in airport_map.items()}
            df[coord] = pd.to_numeric(df[coord], errors="coerce").fillna(df["AIRPORT_ID"].map(lookup))

    out = pd.DataFrame(index=df.index)
    add_date_time_features(out, df)
    add_numeric_features(out, df)
    add_missing_flags(out, df)
    add_categorical_features(out, df)
    out = out.drop(columns=list(DROP_COLS), errors="ignore")

    cat_cols = [col for col in CAT_COLS if col in out.columns]
    for col in cat_cols:
        out[col] = out[col].fillna("Unknown").astype(str)

    numeric_cols = [col for col in out.columns if col not in cat_cols]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out, cat_cols


def build_model(args):
    return CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="SqrtBalanced",
        random_seed=RANDOM_STATE,
        od_type="Iter",
        od_wait=args.early_stopping_rounds,
        verbose=args.verbose,
        allow_writing_files=False,
    )


def report_validation(model, valid_pool, y_valid):
    proba = model.predict_proba(valid_pool)[:, 1]
    print(f"[valid] ROC-AUC={roc_auc_score(y_valid, proba):.5f}")
    print(f"[valid] PR-AUC ={average_precision_score(y_valid, proba):.5f}")
    print(f"[valid] probability mean={proba.mean():.5f}")


def run(args):
    start = time.perf_counter()
    train_df, test_df = read_data(args.train, args.test, args.sample_rows)
    y = train_df[TARGET].astype(int)
    print(f"[load] train={train_df.shape} test={test_df.shape} target_rate={y.mean():.5f}")

    airport_map = get_airport_map(pd.concat([train_df, test_df], axis=0))
    X, cat_cols = make_features(train_df, airport_map)
    X_test, _ = make_features(test_df, airport_map)
    X_test = X_test.reindex(columns=X.columns)
    print(f"[features] rows={X.shape[0]} cols={X.shape[1]} cat_cols={len(cat_cols)}")

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=args.valid_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    train_pool = Pool(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_cols)
    valid_pool = Pool(X.iloc[valid_idx], y.iloc[valid_idx], cat_features=cat_cols)

    print("[fit] training CatBoost for probability-ranked submission...")
    model = build_model(args)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    report_validation(model, valid_pool, y.iloc[valid_idx])

    print("[final] refitting on all labeled rows...")
    final_pool = Pool(X, y, cat_features=cat_cols)
    test_pool = Pool(X_test, cat_features=cat_cols)
    final_model = build_model(args)
    final_model.fit(final_pool)

    proba_test = final_model.predict_proba(test_pool)[:, 1]
    submission = pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET: proba_test})
    submission.to_csv(args.out, index=False)
    print(
        f"[submit] wrote {args.out} rows={len(submission)} "
        f"prob_mean={proba_test.mean():.5f} min={proba_test.min():.5f} max={proba_test.max():.5f}"
    )
    print(f"[done] total time={time.perf_counter() - start:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--out", type=Path, default=Path("catboost_auc_submission.csv"))
    parser.add_argument("--sample-rows", type=int, default=None, help="train on a stratified sample for smoke tests")
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=8.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=100)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
