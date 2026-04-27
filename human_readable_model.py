#!/usr/bin/env python3
"""A simpler wildlife-strike damage model.

This script is based on improved_model.py, but the feature engineering is kept
plain on purpose. The idea is to use features that are easy to explain:

- date parts like year/month/day-of-year
- time as minutes after midnight
- numeric columns as numbers, with a few missing-value flags
- one-hot encoding for categorical columns

Run:
    python human_readable_model.py

Optional small cross-validation search:
    python human_readable_model.py --cv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"
RANDOM_STATE = 42

# Columns that are IDs, narrative text, or direct post-strike proxies.
DROP_COLS = [
    ID_COL,
    TARGET,
    "LOCATION",
    "BIRD_BAND_NUMBER",
    "REG",
    "FLT",
    "AIRCRAFT",
    "LUPDATE",
    "REMARKS",
    "COMMENTS",
    "NUM_STRUCK",
    "REMAINS_COLLECTED",
    "REMAINS_SENT",
    "TRANSFER",
    "SOURCE",
    "PERSON",
]

# These columns should be treated as numbers even if pandas reads them as text.
NUMBER_COLS = [
    "LATITUDE",
    "LONGITUDE",
    "HEIGHT",
    "SPEED",
    "DISTANCE",
    "NUM_SEEN",
    "OUT_OF_RANGE_SPECIES",
    "AC_MASS",
    "NUM_ENGS",
    "ENG_1_POS",
    "ENG_2_POS",
    "ENG_3_POS",
    "ENG_4_POS",
]

# These columns get a simple 0/1 flag when the original value was blank.
MISSING_FLAG_COLS = [
    "HEIGHT",
    "SPEED",
    "DISTANCE",
    "TIME",
    "PHASE_OF_FLIGHT",
    "SKY",
    "PRECIPITATION",
    "SIZE",
]

# I keep only the more common categoricals as one-hot columns so the feature
# table stays readable and does not explode into thousands of dummy columns.
CATEGORY_COLS = [
    "TIME_OF_DAY",
    "STATE",
    "FAAREGION",
    "OPID",
    "AC_CLASS",
    "TYPE_ENG",
    "PHASE_OF_FLIGHT",
    "SKY",
    "WARNED",
    "SIZE",
    "PRECIPITATION",
    "AMA",
    "AMO",
    "EMA",
    "EMO",
    "LAT_BIN",
    "LON_BIN",
]

# Clean high-cardinality context columns. These are not narrative free text; they
# identify things like airport, runway, operator, and species.
HIGH_CARD_COLS = [
    "AIRPORT_ID",
    "SPECIES_ID",
    "OPERATOR",
    "RUNWAY",
]

# Smoothed historical-rate features are built out-of-fold for training rows, so
# validation rows do not leak into the encodings used to score them.
TARGET_RATE_COLS = [
    "AIRPORT_ID",
    "SPECIES_ID",
    "OPERATOR",
    "RUNWAY",
    "OPID",
    "STATE",
    "FAAREGION",
    "AC_CLASS",
    "TYPE_ENG",
    "PHASE_OF_FLIGHT",
    "SKY",
    "PRECIPITATION",
    "WARNED",
    "SIZE",
]


def read_data(train_path, test_path):
    """Load the train and test CSV files."""
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    return train, test


def time_to_minutes(series):
    """Turn times like '13:45' into minutes after midnight."""
    parts = series.fillna("").astype(str).str.split(":", n=1, expand=True)
    hour = pd.to_numeric(parts[0], errors="coerce")
    if 1 in parts.columns:
        minute = pd.to_numeric(parts[1], errors="coerce").fillna(0)
    else:
        minute = pd.Series(0, index=series.index)
    return hour * 60 + minute


def add_date_features(out, df):
    """Add simple date columns from INCIDENT_DATE."""
    dates = pd.to_datetime(df["INCIDENT_DATE"], format="mixed", errors="coerce")
    out["YEAR"] = dates.dt.year
    out["MONTH"] = dates.dt.month
    out["DAY_OF_YEAR"] = dates.dt.dayofyear
    month = dates.dt.month.fillna(6)
    day_of_year = dates.dt.dayofyear.fillna(180)
    out["MONTH_SIN"] = np.sin(2 * np.pi * month / 12.0)
    out["MONTH_COS"] = np.cos(2 * np.pi * month / 12.0)
    out["DAY_OF_YEAR_SIN"] = np.sin(2 * np.pi * day_of_year / 366.0)
    out["DAY_OF_YEAR_COS"] = np.cos(2 * np.pi * day_of_year / 366.0)
    return out


def add_time_features(out, df):
    """Add time as one numeric column."""
    if "TIME" in df.columns:
        minutes = time_to_minutes(df["TIME"])
        out["MINUTES"] = minutes
        filled = minutes.fillna(720)
        out["TIME_SIN"] = np.sin(2 * np.pi * filled / 1440.0)
        out["TIME_COS"] = np.cos(2 * np.pi * filled / 1440.0)
    return out


def add_numeric_features(out, df):
    """Copy numeric columns and add a few simple log versions."""
    for col in NUMBER_COLS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    # Log versions help because these columns have a few very large values.
    for col in ["HEIGHT", "SPEED", "DISTANCE"]:
        if col in out.columns:
            out["LOG_" + col] = np.log1p(out[col].clip(lower=0))

    if "NUM_SEEN" in out.columns:
        out["LOG_NUM_SEEN"] = np.log1p(out["NUM_SEEN"].clip(lower=0))

    if "LATITUDE" in out.columns and "LONGITUDE" in out.columns:
        lat = out["LATITUDE"]
        lon = out["LONGITUDE"]
        out["ABS_LATITUDE"] = lat.abs()
        out["ABS_LONGITUDE"] = lon.abs()
        out["DIST_FROM_US_CENTER"] = np.sqrt((lat - 39.5) ** 2 + (lon + 98.35) ** 2)

    speed = out.get("SPEED", pd.Series(np.nan, index=df.index))
    mass = pd.to_numeric(df.get("AC_MASS", pd.Series(np.nan, index=df.index)), errors="coerce")
    size = df.get("SIZE", pd.Series("", index=df.index)).map({"Small": 1, "Medium": 2, "Large": 3}).fillna(1)
    out["ENERGY_PROXY"] = mass * speed.pow(2)
    out["MOMENTUM_PROXY"] = mass * speed
    out["SIZE_SPEED"] = size * speed

    return out


def add_missing_flags(out, df):
    """Mark a few important columns that were blank in the original data."""
    for col in MISSING_FLAG_COLS:
        if col in df.columns:
            out[col + "_MISSING"] = df[col].isna().astype(int)
    return out


def add_category_features(out, df):
    """Add selected categorical columns before one-hot encoding."""
    if "LATITUDE" in df.columns:
        out["LAT_BIN"] = pd.to_numeric(df["LATITUDE"], errors="coerce").round().astype("Int64").astype(str)
    if "LONGITUDE" in df.columns:
        out["LON_BIN"] = pd.to_numeric(df["LONGITUDE"], errors="coerce").round().astype("Int64").astype(str)

    for col in CATEGORY_COLS:
        if col in df.columns:
            out[col] = df[col].fillna("Unknown").astype(str)
    for col in HIGH_CARD_COLS:
        if col in df.columns:
            out[col] = df[col].fillna("Unknown").astype(str)
    return out


def make_features(df):
    """Build the plain feature table before final encoding/filling."""
    out = pd.DataFrame(index=df.index)
    out = add_date_features(out, df)
    out = add_time_features(out, df)
    out = add_numeric_features(out, df)
    out = add_missing_flags(out, df)
    out = add_category_features(out, df)
    return out.drop(columns=DROP_COLS, errors="ignore")


def limit_categories(X_train, X_val=None, X_test=None, top_n=100):
    """Keep one-hot columns readable by grouping rare values as Other."""
    cat_cols = [c for c in CATEGORY_COLS if c in X_train.columns]
    for col in cat_cols:
        top_values = X_train[col].value_counts().index[:top_n]
        X_train[col] = X_train[col].where(X_train[col].isin(top_values), "Other")
        if X_val is not None and col in X_val.columns:
            X_val[col] = X_val[col].where(X_val[col].isin(top_values), "Other")
        if X_test is not None and col in X_test.columns:
            X_test[col] = X_test[col].where(X_test[col].isin(top_values), "Other")
    return X_train, X_val, X_test


def smoothed_rates(keys, y, global_rate, smooth=80):
    """Return a smoothed positive-rate map for one categorical key."""
    stats = pd.DataFrame({"key": keys, "y": y}).groupby("key")["y"].agg(["sum", "count"])
    return (stats["sum"] + smooth * global_rate) / (stats["count"] + smooth)


def add_history_features(X_train, X_val, X_test, train_df, val_df, test_df, y_train):
    """Add clean frequency and out-of-fold historical-rate features."""
    global_rate = float(y_train.mean())
    y_values = y_train.to_numpy()
    new_train = {}
    new_val = {} if X_val is not None else None
    new_test = {}

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for col in TARGET_RATE_COLS:
        if col not in train_df.columns:
            continue

        train_keys = train_df[col].fillna("Unknown").astype(str).reset_index(drop=True)
        oof = np.full(len(train_df), global_rate, dtype="float32")
        for fit_idx, hold_idx in splitter.split(train_df, y_train):
            rates = smoothed_rates(train_keys.iloc[fit_idx], y_values[fit_idx], global_rate)
            hold_keys = train_keys.iloc[hold_idx]
            oof[hold_idx] = hold_keys.map(rates).fillna(global_rate).astype("float32").to_numpy()

        rates = smoothed_rates(train_keys, y_values, global_rate)
        new_train["RATE_" + col] = oof
        if X_val is not None and val_df is not None and col in val_df.columns:
            new_val["RATE_" + col] = (
                val_df[col].fillna("Unknown").astype(str).map(rates).fillna(global_rate).astype("float32").to_numpy()
            )
        if col in test_df.columns:
            new_test["RATE_" + col] = (
                test_df[col].fillna("Unknown").astype(str).map(rates).fillna(global_rate).astype("float32").to_numpy()
            )

    for col in HIGH_CARD_COLS:
        if col not in train_df.columns:
            continue
        counts = train_df[col].fillna("Unknown").astype(str).value_counts()
        total = len(train_df)
        for raw_df, store in [(train_df, new_train), (val_df, new_val), (test_df, new_test)]:
            if raw_df is None or store is None or col not in raw_df.columns:
                continue
            values = raw_df[col].fillna("Unknown").astype(str).map(counts).fillna(0).astype("float32")
            store["LOG_COUNT_" + col] = np.log1p(values).to_numpy()
            store["FREQ_" + col] = (values / total).to_numpy()

    X_train = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(new_train)], axis=1)
    if X_val is not None and new_val is not None:
        X_val = pd.concat([X_val.reset_index(drop=True), pd.DataFrame(new_val)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(new_test)], axis=1)

    return X_train, X_val, X_test


def prepare_train_val_test(train_df, val_df, test_df, y_train):
    """Create matching train, validation, and test feature matrices."""
    X_train = make_features(train_df)
    X_val = make_features(val_df)
    X_test = make_features(test_df)

    X_train, X_val, X_test = limit_categories(X_train, X_val, X_test)
    X_train, X_val, X_test = add_history_features(X_train, X_val, X_test, train_df, val_df, test_df, y_train)
    X_train = X_train.drop(columns=HIGH_CARD_COLS, errors="ignore")
    X_val = X_val.drop(columns=HIGH_CARD_COLS, errors="ignore")
    X_test = X_test.drop(columns=HIGH_CARD_COLS, errors="ignore")

    combined = pd.concat([X_train, X_val, X_test], axis=0)
    combined = pd.get_dummies(combined, columns=[c for c in CATEGORY_COLS if c in combined], dtype=float)
    combined = combined.replace([np.inf, -np.inf], np.nan)

    train_end = len(X_train)
    val_end = train_end + len(X_val)
    X_train = combined.iloc[:train_end].copy()
    X_val = combined.iloc[train_end:val_end].copy()
    X_test = combined.iloc[val_end:].copy()

    medians = X_train.median(numeric_only=True).fillna(0)
    X_train = X_train.fillna(medians).fillna(0)
    X_val = X_val.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)

    return X_train.astype("float32"), X_val.astype("float32"), X_test.astype("float32")


def prepare_full_train_test(train_df, test_df, y_train):
    """Create matching full-train and test matrices for the final submission."""
    X_train = make_features(train_df)
    X_test = make_features(test_df)

    X_train, _, X_test = limit_categories(X_train, None, X_test)
    X_train, _, X_test = add_history_features(X_train, None, X_test, train_df, None, test_df, y_train)
    X_train = X_train.drop(columns=HIGH_CARD_COLS, errors="ignore")
    X_test = X_test.drop(columns=HIGH_CARD_COLS, errors="ignore")

    combined = pd.concat([X_train, X_test], axis=0)
    combined = pd.get_dummies(combined, columns=[c for c in CATEGORY_COLS if c in combined], dtype=float)
    combined = combined.replace([np.inf, -np.inf], np.nan)

    X_train = combined.iloc[: len(X_train)].copy()
    X_test = combined.iloc[len(X_train) :].copy()

    medians = X_train.median(numeric_only=True).fillna(0)
    X_train = X_train.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)

    return X_train.astype("float32"), X_test.astype("float32")


def build_model(params=None):
    """Build the boosted-tree model."""
    settings = {
        "max_iter": 650,
        "learning_rate": 0.04,
        "max_leaf_nodes": 63,
        "l2_regularization": 0.05,
    }
    if params:
        settings.update(params)

    return HistGradientBoostingClassifier(
        **settings,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )


def best_threshold(y_true, proba):
    """Choose the probability cutoff that gives the best F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best]), float(f1[best]), float(precision[best]), float(recall[best])


def try_cv(train_df, sample_rows):
    """Try a few parameter settings with 3-fold cross-validation."""
    if sample_rows and len(train_df) > sample_rows:
        train_df, _ = train_test_split(
            train_df,
            train_size=sample_rows,
            stratify=train_df[TARGET],
            random_state=RANDOM_STATE,
        )
        print(f"[cv] using a stratified sample of {len(train_df)} rows")

    y = train_df[TARGET].astype(int)
    settings_to_try = [
        {"max_iter": 450, "learning_rate": 0.05, "max_leaf_nodes": 31},
        {"max_iter": 650, "learning_rate": 0.04, "max_leaf_nodes": 63},
        {"max_iter": 800, "learning_rate": 0.03, "max_leaf_nodes": 63, "l2_regularization": 0.08},
    ]

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    best_params = settings_to_try[0]
    best_score = -1

    for params in settings_to_try:
        scores = []
        print("[cv] trying", params)
        for train_idx, val_idx in splitter.split(train_df, y):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]
            y_fold_train = fold_train[TARGET].astype(int)
            X_train, X_val, _ = prepare_train_val_test(fold_train, fold_val, fold_val, y_fold_train)

            model = build_model(params)
            model.fit(X_train, y_fold_train)
            proba = model.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(fold_val[TARGET], proba))

        mean_score = float(np.mean(scores))
        print(f"[cv] PR-AUC mean={mean_score:.5f} scores={[round(s, 5) for s in scores]}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    print("[cv] best params:", best_params)
    return best_params


def evaluate_holdout(train_df, test_df, params):
    """Score the model on an 80/20 holdout split."""
    train_part, val_part = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df[TARGET],
        random_state=RANDOM_STATE,
    )

    y_train = train_part[TARGET].astype(int)
    y_val = val_part[TARGET].astype(int)
    X_train, X_val, _ = prepare_train_val_test(train_part, val_part, test_df, y_train)

    print(f"[features] train={X_train.shape} val={X_val.shape}")
    model = build_model(params)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)[:, 1]
    threshold, f1, precision, recall = best_threshold(y_val, proba)
    pred = (proba >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "pr_auc": float(average_precision_score(y_val, proba)),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
    }

    print("[holdout]")
    print(f"ROC-AUC : {metrics['roc_auc']:.5f}")
    print(f"PR-AUC  : {metrics['pr_auc']:.5f}")
    print(
        f"best threshold: {threshold:.5f}  "
        f"F1={f1:.5f}  precision={precision:.3f}  recall={recall:.3f}"
    )
    print(confusion_matrix(y_val, pred))
    print(classification_report(y_val, pred, digits=4))
    return threshold, metrics


def make_submission(train_df, test_df, out_path, params, threshold):
    """Refit on all training rows and write the Kaggle submission."""
    y_train = train_df[TARGET].astype(int)
    X_train, X_test = prepare_full_train_test(train_df, test_df, y_train)

    print(f"[final] train={X_train.shape} test={X_test.shape}")
    model = build_model(params)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    submission = pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET: pred})
    submission.to_csv(out_path, index=False)
    print(f"[submit] wrote {out_path} shape={submission.shape} pos_rate={pred.mean():.5f}")


def run(args):
    start = time.perf_counter()
    train_df, test_df = read_data(args.train, args.test)
    print(f"[load] train={train_df.shape} test={test_df.shape}")
    print(f"[target] positive rate={train_df[TARGET].mean():.5f}")

    params = try_cv(train_df, args.cv_sample) if args.cv else None
    threshold, _ = evaluate_holdout(train_df, test_df, params)
    make_submission(train_df, test_df, args.out, params, threshold)

    print(f"[done] total time={time.perf_counter() - start:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--out", type=Path, default=Path("human_submission.csv"))
    parser.add_argument("--cv", action="store_true", help="try a small 3-fold parameter search")
    parser.add_argument("--cv-sample", type=int, default=30000, help="rows to use for the CV search")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
