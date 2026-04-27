#!/usr/bin/env python3
"""
AdaBoosted Decision Tree Wildlife Strike Model.

This is a copy of geo_text_model.py's feature pipeline, but swaps the
HistGradientBoostingClassifier for AdaBoost over shallow decision trees.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# --- Configuration & Constants ---
TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"
RANDOM_STATE = 42

# Columns to exclude from the final feature set
DROP_COLS = [
    ID_COL, TARGET, "LOCATION", "BIRD_BAND_NUMBER", "REG", "FLT",
    "AIRCRAFT", "LUPDATE", "COMMENTS", "REMARKS", "AIRPORT", "OPERATOR", "SPECIES"
]

# Physical and environmental numeric fields
NUMERIC_COLS = [
    "LATITUDE", "LONGITUDE", "HEIGHT", "SPEED", "DISTANCE",
    "NUM_SEEN", "NUM_STRUCK", "OUT_OF_RANGE_SPECIES",
    "AC_MASS", "NUM_ENGS", "ENG_1_POS", "ENG_2_POS", "ENG_3_POS", "ENG_4_POS",
    "REMAINS_COLLECTED", "REMAINS_SENT"
]

# Standard categorical fields
CATEGORY_COLS = [
    "TIME_OF_DAY", "STATE", "FAAREGION", "OPID", "AC_CLASS", "TYPE_ENG",
    "PHASE_OF_FLIGHT", "SKY", "WARNED", "SIZE", "SOURCE", "PERSON",
    "PRECIPITATION", "AMA", "AMO", "EMA", "EMO"
]

# High-cardinality fields requiring specific encoding
HIGH_CARD_COLS = ["AIRPORT_ID", "SPECIES_ID", "OPERATOR", "RUNWAY"]

# Focused list of high-signal damage keywords (human-interpretable)
DAMAGE_WORDS = [
    "damage", "engine", "wing", "crack",
    "dent", "repair", "replaced"
]


def read_data(train_path, test_path):
    """Loads the strike dataset."""
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    return train, test


def get_airport_map(df):
    """
    Creates a lookup table of AIRPORT_ID -> (LATITUDE, LONGITUDE).
    Uses the median coordinates from all records associated with a specific airport.
    """
    df = df.copy()
    for col in ["LATITUDE", "LONGITUDE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df[
        (df["AIRPORT_ID"].notna())
        & (df["AIRPORT_ID"] != "ZZZZ")
        & (df["LATITUDE"].notna())
        & (df["LONGITUDE"].notna())
    ]

    return valid.groupby("AIRPORT_ID")[["LATITUDE", "LONGITUDE"]].median().to_dict("index")


def impute_missing_data(df, airport_map):
    """
    Fills in missing geographic coordinates using airport medians.

    For rows missing Lat/Lon, look up the median coordinates for the
    identified airport.
    """
    df = df.copy()

    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")

    for coord in ["LATITUDE", "LONGITUDE"]:
        lookup = {k: v[coord] for k, v in airport_map.items()}
        df[coord] = df[coord].fillna(df["AIRPORT_ID"].map(lookup))

    return df


def add_features(df, airport_map):
    """
    Main feature engineering pipeline.
    Constructs temporal, physical, and text-based factors.
    """
    text_corpus = pd.Series("", index=df.index)
    for col in ["REMARKS", "COMMENTS"]:
        if col in df.columns:
            text_corpus = text_corpus + " " + df[col].fillna("").astype(str).str.lower()

    df = impute_missing_data(df, airport_map)
    features = {}

    # 1. Temporal Signal
    incident_date = pd.to_datetime(df["INCIDENT_DATE"], format="mixed", errors="coerce")
    year = incident_date.dt.year.fillna(2015)
    features["YEAR"] = year
    month = incident_date.dt.month.fillna(6)
    features["MONTH"] = month

    # 2. Geography
    lat = df["LATITUDE"]
    lon = df["LONGITUDE"]
    features["LATITUDE"] = lat
    features["LONGITUDE"] = lon
    features["LAT_SIN"] = np.sin(np.deg2rad(lat))
    features["LAT_COS"] = np.cos(np.deg2rad(lat))
    features["LON_SIN"] = np.sin(np.deg2rad(lon))
    features["LON_COS"] = np.cos(np.deg2rad(lon))
    features["LAT_X_LON"] = lat * lon

    # 3. Numeric Physical Evidence
    for col in NUMERIC_COLS:
        if col in df.columns and col not in features:
            features[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["HEIGHT", "SPEED", "DISTANCE"]:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
        features["LOG_" + col] = np.log1p(np.maximum(0, vals))

    # 4. Text-Based Damage Indicators (POST-STRIKE SIGNAL)
    features["TEXT_LENGTH"] = text_corpus.str.len().clip(0, 1000)
    for word in DAMAGE_WORDS:
        features["HAS_" + word.upper()] = text_corpus.str.contains(word, regex=False).astype(int)

    # 5. Categorical Strings
    for col in CATEGORY_COLS + HIGH_CARD_COLS:
        if col in df.columns:
            features[col] = df[col].fillna("Unknown").astype(str)

    return pd.DataFrame(features, index=df.index)


def prepare_matrices(train_df, test_df, airport_map):
    """Handles encoding and final cleanup before training."""
    X_train = add_features(train_df, airport_map)
    X_test = add_features(test_df, airport_map)

    cat_cols = [col for col in CATEGORY_COLS if col in X_train.columns]
    for col in cat_cols:
        top = X_train[col].value_counts().index[:50]
        X_train[col] = X_train[col].where(X_train[col].isin(top), "Other")
        X_test[col] = X_test[col].where(X_test[col].isin(top), "Other")

    combined = pd.concat([X_train, X_test], axis=0)
    combined = pd.get_dummies(combined, columns=cat_cols, dtype=float)

    X_train = combined.iloc[:len(X_train)].copy()
    X_test = combined.iloc[len(X_train):].copy()

    for col in HIGH_CARD_COLS:
        freq = X_train[col].value_counts(normalize=True).to_dict()
        X_train["FREQ_" + col] = X_train[col].map(freq).fillna(0)
        X_test["FREQ_" + col] = X_test[col].map(freq).fillna(0)
        X_train.drop(columns=[col], inplace=True)
        X_test.drop(columns=[col], inplace=True)

    medians = X_train.median(numeric_only=True).fillna(0)
    X_train = X_train.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)

    return X_train.astype("float32"), X_test.astype("float32")


def build_model():
    """AdaBoost over shallow decision trees."""
    base_tree = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=250,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )


def run(args):
    print("[1/5] Loading data...")
    train_full, test_df = read_data(args.train, args.test)

    print("[2/5] Building airport geographic map...")
    # Used by impute_missing_data() to fill missing coordinates from other
    # rows with the same AIRPORT_ID instead of falling back to global medians.
    airport_map = get_airport_map(pd.concat([train_full, test_df]))

    train_part, val_part = train_test_split(
        train_full, test_size=0.2, stratify=train_full[TARGET], random_state=RANDOM_STATE
    )

    print("[3/5] Preparing feature matrices...")
    X_train_eval, X_val_eval = prepare_matrices(train_part, val_part, airport_map)
    y_train_eval = train_part[TARGET].astype(int)
    y_val_eval = val_part[TARGET].astype(int)

    print(f"[4/5] Training AdaBoosted trees on {X_train_eval.shape[1]} features...")
    model = build_model()
    model.fit(X_train_eval, y_train_eval)

    proba = model.predict_proba(X_val_eval)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val_eval, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.argmax(f1[:-1])
    best_thresh = thresholds[best_idx]

    print(f"\nEvaluation Results (Threshold={best_thresh:.3f}):")
    print(f"ROC-AUC : {roc_auc_score(y_val_eval, proba):.5f}")
    print(f"PR-AUC  : {average_precision_score(y_val_eval, proba):.5f}")
    print(f"Best F1 : {f1[best_idx]:.5f}")

    print("\n[5/5] Generating final submission...")
    X_train_final, X_test_final = prepare_matrices(train_full, test_df, airport_map)
    y_train_final = train_full[TARGET].astype(int)

    model.fit(X_train_final, y_train_final)
    proba_final = model.predict_proba(X_test_final)[:, 1]
    preds = (proba_final >= best_thresh).astype(int)

    pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: preds}).to_csv(args.out, index=False)
    print(f"Done! Saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--out", type=Path, default=Path("boosted_tree_submission.csv"))
    args = parser.parse_args()
    run(args)
