#!/usr/bin/env python3
import argparse
import time
import re
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
from sklearn.model_selection import train_test_split

TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"
RANDOM_STATE = 42

DROP_COLS = [ID_COL, TARGET, "LOCATION", "BIRD_BAND_NUMBER", "REG", "FLT", "AIRCRAFT", "LUPDATE", "COMMENTS", "REMARKS"]
NUMBER_COLS = ["LATITUDE", "LONGITUDE", "HEIGHT", "SPEED", "DISTANCE", "NUM_SEEN", "NUM_STRUCK", "OUT_OF_RANGE_SPECIES", "AC_MASS", "NUM_ENGS", "ENG_1_POS", "ENG_2_POS", "ENG_3_POS", "ENG_4_POS", "REMAINS_COLLECTED", "REMAINS_SENT"]
MISSING_FLAG_COLS = ["HEIGHT", "SPEED", "DISTANCE", "TIME", "PHASE_OF_FLIGHT", "SKY", "PRECIPITATION"]
CATEGORY_COLS = ["TIME_OF_DAY", "STATE", "FAAREGION", "OPID", "AC_CLASS", "TYPE_ENG", "PHASE_OF_FLIGHT", "SKY", "WARNED", "SIZE", "SOURCE", "PERSON", "LAT_BIN", "LON_BIN", "PRECIPITATION", "AMA", "AMO", "EMA", "EMO"]
HIGH_CARD_COLS = ["AIRPORT_ID", "SPECIES_ID", "OPERATOR", "RUNWAY"]

DAMAGE_WORDS = [
    "damage", "damaged", "engine", "wing", "windshield", "radome", "nose", 
    "ingest", "ingested", "ingestion", "blood", "dent", "hole", "crack", 
    "blade", "intake", "repair", "replaced", "shattered", "bent", 
    "embedded", "removed", "core", "fuselage", "fan", "fire", "shut",
    "vibration", "destroyed", "replacement", "collapsed", "punctured",
    "smashed", "struck", "hit", "evidence", "found", "impact", "maintenance",
    "out of service", "aborted", "emergency", "inspection",
    "deformation", "broken", "leak", "leaking", "loss", "lost", "failure",
    "leading edge", "species identification", "engine manufacturer", "engine model",
    "data entry", "mor remarks", "heavy", "substantial", "significant",
    "cowling", "spinner", "propeller", "stabilizer", "elevator", "rudder",
    "flap", "slat", "tire", "gear", "brake", "light", "lens",
    "bird strike", "id by", "smithsonian wildlife", "found on", "on rwy", "strike occurred", "flock of",
    "parts of", "pieces of", "multiple strikes", "substantial damage"
]

def read_data(train_path, test_path):
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    return train, test

def get_airport_map(df):
    df = df.copy()
    for col in ["LATITUDE", "LONGITUDE"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    valid = df[(df["AIRPORT_ID"].notna()) & (df["AIRPORT_ID"] != "ZZZZ") & (df["LATITUDE"].notna()) & (df["LONGITUDE"].notna())]
    coords = valid.groupby("AIRPORT_ID")[["LATITUDE", "LONGITUDE"]].median()
    return coords.to_dict('index')

def impute_data(df, airport_map):
    df = df.copy()
    loc = df["LOCATION"].fillna("").astype(str)
    extracted = loc.str.extract(r'\b(K[A-Z]{3})\b', expand=False)
    extracted = extracted.apply(lambda x: "K" + x if isinstance(x, str) and len(x) == 3 else x)
    mask_recover = (df["AIRPORT_ID"] == "ZZZZ") & extracted.notna() & extracted.isin(airport_map.keys())
    df.loc[mask_recover, "AIRPORT_ID"] = extracted[mask_recover]
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors='coerce')
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors='coerce')
    for coord in ["LATITUDE", "LONGITUDE"]:
        m = {k: v[coord] for k, v in airport_map.items()}
        df[coord] = df[coord].fillna(df["AIRPORT_ID"].map(m))
    return df

def time_to_minutes(series):
    parts = series.fillna("").astype(str).str.split(":", n=1, expand=True)
    hour = pd.to_numeric(parts[0], errors="coerce")
    minute = pd.to_numeric(parts[1], errors="coerce").fillna(0) if 1 in parts.columns else pd.Series(0, index=series.index)
    return hour * 60 + minute

def make_features(df, airport_map):
    text = pd.Series("", index=df.index)
    for col in ["REMARKS", "COMMENTS"]:
        if col in df.columns:
            text = text + " " + df[col].fillna("").astype(str).str.lower()
    
    df = impute_data(df, airport_map)
    out_dict = {}
    
    dates = pd.to_datetime(df["INCIDENT_DATE"], format="mixed", errors="coerce")
    year = dates.dt.year.fillna(2015)
    out_dict["YEAR"] = year
    out_dict["YEAR_SQ"] = (year - 1990)**2
    out_dict["MONTH"] = dates.dt.month.fillna(6)
    
    if "TIME" in df.columns:
        out_dict["MINUTES"] = time_to_minutes(df["TIME"])
        
    for col in NUMBER_COLS:
        if col in df.columns:
            out_dict[col] = pd.to_numeric(df[col], errors="coerce")
            
    for col in ["HEIGHT", "SPEED", "DISTANCE"]:
        if col in out_dict:
            vals = pd.to_numeric(out_dict[col], errors='coerce').fillna(0)
            out_dict["LOG_" + col] = np.log1p(np.maximum(0, vals))
            
    if "LATITUDE" in out_dict and "LONGITUDE" in out_dict:
        lat = out_dict["LATITUDE"]
        lon = out_dict["LONGITUDE"]
        out_dict["ABS_LAT"] = np.abs(lat)
        out_dict["ABS_LON"] = np.abs(lon)
        out_dict["DIST_FROM_US_CENTER"] = np.sqrt((lat - 39.5)**2 + (lon + 98.35)**2)
        out_dict["LAT_X_LON"] = lat * lon
        
    for col in MISSING_FLAG_COLS:
        if col in df.columns:
            out_dict[col + "_MISSING"] = df[col].isna().astype(int)
            
    for col in CATEGORY_COLS:
        if col in df.columns:
            out_dict[col] = df[col].fillna("Unknown").astype(str)
            
    for col in HIGH_CARD_COLS:
        if col in df.columns:
            out_dict[col] = df[col].fillna("Unknown").astype(str)
            
    out_dict["TEXT_LENGTH"] = text.str.len().clip(0, 1000)
    for word in DAMAGE_WORDS:
        out_dict["TEXT_HAS_" + word.upper().replace(" ", "_")] = text.str.contains(word, regex=False).astype(int)
            
    out = pd.DataFrame(out_dict, index=df.index)
    return out.drop(columns=DROP_COLS, errors="ignore")

def freq_encode(train_X, val_X, test_X):
    for col in HIGH_CARD_COLS:
        if col in train_X.columns:
            freq = train_X[col].value_counts(normalize=True).to_dict()
            train_X["FREQ_" + col] = train_X[col].map(freq).fillna(0)
            val_X["FREQ_" + col] = val_X[col].map(freq).fillna(0)
            test_X["FREQ_" + col] = test_X[col].map(freq).fillna(0)
    return train_X.drop(columns=HIGH_CARD_COLS, errors="ignore"), val_X.drop(columns=HIGH_CARD_COLS, errors="ignore"), test_X.drop(columns=HIGH_CARD_COLS, errors="ignore")

def prepare_train_val_test(train_df, val_df, test_df, airport_map):
    X_train = make_features(train_df, airport_map)
    X_val = make_features(val_df, airport_map)
    X_test = make_features(test_df, airport_map)
    
    cat_cols_to_ohe = [c for c in CATEGORY_COLS if c in X_train.columns]
    for col in cat_cols_to_ohe:
        top = X_train[col].value_counts().index[:50]
        X_train[col] = X_train[col].where(X_train[col].isin(top), "Other")
        X_val[col] = X_val[col].where(X_val[col].isin(top), "Other")
        X_test[col] = X_test[col].where(X_test[col].isin(top), "Other")
        
    X_train, X_val, X_test = freq_encode(X_train, X_val, X_test)
    combined = pd.concat([X_train, X_val, X_test], axis=0)
    combined = pd.get_dummies(combined, columns=cat_cols_to_ohe, dtype=float)
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

def prepare_full_train_test(train_df, test_df, airport_map):
    X_train = make_features(train_df, airport_map)
    X_test = make_features(test_df, airport_map)
    cat_cols_to_ohe = [c for c in CATEGORY_COLS if c in X_train.columns]
    for col in cat_cols_to_ohe:
        top = X_train[col].value_counts().index[:50]
        X_train[col] = X_train[col].where(X_train[col].isin(top), "Other")
        X_test[col] = X_test[col].where(X_test[col].isin(top), "Other")
    X_train, X_test, _ = freq_encode(X_train, X_test, X_test)
    combined = pd.concat([X_train, X_test], axis=0)
    combined = pd.get_dummies(combined, columns=cat_cols_to_ohe, dtype=float)
    X_train = combined.iloc[: len(X_train)].copy()
    X_test = combined.iloc[len(X_train) :].copy()
    medians = X_train.median(numeric_only=True).fillna(0)
    X_train = X_train.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)
    return X_train.astype("float32"), X_test.astype("float32")

def build_model():
    return HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.03, max_leaf_nodes=127, l2_regularization=0.2,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=RANDOM_STATE,
    )

def best_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best]), float(f1[best]), float(precision[best]), float(recall[best])

def evaluate_holdout(train_df, test_df, airport_map):
    train_part, val_part = train_test_split(train_df, test_size=0.2, stratify=train_df[TARGET], random_state=RANDOM_STATE)
    X_train, X_val, _ = prepare_train_val_test(train_part, val_part, test_df, airport_map)
    y_train = train_part[TARGET].astype(int)
    y_val = val_part[TARGET].astype(int)
    model = build_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    threshold, f1, precision, recall = best_threshold(y_val, proba)
    metrics = {"roc_auc": float(roc_auc_score(y_val, proba)), "pr_auc": float(average_precision_score(y_val, proba)), "f1": f1, "precision": precision, "recall": recall, "threshold": threshold}
    print(f"ROC-AUC : {metrics['roc_auc']:.5f}")
    print(f"PR-AUC  : {metrics['pr_auc']:.5f}")
    print(f"best threshold: {threshold:.5f}  F1={f1:.5f}  precision={precision:.3f}  recall={recall:.3f}")
    return threshold, metrics

def run(args):
    train_df, test_df = read_data(args.train, args.test)
    all_data = pd.concat([train_df, test_df], axis=0)
    airport_map = get_airport_map(all_data)
    
    threshold, metrics = evaluate_holdout(train_df, test_df, airport_map)
    X_train, X_test = prepare_full_train_test(train_df, test_df, airport_map)
    y_train = train_df[TARGET].astype(int)
    model = build_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    submission = pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET: pred})
    submission.to_csv(args.out, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--out", type=Path, default=Path("mega_push_v2_submission.csv"))
    args = parser.parse_args()
    run(args)
