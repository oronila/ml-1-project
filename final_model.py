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
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder

TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"
RANDOM_STATE = 42

DROP_COLS = [ID_COL, TARGET, "LOCATION", "BIRD_BAND_NUMBER", "REG", "FLT", "AIRCRAFT", "LUPDATE", "COMMENTS", "REMARKS", "AIRPORT", "OPERATOR", "SPECIES"]
NUMBER_COLS = ["LATITUDE", "LONGITUDE", "HEIGHT", "SPEED", "DISTANCE", "NUM_SEEN", "NUM_STRUCK", "OUT_OF_RANGE_SPECIES", "AC_MASS", "NUM_ENGS", "ENG_1_POS", "ENG_2_POS", "ENG_3_POS", "ENG_4_POS", "REMAINS_COLLECTED", "REMAINS_SENT", "TRANSFER"]
CATEGORY_COLS = ["TIME_OF_DAY", "STATE", "FAAREGION", "OPID", "AC_CLASS", "TYPE_ENG", "PHASE_OF_FLIGHT", "SKY", "WARNED", "SIZE", "SOURCE", "PERSON", "PRECIPITATION"]

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
    df = impute_data(df, airport_map)
    out_dict = {}
    d = pd.to_datetime(df["INCIDENT_DATE"], format="mixed", errors="coerce")
    year = d.dt.year.fillna(2015)
    out_dict["YEAR"] = year
    out_dict["YEAR_SQ"] = (year - 1990)**2
    out_dict["MONTH_SIN"] = np.sin(2 * np.pi * d.dt.month.fillna(6) / 12.0)
    out_dict["MONTH_COS"] = np.cos(2 * np.pi * d.dt.month.fillna(6) / 12.0)
    
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
        out_dict["DIST_FROM_CENTER"] = np.sqrt((out_dict["LATITUDE"] - 39.5)**2 + (out_dict["LONGITUDE"] + 98.35)**2)

    for col in CATEGORY_COLS:
        if col in df.columns:
            out_dict[col] = df[col].fillna("Unknown").astype(str)
            
    return pd.DataFrame(out_dict, index=df.index)

def prepare_data(train_df, test_df, airport_map):
    X_train = make_features(train_df, airport_map)
    X_test = make_features(test_df, airport_map)
    
    # Efficient Categorical Handling
    for col in CATEGORY_COLS:
        top = X_train[col].value_counts().index[:50]
        X_train[col] = X_train[col].where(X_train[col].isin(top), "Other")
        X_test[col] = X_test[col].where(X_test[col].isin(top), "Other")
        
    combined = pd.concat([X_train, X_test], axis=0)
    combined = pd.get_dummies(combined, columns=CATEGORY_COLS, dtype=float)
    
    X_train = combined.iloc[:len(X_train)].copy()
    X_test = combined.iloc[len(X_train):].copy()
    
    medians = X_train.median(numeric_only=True).fillna(0)
    X_train = X_train.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)
    
    return X_train.astype("float32"), X_test.astype("float32")

def get_best_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.argmax(f1[:-1])
    return thresholds[best_idx], f1[best_idx]

def run(args):
    print("[load] reading data...")
    train_full, test_df = read_data(args.train, args.test)
    all_data = pd.concat([train_full, test_df], axis=0)
    airport_map = get_airport_map(all_data)

    X_train_full, X_test = prepare_data(train_full, test_df, airport_map)
    y_train_full = train_full[TARGET].astype(int)
    
    pos_weight = (len(y_train_full) - y_train_full.sum()) / y_train_full.sum()
    sample_weights = np.ones(len(y_train_full))
    sample_weights[y_train_full == 1] = pos_weight

    # Fast 5-fold CV Search
    print(f"[search] starting 5-fold stratified CV search on {X_train_full.shape[1]} features...")
    param_grid = {
        'max_iter': [400, 600],
        'learning_rate': [0.03, 0.05],
        'max_leaf_nodes': [63, 127],
        'l2_regularization': [0.1, 1.0],
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        param_distributions=param_grid,
        n_iter=4, # Fewer iters for speed
        scoring='average_precision',
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1 # Use all cores
    )
    
    search.fit(X_train_full, y_train_full, sample_weight=sample_weights)
    print(f"[search] best params: {search.best_params_}")
    
    model = search.best_estimator_
    
    # Threshold Tune
    X_t, X_v, y_t, y_v, w_t, _ = train_test_split(
        X_train_full, y_train_full, sample_weights, test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE
    )
    model.fit(X_t, y_t, sample_weight=w_t)
    proba_v = model.predict_proba(X_v)[:, 1]
    best_thresh, best_f1 = get_best_threshold(y_v, proba_v)
    print(f"[tune] best holdout F1: {best_f1:.5f} at threshold {best_thresh:.3f}")

    # Final Prediction
    print("[final] refitting and predicting...")
    model.fit(X_train_full, y_train_full, sample_weight=sample_weights)
    proba_test = model.predict_proba(X_test)[:, 1]
    preds = (proba_test >= best_thresh).astype(int)
    
    pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: preds}).to_csv(args.out, index=False)
    print(f"[done] wrote {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--out", type=Path, default=Path("optimized_physics_submission.csv"))
    args = parser.parse_args()
    run(args)
