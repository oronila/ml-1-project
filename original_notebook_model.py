#!/usr/bin/env python3
"""Recreate the basic model from main.ipynb as a standalone script.

This intentionally mirrors the notebook's simple preprocessing:
- drop the same columns
- convert INCIDENT_DATE to days since the train minimum date
- convert TIME to minutes
- coerce LATITUDE/LONGITUDE to numeric and add coordinate interaction features
- fill numeric missing values with -1 and categoricals with "Unknown"
- ordinal-encode object columns
- train AdaBoost over a depth-15 balanced decision tree
- write hard 0/1 predictions using clf.predict()

Run:
    python original_notebook_model.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"
RANDOM_STATE = 42

COLS_TO_DROP = [
    "REMARKS",
    "LOCATION",
    "ENG_3_POS",
    "BIRD_BAND_NUMBER",
    "ENG_4_POS",
    "ENROUTE_STATE",
    "PRECIPITATION",
    "COMMENTS",
    "TRANSFER",
    "SOURCE",
    "LUPDATE",
    "RUNWAY",
    "FLT",
    "AIRCRAFT",
    ID_COL,
    "REG",
    "PERSON",
]


def read_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    return train, test


def date_anchor(train_df: pd.DataFrame) -> pd.Timestamp:
    dates = pd.to_datetime(
        train_df["INCIDENT_DATE"],
        format="mixed",
        dayfirst=False,
        errors="coerce",
    ).dt.normalize()
    return dates.min()


def time_to_minutes(series: pd.Series) -> pd.Series:
    parts = series.astype(str).str.split(":", n=1, expand=True)
    hour = pd.to_numeric(parts[0], errors="coerce")
    if 1 in parts.columns:
        minute = pd.to_numeric(parts[1], errors="coerce").fillna(0)
    else:
        minute = pd.Series(0, index=series.index)
    return (hour * 60 + minute).where(hour.between(0, 23) & minute.between(0, 59))


def add_coordinate_features(out: pd.DataFrame) -> None:
    """Keep coordinates numeric and add simple geographic transforms."""
    if "LATITUDE" not in out.columns or "LONGITUDE" not in out.columns:
        return

    lat = pd.to_numeric(out["LATITUDE"], errors="coerce")
    lon = pd.to_numeric(out["LONGITUDE"], errors="coerce")
    out["LATITUDE"] = lat
    out["LONGITUDE"] = lon

    lat_filled = lat.fillna(-1)
    lon_filled = lon.fillna(-1)
    out["LATITUDE_X_LONGITUDE"] = lat_filled * lon_filled
    out["LATITUDE_SQUARED"] = lat_filled * lat_filled
    out["LONGITUDE_SQUARED"] = lon_filled * lon_filled
    out["LATITUDE_SIN"] = np.sin(np.deg2rad(lat_filled))
    out["LATITUDE_COS"] = np.cos(np.deg2rad(lat_filled))
    out["LONGITUDE_SIN"] = np.sin(np.deg2rad(lon_filled))
    out["LONGITUDE_COS"] = np.cos(np.deg2rad(lon_filled))


def make_notebook_features(df: pd.DataFrame, anchor: pd.Timestamp, has_target: bool) -> pd.DataFrame:
    out = df.drop(columns=COLS_TO_DROP, errors="ignore").copy()

    dates = pd.to_datetime(
        out["INCIDENT_DATE"],
        format="mixed",
        dayfirst=False,
        errors="coerce",
    ).dt.normalize()
    out["INCIDENT_DATE"] = (dates - anchor).dt.days

    out = out.drop(columns=["INCIDENT_YEAR"], errors="ignore")
    out["TIME"] = time_to_minutes(out["TIME"])
    out = out.drop(columns=["TIME_OF_DAY"], errors="ignore")

    # Keep IDs instead of verbose redundant names, matching the notebook.
    out = out.drop(columns=["AIRPORT", "OPERATOR", "SPECIES"], errors="ignore")
    add_coordinate_features(out)

    if not has_target:
        out = out.drop(columns=[TARGET], errors="ignore")

    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].fillna(-1)

    for col in out.select_dtypes(include=["str", "object", "category"]).columns:
        out[col] = out[col].fillna("Unknown")

    return out


def as_cat_str(series: pd.Series) -> pd.Series:
    values = series.map(
        lambda value: "Unknown"
        if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value)
        else value
    )
    values = values.astype("string[python]").astype(str)
    return values.replace({"nan": "Unknown", "None": "Unknown", "<NA>": "Unknown", "": "Unknown"})


def encode_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    X_train = X_train.copy()
    X_test = X_test.copy()
    text_cols = list(X_train.select_dtypes(include=["str", "object", "string", "category"]).columns)

    if text_cols:
        for col in text_cols:
            X_train[col] = as_cat_str(X_train[col])
            X_test[col] = as_cat_str(X_test[col])

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.int64,
        )
        encoder.fit(X_train[text_cols])
        X_train[text_cols] = encoder.transform(X_train[text_cols]).astype(np.int64)
        X_test[text_cols] = encoder.transform(X_test[text_cols]).astype(np.int64)

    for col in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[col]) or not pd.api.types.is_numeric_dtype(X_test[col]):
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            median = X_train[col].median()
            if pd.isna(median):
                median = -1
            X_train[col] = X_train[col].fillna(median)
            X_test[col] = X_test[col].fillna(median)

    return X_train.astype(float), X_test.astype(float), text_cols


def build_model() -> AdaBoostClassifier:
    base = DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return AdaBoostClassifier(
        estimator=base,
        n_estimators=300,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )


def evaluate_holdout(features: pd.DataFrame) -> None:
    y = features[TARGET].astype(int)
    X = features.drop(columns=[TARGET], errors="ignore")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_valid, text_cols = encode_train_test(X_train, X_valid)

    print(f"[holdout] X_train={X_train.shape} X_valid={X_valid.shape} label_encoded={text_cols}")
    model = build_model()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_valid)[:, 1]
    pred = model.predict(X_valid)
    print(f"[holdout] ROC-AUC={roc_auc_score(y_valid, proba):.5f}")
    print(f"[holdout] PR-AUC ={average_precision_score(y_valid, proba):.5f}")
    print(f"[holdout] pred_pos_rate={pred.mean():.5f}")
    print(classification_report(y_valid, pred, digits=4))


def make_submission(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    test_ids: pd.Series,
    out_path: Path,
) -> None:
    y_train = train_features[TARGET].astype(int)
    X_train = train_features.drop(columns=[TARGET], errors="ignore")
    X_train, X_test, text_cols = encode_train_test(X_train, test_features)

    print(f"[final] X_train={X_train.shape} X_test={X_test.shape} label_encoded={text_cols}")
    model = build_model()
    model.fit(X_train, y_train)

    pred = model.predict(X_test).astype(int)
    submission = pd.DataFrame({ID_COL: test_ids.values, TARGET: pred})
    submission.to_csv(out_path, index=False)
    print(
        f"[submit] wrote {out_path} rows={len(submission)} "
        f"positives={int(pred.sum())} pos_rate={pred.mean():.6f}"
    )


def run(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    train_df, test_df = read_data(args.train, args.test)
    anchor = date_anchor(train_df)

    train_features = make_notebook_features(train_df, anchor, has_target=True)
    test_features = make_notebook_features(test_df, anchor, has_target=False)
    test_features = test_features.reindex(columns=train_features.drop(columns=[TARGET]).columns)

    print(f"[load] train={train_df.shape} test={test_df.shape}")
    print(f"[features] train={train_features.shape} test={test_features.shape}")

    if not args.skip_holdout:
        evaluate_holdout(train_features)

    make_submission(train_features, test_features, test_df[ID_COL], args.out)
    print(f"[done] total time={time.perf_counter() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--out", type=Path, default=Path("original_notebook_submission.csv"))
    parser.add_argument("--skip-holdout", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
