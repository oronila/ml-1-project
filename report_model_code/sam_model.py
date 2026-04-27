#!/usr/bin/env python3
"""Standalone submission script for the modeling pipeline in sam.ipynb.

The notebook's final submission cell points to a missing scripts/make_submission.py,
so this script recreates the notebook preprocessing and trains the base
HistGradientBoostingClassifier on all labeled rows.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"
RANDOM_STATE = 42

COLS_TO_DROP = [
    "REMARKS",
    "NUM_STRUCK",
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
    "NUM_SEEN",
    "NUM_STRUCK",
]


def read_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    return train, test


def add_time_features(df: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    incident_dt = pd.to_datetime(df["INCIDENT_DATE"], format="mixed", dayfirst=False, errors="coerce")
    df["INCIDENT_DATE"] = (incident_dt - anchor).dt.days
    df["INCIDENT_MONTH"] = incident_dt.dt.month.fillna(-1).astype(int)
    df["INCIDENT_DAYOFWEEK"] = incident_dt.dt.dayofweek.fillna(-1).astype(int)
    df["INCIDENT_IS_WEEKEND"] = incident_dt.dt.dayofweek.isin([5, 6]).astype(int).fillna(0).astype(int)
    df["INCIDENT_DAYOFYEAR"] = incident_dt.dt.dayofyear.fillna(-1).astype(int)
    df = df.drop(columns=["INCIDENT_YEAR"], errors="ignore")

    parts = df["TIME"].astype(str).str.split(":", n=1, expand=True)
    hour = pd.to_numeric(parts[0], errors="coerce")
    if 1 in parts.columns:
        minute = pd.to_numeric(parts[1], errors="coerce").fillna(0)
    else:
        minute = pd.Series(0, index=df.index)

    df["TIME"] = (hour * 60 + minute).where(hour.between(0, 23) & minute.between(0, 59))
    df["TIME_HOUR"] = hour
    df["TIME_MINUTE"] = minute
    df["TIME_SIN"] = np.sin(2 * np.pi * df["TIME"] / 1440)
    df["TIME_COS"] = np.cos(2 * np.pi * df["TIME"] / 1440)
    df = df.drop(columns=["TIME_OF_DAY"], errors="ignore")
    return df


def make_raw_features(df: pd.DataFrame, anchor: pd.Timestamp, has_target: bool) -> pd.DataFrame:
    out = df.drop(columns=COLS_TO_DROP, errors="ignore").copy()
    out = add_time_features(out, anchor)

    # AIRPORT_ID, OPID, and SPECIES_ID replace these verbose redundant names.
    out = out.drop(columns=["AIRPORT", "OPERATOR", "SPECIES"], errors="ignore")
    if not has_target:
        out = out.drop(columns=[TARGET], errors="ignore")

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


def encode_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    high_card_threshold: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    text_cols = list(X_train.select_dtypes(include=["str", "object", "string", "category"]).columns)
    high_card_cols: list[str] = []
    low_card_cols: list[str] = []

    if text_cols:
        for col in text_cols:
            X_train[col] = as_cat_str(X_train[col])
            X_test[col] = as_cat_str(X_test[col])

        high_card_cols = [
            col for col in text_cols if X_train[col].nunique(dropna=False) > high_card_threshold
        ]
        low_card_cols = [
            col for col in text_cols if X_train[col].nunique(dropna=False) <= high_card_threshold
        ]

        for col in high_card_cols:
            freq = X_train[col].value_counts(normalize=True)
            X_train[col] = X_train[col].map(freq).fillna(0.0)
            X_test[col] = X_test[col].map(freq).fillna(0.0)

        if low_card_cols:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoder.fit(X_train[low_card_cols])
            encoded_train = pd.DataFrame(
                encoder.transform(X_train[low_card_cols]),
                columns=encoder.get_feature_names_out(low_card_cols),
                index=X_train.index,
            )
            encoded_test = pd.DataFrame(
                encoder.transform(X_test[low_card_cols]),
                columns=encoder.get_feature_names_out(low_card_cols),
                index=X_test.index,
            )
            X_train = X_train.drop(columns=low_card_cols).join(encoded_train)
            X_test = X_test.drop(columns=low_card_cols).join(encoded_test)

    for col in X_train.columns:
        if X_train[col].dtype == "object" or str(X_train[col].dtype).startswith("string"):
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    for col in list(X_train.columns):
        if X_train[col].isna().any():
            X_train[f"{col}_MISSING"] = X_train[col].isna().astype(int)
            X_test[f"{col}_MISSING"] = X_test[col].isna().astype(int)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy="median")
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

    return X_train.astype(float), X_test.astype(float), text_cols, high_card_cols, low_card_cols


def build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=RANDOM_STATE,
    )


def evaluate_holdout(train_features: pd.DataFrame) -> None:
    y = train_features[TARGET].astype(int)
    X = train_features.drop(columns=[TARGET], errors="ignore")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_valid, text_cols, high_card_cols, low_card_cols = encode_features(X_train, X_valid)
    print(f"[holdout] X_train={X_train.shape} X_valid={X_valid.shape}")
    print(f"[holdout] text_cols={text_cols}")
    print(f"[holdout] high_card={high_card_cols}")
    print(f"[holdout] low_card={low_card_cols}")

    model = build_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_valid)[:, 1]
    pred = model.predict(X_valid)

    print(f"[holdout] ROC-AUC={roc_auc_score(y_valid, proba):.6f}")
    print(f"[holdout] PR-AUC ={average_precision_score(y_valid, proba):.6f}")
    print(f"[holdout] pred_pos_rate={pred.mean():.6f}")
    print("[holdout] confusion_matrix [[TN FP],[FN TP]]")
    print(confusion_matrix(y_valid, pred))
    print(classification_report(y_valid, pred, digits=4))


def make_submission(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    test_ids: pd.Series,
    out_path: Path,
) -> None:
    y_train = train_features[TARGET].astype(int)
    X_train = train_features.drop(columns=[TARGET], errors="ignore")
    X_train, X_test, text_cols, high_card_cols, low_card_cols = encode_features(X_train, test_features)
    print(f"[final] X_train={X_train.shape} X_test={X_test.shape}")
    print(f"[final] text_cols={text_cols}")
    print(f"[final] high_card={high_card_cols}")
    print(f"[final] low_card={low_card_cols}")

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
    anchor = pd.to_datetime(
        train_df["INCIDENT_DATE"],
        format="mixed",
        dayfirst=False,
        errors="coerce",
    ).min()

    train_features = make_raw_features(train_df, anchor, has_target=True)
    test_features = make_raw_features(test_df, anchor, has_target=False)
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
    parser.add_argument("--out", type=Path, default=Path("sam_submission.csv"))
    parser.add_argument("--skip-holdout", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
