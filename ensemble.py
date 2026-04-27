#!/usr/bin/env python3
"""Simple ensemble using the cleaning pattern from sam.ipynb.

This script intentionally keeps the preprocessing easy to explain:

1. Drop text/id columns that are not useful for modeling.
2. Convert incident date and time into numeric features.
3. Fill missing categorical values with "Unknown".
4. Frequency-encode high-cardinality categoricals.
5. One-hot encode low-cardinality categoricals.
6. Add missing-value flags and median-impute numeric columns.
7. Average a few simple tree-based classifiers.

Run:
    python sam_style_ensemble.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


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


def threshold_for_target_rate(scores: np.ndarray, target_rate: float) -> float:
    k = int(round(target_rate * len(scores)))
    k = max(1, min(len(scores) - 1, k))
    return float(np.sort(scores)[::-1][k - 1])


def add_time_features(df: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()

    incident_dt = pd.to_datetime(out["INCIDENT_DATE"], format="mixed", dayfirst=False, errors="coerce")
    out["INCIDENT_DATE"] = (incident_dt - anchor).dt.days
    out["INCIDENT_MONTH"] = incident_dt.dt.month.fillna(-1).astype(int)
    out["INCIDENT_DAYOFWEEK"] = incident_dt.dt.dayofweek.fillna(-1).astype(int)
    out["INCIDENT_IS_WEEKEND"] = incident_dt.dt.dayofweek.isin([5, 6]).astype(int)
    out["INCIDENT_DAYOFYEAR"] = incident_dt.dt.dayofyear.fillna(-1).astype(int)
    out = out.drop(columns=["INCIDENT_YEAR"], errors="ignore")

    parts = out["TIME"].astype(str).str.split(":", n=1, expand=True)
    hour = pd.to_numeric(parts[0], errors="coerce")
    minute = pd.to_numeric(parts[1], errors="coerce").fillna(0) if 1 in parts.columns else 0
    out["TIME"] = (hour * 60 + minute).where(hour.between(0, 23) & minute.between(0, 59))
    out["TIME_HOUR"] = hour
    out["TIME_MINUTE"] = minute
    out["TIME_SIN"] = np.sin(2 * np.pi * out["TIME"] / 1440)
    out["TIME_COS"] = np.cos(2 * np.pi * out["TIME"] / 1440)
    out = out.drop(columns=["TIME_OF_DAY"], errors="ignore")
    return out


def make_raw_features(df: pd.DataFrame, anchor: pd.Timestamp, has_target: bool) -> pd.DataFrame:
    out = df.drop(columns=COLS_TO_DROP, errors="ignore").copy()
    out = add_time_features(out, anchor)

    # Keep compact code columns instead of long descriptive text names.
    out = out.drop(columns=["AIRPORT", "OPERATOR", "SPECIES"], errors="ignore")
    if not has_target:
        out = out.drop(columns=[TARGET], errors="ignore")

    for col in out.select_dtypes(include=["str", "object", "category"]).columns:
        out[col] = out[col].fillna("Unknown")

    return out


def as_category_string(series: pd.Series) -> pd.Series:
    values = series.map(lambda value: "Unknown" if pd.isna(value) else value)
    values = values.astype("string[python]").astype(str)
    return values.replace({"nan": "Unknown", "None": "Unknown", "<NA>": "Unknown", "": "Unknown"})


def encode_features(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    high_card_threshold: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_other = X_other.copy()

    text_cols = list(X_train.select_dtypes(include=["str", "object", "string", "category"]).columns)
    if text_cols:
        for col in text_cols:
            X_train[col] = as_category_string(X_train[col])
            X_other[col] = as_category_string(X_other[col])

        high_card_cols = [
            col for col in text_cols if X_train[col].nunique(dropna=False) > high_card_threshold
        ]
        low_card_cols = [
            col for col in text_cols if X_train[col].nunique(dropna=False) <= high_card_threshold
        ]

        for col in high_card_cols:
            freq = X_train[col].value_counts(normalize=True)
            X_train[col] = X_train[col].map(freq).fillna(0.0)
            X_other[col] = X_other[col].map(freq).fillna(0.0)

        if low_card_cols:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoder.fit(X_train[low_card_cols])
            train_encoded = pd.DataFrame(
                encoder.transform(X_train[low_card_cols]),
                columns=encoder.get_feature_names_out(low_card_cols),
                index=X_train.index,
            )
            other_encoded = pd.DataFrame(
                encoder.transform(X_other[low_card_cols]),
                columns=encoder.get_feature_names_out(low_card_cols),
                index=X_other.index,
            )
            X_train = X_train.drop(columns=low_card_cols).join(train_encoded)
            X_other = X_other.drop(columns=low_card_cols).join(other_encoded)

    for col in X_train.columns:
        if X_train[col].dtype == "object" or str(X_train[col].dtype).startswith("string"):
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_other[col] = pd.to_numeric(X_other[col], errors="coerce")

    for col in list(X_train.columns):
        if X_train[col].isna().any():
            X_train[col + "_MISSING"] = X_train[col].isna().astype(int)
            X_other[col + "_MISSING"] = X_other[col].isna().astype(int)

    num_cols = X_train.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy="median")
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_other[num_cols] = imputer.transform(X_other[num_cols])

    return X_train.astype("float32"), X_other.astype("float32")


def build_models() -> list[tuple[str, object, float]]:
    base_tree = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=120,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return [
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            0.25,
        ),
        (
            "cat",
            CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0,
                allow_writing_files=False,
            ),
            0.22,
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=450,
                learning_rate=0.04,
                max_depth=4,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                scale_pos_weight=6.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            0.18,
        ),
        (
            "ada",
            AdaBoostClassifier(
                estimator=base_tree,
                n_estimators=180,
                learning_rate=0.08,
                random_state=RANDOM_STATE,
            ),
            0.20,
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=20,
                min_samples_leaf=25,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            0.10,
        ),
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=250,
                max_depth=22,
                min_samples_leaf=25,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            0.05,
        ),
    ]


def evaluate_holdout(train_features: pd.DataFrame) -> None:
    y = train_features[TARGET].astype(int)
    X = train_features.drop(columns=[TARGET], errors="ignore")
    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_valid = encode_features(X_train_raw, X_valid_raw)

    blend = np.zeros(len(X_valid), dtype=float)
    total_weight = 0.0
    print(f"[holdout] train={X_train.shape} valid={X_valid.shape}")
    for name, model, weight in build_models():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_valid)[:, 1]
        blend += weight * proba
        total_weight += weight
        print(
            f"[holdout:{name}] ROC-AUC={roc_auc_score(y_valid, proba):.5f} "
            f"PR-AUC={average_precision_score(y_valid, proba):.5f}"
        )

    blend = blend / total_weight
    threshold = threshold_for_target_rate(blend, 0.18)
    pred = (blend >= threshold).astype(int)
    print(f"[holdout:ensemble] ROC-AUC={roc_auc_score(y_valid, blend):.5f}")
    print(f"[holdout:ensemble] PR-AUC ={average_precision_score(y_valid, blend):.5f}")
    print(f"[holdout:ensemble] F1@18% ={f1_score(y_valid, pred):.5f}")
    print(classification_report(y_valid, pred, digits=4))


def make_submission(train_features: pd.DataFrame, test_features: pd.DataFrame, test_ids: pd.Series, args) -> None:
    y_train = train_features[TARGET].astype(int)
    X_train_raw = train_features.drop(columns=[TARGET], errors="ignore")
    X_train, X_test = encode_features(X_train_raw, test_features)

    blend = np.zeros(len(X_test), dtype=float)
    total_weight = 0.0
    print(f"[final] train={X_train.shape} test={X_test.shape}")
    for name, model, weight in build_models():
        model.fit(X_train, y_train)
        blend += weight * model.predict_proba(X_test)[:, 1]
        total_weight += weight
        print(f"[final:{name}] fitted")

    blend = blend / total_weight
    threshold = threshold_for_target_rate(blend, args.target_rate)
    pred = (blend >= threshold).astype(int)
    submission = pd.DataFrame({ID_COL: test_ids.values, TARGET: pred})
    submission.to_csv(args.out, index=False)
    np.save(args.out.with_suffix(".proba.npy"), blend)
    print(
        f"[submit] wrote {args.out} rows={len(submission)} "
        f"positives={int(pred.sum())} pos_rate={pred.mean():.6f}"
    )


def run(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    train_df = pd.read_csv(args.train, low_memory=False)
    test_df = pd.read_csv(args.test, low_memory=False)
    anchor = pd.to_datetime(
        train_df["INCIDENT_DATE"],
        format="mixed",
        dayfirst=False,
        errors="coerce",
    ).min()

    train_features = make_raw_features(train_df, anchor, has_target=True)
    test_features = make_raw_features(test_df, anchor, has_target=False)
    test_features = test_features.reindex(columns=train_features.drop(columns=[TARGET]).columns)
    print(f"[features] train={train_features.shape} test={test_features.shape}")

    if not args.skip_holdout:
        evaluate_holdout(train_features)

    make_submission(train_features, test_features, test_df[ID_COL], args)
    print(f"[done] total time={time.perf_counter() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--out", type=Path, default=Path("sam_style_cat_xgb_ensemble_submission.csv"))
    parser.add_argument("--target-rate", type=float, default=0.18)
    parser.add_argument("--skip-holdout", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
