#!/usr/bin/env python3
"""
Best model — aggressive feature engineering guided by ablation results.

Key insight: Bird info (SPECIES_ID + SIZE) alone = 0.35 PR-AUC.
Flight conditions and aircraft type are next. Focus engineering there.

Strategy:
- Rich target encoding for SPECIES_ID (the #1 predictor)
- Physics-based interaction features (kinetic energy, momentum)
- Multiple target-encoding smoothing levels
- Aggressive CatBoost + LightGBM ensemble with 5-fold averaging
- No text, no SOURCE/PERSON (leaky)
"""

import time
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

# Bird mass estimates in kg by SIZE category
BIRD_MASS_KG = {"Small": 0.08, "Medium": 0.5, "Large": 3.5}

# NUM_STRUCK ordinal mapping (Excel corrupted "2-10" to "10-Feb")
NUM_STRUCK_MAP = {"1": 1, "10-Feb": 5, "11-100": 50, "More than 100": 150}

# Columns for CatBoost native categorical
CB_CAT_COLS = [
    "TIME_OF_DAY", "STATE", "FAAREGION", "OPID",
    "AC_CLASS", "TYPE_ENG", "PHASE_OF_FLIGHT",
    "SKY", "PRECIPITATION", "WARNED", "SIZE",
    "AMA", "AMO", "EMA", "EMO",
    "ENROUTE_STATE", "RUNWAY",
    "AIRPORT_ID", "SPECIES_ID", "OPERATOR",
    "NUM_STRUCK",
]

# Columns to target-encode for LightGBM
TE_COLS = [
    "SPECIES_ID", "AIRPORT_ID", "OPERATOR", "RUNWAY",
    "OPID", "STATE", "FAAREGION",
    "AC_CLASS", "TYPE_ENG", "PHASE_OF_FLIGHT",
    "SKY", "PRECIPITATION", "WARNED", "SIZE",
    "NUM_STRUCK",
]

# Low-cardinality categoricals kept native for LightGBM
LGB_CAT_COLS = ["TIME_OF_DAY", "AMA", "AMO", "EMA", "EMO", "ENROUTE_STATE"]


def get_airport_map(df):
    tmp = df.copy()
    for col in ["LATITUDE", "LONGITUDE"]:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    valid = tmp[
        tmp["AIRPORT_ID"].notna() & (tmp["AIRPORT_ID"] != "ZZZZ")
        & tmp["LATITUDE"].notna() & tmp["LONGITUDE"].notna()
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

    # ---- DATE/TIME (ablation: 0.167 standalone) ----
    dates = pd.to_datetime(df["INCIDENT_DATE"], format="mixed", errors="coerce")
    out["YEAR"] = dates.dt.year
    out["INCIDENT_YEAR"] = pd.to_numeric(df["INCIDENT_YEAR"], errors="coerce")
    month = dates.dt.month.fillna(6)
    out["MONTH_SIN"] = np.sin(2 * np.pi * month / 12.0)
    out["MONTH_COS"] = np.cos(2 * np.pi * month / 12.0)
    doy = dates.dt.dayofyear.fillna(180)
    out["DOY_SIN"] = np.sin(2 * np.pi * doy / 366.0)
    out["DOY_COS"] = np.cos(2 * np.pi * doy / 366.0)

    minutes = time_to_minutes(df["TIME"])
    out["MINUTES"] = minutes
    filled_min = minutes.fillna(720)
    out["TIME_SIN"] = np.sin(2 * np.pi * filled_min / 1440.0)
    out["TIME_COS"] = np.cos(2 * np.pi * filled_min / 1440.0)
    out["TIME_MISSING"] = df["TIME"].isna().astype("int8")

    # ---- BIRD INFO (ablation: 0.352 standalone — #1) ----
    out["NUM_SEEN"] = pd.to_numeric(df["NUM_SEEN"], errors="coerce")
    out["LOG_NUM_SEEN"] = np.log1p(out["NUM_SEEN"].clip(lower=0))
    out["OUT_OF_RANGE_SPECIES"] = pd.to_numeric(df["OUT_OF_RANGE_SPECIES"], errors="coerce")

    # Bird mass estimate from SIZE
    size_str = df["SIZE"].fillna("Unknown")
    bird_mass = size_str.map(BIRD_MASS_KG).fillna(0.3)  # default ~medium
    out["BIRD_MASS_KG"] = bird_mass
    out["SIZE_ORD"] = size_str.map({"Small": 1, "Medium": 2, "Large": 3}).fillna(0)

    # ---- FLIGHT CONDITIONS (ablation: 0.189 — #3) ----
    height = pd.to_numeric(df["HEIGHT"], errors="coerce")
    speed = pd.to_numeric(df["SPEED"], errors="coerce")
    distance = pd.to_numeric(df["DISTANCE"], errors="coerce")
    out["HEIGHT"] = height
    out["SPEED"] = speed
    out["DISTANCE"] = distance
    out["LOG_HEIGHT"] = np.log1p(height.clip(lower=0))
    out["LOG_SPEED"] = np.log1p(speed.clip(lower=0))
    out["LOG_DISTANCE"] = np.log1p(distance.clip(lower=0))
    out["HEIGHT_MISSING"] = height.isna().astype("int8")
    out["SPEED_MISSING"] = speed.isna().astype("int8")
    out["DISTANCE_MISSING"] = distance.isna().astype("int8")

    # ---- AIRCRAFT (ablation: 0.170 — #4) ----
    ac_mass = pd.to_numeric(df["AC_MASS"], errors="coerce")
    num_engs = pd.to_numeric(df["NUM_ENGS"], errors="coerce")
    out["AC_MASS"] = ac_mass
    out["NUM_ENGS"] = num_engs
    for col in ["ENG_1_POS", "ENG_2_POS", "ENG_3_POS", "ENG_4_POS"]:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- PHYSICS INTERACTIONS (Bird × Flight × Aircraft) ----
    # Kinetic energy of bird impact: 0.5 * bird_mass * speed^2
    # Speed in knots → m/s: multiply by 0.5144
    speed_ms = speed * 0.5144
    out["KINETIC_ENERGY"] = 0.5 * bird_mass * speed_ms ** 2
    out["LOG_KE"] = np.log1p(out["KINETIC_ENERGY"].clip(lower=0))

    # Momentum: bird_mass * speed
    out["MOMENTUM"] = bird_mass * speed_ms
    out["LOG_MOMENTUM"] = np.log1p(out["MOMENTUM"].clip(lower=0))

    # Bird-to-aircraft mass ratio (higher = more dangerous per unit)
    # AC_MASS is a category (1-4), map to rough kg: 1=2000, 2=6000, 3=30000, 4=150000
    ac_mass_kg = ac_mass.map({1: 2000, 2: 6000, 3: 30000, 4: 150000}).fillna(15000)
    out["MASS_RATIO"] = bird_mass / ac_mass_kg
    out["LOG_MASS_RATIO"] = np.log1p(out["MASS_RATIO"])

    # Size × speed (simple but effective)
    out["SIZE_SPEED"] = out["SIZE_ORD"] * speed
    out["SIZE_HEIGHT"] = out["SIZE_ORD"] * height

    # Energy per engine (single engine more vulnerable)
    out["KE_PER_ENGINE"] = out["KINETIC_ENERGY"] / num_engs.clip(lower=1)

    # Speed × height interaction (fast + low = takeoff/landing damage)
    out["SPEED_X_HEIGHT"] = speed * height

    # Aircraft mass × speed^2 (total kinetic energy at impact)
    out["AC_ENERGY"] = ac_mass * speed ** 2

    # ---- LOCATION (ablation: 0.097 + AIRPORT_ID 0.153) ----
    lat = df["LATITUDE"].astype(float)
    lon = df["LONGITUDE"].astype(float)
    out["LATITUDE"] = lat
    out["LONGITUDE"] = lon
    out["LAT_SIN"] = np.sin(np.deg2rad(lat))
    out["LAT_COS"] = np.cos(np.deg2rad(lat))
    out["LON_SIN"] = np.sin(np.deg2rad(lon))
    out["LON_COS"] = np.cos(np.deg2rad(lon))
    out["DIST_FROM_US_CENTER"] = np.sqrt((lat - 39.5) ** 2 + (lon + 98.35) ** 2)
    out["LAT_X_LON"] = lat * lon
    out["LAT_MISSING"] = lat.isna().astype("int8")

    # ---- POST-STRIKE (ablation: 0.109) ----
    out["REMAINS_COLLECTED"] = pd.to_numeric(df["REMAINS_COLLECTED"], errors="coerce")
    out["NUM_STRUCK_ORD"] = df["NUM_STRUCK"].map(NUM_STRUCK_MAP)
    out["NUM_STRUCK_MISSING"] = df["NUM_STRUCK"].isna().astype("int8")

    # ---- MISSING FLAGS for key fields ----
    out["PHASE_MISSING"] = df["PHASE_OF_FLIGHT"].isna().astype("int8")
    out["SKY_MISSING"] = df["SKY"].isna().astype("int8")
    out["SIZE_MISSING"] = df["SIZE"].isna().astype("int8")

    # ---- ALL CATEGORICALS (kept as string) ----
    all_cats = set(CB_CAT_COLS) | set(TE_COLS) | set(LGB_CAT_COLS)
    for col in all_cats:
        if col in df.columns:
            out[col] = df[col].fillna("_m_").astype(str)

    return out


def target_encode_oof(X_train, y_train, X_test, cols, smooth=50):
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

        # Frequency
        counts = pd.Series(train_keys).value_counts()
        X_train[f"FREQ_{col}"] = pd.Series(train_keys).map(counts).fillna(0).astype("float32").values
        X_test[f"FREQ_{col}"] = pd.Series(X_test[col].values).map(counts).fillna(0).astype("float32").values

        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

    return X_train, X_test


def target_encode_multi_smooth(X_train, y_train, X_test, col, smoothings):
    """Multiple smoothing levels for the most important categorical."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    global_mean = float(y_train.mean())
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_arr = y_train.values
    train_keys = X_train[col].values

    for smooth in smoothings:
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

        X_train[f"TE_{col}_s{smooth}"] = oof.astype("float32")
        X_test[f"TE_{col}_s{smooth}"] = test_enc.astype("float32")

    return X_train, X_test


def prepare_lgbm(X_all, y, X_test):
    X_tr, X_te = X_all.copy(), X_test.copy()

    # Multi-smooth target encoding for SPECIES_ID (the #1 feature)
    X_tr, X_te = target_encode_multi_smooth(X_tr, y, X_te, "SPECIES_ID", [10, 50, 200])

    # Regular target encoding for the rest
    te_cols_rest = [c for c in TE_COLS if c != "SPECIES_ID"]
    X_tr, X_te = target_encode_oof(X_tr, y, X_te, te_cols_rest)

    # Drop SPECIES_ID string column (already encoded)
    if "SPECIES_ID" in X_tr.columns:
        X_tr = X_tr.drop(columns=["SPECIES_ID"])
        X_te = X_te.drop(columns=["SPECIES_ID"])

    for col in LGB_CAT_COLS:
        if col in X_tr.columns:
            X_tr[col] = X_tr[col].astype("category")
            X_te[col] = X_te[col].astype("category")

    # Drop any remaining string columns
    str_cols = X_tr.select_dtypes(include="object").columns.tolist()
    X_tr = X_tr.drop(columns=str_cols, errors="ignore")
    X_te = X_te.drop(columns=str_cols, errors="ignore")

    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    X_te = X_te.replace([np.inf, -np.inf], np.nan)

    cat_features = [c for c in LGB_CAT_COLS if c in X_tr.columns]
    return X_tr, X_te, cat_features


def prepare_catboost(X_all, X_test):
    X_tr, X_te = X_all.copy(), X_test.copy()
    cat_cols = [c for c in CB_CAT_COLS if c in X_tr.columns]
    for col in cat_cols:
        X_tr[col] = X_tr[col].fillna("_m_").astype(str)
        X_te[col] = X_te[col].fillna("_m_").astype(str)

    num_cols = [c for c in X_tr.columns if c not in cat_cols]
    X_tr[num_cols] = X_tr[num_cols].replace([np.inf, -np.inf], np.nan)
    X_te[num_cols] = X_te[num_cols].replace([np.inf, -np.inf], np.nan)
    X_te = X_te.reindex(columns=X_tr.columns)
    return X_tr, X_te, cat_cols


def run_lgbm(X_all, y, X_test, cat_features):
    # Two LGB configs for diversity
    configs = [
        {  # Config A: deeper, more regularized
            "objective": "binary", "metric": "average_precision",
            "boosting_type": "gbdt", "learning_rate": 0.03,
            "num_leaves": 127, "min_data_in_leaf": 50,
            "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 5,
            "lambda_l1": 0.1, "lambda_l2": 3.0, "min_gain_to_split": 0.01,
            "verbose": -1, "random_state": RANDOM_STATE, "n_estimators": 3000,
        },
        {  # Config B: shallower, less regularized
            "objective": "binary", "metric": "average_precision",
            "boosting_type": "gbdt", "learning_rate": 0.05,
            "num_leaves": 63, "min_data_in_leaf": 30,
            "feature_fraction": 0.75, "bagging_fraction": 0.85, "bagging_freq": 5,
            "lambda_l1": 0.05, "lambda_l2": 1.0, "min_gain_to_split": 0.005,
            "verbose": -1, "random_state": RANDOM_STATE + 100, "n_estimators": 3000,
        },
    ]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_oof = []
    all_test = []

    for ci, params in enumerate(configs):
        oof = np.zeros(len(X_all))
        test_preds = np.zeros(len(X_test))

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

        pr = average_precision_score(y, oof)
        roc = roc_auc_score(y, oof)
        print(f"  LGB config {ci}: ROC={roc:.5f}  PR-AUC={pr:.5f}")
        all_oof.append(oof)
        all_test.append(test_preds)

    return all_oof, all_test


def run_catboost(X_all, y, X_test, cat_cols):
    configs = [
        {  # Config A
            "iterations": 3000, "learning_rate": 0.035, "depth": 6,
            "l2_leaf_reg": 5.0, "random_strength": 1.0,
        },
        {  # Config B: deeper
            "iterations": 3000, "learning_rate": 0.03, "depth": 7,
            "l2_leaf_reg": 3.0, "random_strength": 0.5,
        },
    ]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_oof = []
    all_test = []

    for ci, cfg in enumerate(configs):
        oof = np.zeros(len(X_all))
        test_preds = np.zeros(len(X_test))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y)):
            train_pool = Pool(X_all.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols)
            valid_pool = Pool(X_all.iloc[va_idx], y.iloc[va_idx], cat_features=cat_cols)
            test_pool = Pool(X_test, cat_features=cat_cols)

            model = CatBoostClassifier(
                **cfg,
                loss_function="Logloss", eval_metric="PRAUC",
                auto_class_weights="SqrtBalanced",
                random_seed=RANDOM_STATE + fold + ci * 100,
                od_type="Iter", od_wait=150,
                verbose=0, allow_writing_files=False,
            )
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            p = model.predict_proba(valid_pool)[:, 1]
            oof[va_idx] = p
            test_preds += model.predict_proba(test_pool)[:, 1] / N_FOLDS

        pr = average_precision_score(y, oof)
        roc = roc_auc_score(y, oof)
        print(f"  CB config {ci}:  ROC={roc:.5f}  PR-AUC={pr:.5f}")
        all_oof.append(oof)
        all_test.append(test_preds)

    return all_oof, all_test


def run():
    start = time.perf_counter()

    train_df = pd.read_csv("train.csv", low_memory=False)
    test_df = pd.read_csv("test.csv", low_memory=False)
    y = train_df[TARGET].astype(int)
    print(f"[load] train={train_df.shape} test={test_df.shape} pos_rate={y.mean():.4f}")

    airport_map = get_airport_map(pd.concat([train_df, test_df], axis=0))
    X_all = make_features(train_df, airport_map)
    X_test = make_features(test_df, airport_map)
    print(f"[features] {X_all.shape[1]} columns")

    # --- LightGBM ---
    print("\n--- LightGBM (2 configs × 5 folds) ---")
    X_lgb_tr, X_lgb_te, lgb_cats = prepare_lgbm(X_all, y, X_test)
    print(f"[lgbm] features={X_lgb_tr.shape[1]} native_cat={len(lgb_cats)}")
    lgb_oofs, lgb_tests = run_lgbm(X_lgb_tr, y, X_lgb_te, lgb_cats)

    # --- CatBoost ---
    print("\n--- CatBoost (2 configs × 5 folds) ---")
    X_cb_tr, X_cb_te, cb_cats = prepare_catboost(X_all, X_test)
    print(f"[catboost] features={X_cb_tr.shape[1]} cat_features={len(cb_cats)}")
    cb_oofs, cb_tests = run_catboost(X_cb_tr, y, X_cb_te, cb_cats)

    # --- Ensemble: equal weight average of all 4 model variants ---
    all_oofs = lgb_oofs + cb_oofs
    all_tests = lgb_tests + cb_tests
    n_models = len(all_oofs)

    ens_oof = np.mean(all_oofs, axis=0)
    ens_test = np.mean(all_tests, axis=0)

    print(f"\n--- Ensemble ({n_models} models) ---")
    print(f"  ROC-AUC = {roc_auc_score(y, ens_oof):.5f}")
    print(f"  PR-AUC  = {average_precision_score(y, ens_oof):.5f}")

    precision, recall, thresholds = precision_recall_curve(y, ens_oof)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1[:-1]))
    best_thresh = float(thresholds[best_idx])
    print(f"  Best F1 = {f1[best_idx]:.5f}  threshold={best_thresh:.4f}")
    print(f"  precision={precision[best_idx]:.3f}  recall={recall[best_idx]:.3f}")

    oof_pred = (ens_oof >= best_thresh).astype(int)
    print(f"\n{confusion_matrix(y, oof_pred)}")
    print(classification_report(y, oof_pred, digits=4))

    # --- Write submissions (both binary and probability) ---
    preds_binary = (ens_test >= best_thresh).astype(int)
    pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: preds_binary}).to_csv(
        "best_submission.csv", index=False
    )
    pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: ens_test}).to_csv(
        "best_submission_proba.csv", index=False
    )
    print(f"\n[submit] binary pos_rate={preds_binary.mean():.4f}")
    print(f"[submit] proba mean={ens_test.mean():.4f} min={ens_test.min():.4f} max={ens_test.max():.4f}")
    print(f"\n[done] time={time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    run()
