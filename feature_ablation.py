#!/usr/bin/env python3
"""
Feature group ablation study.

Trains a quick CatBoost model for each feature group individually,
then cumulatively, to see which groups contribute the most to PR-AUC.
"""

import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "INDICATED_DAMAGE"
RANDOM_STATE = 42


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


def build_feature_groups(df, airport_map):
    """Build feature groups as separate dicts. Each group is (features_df, cat_cols)."""
    df = impute_coords(df, airport_map)
    groups = {}

    # 1. Location
    lat = df["LATITUDE"]
    lon = df["LONGITUDE"]
    loc = pd.DataFrame({
        "LATITUDE": lat, "LONGITUDE": lon,
        "LAT_SIN": np.sin(np.deg2rad(lat)), "LAT_COS": np.cos(np.deg2rad(lat)),
        "LON_SIN": np.sin(np.deg2rad(lon)), "LON_COS": np.cos(np.deg2rad(lon)),
        "DIST_FROM_US_CENTER": np.sqrt((lat - 39.5)**2 + (lon + 98.35)**2),
    }, index=df.index)
    groups["Location (lat/lon)"] = (loc, [])

    # 2. Airport ID
    aid = pd.DataFrame({"AIRPORT_ID": df["AIRPORT_ID"].fillna("_m_").astype(str)}, index=df.index)
    groups["AIRPORT_ID"] = (aid, ["AIRPORT_ID"])

    # 3. Date/Time
    dates = pd.to_datetime(df["INCIDENT_DATE"], format="mixed", errors="coerce")
    minutes = time_to_minutes(df["TIME"])
    dt = pd.DataFrame({
        "YEAR": dates.dt.year,
        "MONTH_SIN": np.sin(2 * np.pi * dates.dt.month.fillna(6) / 12.0),
        "MONTH_COS": np.cos(2 * np.pi * dates.dt.month.fillna(6) / 12.0),
        "MINUTES": minutes,
        "TIME_SIN": np.sin(2 * np.pi * minutes.fillna(720) / 1440.0),
        "TIME_COS": np.cos(2 * np.pi * minutes.fillna(720) / 1440.0),
        "TIME_OF_DAY": df["TIME_OF_DAY"].fillna("_m_").astype(str),
    }, index=df.index)
    groups["Date/Time"] = (dt, ["TIME_OF_DAY"])

    # 4. Bird info (species, size, num_seen)
    bird = pd.DataFrame({
        "SPECIES_ID": df["SPECIES_ID"].fillna("_m_").astype(str),
        "SIZE": df["SIZE"].fillna("_m_").astype(str),
        "NUM_SEEN": pd.to_numeric(df["NUM_SEEN"], errors="coerce"),
        "OUT_OF_RANGE_SPECIES": pd.to_numeric(df["OUT_OF_RANGE_SPECIES"], errors="coerce"),
    }, index=df.index)
    groups["Bird (species/size/count)"] = (bird, ["SPECIES_ID", "SIZE"])

    # 5. Aircraft info
    ac = pd.DataFrame({
        "AC_CLASS": df["AC_CLASS"].fillna("_m_").astype(str),
        "AC_MASS": pd.to_numeric(df["AC_MASS"], errors="coerce"),
        "TYPE_ENG": df["TYPE_ENG"].fillna("_m_").astype(str),
        "NUM_ENGS": pd.to_numeric(df["NUM_ENGS"], errors="coerce"),
        "AMA": df["AMA"].fillna("_m_").astype(str),
        "AMO": df["AMO"].fillna("_m_").astype(str),
        "EMA": df["EMA"].fillna("_m_").astype(str),
        "EMO": df["EMO"].fillna("_m_").astype(str),
    }, index=df.index)
    groups["Aircraft (class/mass/engine)"] = (ac, ["AC_CLASS", "TYPE_ENG", "AMA", "AMO", "EMA", "EMO"])

    # 6. Flight conditions
    flight = pd.DataFrame({
        "PHASE_OF_FLIGHT": df["PHASE_OF_FLIGHT"].fillna("_m_").astype(str),
        "HEIGHT": pd.to_numeric(df["HEIGHT"], errors="coerce"),
        "SPEED": pd.to_numeric(df["SPEED"], errors="coerce"),
        "DISTANCE": pd.to_numeric(df["DISTANCE"], errors="coerce"),
        "LOG_HEIGHT": np.log1p(pd.to_numeric(df["HEIGHT"], errors="coerce").clip(lower=0)),
        "LOG_SPEED": np.log1p(pd.to_numeric(df["SPEED"], errors="coerce").clip(lower=0)),
    }, index=df.index)
    groups["Flight (phase/height/speed)"] = (flight, ["PHASE_OF_FLIGHT"])

    # 7. Weather
    wx = pd.DataFrame({
        "SKY": df["SKY"].fillna("_m_").astype(str),
        "PRECIPITATION": df["PRECIPITATION"].fillna("_m_").astype(str),
    }, index=df.index)
    groups["Weather (sky/precip)"] = (wx, ["SKY", "PRECIPITATION"])

    # 8. Operator
    op = pd.DataFrame({
        "OPID": df["OPID"].fillna("_m_").astype(str),
        "OPERATOR": df["OPERATOR"].fillna("_m_").astype(str),
    }, index=df.index)
    groups["Operator"] = (op, ["OPID", "OPERATOR"])

    # 9. Warned
    warned = pd.DataFrame({
        "WARNED": df["WARNED"].fillna("_m_").astype(str),
    }, index=df.index)
    groups["Warned"] = (warned, ["WARNED"])

    # 10. Post-strike (remains, num_struck)
    NUM_STRUCK_MAP = {"1": 1, "10-Feb": 5, "11-100": 50, "More than 100": 150}
    ps = pd.DataFrame({
        "REMAINS_COLLECTED": pd.to_numeric(df["REMAINS_COLLECTED"], errors="coerce"),
        "REMAINS_SENT": pd.to_numeric(df["REMAINS_SENT"], errors="coerce"),
        "NUM_STRUCK": df["NUM_STRUCK"].fillna("_m_").astype(str),
        "NUM_STRUCK_ORD": df["NUM_STRUCK"].map(NUM_STRUCK_MAP),
    }, index=df.index)
    groups["Post-strike (remains/struck)"] = (ps, ["NUM_STRUCK"])

    # 11. Reporting (SOURCE, PERSON) - known leaky
    rpt = pd.DataFrame({
        "SOURCE": df["SOURCE"].fillna("_m_").astype(str),
        "PERSON": df["PERSON"].fillna("_m_").astype(str),
    }, index=df.index)
    groups["Reporting (SOURCE/PERSON)"] = (rpt, ["SOURCE", "PERSON"])

    # 12. Runway / State / Region
    geo_admin = pd.DataFrame({
        "RUNWAY": df["RUNWAY"].fillna("_m_").astype(str),
        "STATE": df["STATE"].fillna("_m_").astype(str),
        "FAAREGION": df["FAAREGION"].fillna("_m_").astype(str),
        "ENROUTE_STATE": df["ENROUTE_STATE"].fillna("_m_").astype(str),
    }, index=df.index)
    groups["Geo-admin (state/region/runway)"] = (geo_admin, ["RUNWAY", "STATE", "FAAREGION", "ENROUTE_STATE"])

    return groups


def quick_eval(X, y, cat_cols, n_folds=3):
    """Fast 3-fold eval with CatBoost."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X))

    X = X.replace([np.inf, -np.inf], np.nan)

    for tr_idx, va_idx in skf.split(X, y):
        model = CatBoostClassifier(
            iterations=1000, learning_rate=0.08, depth=6,
            l2_leaf_reg=5.0, loss_function="Logloss", eval_metric="PRAUC",
            auto_class_weights="SqrtBalanced", random_seed=RANDOM_STATE,
            od_type="Iter", od_wait=80, verbose=0, allow_writing_files=False,
        )
        train_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols)
        valid_pool = Pool(X.iloc[va_idx], y.iloc[va_idx], cat_features=cat_cols)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        oof[va_idx] = model.predict_proba(valid_pool)[:, 1]

    return roc_auc_score(y, oof), average_precision_score(y, oof)


def run():
    start = time.perf_counter()
    train_df = pd.read_csv("train.csv", low_memory=False)
    y = train_df[TARGET].astype(int)
    airport_map = get_airport_map(train_df)

    groups = build_feature_groups(train_df, airport_map)

    # --- Part 1: Each group alone ---
    print("=" * 70)
    print("PART 1: Each feature group ALONE")
    print("=" * 70)
    solo_results = {}
    for name, (feat_df, cat_cols) in groups.items():
        t0 = time.perf_counter()
        roc, pr = quick_eval(feat_df, y, cat_cols)
        elapsed = time.perf_counter() - t0
        solo_results[name] = (roc, pr)
        print(f"  {name:40s}  ROC={roc:.4f}  PR-AUC={pr:.4f}  ({elapsed:.0f}s)")

    # Sort by PR-AUC
    print("\nRanked by PR-AUC (standalone):")
    for name, (roc, pr) in sorted(solo_results.items(), key=lambda x: -x[1][1]):
        print(f"  {pr:.4f}  {name}")

    # --- Part 2: Cumulative addition (greedy, add best group first) ---
    print("\n" + "=" * 70)
    print("PART 2: Cumulative (greedy add best remaining group)")
    print("=" * 70)

    remaining = dict(groups)
    combined_feat = pd.DataFrame(index=train_df.index)
    combined_cats = []
    cumulative_results = []

    # Baseline: random
    print(f"  {'(baseline - random)':40s}  ROC=0.5000  PR-AUC={y.mean():.4f}")

    while remaining:
        best_name = None
        best_pr = -1
        best_roc = -1

        for name, (feat_df, cat_cols) in remaining.items():
            trial_feat = pd.concat([combined_feat, feat_df], axis=1)
            trial_cats = combined_cats + cat_cols
            roc, pr = quick_eval(trial_feat, y, trial_cats)
            if pr > best_pr:
                best_pr = pr
                best_roc = roc
                best_name = name

        feat_df, cat_cols = remaining.pop(best_name)
        combined_feat = pd.concat([combined_feat, feat_df], axis=1)
        combined_cats = combined_cats + cat_cols
        cumulative_results.append((best_name, best_roc, best_pr))
        print(f"  +{best_name:39s}  ROC={best_roc:.4f}  PR-AUC={best_pr:.4f}  (cols={combined_feat.shape[1]})")

    print(f"\n[done] total time={time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    run()
