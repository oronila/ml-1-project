"""Shared feature engineering: NO text fields. Physics + geo + categoricals only.

Hard rule: NEVER use REMARKS, COMMENTS, AIRPORT name, OPERATOR name, or SPECIES name
strings. Only use codes (AIRPORT_ID, OPID, SPECIES_ID).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"

TEXT_BANNED = [
    "REMARKS", "COMMENTS", "AIRPORT", "OPERATOR", "SPECIES",
    "LOCATION", "AIRCRAFT", "REG", "FLT", "PERSON", "SOURCE",
    "BIRD_BAND_NUMBER", "LUPDATE", "TRANSFER", "ENROUTE_STATE",
]

NUMERIC_RAW = [
    "HEIGHT", "SPEED", "DISTANCE", "AC_MASS", "NUM_ENGS",
    "OUT_OF_RANGE_SPECIES", "REMAINS_COLLECTED", "REMAINS_SENT",
    "INCIDENT_MONTH", "EMA", "EMO",
    "ENG_1_POS", "ENG_2_POS", "ENG_3_POS", "ENG_4_POS",
]

CATEGORICAL_LOWCARD = [
    "TIME_OF_DAY", "STATE", "FAAREGION", "AC_CLASS", "TYPE_ENG",
    "PHASE_OF_FLIGHT", "SKY", "PRECIPITATION", "WARNED", "SIZE",
    "AMA", "AMO",
]

CATEGORICAL_HIGHCARD = ["AIRPORT_ID", "OPID", "SPECIES_ID", "RUNWAY"]


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _time_to_minutes(series: pd.Series) -> pd.Series:
    parts = series.astype(str).str.split(":", n=1, expand=True)
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce") if 1 in parts.columns else pd.Series(0, index=series.index)
    m = m.fillna(0)
    valid = h.between(0, 23) & m.between(0, 59)
    return (h * 60 + m).where(valid)


def _num_to_struck(series: pd.Series) -> pd.Series:
    """NUM_SEEN/NUM_STRUCK come as awkward strings like '10-Feb' meaning 2-10."""
    s = series.astype(str)
    s_low = s.str.lower()
    mp = {
        "1": 1.0, "2": 2.0, "11-50": 30.0, "51-100": 75.0, "100+": 150.0,
        "10-feb": 6.0, "100-mar": 50.0, "11-may": 8.0, "5-mar": 4.0,
    }
    out = s_low.map(mp)
    leftover = out.isna()
    out.loc[leftover] = pd.to_numeric(s.loc[leftover], errors="coerce")
    return out


def _airport_coord_map(combined: pd.DataFrame) -> dict:
    df = combined[["AIRPORT_ID", "LATITUDE", "LONGITUDE"]].copy()
    df["LATITUDE"] = _to_num(df["LATITUDE"])
    df["LONGITUDE"] = _to_num(df["LONGITUDE"])
    valid = df[df["AIRPORT_ID"].notna() & (df["AIRPORT_ID"] != "ZZZZ") & df["LATITUDE"].notna() & df["LONGITUDE"].notna()]
    return valid.groupby("AIRPORT_ID")[["LATITUDE", "LONGITUDE"]].median().to_dict("index")


def _build_features(df: pd.DataFrame, airport_map: dict, anchor: pd.Timestamp) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # date features
    d = pd.to_datetime(df["INCIDENT_DATE"], format="mixed", errors="coerce")
    out["DAYS_SINCE_ANCHOR"] = (d - anchor).dt.days.astype("float32")
    out["YEAR"] = d.dt.year.astype("float32")
    out["MONTH"] = d.dt.month.astype("float32")
    out["DAY_OF_WEEK"] = d.dt.dayofweek.astype("float32")
    out["DAY_OF_YEAR"] = d.dt.dayofyear.astype("float32")
    out["MONTH_SIN"] = np.sin(2 * np.pi * d.dt.month.fillna(6) / 12)
    out["MONTH_COS"] = np.cos(2 * np.pi * d.dt.month.fillna(6) / 12)

    # time
    if "TIME" in df.columns:
        t = _time_to_minutes(df["TIME"])
        out["TIME_MINUTES"] = t.astype("float32")
        out["HOUR_SIN"] = np.sin(2 * np.pi * t.fillna(720) / 1440)
        out["HOUR_COS"] = np.cos(2 * np.pi * t.fillna(720) / 1440)
        out["TIME_MISSING"] = t.isna().astype("float32")

    # lat/lon with airport-id imputation
    lat = _to_num(df["LATITUDE"])
    lon = _to_num(df["LONGITUDE"])
    lat_lookup = {k: v["LATITUDE"] for k, v in airport_map.items()}
    lon_lookup = {k: v["LONGITUDE"] for k, v in airport_map.items()}
    lat = lat.fillna(df["AIRPORT_ID"].map(lat_lookup))
    lon = lon.fillna(df["AIRPORT_ID"].map(lon_lookup))
    out["LATITUDE"] = lat.astype("float32")
    out["LONGITUDE"] = lon.astype("float32")
    out["LAT_SIN"] = np.sin(np.deg2rad(lat.fillna(0)))
    out["LAT_COS"] = np.cos(np.deg2rad(lat.fillna(0)))
    out["LON_SIN"] = np.sin(np.deg2rad(lon.fillna(0)))
    out["LON_COS"] = np.cos(np.deg2rad(lon.fillna(0)))
    out["LAT_X_LON"] = (lat.fillna(0) * lon.fillna(0)).astype("float32")
    out["LAT_MISSING"] = lat.isna().astype("float32")

    # numeric raw
    for c in NUMERIC_RAW:
        if c in df.columns:
            out[c] = _to_num(df[c]).astype("float32")
            out[c + "_MISSING"] = df[c].isna().astype("float32")

    # NUM_SEEN / NUM_STRUCK -- careful: NUM_STRUCK is post-strike but informative
    if "NUM_SEEN" in df.columns:
        out["NUM_SEEN"] = _num_to_struck(df["NUM_SEEN"]).astype("float32")
        out["NUM_SEEN_MISSING"] = df["NUM_SEEN"].isna().astype("float32")
    if "NUM_STRUCK" in df.columns:
        out["NUM_STRUCK"] = _num_to_struck(df["NUM_STRUCK"]).astype("float32")
        out["NUM_STRUCK_MISSING"] = df["NUM_STRUCK"].isna().astype("float32")

    # log transforms for skewed
    for c in ["HEIGHT", "SPEED", "DISTANCE", "AC_MASS"]:
        if c in out.columns:
            v = out[c].fillna(0).clip(lower=0)
            out["LOG_" + c] = np.log1p(v).astype("float32")

    # kinetic-energy style proxy
    if "AC_MASS" in out.columns and "SPEED" in out.columns:
        m = out["AC_MASS"].fillna(0).clip(lower=0)
        s = out["SPEED"].fillna(0).clip(lower=0)
        out["KE_PROXY"] = (m * s * s).astype("float32")
        out["LOG_KE_PROXY"] = np.log1p(out["KE_PROXY"]).astype("float32")

    # categorical low-card -> string
    for c in CATEGORICAL_LOWCARD:
        if c in df.columns:
            out[c] = df[c].fillna("Unknown").astype(str)

    # high-card categoricals: keep code + frequency
    for c in CATEGORICAL_HIGHCARD:
        if c in df.columns:
            out[c] = df[c].fillna("Unknown").astype(str)

    return out


def make_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Build train and test feature frames consistently, with banned cols dropped."""
    combined_geo = pd.concat([train_df[["AIRPORT_ID", "LATITUDE", "LONGITUDE"]],
                              test_df[["AIRPORT_ID", "LATITUDE", "LONGITUDE"]]])
    airport_map = _airport_coord_map(combined_geo)
    d_anchor = pd.to_datetime(train_df["INCIDENT_DATE"], format="mixed", errors="coerce").min()

    X_train = _build_features(train_df, airport_map, d_anchor)
    X_test = _build_features(test_df, airport_map, d_anchor)

    # frequency encode high-card cols based on train+test combined
    for c in CATEGORICAL_HIGHCARD:
        if c not in X_train.columns:
            continue
        freq = pd.concat([X_train[c], X_test[c]]).value_counts(normalize=True).to_dict()
        X_train["FREQ_" + c] = X_train[c].map(freq).fillna(0).astype("float32")
        X_test["FREQ_" + c] = X_test[c].map(freq).fillna(0).astype("float32")

    return X_train, X_test


def encode_for_lgb(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Encode all categoricals as integer codes (LightGBM-style). Returns (X_train, X_test, cat_cols)."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    cat_cols = []
    for c in CATEGORICAL_LOWCARD + CATEGORICAL_HIGHCARD:
        if c not in X_train.columns:
            continue
        # union categories from train+test so test never gets unknown
        cats = pd.Index(pd.concat([X_train[c], X_test[c]]).astype(str).unique())
        cat_dtype = pd.CategoricalDtype(categories=cats)
        X_train[c] = X_train[c].astype(cat_dtype).cat.codes.astype("int32")
        X_test[c] = X_test[c].astype(cat_dtype).cat.codes.astype("int32")
        cat_cols.append(c)
    # ensure float32 elsewhere
    for c in X_train.columns:
        if c in cat_cols:
            continue
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce").astype("float32")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").astype("float32")
    return X_train, X_test, cat_cols


def encode_for_dense(X_train: pd.DataFrame, X_test: pd.DataFrame, top_k: int = 50):
    """One-hot encode low-card categoricals, frequency encode high-card. Returns dense float32."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    keep_cats = [c for c in CATEGORICAL_LOWCARD if c in X_train.columns]
    for c in keep_cats:
        top = X_train[c].value_counts().index[:top_k]
        X_train[c] = X_train[c].where(X_train[c].isin(top), "Other")
        X_test[c] = X_test[c].where(X_test[c].isin(top), "Other")
    combined = pd.concat([X_train, X_test], axis=0)
    combined = pd.get_dummies(combined, columns=keep_cats, dtype="float32")
    # drop remaining string columns (high-card raw - we already have FREQ_)
    for c in CATEGORICAL_HIGHCARD:
        if c in combined.columns:
            combined = combined.drop(columns=[c])
    X_train = combined.iloc[: len(X_train)].copy()
    X_test = combined.iloc[len(X_train):].copy()
    medians = X_train.median(numeric_only=True).fillna(0)
    X_train = X_train.fillna(medians).fillna(0).astype("float32")
    X_test = X_test.fillna(medians).fillna(0).astype("float32")
    return X_train, X_test


def threshold_for_target_rate(proba: np.ndarray, target_rate: float) -> float:
    """Return the threshold that yields exactly `target_rate` positive rate on `proba`."""
    if target_rate <= 0:
        return 1.0
    if target_rate >= 1:
        return 0.0
    n = len(proba)
    k = int(round(target_rate * n))
    k = max(1, min(n - 1, k))
    sorted_p = np.sort(proba)[::-1]
    return sorted_p[k - 1]


def write_submission(test_ids: pd.Series, preds: np.ndarray, path: Path) -> None:
    sub = pd.DataFrame({ID_COL: test_ids.values, TARGET: preds.astype(int)})
    sub.to_csv(path, index=False)
    pos = int(preds.sum())
    print(f"[submit] wrote {path} rows={len(sub)} positives={pos} pos_rate={pos / len(sub):.4f}")
