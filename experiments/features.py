"""Shared feature engineering — no text from REMARKS/COMMENTS."""
import numpy as np
import pandas as pd

TARGET = "INDICATED_DAMAGE"
ID_COL = "INDEX_NR"

# Columns to ignore entirely (text + leakage-prone identifiers)
TEXT_COLS = ["REMARKS", "COMMENTS"]
DROP_COLS = [ID_COL, TARGET, "LOCATION", "BIRD_BAND_NUMBER", "REG", "FLT",
             "AIRCRAFT", "LUPDATE"] + TEXT_COLS

NUMBER_COLS = ["LATITUDE", "LONGITUDE", "HEIGHT", "SPEED", "DISTANCE",
               "NUM_SEEN", "NUM_STRUCK", "OUT_OF_RANGE_SPECIES",
               "AC_MASS", "NUM_ENGS",
               "ENG_1_POS", "ENG_2_POS", "ENG_3_POS", "ENG_4_POS",
               "REMAINS_COLLECTED", "REMAINS_SENT", "TRANSFER"]

LOW_CARD_CAT = ["TIME_OF_DAY", "FAAREGION", "AC_CLASS", "TYPE_ENG",
                "PHASE_OF_FLIGHT", "SKY", "WARNED", "SIZE", "SOURCE",
                "PRECIPITATION"]
HIGH_CARD_CAT = ["STATE", "OPID", "AIRPORT_ID", "SPECIES_ID",
                 "AMA", "AMO", "EMA", "EMO", "ENROUTE_STATE", "PERSON",
                 "RUNWAY", "OPERATOR", "AIRPORT", "SPECIES"]


def get_airport_map(df):
    df = df.copy()
    for col in ["LATITUDE", "LONGITUDE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = df[(df["AIRPORT_ID"].notna()) & (df["AIRPORT_ID"] != "ZZZZ")
              & (df["LATITUDE"].notna()) & (df["LONGITUDE"].notna())]
    coords = valid.groupby("AIRPORT_ID")[["LATITUDE", "LONGITUDE"]].median()
    return coords.to_dict("index")


def time_to_minutes(series):
    s = series.fillna("").astype(str).str.replace(":", "", regex=False)
    s = pd.to_numeric(s, errors="coerce")
    h = (s // 100).clip(0, 23)
    m = (s % 100).clip(0, 59)
    return h * 60 + m


def make_base(df, airport_map):
    df = df.copy()
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    # lat/lon imputation from airport_id
    for coord in ["LATITUDE", "LONGITUDE"]:
        m = {k: v[coord] for k, v in airport_map.items()}
        df[coord] = df[coord].fillna(df["AIRPORT_ID"].map(m))

    out = {}
    d = pd.to_datetime(df["INCIDENT_DATE"], format="mixed", errors="coerce")
    year = pd.to_numeric(df.get("INCIDENT_YEAR"), errors="coerce")
    year = year.fillna(d.dt.year).fillna(2015)
    out["YEAR"] = year
    out["YEAR_REL"] = year - 2015
    month = pd.to_numeric(df.get("INCIDENT_MONTH"), errors="coerce").fillna(d.dt.month).fillna(6)
    out["MONTH"] = month
    out["MONTH_SIN"] = np.sin(2 * np.pi * month / 12.0)
    out["MONTH_COS"] = np.cos(2 * np.pi * month / 12.0)
    out["DAY_OF_YEAR"] = d.dt.dayofyear.fillna(180)

    if "TIME" in df.columns:
        mins = time_to_minutes(df["TIME"])
        out["MINUTES"] = mins
        out["HOUR_SIN"] = np.sin(2 * np.pi * mins / 1440.0)
        out["HOUR_COS"] = np.cos(2 * np.pi * mins / 1440.0)

    for col in NUMBER_COLS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["HEIGHT", "SPEED", "DISTANCE"]:
        if col in out:
            v = out[col].fillna(0).clip(lower=0)
            out["LOG_" + col] = np.log1p(v)

    # Physics-ish kinetic energy proxy
    if "SPEED" in out and "AC_MASS" in out:
        sp = out["SPEED"].fillna(out["SPEED"].median())
        ms = out["AC_MASS"].fillna(out["AC_MASS"].median())
        out["KE_PROXY"] = ms * (sp ** 2)
        out["LOG_KE"] = np.log1p(out["KE_PROXY"].clip(lower=0))

    if "LATITUDE" in out and "LONGITUDE" in out:
        out["LAT_BIN"] = (out["LATITUDE"] / 2).round() * 2
        out["LON_BIN"] = (out["LONGITUDE"] / 2).round() * 2

    # Missingness flags for high-impact numeric cols
    for col in ["SPEED", "HEIGHT", "DISTANCE", "AC_MASS"]:
        if col in df.columns:
            out["MISS_" + col] = df[col].isna().astype("int8")

    # Categorical raw strings — caller decides encoding
    for col in LOW_CARD_CAT + HIGH_CARD_CAT:
        if col in df.columns:
            out[col] = df[col].fillna("Unknown").astype(str)

    return pd.DataFrame(out, index=df.index)


def freq_encode(train_series, test_series, smooth=1):
    counts = train_series.value_counts()
    total = counts.sum() + smooth * len(counts)
    rates = (counts + smooth) / total
    return train_series.map(rates).fillna(smooth / total), \
           test_series.map(rates).fillna(smooth / total)


def kfold_target_encode(train_series, target, test_series, n_splits=5, smooth=20, seed=42):
    """Out-of-fold target encoding to prevent leakage."""
    from sklearn.model_selection import StratifiedKFold
    prior = target.mean()
    oof = np.full(len(train_series), prior, dtype="float32")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(train_series, target)):
        tr_s = train_series.iloc[tr_idx]
        tr_y = target.iloc[tr_idx]
        agg = tr_y.groupby(tr_s).agg(["sum", "count"])
        enc = (agg["sum"] + smooth * prior) / (agg["count"] + smooth)
        oof[val_idx] = train_series.iloc[val_idx].map(enc).fillna(prior).values
    # full-train fit for test
    agg = target.groupby(train_series).agg(["sum", "count"])
    full_enc = (agg["sum"] + smooth * prior) / (agg["count"] + smooth)
    test_enc = test_series.map(full_enc).fillna(prior).values.astype("float32")
    return oof, test_enc
