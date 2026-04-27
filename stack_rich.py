"""Rich-feature stacker: meta-LGB sees [base_logits + key raw features].

If base models miss something specific (a missing-data pattern, a categorical bin),
the meta can recover it by combining base scores with the raw signal.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import feat_pipeline as fp

EPS = 1e-7
KEY_FEATURES = [
    "SIZE", "PHASE_OF_FLIGHT", "TYPE_ENG", "AC_CLASS",
    "AC_MASS", "HEIGHT", "SPEED", "DISTANCE",
    "TIME_MINUTES", "MONTH", "YEAR",
    "LATITUDE", "LONGITUDE",
    "WARNED", "FAAREGION",
    "FREQ_AIRPORT_ID", "FREQ_OPID", "FREQ_SPECIES_ID",
    "NUM_STRUCK", "NUM_SEEN",
    "REMAINS_COLLECTED", "REMAINS_SENT",
    "AMA", "AMO",
    "AC_MASS_MISSING", "SPEED_MISSING",
]


def to_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def isotonic_cv(oof, y, test_proba, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_cal = np.zeros_like(oof); test_cals = []
    for tr, va in skf.split(oof, y):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[tr], y[tr])
        oof_cal[va] = iso.transform(oof[va])
        test_cals.append(iso.transform(test_proba))
    return oof_cal, np.mean(test_cals, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rate", type=float, default=0.18)
    ap.add_argument("--out", default="stack_rich_submission.csv")
    args = ap.parse_args()

    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    y = train[fp.TARGET].astype(int).values

    base_paths = {
        "lgbm": "lgbm_target_rate_submission",
        "lgbm_v2": "lgbm_v2_submission",
        "lgbm_int": "lgbm_interactions_submission",
        "lgbm_te": "lgbm_targetenc_submission",
        "xgb": "xgb_target_rate_submission",
        "xgb_seed": "xgb_seedavg_submission",
        "rf": "rf_target_rate_submission",
        "et": "et_target_rate_submission",
        "hgb": "hgb_target_rate_submission",
        "cat": "cat_target_rate_submission",
        "cat_seed": "cat_seedavg_submission",
        "cat_deep": "cat_deep_submission",
        "cat_grand": "cat_grand_submission",
        "ada": "adaboost_v2_submission",
    }
    base_paths = {k: v for k, v in base_paths.items() if Path(v + ".oof.npy").exists()}
    oofs, probas, names = [], [], []
    for n, b in base_paths.items():
        oofs.append(np.load(b + ".oof.npy"))
        probas.append(np.load(b + ".proba.npy"))
        names.append(n)
    oof_mat = np.array(oofs).T
    proba_mat = np.array(probas).T
    print(f"== {len(names)} bases loaded ==")

    # CV-isotonic each base
    iso_oof = np.zeros_like(oof_mat); iso_test = np.zeros_like(proba_mat)
    for i in range(len(names)):
        iso_oof[:, i], iso_test[:, i] = isotonic_cv(oof_mat[:, i], y, proba_mat[:, i])

    # logit-transformed iso probas
    Z_oof = to_logit(iso_oof); Z_test = to_logit(iso_test)

    # build the raw-feature side
    X_train_raw, X_test_raw = fp.make_features(train, test)
    X_train_lgb, X_test_lgb, cat_cols_all = fp.encode_for_lgb(X_train_raw, X_test_raw)
    avail = [c for c in KEY_FEATURES if c in X_train_lgb.columns]
    print(f"raw key features used: {len(avail)}")
    Xraw_tr = X_train_lgb[avail].astype("float32")
    Xraw_te = X_test_lgb[avail].astype("float32")
    cat_cols_meta = [c for c in cat_cols_all if c in avail]

    # combine: [base_logits | raw_features]
    base_cols = [f"BASE_{n}" for n in names]
    train_meta = pd.concat([pd.DataFrame(Z_oof, columns=base_cols, index=Xraw_tr.index),
                             Xraw_tr.reset_index(drop=True)], axis=1)
    test_meta = pd.concat([pd.DataFrame(Z_test, columns=base_cols, index=Xraw_te.index),
                            Xraw_te.reset_index(drop=True)], axis=1)
    print(f"meta features: {train_meta.shape[1]}  (base={len(names)}  raw={len(avail)})")

    # train meta-LGB
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(y)); iters = []
    params = dict(objective="binary", metric="average_precision", learning_rate=0.025,
                  num_leaves=31, min_data_in_leaf=200, feature_fraction=0.85,
                  bagging_fraction=0.85, bagging_freq=5, lambda_l2=2.0,
                  verbose=-1, seed=42)
    for fold, (tr, va) in enumerate(skf.split(train_meta, y), 1):
        ds_t = lgb.Dataset(train_meta.iloc[tr], y[tr], categorical_feature=cat_cols_meta)
        ds_v = lgb.Dataset(train_meta.iloc[va], y[va], categorical_feature=cat_cols_meta, reference=ds_t)
        b = lgb.train(params, ds_t, num_boost_round=2000, valid_sets=[ds_v],
                      callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        iters.append(b.best_iteration)
        meta_oof[va] = b.predict(train_meta.iloc[va], num_iteration=b.best_iteration)
        print(f"  fold{fold}: iter={b.best_iteration} auc={roc_auc_score(y[va], meta_oof[va]):.5f}")

    print(f"\n[stack-rich] OOF AUC={roc_auc_score(y, meta_oof):.5f}  AP={average_precision_score(y, meta_oof):.5f}")
    fi = int(np.mean(iters) * 1.05)
    print(f"  final iter={fi}")
    final = lgb.train(params, lgb.Dataset(train_meta, y, categorical_feature=cat_cols_meta), num_boost_round=fi)
    proba_test = final.predict(test_meta)

    # show top-20 feature importances
    fi_ser = pd.Series(final.feature_importance("gain"), index=train_meta.columns).sort_values(ascending=False)
    print("\nTop 20 meta features by gain:")
    print(fi_ser.head(20).to_string())

    for tr in [0.15, 0.175, 0.18, 0.185, 0.19]:
        thr = fp.threshold_for_target_rate(meta_oof, tr)
        pred = (meta_oof >= thr).astype(int)
        print(f"  target={tr:.3f} thr={thr:.4f} f1={f1_score(y, pred):.4f}")

    thr = fp.threshold_for_target_rate(proba_test, args.target_rate)
    preds = (proba_test >= thr).astype(int)
    fp.write_submission(test[fp.ID_COL], preds, Path(args.out))
    np.save(Path(args.out).with_suffix(".proba.npy"), proba_test)
    np.save(Path(args.out).with_suffix(".oof.npy"), meta_oof)


if __name__ == "__main__":
    main()
