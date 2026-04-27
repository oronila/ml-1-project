"""Train each base model and save the trained estimator + features as artifacts.

Saves:
- models/<name>.joblib              — trained sklearn-style estimator (or list of K-fold estimators)
- models/<name>.meta.json           — params, OOF metrics, feature names
- models/feature_pipeline.joblib    — pickled feat_pipeline state (anchor date, airport map)

Once run, `predict_with_saved.py` can load any model and produce predictions on a
new test CSV without retraining.

Run:
    python save_models.py --models cat_seed,xgb_seed,hgb,lgbm_v2,ada
    python save_models.py --models all
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import feat_pipeline as fp

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def _save_artifact(name, estimators, oof, proba_test, feat_names, meta):
    """estimators: list of K models (one per fold) for refittable. proba_test: averaged across folds."""
    joblib.dump(estimators, MODEL_DIR / f"{name}.joblib")
    np.save(MODEL_DIR / f"{name}.oof.npy", oof)
    np.save(MODEL_DIR / f"{name}.proba.npy", proba_test)
    json_meta = dict(meta)
    json_meta["feat_names"] = list(feat_names)
    json_meta["oof_auc"] = float(roc_auc_score(meta["_y"], oof))
    json_meta["oof_ap"] = float(average_precision_score(meta["_y"], oof))
    json_meta.pop("_y", None)
    (MODEL_DIR / f"{name}.meta.json").write_text(json.dumps(json_meta, indent=2, default=str))
    print(f"  [saved] {name}: oof_auc={json_meta['oof_auc']:.5f} oof_ap={json_meta['oof_ap']:.5f}")


def train_lgbm(X_train, y, X_test, cat_cols, name, params, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    proba_test = np.zeros(len(X_test))
    estimators = []
    for fold, (tr, va) in enumerate(skf.split(X_train, y), 1):
        ds_t = lgb.Dataset(X_train.iloc[tr], y.iloc[tr], categorical_feature=cat_cols)
        ds_v = lgb.Dataset(X_train.iloc[va], y.iloc[va], categorical_feature=cat_cols, reference=ds_t)
        b = lgb.train(params, ds_t, num_boost_round=2500, valid_sets=[ds_v],
                      callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        estimators.append(b)
        oof[va] = b.predict(X_train.iloc[va], num_iteration=b.best_iteration)
        proba_test += b.predict(X_test, num_iteration=b.best_iteration) / 5
        print(f"    {name} fold{fold}: iter={b.best_iteration} auc={roc_auc_score(y.iloc[va], oof[va]):.5f}")
    _save_artifact(name, estimators, oof, proba_test, X_train.columns,
                   {"_y": y.values, "model_type": "lightgbm", "params": params, "seed": seed})


def train_xgb(X_train, y, X_test, name, params, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    proba_test = np.zeros(len(X_test))
    estimators = []
    for fold, (tr, va) in enumerate(skf.split(X_train, y), 1):
        dtr = xgb.DMatrix(X_train.iloc[tr], label=y.iloc[tr])
        dva = xgb.DMatrix(X_train.iloc[va], label=y.iloc[va])
        bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "va")],
                        early_stopping_rounds=80, verbose_eval=0)
        estimators.append(bst)
        oof[va] = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        proba_test += bst.predict(xgb.DMatrix(X_test), iteration_range=(0, bst.best_iteration + 1)) / 5
        print(f"    {name} fold{fold}: iter={bst.best_iteration} auc={roc_auc_score(y.iloc[va], oof[va]):.5f}")
    _save_artifact(name, estimators, oof, proba_test, X_train.columns,
                   {"_y": y.values, "model_type": "xgboost", "params": params, "seed": seed})


def train_cat(X_train_raw, y, X_test_raw, cat_cols, name, depth=8, lr=0.04, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    proba_test = np.zeros(len(X_test_raw))
    estimators = []
    for fold, (tr, va) in enumerate(skf.split(X_train_raw, y), 1):
        Xt = X_train_raw.iloc[tr]; Xv = X_train_raw.iloc[va]
        yt = y.iloc[tr]; yv = y.iloc[va]
        pool_t = Pool(Xt, yt, cat_features=cat_cols)
        pool_v = Pool(Xv, yv, cat_features=cat_cols)
        cb = CatBoostClassifier(
            iterations=2500, learning_rate=lr, depth=depth, l2_leaf_reg=3.0,
            random_strength=1.5, bagging_temperature=0.4, border_count=128,
            loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
            random_seed=seed, verbose=0, early_stopping_rounds=80, allow_writing_files=False,
        )
        cb.fit(pool_t, eval_set=pool_v)
        estimators.append(cb)
        oof[va] = cb.predict_proba(pool_v)[:, 1]
        proba_test += cb.predict_proba(Pool(X_test_raw, cat_features=cat_cols))[:, 1] / 5
        print(f"    {name} fold{fold}: iter={cb.tree_count_} auc={roc_auc_score(yv, oof[va]):.5f}")
    _save_artifact(name, estimators, oof, proba_test, X_train_raw.columns,
                   {"_y": y.values, "model_type": "catboost", "depth": depth, "lr": lr, "seed": seed})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="cat_seed,xgb_seed,hgb,lgbm_v2,ada,cat_grand",
                    help="comma-separated; or 'all'")
    args = ap.parse_args()

    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    X_train_raw, X_test_raw = fp.make_features(train, test)
    X_train, X_test, cat_cols = fp.encode_for_lgb(X_train_raw, X_test_raw)
    y = train[fp.TARGET].astype(int)
    pos_w = (1 - y.mean()) / y.mean()
    print(f"X_train={X_train.shape} pos_w={pos_w:.3f}")

    # save the feature pipeline state too
    pipeline_state = {
        "anchor": pd.to_datetime(train["INCIDENT_DATE"], format="mixed", errors="coerce").min(),
        "airport_map": fp._airport_coord_map(pd.concat([train[["AIRPORT_ID","LATITUDE","LONGITUDE"]], test[["AIRPORT_ID","LATITUDE","LONGITUDE"]]])),
        "cat_cols": cat_cols,
        "feat_names": list(X_train.columns),
    }
    joblib.dump(pipeline_state, MODEL_DIR / "feature_pipeline.joblib")
    print(f"[saved] models/feature_pipeline.joblib")

    requested = args.models.split(",")
    if "all" in requested:
        requested = ["lgbm", "lgbm_v2", "xgb_seed", "hgb", "rf", "et", "cat_seed", "ada"]

    if "lgbm_v2" in requested:
        print("\n=== lgbm_v2 ===")
        train_lgbm(X_train, y, X_test, cat_cols, "lgbm_v2",
                   dict(objective="binary", metric="average_precision", learning_rate=0.025,
                        num_leaves=255, min_data_in_leaf=40, feature_fraction=0.7,
                        bagging_fraction=0.8, bagging_freq=5, lambda_l1=0.5, lambda_l2=2.0,
                        scale_pos_weight=pos_w, path_smooth=10.0, verbose=-1, seed=99), seed=99)

    if "xgb_seed" in requested:
        print("\n=== xgb_seed (single seed=42) ===")
        train_xgb(X_train, y, X_test, "xgb_seed",
                  dict(objective="binary:logistic", eval_metric=["aucpr", "auc"], eta=0.04,
                       max_depth=8, min_child_weight=8, subsample=0.85, colsample_bytree=0.8,
                       reg_lambda=1.0, scale_pos_weight=pos_w, tree_method="hist", seed=42,
                       nthread=-1, verbosity=0), seed=42)

    if "cat_seed" in requested:
        # CatBoost requires raw frame with string cats
        cat_only_cols = [c for c in fp.CATEGORICAL_LOWCARD + fp.CATEGORICAL_HIGHCARD if c in X_train_raw.columns]
        Xt_raw = X_train_raw.copy(); Xs_raw = X_test_raw.copy()
        for c in cat_only_cols:
            Xt_raw[c] = Xt_raw[c].fillna("Unknown").astype(str)
            Xs_raw[c] = Xs_raw[c].fillna("Unknown").astype(str)
        for c in [c for c in Xt_raw.columns if c not in cat_only_cols]:
            Xt_raw[c] = pd.to_numeric(Xt_raw[c], errors="coerce").astype("float32")
            Xs_raw[c] = pd.to_numeric(Xs_raw[c], errors="coerce").astype("float32")
        print("\n=== cat_seed (single seed=42) ===")
        train_cat(Xt_raw, y, Xs_raw, cat_only_cols, "cat_seed", depth=8, lr=0.04, seed=42)

    if "cat_grand" in requested:
        cat_only_cols = [c for c in fp.CATEGORICAL_LOWCARD + fp.CATEGORICAL_HIGHCARD if c in X_train_raw.columns]
        Xt_raw = X_train_raw.copy(); Xs_raw = X_test_raw.copy()
        for c in cat_only_cols:
            Xt_raw[c] = Xt_raw[c].fillna("Unknown").astype(str)
            Xs_raw[c] = Xs_raw[c].fillna("Unknown").astype(str)
        for c in [c for c in Xt_raw.columns if c not in cat_only_cols]:
            Xt_raw[c] = pd.to_numeric(Xt_raw[c], errors="coerce").astype("float32")
            Xs_raw[c] = pd.to_numeric(Xs_raw[c], errors="coerce").astype("float32")
        for s in [99, 1337]:
            print(f"\n=== cat_grand_s{s} ===")
            train_cat(Xt_raw, y, Xs_raw, cat_only_cols, f"cat_grand_s{s}", depth=8, lr=0.04, seed=s)

    if "hgb" in requested:
        print("\n=== hgb ===")
        native_cats = [c for c in cat_cols
                       if (max(X_train[c].max(), X_test[c].max()) + 1) <= 255 and c in fp.CATEGORICAL_LOWCARD]
        cat_mask = [c in native_cats for c in X_train.columns]
        sw = np.where(y == 1, pos_w, 1.0)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y)); proba_test = np.zeros(len(X_test)); ests = []
        for fold, (tr, va) in enumerate(skf.split(X_train, y), 1):
            h = HistGradientBoostingClassifier(
                max_iter=1000, learning_rate=0.04, max_leaf_nodes=63, min_samples_leaf=40,
                l2_regularization=1.0, categorical_features=cat_mask,
                early_stopping=True, validation_fraction=0.15, n_iter_no_change=80,
                random_state=42 + fold)
            h.fit(X_train.iloc[tr], y.iloc[tr], sample_weight=sw[tr])
            ests.append(h)
            oof[va] = h.predict_proba(X_train.iloc[va])[:, 1]
            proba_test += h.predict_proba(X_test)[:, 1] / 5
            print(f"    hgb fold{fold}: n_iter={h.n_iter_} auc={roc_auc_score(y.iloc[va], oof[va]):.5f}")
        _save_artifact("hgb", ests, oof, proba_test, X_train.columns,
                       {"_y": y.values, "model_type": "hgb"})

    if "ada" in requested:
        print("\n=== ada ===")
        Xt = X_train.fillna(-1).astype("float32"); Xs = X_test.fillna(-1).astype("float32")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof = np.zeros(len(y)); proba_test = np.zeros(len(Xs)); ests = []
        for fold, (tr, va) in enumerate(skf.split(Xt, y), 1):
            base = DecisionTreeClassifier(max_depth=15, min_samples_leaf=100, class_weight="balanced", random_state=42)
            a = AdaBoostClassifier(estimator=base, n_estimators=200, learning_rate=0.1, random_state=42)
            a.fit(Xt.iloc[tr], y.iloc[tr])
            ests.append(a)
            oof[va] = a.predict_proba(Xt.iloc[va])[:, 1]
            proba_test += a.predict_proba(Xs)[:, 1] / 3
            print(f"    ada fold{fold}: auc={roc_auc_score(y.iloc[va], oof[va]):.5f}")
        _save_artifact("ada", ests, oof, proba_test, Xt.columns,
                       {"_y": y.values, "model_type": "adaboost"})

    if "rf" in requested:
        print("\n=== rf ===")
        Xt, Xs = fp.encode_for_dense(X_train_raw, X_test_raw, top_k=40)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y)); proba_test = np.zeros(len(Xs)); ests = []
        for fold, (tr, va) in enumerate(skf.split(Xt, y), 1):
            r = RandomForestClassifier(n_estimators=300, max_depth=22, min_samples_leaf=20,
                                        max_features="sqrt", class_weight="balanced", n_jobs=-1,
                                        random_state=42 + fold)
            r.fit(Xt.iloc[tr], y.iloc[tr])
            ests.append(r)
            oof[va] = r.predict_proba(Xt.iloc[va])[:, 1]
            proba_test += r.predict_proba(Xs)[:, 1] / 5
            print(f"    rf fold{fold}: auc={roc_auc_score(y.iloc[va], oof[va]):.5f}")
        _save_artifact("rf", ests, oof, proba_test, Xt.columns,
                       {"_y": y.values, "model_type": "rf"})

    print("\n[done] models saved in", MODEL_DIR.resolve())


if __name__ == "__main__":
    main()
