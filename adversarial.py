"""Adversarial validation: train classifier to distinguish train rows from test rows.

If AUC ~ 0.5, distributions are similar.
If AUC > 0.7, there's significant shift.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import feat_pipeline as fp


def main():
    train = pd.read_csv("train.csv", low_memory=False)
    test = pd.read_csv("test.csv", low_memory=False)
    X_train_raw, X_test_raw = fp.make_features(train, test)
    X_train, X_test, cat_cols = fp.encode_for_lgb(X_train_raw, X_test_raw)

    X_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_all = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_all))
    feat_imp = pd.Series(0.0, index=X_all.columns)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all), 1):
        ds_t = lgb.Dataset(X_all.iloc[tr_idx], label=y_all[tr_idx], categorical_feature=cat_cols)
        ds_v = lgb.Dataset(X_all.iloc[va_idx], label=y_all[va_idx], categorical_feature=cat_cols, reference=ds_t)
        params = dict(objective="binary", metric="auc", learning_rate=0.1, num_leaves=63,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                      verbose=-1, seed=42)
        b = lgb.train(params, ds_t, num_boost_round=300, valid_sets=[ds_v],
                      callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
        oof[va_idx] = b.predict(X_all.iloc[va_idx], num_iteration=b.best_iteration)
        for name, imp in zip(X_all.columns, b.feature_importance("gain")):
            feat_imp[name] += imp / 5
        print(f"  fold{fold}: auc={roc_auc_score(y_all[va_idx], oof[va_idx]):.5f}")

    auc = roc_auc_score(y_all, oof)
    print(f"\nAdversarial AUC = {auc:.5f}")
    if auc > 0.6:
        print(">>> distribution shift detected!")
    print("\nTop 25 most-shifted features:")
    print(feat_imp.sort_values(ascending=False).head(25).to_string())


if __name__ == "__main__":
    main()
