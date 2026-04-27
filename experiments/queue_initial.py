"""Drop a bunch of initial experiment configs into experiments/queue/."""
import json
import sys
from pathlib import Path

QUEUE = Path(__file__).resolve().parent / "queue"
QUEUE.mkdir(exist_ok=True)

CONFIGS = [
    # === LightGBM variants ===
    dict(framework="lgbm", name="lgbm_native_cat",
         params=dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                     feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                     lambda_l1=0.1, lambda_l2=0.5, cat_smooth=20, cat_l2=10, n_jobs=-1),
         te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID"],
         drop_high_card=False, num_boost_round=2000, early_stop=80),
    dict(framework="lgbm", name="lgbm_strong_reg",
         params=dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=400,
                     feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5,
                     lambda_l1=0.5, lambda_l2=2.0, max_depth=8, n_jobs=-1),
         te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID", "STATE", "AMA", "AMO"],
         drop_high_card=True, num_boost_round=3000, early_stop=120),
    dict(framework="lgbm", name="lgbm_shallow_te_only",
         params=dict(learning_rate=0.04, num_leaves=31, min_data_in_leaf=300,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                     lambda_l1=0.3, lambda_l2=1.5, max_depth=6, n_jobs=-1),
         te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID", "STATE", "AMA", "AMO", "EMA", "EMO"],
         drop_high_card=True, num_boost_round=2500, early_stop=100),
    dict(framework="lgbm", name="lgbm_physics_only",
         # physics-only feature filtering will happen via te_cols=[] and
         # drop_high_card=True; coarse but lets us check non-leakable features
         params=dict(learning_rate=0.04, num_leaves=63, min_data_in_leaf=300,
                     feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                     lambda_l1=0.2, lambda_l2=1.0, n_jobs=-1),
         te_cols=[],
         drop_high_card=True, num_boost_round=2000, early_stop=100),

    # === XGBoost variants (GPU) ===
    dict(framework="xgb", name="xgb_strong_reg",
         params=dict(eta=0.03, max_depth=6, min_child_weight=50,
                     subsample=0.7, colsample_bytree=0.7,
                     reg_alpha=0.5, reg_lambda=3.0, gamma=0.1),
         te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID", "STATE", "AMA", "AMO"],
         num_boost_round=3000, early_stop=150),
    dict(framework="xgb", name="xgb_shallow",
         params=dict(eta=0.05, max_depth=5, min_child_weight=80,
                     subsample=0.8, colsample_bytree=0.8,
                     reg_alpha=0.3, reg_lambda=2.0, gamma=0.05),
         te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID", "STATE", "AMA", "AMO", "EMA", "EMO"],
         num_boost_round=3500, early_stop=150),
    dict(framework="xgb", name="xgb_deep",
         params=dict(eta=0.04, max_depth=10, min_child_weight=10,
                     subsample=0.8, colsample_bytree=0.8,
                     reg_alpha=0.3, reg_lambda=2.0, gamma=0.05),
         te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID", "STATE", "AMA", "AMO", "EMA", "EMO"],
         num_boost_round=2500, early_stop=120),

    # === CatBoost variants (GPU) ===
    dict(framework="cat", name="cat_native",
         params=dict(learning_rate=0.05, depth=6, l2_leaf_reg=3.0),
         te_cols=[],
         drop_high_card=False, iterations=3000, early_stop=120),
    dict(framework="cat", name="cat_te_extra",
         params=dict(learning_rate=0.04, depth=7, l2_leaf_reg=5.0),
         te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID"],
         drop_high_card=False, iterations=3000, early_stop=120),
    dict(framework="cat", name="cat_strong_reg",
         params=dict(learning_rate=0.03, depth=5, l2_leaf_reg=8.0,
                     bagging_temperature=1.0, random_strength=2.0),
         te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID", "STATE", "AMA", "AMO"],
         drop_high_card=True, iterations=3500, early_stop=150),
]


def write():
    for i, cfg in enumerate(CONFIGS):
        # zero-pad index to keep order
        path = QUEUE / f"{i:03d}_{cfg['name']}.json"
        path.write_text(json.dumps(cfg, indent=2))
        print(f"queued: {path.name}")


if __name__ == "__main__":
    write()
