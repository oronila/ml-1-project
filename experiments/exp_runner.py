"""Run a list of LGBM configs sequentially, with unbuffered output to a log file."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_lgbm import run

# Force unbuffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

CONFIGS = [
    dict(
        name="baseline_lgbm",
        params=dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                    lambda_l1=0.1, lambda_l2=0.5, n_jobs=-1),
        te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID", "STATE"],
        drop_high_card=True, num_boost_round=1500, early_stop=80,
    ),
    dict(
        name="lgbm_native_cat",
        params=dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                    lambda_l1=0.1, lambda_l2=0.5, cat_smooth=20, cat_l2=10, n_jobs=-1),
        te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID"],
        drop_high_card=False, num_boost_round=2000, early_stop=80,
    ),
    dict(
        name="lgbm_strong_reg",
        params=dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=400,
                    feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=5,
                    lambda_l1=0.5, lambda_l2=2.0, max_depth=8, n_jobs=-1),
        te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID", "STATE", "AMA", "AMO"],
        drop_high_card=True, num_boost_round=2500, early_stop=120,
    ),
    dict(
        name="lgbm_deep_native",
        params=dict(learning_rate=0.04, num_leaves=127, min_data_in_leaf=150,
                    feature_fraction=0.8, bagging_fraction=0.85, bagging_freq=5,
                    lambda_l1=0.2, lambda_l2=1.0, cat_smooth=20, cat_l2=10,
                    min_gain_to_split=0.01, n_jobs=-1),
        te_cols=["SPECIES_ID", "OPID"],
        drop_high_card=False, num_boost_round=2500, early_stop=100,
    ),
]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        names = sys.argv[1].split(",")
        configs = [c for c in CONFIGS if c["name"] in names]
    else:
        configs = CONFIGS
    for cfg in configs:
        print(f"\n=== {cfg['name']} ===", flush=True)
        t0 = time.time()
        try:
            run(cfg["name"], cfg["params"], cfg["te_cols"],
                cfg.get("drop_high_card", False),
                num_boost_round=cfg.get("num_boost_round", 2000),
                early_stop=cfg.get("early_stop", 100))
        except Exception as e:
            print(f"[{cfg['name']}] FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()
        print(f"=== {cfg['name']} took {time.time()-t0:.1f}s ===", flush=True)
