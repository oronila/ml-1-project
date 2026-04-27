"""Run multiple XGB configs (GPU)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(line_buffering=True)
from run_xgb import run

CONFIGS = [
    dict(
        name="xgb_baseline",
        params=dict(eta=0.05, max_depth=8, min_child_weight=20,
                    subsample=0.85, colsample_bytree=0.85,
                    reg_alpha=0.1, reg_lambda=1.0),
        te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID", "STATE", "AMA", "AMO"],
        num_boost_round=2000, early_stop=100,
    ),
    dict(
        name="xgb_strong_reg",
        params=dict(eta=0.03, max_depth=6, min_child_weight=50,
                    subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=0.5, reg_lambda=3.0, gamma=0.1),
        te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID", "STATE", "AMA", "AMO"],
        num_boost_round=3000, early_stop=150,
    ),
    dict(
        name="xgb_deep",
        params=dict(eta=0.04, max_depth=10, min_child_weight=10,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.3, reg_lambda=2.0, gamma=0.05),
        te_cols=["SPECIES_ID", "OPID", "AIRPORT_ID", "STATE", "AMA", "AMO", "EMA", "EMO"],
        num_boost_round=2500, early_stop=120,
    ),
]

if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    for cfg in CONFIGS:
        if names and cfg["name"] not in names: continue
        print(f"\n=== {cfg['name']} ===", flush=True)
        t0 = time.time()
        try:
            run(cfg["name"], cfg["params"], cfg["te_cols"],
                num_boost_round=cfg["num_boost_round"], early_stop=cfg["early_stop"])
        except Exception as e:
            import traceback; traceback.print_exc()
        print(f"=== {cfg['name']} took {time.time()-t0:.1f}s ===", flush=True)
