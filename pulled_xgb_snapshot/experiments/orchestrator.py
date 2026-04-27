"""Long-running orchestrator: drains experiments/queue/*.json one at a time.

Each queued config file is a JSON dict with keys:
  framework: lgbm | xgb | cat
  name: experiment name (also output prefix)
  + framework-specific kwargs (params, te_cols, etc.)

After successful run, the queue file is moved to experiments/queue_done/.
On error, moved to experiments/queue_failed/ with error appended.

This script never exits — it polls the queue dir indefinitely.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from run_lgbm import run as run_lgbm
from run_xgb import run as run_xgb
from run_cat import run as run_cat

ROOT = Path(__file__).resolve().parent
QUEUE = ROOT / "queue"
DONE = ROOT / "queue_done"
FAILED = ROOT / "queue_failed"
QUEUE.mkdir(exist_ok=True)
DONE.mkdir(exist_ok=True)
FAILED.mkdir(exist_ok=True)


def execute(cfg):
    fw = cfg["framework"]
    name = cfg["name"]
    if fw == "lgbm":
        run_lgbm(name, cfg.get("params", {}), cfg.get("te_cols", []),
                 drop_high_card=cfg.get("drop_high_card", False),
                 n_splits=cfg.get("n_splits", 5),
                 num_boost_round=cfg.get("num_boost_round", 2000),
                 early_stop=cfg.get("early_stop", 100))
    elif fw == "xgb":
        run_xgb(name, cfg.get("params", {}), cfg.get("te_cols", []),
                n_splits=cfg.get("n_splits", 5),
                num_boost_round=cfg.get("num_boost_round", 2000),
                early_stop=cfg.get("early_stop", 100))
    elif fw == "cat":
        run_cat(name, cfg.get("params", {}), cfg.get("te_cols", []),
                drop_high_card=cfg.get("drop_high_card", False),
                n_splits=cfg.get("n_splits", 5),
                iterations=cfg.get("iterations", 2500),
                early_stop=cfg.get("early_stop", 100))
    else:
        raise ValueError(f"unknown framework {fw}")


def main():
    print("[orch] started, polling queue/", flush=True)
    while True:
        files = sorted(QUEUE.glob("*.json"))
        if not files:
            time.sleep(15)
            continue
        f = files[0]
        try:
            cfg = json.loads(f.read_text())
        except Exception as e:
            print(f"[orch] bad json {f.name}: {e}", flush=True)
            f.rename(FAILED / f.name)
            continue
        print(f"\n[orch] >>> {cfg.get('name')} ({cfg.get('framework')})", flush=True)
        t0 = time.time()
        try:
            execute(cfg)
            f.rename(DONE / f.name)
            print(f"[orch] <<< {cfg.get('name')} OK in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[orch] !!! {cfg.get('name')} FAILED: {e}", flush=True)
            traceback.print_exc()
            tgt = FAILED / f.name
            tgt.write_text(json.dumps({**cfg, "_error": str(e)}, indent=2))
            f.unlink()


if __name__ == "__main__":
    main()
