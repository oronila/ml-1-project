"""Print a leaderboard of all experiment results."""
import json
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parent / "results"
rows = []
for f in sorted(R.glob("*.json")):
    try:
        d = json.loads(f.read_text())
        rows.append(dict(
            name=d.get("name"),
            val_auc=d.get("val_auc"),
            val_ap=d.get("val_ap"),
            train_auc=d.get("train_auc"),
            gap=d.get("overfit_gap"),
            f1=d.get("oof_f1"),
            thr_f1=d.get("oof_thr_f1"),
            thr7=d.get("thr_rate7"),
            time_s=d.get("elapsed_s"),
            n_feat=d.get("n_features"),
        ))
    except Exception:
        pass

df = pd.DataFrame(rows)
if df.empty:
    print("no results yet")
else:
    df = df.sort_values("val_ap", ascending=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))
