"""Winner-swap: keep 18.15% positive rate, swap K lowest-confidence winner positives
with K highest-confidence stacker non-winner predictions.

Anchor: original_submission.csv (user's best leaderboard score, 18.15% rate).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import feat_pipeline as fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap-k", type=int, default=300, help="number of swaps")
    ap.add_argument("--out", default="winner_swap_submission.csv")
    args = ap.parse_args()

    test = pd.read_csv("test.csv", low_memory=False)
    proba = np.load("final_stack_submission.proba.npy")
    on = pd.read_csv("original_submission.csv")

    df = pd.DataFrame({fp.ID_COL: test[fp.ID_COL].values, "p": proba})
    df = df.merge(on[[fp.ID_COL, "INDICATED_DAMAGE"]].rename(columns={"INDICATED_DAMAGE": "winner"}), on=fp.ID_COL, how="left")

    n_winner = int((df["winner"] == 1).sum())
    print(f"winner positives: {n_winner}  K swaps: {args.swap_k}")

    # Of winner's positives, identify the K with lowest stacker probability (most disagreement)
    winner_pos = df[df["winner"] == 1].sort_values("p")
    drop_ids = set(winner_pos.head(args.swap_k)[fp.ID_COL].values)

    # Of non-winners, take top-K by stacker probability
    non_winner = df[df["winner"] == 0].sort_values("p", ascending=False)
    add_ids = set(non_winner.head(args.swap_k)[fp.ID_COL].values)

    df["pred"] = df["winner"].astype(int)
    df.loc[df[fp.ID_COL].isin(drop_ids), "pred"] = 0
    df.loc[df[fp.ID_COL].isin(add_ids), "pred"] = 1

    n_pred = int(df["pred"].sum())
    print(f"final positives: {n_pred}  rate={n_pred / len(df):.4f}")

    sub = df[[fp.ID_COL, "pred"]].rename(columns={"pred": fp.TARGET})
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}")

    # show which records moved
    print(f"\nTop-5 winner positives DROPPED (lowest stacker confidence):")
    print(winner_pos.head(5).to_string(index=False))
    print(f"\nTop-5 stacker positives ADDED (highest stacker confidence among non-winner):")
    print(non_winner.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
