"""Winner-anchored submission: prefer all of original_notebook's positives, then fill with our stacker's top.

The user said original_notebook is the best leaderboard performer (5840 positives, 17.1% rate).
Our stacker has higher OOF AUC. Combine: take all winner positives, fill remaining slots with
the top non-winner predictions from our stacker, up to target positive rate.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import feat_pipeline as fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rate", type=float, default=0.18)
    ap.add_argument("--out", default="winner_anchored_submission.csv")
    args = ap.parse_args()

    test = pd.read_csv("test.csv", low_memory=False)
    proba = np.load("final_stack_submission.proba.npy")
    # original_submission.csv is the user's best leaderboard score (18.15% pos rate)
    on = pd.read_csv("original_submission.csv")

    # align on INDEX_NR
    df = pd.DataFrame({fp.ID_COL: test[fp.ID_COL].values, "p": proba})
    df = df.merge(on[[fp.ID_COL, "INDICATED_DAMAGE"]].rename(columns={"INDICATED_DAMAGE": "winner"}), on=fp.ID_COL, how="left")

    n_target = int(round(args.target_rate * len(df)))
    print(f"target positives: {n_target}  ({args.target_rate:.4f} rate)")

    # Step 1: take all winner positives
    win_pos_mask = df["winner"] == 1
    n_winner = int(win_pos_mask.sum())
    print(f"winner positives: {n_winner}")

    if n_target <= n_winner:
        # Just take top-n_target by stacker proba among winner positives
        cand = df[win_pos_mask].sort_values("p", ascending=False).head(n_target)
        df["pred"] = 0
        df.loc[cand.index, "pred"] = 1
    else:
        # Take ALL winner positives + top (n_target - n_winner) from non-winner by stacker proba
        n_remaining = n_target - n_winner
        non_winner = df[~win_pos_mask].sort_values("p", ascending=False).head(n_remaining)
        df["pred"] = 0
        df.loc[win_pos_mask, "pred"] = 1
        df.loc[non_winner.index, "pred"] = 1

    sub = df[[fp.ID_COL, "pred"]].rename(columns={"pred": fp.TARGET})
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out} positives={int(sub[fp.TARGET].sum())} rate={sub[fp.TARGET].mean():.4f}")


if __name__ == "__main__":
    main()
