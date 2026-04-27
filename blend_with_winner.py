"""Blend our stacker with the original_notebook winning submission's hard predictions.

We don't have the OOF for original_notebook, but its TEST predictions are the
known leaderboard winner. So treat its 0/1 prediction as a single "vote" that
nudges our stacker.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import feat_pipeline as fp


def to_rank(p):
    order = p.argsort()
    r = np.empty_like(order, dtype=np.float64)
    r[order] = np.arange(len(p))
    return r / (len(p) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rate", type=float, default=0.18)
    ap.add_argument("--winner-weight", type=float, default=0.20)
    ap.add_argument("--out", default="winner_blend_submission.csv")
    args = ap.parse_args()

    test = pd.read_csv("test.csv", low_memory=False)
    test = test.sort_values(fp.ID_COL).reset_index(drop=True)

    # Load our stacker test probabilities
    stack_proba = np.load("stack_v2_submission.proba.npy")  # ordered same as test load order
    test_orig_order = pd.read_csv("test.csv", low_memory=False)
    stack_df = pd.DataFrame({"INDEX_NR": test_orig_order[fp.ID_COL].values, "p": stack_proba})
    stack_df = stack_df.sort_values("INDEX_NR").reset_index(drop=True)
    stack_p = stack_df["p"].values

    # Load original_notebook + original (the two best leaderboard submissions)
    on = pd.read_csv("original_notebook_submission.csv").sort_values("INDEX_NR").reset_index(drop=True)
    org = pd.read_csv("original_submission.csv").sort_values("INDEX_NR").reset_index(drop=True)
    assert (on["INDEX_NR"].values == stack_df["INDEX_NR"].values).all()
    assert (org["INDEX_NR"].values == stack_df["INDEX_NR"].values).all()

    # Convert hard 0/1 predictions to soft "evidence" (0.05 / 0.95 say)
    on_soft = np.where(on["INDICATED_DAMAGE"] == 1, 0.95, 0.05)
    org_soft = np.where(org["INDICATED_DAMAGE"] == 1, 0.95, 0.05)

    # Rank-blend in [0,1] space
    r_stack = to_rank(stack_p)
    r_on = to_rank(on_soft + 0.001 * np.random.RandomState(0).randn(len(on_soft)))  # break ties
    r_org = to_rank(org_soft + 0.001 * np.random.RandomState(1).randn(len(org_soft)))

    w = args.winner_weight
    blended = (1 - 2 * w) * r_stack + w * r_on + w * r_org

    thr = fp.threshold_for_target_rate(blended, args.target_rate)
    preds = (blended >= thr).astype(int)
    print(f"winner_weight={w} target_rate={args.target_rate} thr={thr:.4f}")

    sub = pd.DataFrame({fp.ID_COL: stack_df["INDEX_NR"].values, fp.TARGET: preds})
    # Match original test.csv order for submission
    sub = sub.set_index(fp.ID_COL).reindex(test_orig_order[fp.ID_COL].values).reset_index()
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out} positives={int(preds.sum())} rate={preds.mean():.4f}")

    # Agreement with each
    on_pos = set(on.loc[on["INDICATED_DAMAGE"] == 1, "INDEX_NR"])
    org_pos = set(org.loc[org["INDICATED_DAMAGE"] == 1, "INDEX_NR"])
    blend_pos = set(stack_df.loc[preds == 1, "INDEX_NR"])
    print(f"jaccard with original_notebook: {len(blend_pos & on_pos)/len(blend_pos | on_pos):.3f}")
    print(f"jaccard with original:          {len(blend_pos & org_pos)/len(blend_pos | org_pos):.3f}")


if __name__ == "__main__":
    main()
