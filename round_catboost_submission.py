#!/usr/bin/env python3
"""Convert CatBoost probability submission to required 0/1 labels."""

import argparse
from pathlib import Path

import pandas as pd


TARGET = "INDICATED_DAMAGE"


def run(input_path: Path, output_path: Path) -> None:
    submission = pd.read_csv(input_path)
    submission[TARGET] = submission[TARGET].round().astype(int)
    submission.to_csv(output_path, index=False)
    print(
        f"wrote {output_path} rows={len(submission)} "
        f"positives={int(submission[TARGET].sum())} "
        f"pos_rate={submission[TARGET].mean():.5f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("catboost_auc_submission.csv"))
    parser.add_argument("--out", type=Path, default=Path("catboost_binary_submission.csv"))
    args = parser.parse_args()
    run(args.input, args.out)


if __name__ == "__main__":
    main()
