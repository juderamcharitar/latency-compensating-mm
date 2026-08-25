"""
ablation.py — do the LOB microstructure features add anything beyond
              knowing the quote's offset, latency, and side?

This is the experiment the thesis depends on. If Arm B matches Arm A, then
fill probability is essentially a function of how far you quote from mid,
and "AI reads microstructure to substitute for speed" is not supported by
this data. If Arm A clearly beats Arm B, the LOB features carry real signal.

Three arms, identical architecture / data / schedule, one variable changed:

  A  full   : LOB features + [latency, offset, side]      (current model)
  B  cond   : LOB features ZEROED, conditioning only      (offset/latency floor)
  C  lob    : LOB features + latency only, offset/side removed  (the old buggy model)

Run:  python ablation.py --epochs 40 --seeds 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from loguru import logger

sys.path.insert(0, str(Path("src")))
import fill_predictor as fp


class AblatedDataset(fp.FillDataset):
    """FillDataset with a chosen input stream suppressed."""

    def __init__(self, *args, arm: str = "full", **kwargs):
        super().__init__(*args, **kwargs)
        self.arm = arm

    def __getitem__(self, i):
        x, cond, t_obs, delta = super().__getitem__(i)

        if self.arm == "cond":
            # Kill the LOB window; keep [latency, offset, side]
            x = torch.zeros_like(x)
        elif self.arm == "lob":
            # Keep LOB; strip offset and side back to their pre-fix defaults
            cond = cond.clone()
            cond[1] = 1.0 / fp.MAX_OFFSET_TICKS
            cond[2] = 0.5
        return x, cond, t_obs, delta


def run_arm(arm: str, features, labels, n_features, epochs, seed) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(labels)
    tr, va = int(n * 0.70), int(n * 0.85)

    mk = lambda lab, sh: DataLoader(
        AblatedDataset(features, lab, arm=arm),
        batch_size=fp.BATCH_SIZE, shuffle=sh, num_workers=0,
    )
    train_loader = mk(labels.iloc[:tr], True)
    val_loader   = mk(labels.iloc[tr:va], False)
    test_loader  = mk(labels.iloc[va:], False)

    model = fp.LSTMFillPredictor(n_features=n_features)
    fp.train_model(model, train_loader, val_loader, epochs, f"{arm}_s{seed}")
    m = fp.evaluate_model(model, test_loader, f"{arm}_s{seed}")
    m["arm"], m["seed"] = arm, seed
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2,
                    help="repeats per arm — needed because run-to-run "
                         "variance here is large (~0.04 observed)")
    args = ap.parse_args()

    features, labels, feat_cols = fp.load_data()
    n_features = len(feat_cols)

    rows = []
    for arm in ("full", "cond", "lob"):
        for seed in range(args.seeds):
            logger.info(f"=== ARM {arm} | seed {seed} ===")
            rows.append(run_arm(arm, features, labels, n_features,
                                args.epochs, seed))

    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/ablation_results.csv", index=False)

    order = [a for a in ("full", "cond", "lob") if a in set(df["arm"])]

    # AUC is the metric the abstention rule consumes: can the model tell
    # a quote that fills from one that doesn't. c_index is reported
    # alongside it but measures fill TIMING among filled orders only.
    for metric in ("auc", "c_index"):
        if metric not in df.columns:
            continue
        s = df.groupby("arm")[metric].agg(["mean", "std", "min", "max"]).loc[order]
        print("\n" + "=" * 66)
        print(f"ABLATION — {metric} by arm  (n={args.seeds} seeds)")
        print("=" * 66)
        print(s.to_string())
        print(f"  full - cond : {s.loc['full','mean'] - s.loc['cond','mean']:+.4f}")
        print(f"  full - lob  : {s.loc['full','mean'] - s.loc['lob','mean']:+.4f}")

    summary = df.groupby("arm")["c_index"].agg(["mean", "std", "min", "max"]).loc[order]
    print()

    full_m = summary.loc["full", "mean"]
    cond_m = summary.loc["cond", "mean"]
    lob_m  = summary.loc["lob",  "mean"]
    spread = float(df.groupby("arm")["c_index"].std().max())

    print("=" * 66)
    print("VERDICT (on c_index)")
    print("=" * 66)
    if full_m - cond_m > max(0.02, 2 * spread):
        print("LOB features add signal beyond offset/latency alone.")
        print("The microstructure-prediction claim is supported on this data.")
    elif full_m - cond_m > spread:
        print("LOB features add a small edge, but it is comparable to run")
        print("variance. Report with the ablation and more seeds, not alone.")
    else:
        print("LOB features add nothing measurable beyond offset/latency.")
        print("Fill probability here is mostly a function of quote distance.")
        print("The paper's central claim is NOT supported by this data and")
        print("the framing needs to change before publication.")
    print("=" * 66)


if __name__ == "__main__":
    main()
