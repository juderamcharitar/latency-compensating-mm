"""
per_asset.py — does the LOB signal exist in any single asset?

The pooled result (three assets combined) showed the LSTM matching a
32-cell lookup table to +0.0003 AUC. This checks whether that null holds
per-asset, or whether pooling washed out a real effect in one book.

SOL is the asset to watch: its fill rate spans 0.654 -> 0.251 across
quote offsets, while BTC and ETH are nearly flat (0.581 -> 0.637).

For each asset, trains the LSTM and fits a train-only lookup table,
then compares both on the same test split.

PRE-REGISTERED READING (decided before running):
  - LSTM - lookup > +0.02 on an asset  -> LOB signal exists there
  - within +/-0.02                     -> no signal beyond quote geometry
  - reported for ALL assets regardless of which way each lands

Run:  python per_asset.py --epochs 40 --seeds 3
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

ASSETS = ["BTCUSD", "ETHUSD", "SOLUSD"]


def load_asset(symbol: str, max_labels: int = 200_000):
    """Load one asset's features and labels independently of load_data()."""
    feat_path = Path(f"data/features/{symbol}_features.parquet")
    lab_path  = Path(f"data/features/{symbol}_labels.parquet")
    if not feat_path.exists() or not lab_path.exists():
        return None, None, None

    fdf = pd.read_parquet(feat_path)
    ldf = pd.read_parquet(lab_path)

    feat_cols = fp.get_feature_cols(fdf)
    feats = np.nan_to_num(
        fdf[feat_cols].to_numpy(dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )

    if len(ldf) > max_labels:
        ldf = ldf.sample(max_labels, random_state=0).sort_index()
    ldf = ldf.reset_index(drop=True)
    return feats, ldf, feat_cols


def lookup_auc(train_lab, test_lab):
    """Train-only lookup table on (offset, latency, side)."""
    keys = [k for k in ("offset", "latency_ms", "side") if k in train_lab.columns]
    cell = train_lab.groupby(keys)["filled"].mean()
    base = float(train_lab["filled"].mean())

    p = test_lab.set_index(keys).index.map(cell).to_numpy(dtype=float)
    p = np.where(np.isnan(p), base, p)
    y = test_lab["filled"].to_numpy(dtype=float)
    return fp._fill_auc(p, y), len(cell)


def lstm_auc(feats, labels, n_features, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(labels)
    tr, va = int(n * 0.70), int(n * 0.85)
    mk = lambda lab, sh: DataLoader(
        fp.FillDataset(feats, lab), batch_size=fp.BATCH_SIZE,
        shuffle=sh, num_workers=0,
    )
    model = fp.LSTMFillPredictor(n_features=n_features)
    fp.train_model(model, mk(labels.iloc[:tr], True),
                   mk(labels.iloc[tr:va], False), epochs, f"s{seed}")
    m = fp.evaluate_model(model, mk(labels.iloc[va:], False), f"s{seed}")
    return m["auc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    rows = []
    for sym in ASSETS:
        feats, labels, feat_cols = load_asset(sym)
        if feats is None:
            print(f"[skip] {sym}: run lob_features.py for this symbol first")
            continue

        n = len(labels)
        tr, va = int(n * 0.70), int(n * 0.85)
        keep = lambda lab: lab[lab["snapshot_idx"] >= fp.LOOKBACK]
        lut, n_cells = lookup_auc(keep(labels.iloc[:tr]), keep(labels.iloc[va:]))

        aucs = []
        for seed in range(args.seeds):
            logger.info(f"=== {sym} | seed {seed} ===")
            aucs.append(lstm_auc(feats, labels, len(feat_cols),
                                 args.epochs, seed))

        rows.append({
            "asset":      sym,
            "snapshots":  len(feats),
            "labels":     n,
            "lut_cells":  n_cells,
            "lookup_auc": round(lut, 4),
            "lstm_auc":   round(float(np.mean(aucs)), 4),
            "lstm_std":   round(float(np.std(aucs)), 4),
            "gap":        round(float(np.mean(aucs)) - lut, 4),
        })

    if not rows:
        print("No assets processed.")
        return

    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/per_asset_results.csv", index=False)

    print("\n" + "=" * 74)
    print(f"PER-ASSET — LSTM vs train-only lookup table (AUC, {args.seeds} seeds)")
    print("=" * 74)
    print(df.to_string(index=False))
    print("=" * 74)
    for _, r in df.iterrows():
        if r["gap"] > 0.02:
            v = "LOB signal present"
        elif r["gap"] > -0.02:
            v = "no signal beyond quote geometry"
        else:
            v = "LSTM underperforms the table"
        print(f"  {r['asset']}: {r['gap']:+.4f}  ->  {v}")
    print("=" * 74)


if __name__ == "__main__":
    main()
