"""
diagnose.py — is the fill-prediction task learnable, or is the pipeline broken?

Logic: the marginal fill-rate tables already show fill rate varies strongly with
offset and latency (SOL: 0.654 -> 0.251 across offsets). So a model given ONLY
offset + latency must score well above 0.50. If it doesn't, the labels are not
correctly associated with their features and the problem is plumbing, not learning.

Run:  python diagnose.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

FEAT = Path("data/features/SOLUSD_features.parquet")
LAB  = Path("data/features/SOLUSD_labels.parquet")

print("=" * 64)
print("FILL PREDICTOR DIAGNOSTIC")
print("=" * 64)

if not LAB.exists():
    print(f"\nMissing {LAB}. Re-run the pipeline for SOLUSD first:")
    print("  python src/lob_features.py --mode real --data-dir data\\kraken\\raw --symbol SOLUSD")
    raise SystemExit(1)

lab = pd.read_parquet(LAB)
print(f"\nLabels: {len(lab):,} rows")
print(f"Columns: {list(lab.columns)}\n")

# ---- 1. Label sanity -------------------------------------------------------
print("-" * 64)
print("1. LABEL DISTRIBUTION")
print("-" * 64)
if "filled" not in lab.columns:
    print("No 'filled' column. Cannot proceed.")
    raise SystemExit(1)

print(f"filled mean : {lab['filled'].mean():.4f}")
print(f"filled dtype: {lab['filled'].dtype}")
print(f"unique vals : {sorted(lab['filled'].unique())[:5]}")

# ---- 2. Does fill vary with offset/latency in the LABELS themselves? -------
print()
print("-" * 64)
print("2. MARGINAL STRUCTURE IN LABELS (must be non-flat)")
print("-" * 64)
for col in ("offset", "offset_ticks", "latency_ms"):
    if col in lab.columns:
        g = lab.groupby(col)["filled"].agg(["mean", "count"])
        print(f"\nfilled by {col}:")
        print(g.to_string())

# ---- 3. Can offset+latency ALONE predict fill? -----------------------------
print()
print("-" * 64)
print("3. BASELINE: offset + latency ONLY")
print("-" * 64)

pred_cols = [c for c in ("offset", "offset_ticks", "latency_ms", "side") if c in lab.columns]
print(f"Using: {pred_cols}")

if not pred_cols:
    print("No offset/latency columns found in labels — that itself is the bug.")
    raise SystemExit(1)

d = lab[pred_cols + ["filled"]].dropna().copy()
for c in pred_cols:
    if d[c].dtype == object:
        d[c] = pd.factorize(d[c])[0]

# group-mean predictor: predict each row's fill rate from its (offset, latency) cell
cell = d.groupby(pred_cols)["filled"].transform("mean")

y = d["filled"].values.astype(float)
p = cell.values.astype(float)

# Concordance: P(higher predicted score has higher outcome) over discordant pairs
rng = np.random.default_rng(0)
n = min(200_000, len(y))
i = rng.integers(0, len(y), n)
j = rng.integers(0, len(y), n)
mask = y[i] != y[j]
i, j = i[mask], j[mask]
hi = np.where(y[i] > y[j], i, j)   # index of the filled==1 row
lo = np.where(y[i] > y[j], j, i)   # index of the filled==0 row
conc = (p[hi] > p[lo]).mean() + 0.5 * (p[hi] == p[lo]).mean()

print(f"\nPairs evaluated : {len(hi):,}")
print(f"Baseline C-index: {conc:.4f}")

print()
print("=" * 64)
print("VERDICT")
print("=" * 64)
if conc > 0.60:
    print(f"Baseline scores {conc:.4f} using offset+latency alone.")
    print("The task IS learnable. The LSTM scoring ~0.47 means the model")
    print("is not receiving these labels correctly -> PIPELINE BUG.")
elif conc > 0.53:
    print(f"Baseline scores {conc:.4f} — weak but real signal.")
    print("Task is marginally learnable; LSTM underperforming a trivial")
    print("baseline still points at a pipeline or alignment problem.")
else:
    print(f"Baseline scores {conc:.4f} — no signal even from offset+latency,")
    print("despite the marginal tables showing variation. This means rows are")
    print("not matched to their own labels -> PIPELINE BUG in the join.")
print("=" * 64)
