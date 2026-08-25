"""
baseline_check.py — is the LSTM beaten by a lookup table?

Compares, on the IDENTICAL test split the model uses:
  A) LSTM predictions
  B) a lookup table: mean fill rate per (offset, latency, side) cell,
     fitted on train only, applied to test

Same rows, same metric, so the comparison is actually valid.

Run:  python baseline_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from loguru import logger

sys.path.insert(0, str(Path("src")))
import fill_predictor as fp

features, labels, feat_cols = fp.load_data()
n = len(labels)
tr, va = int(n * 0.70), int(n * 0.85)

train_labels = labels.iloc[:tr]
test_labels  = labels.iloc[va:]

# ── Restrict to rows the Dataset actually keeps (needs lookback history) ──
def usable(lab):
    return lab[lab["snapshot_idx"] >= fp.LOOKBACK]

train_u = usable(train_labels)
test_u  = usable(test_labels)

print("=" * 62)
print("BASELINE CHECK — identical test split")
print("=" * 62)
print(f"train rows: {len(train_u):,}")
print(f"test rows : {len(test_u):,}")

# ── B) Lookup table, fitted on TRAIN only ────────────────────────────────
keys = [k for k in ("offset", "latency_ms", "side") if k in train_u.columns]
cell = train_u.groupby(keys)["filled"].mean()
global_rate = float(train_u["filled"].mean())

pred_lut = (
    test_u.set_index(keys).index.map(cell).to_numpy(dtype=float)
)
pred_lut = np.where(np.isnan(pred_lut), global_rate, pred_lut)
y_test = test_u["filled"].to_numpy(dtype=float)

auc_lut = fp._fill_auc(pred_lut, y_test)
print(f"\nLookup table  ({'+'.join(keys)})")
print(f"  cells fitted : {len(cell)}")
print(f"  AUC          : {auc_lut:.4f}")

# ── A) LSTM on the same rows ─────────────────────────────────────────────
model_path = fp.MODELS_DIR / "lstm_fill.pt"
if not model_path.exists():
    print(f"\nNo trained model at {model_path} — run fill_predictor.py first.")
    raise SystemExit(1)

model = fp.LSTMFillPredictor(n_features=len(feat_cols))
model.load_state_dict(torch.load(model_path, map_location=fp.DEVICE))
model.to(fp.DEVICE).eval()

test_ds = fp.FillDataset(features, test_labels)
loader  = DataLoader(test_ds, batch_size=fp.BATCH_SIZE, shuffle=False, num_workers=0)

preds, deltas = [], []
with torch.no_grad():
    for x, cond, t_obs, delta in loader:
        _, survival = model(x.to(fp.DEVICE), cond.to(fp.DEVICE))
        preds.append((1.0 - survival[:, -1]).cpu().numpy())
        deltas.append(delta.numpy())

pred_lstm = np.concatenate(preds)
y_lstm    = np.concatenate(deltas)
auc_lstm  = fp._fill_auc(pred_lstm, y_lstm)

print(f"\nLSTM")
print(f"  rows scored  : {len(y_lstm):,}")
print(f"  AUC          : {auc_lstm:.4f}")

# sanity: both should be scoring the same rows
if len(y_lstm) != len(y_test):
    print(f"\n  [warn] row counts differ ({len(y_lstm):,} vs {len(y_test):,});"
          " comparison may be skewed.")

print()
print("=" * 62)
gap = auc_lstm - auc_lut
print(f"LSTM - lookup : {gap:+.4f}")
print("=" * 62)
if gap > 0.02:
    print("LSTM beats the lookup table. The LOB features are doing")
    print("real work the table cannot capture.")
elif gap > -0.02:
    print("LSTM roughly MATCHES a lookup table on offset/latency/side.")
    print("The 109 LOB features add little over quote geometry here.")
else:
    print("LSTM LOSES to a lookup table. It is failing to extract even")
    print("the quote geometry it is given directly. Architecture or")
    print("objective problem — not a data-volume problem.")
print("=" * 62)
