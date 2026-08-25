"""
gbm_matched.py — does the GBM result replicate, and is it matched to the LSTM?

Two gaps remain after gbm_leakage_check.py:

  (a) The 0.8264 GBM number is BTC-only; the 0.6375 LSTM number is pooled.
      They were never measured on the same rows.
  (b) The GBM result has only been shown on BTC. If it is real microstructure
      signal it should replicate on ETH and SOL.

This script closes both. For each asset it fits, on IDENTICAL snapshot-grouped
splits and scoring identical test rows:

    lookup table  |  GBM  |  LSTM

It also reports the level-2 price importance flagged in the previous run, since
bid_price_2 / ask_price_2 carrying ~25% of importance is unexplained.

READING (fixed before running):
  - GBM - lookup > +0.10 on all three assets  -> replicates; effect is real and
        general; the paper reports it as the headline
  - replicates on some assets only            -> report per-asset, no pooled claim
  - BTC only                                  -> likely BTC-specific artefact,
        investigate before claiming anything

Run:  python gbm_matched.py --epochs 40
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


def load_asset(symbol, max_labels=200_000):
    f = Path(f"data/features/{symbol}_features.parquet")
    l = Path(f"data/features/{symbol}_labels.parquet")
    if not (f.exists() and l.exists()):
        return None, None, None
    fdf, ldf = pd.read_parquet(f), pd.read_parquet(l)
    cols = fp.get_feature_cols(fdf)
    feats = np.nan_to_num(fdf[cols].to_numpy(dtype=np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
    if len(ldf) > max_labels:
        ldf = ldf.sample(max_labels, random_state=0).sort_index()
    ldf = ldf[ldf["snapshot_idx"] >= fp.LOOKBACK].reset_index(drop=True)
    return feats, ldf, cols


def side_num(s):
    if s.dtype == object:
        return (s.astype(str).str.lower().str[0] == "a").astype(float).to_numpy()
    return s.to_numpy(dtype=float)


def design(feats, df):
    lob = feats[df["snapshot_idx"].to_numpy(dtype=int)]
    cond = np.column_stack([
        df["latency_ms"].to_numpy(dtype=float),
        df["offset"].to_numpy(dtype=float),
        side_num(df["side"]),
    ])
    return np.hstack([lob, cond]).astype(np.float32)


def make_gbm():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=-1, random_state=0,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=400, max_depth=6, learning_rate=0.05, random_state=0,
        )


def grouped_split(labels):
    """Split on snapshots so none straddles train/test."""
    snaps = np.sort(labels["snapshot_idx"].unique())
    a, b = snaps[int(len(snaps) * .70)], snaps[int(len(snaps) * .85)]
    return (labels[labels["snapshot_idx"] < a],
            labels[(labels["snapshot_idx"] >= a) & (labels["snapshot_idx"] < b)],
            labels[labels["snapshot_idx"] >= b])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    rows, imps = [], {}

    for sym in ASSETS:
        feats, labels, cols = load_asset(sym)
        if feats is None:
            print(f"[skip] {sym}: run lob_features.py for this symbol first")
            continue

        print("\n" + "=" * 62)
        print(f"{sym} — grouped split, identical test rows")
        print("=" * 62)

        tr, va, te = grouped_split(labels)
        y_te = te["filled"].to_numpy(dtype=float)
        print(f"train/val/test rows : {len(tr):,} / {len(va):,} / {len(te):,}")

        # lookup table
        keys = ["offset", "latency_ms", "side"]
        cell = tr.groupby(keys)["filled"].mean()
        base = float(tr["filled"].mean())
        p = te.set_index(keys).index.map(cell).to_numpy(dtype=float)
        auc_lut = fp._fill_auc(np.where(np.isnan(p), base, p), y_te)

        # GBM
        gbm = make_gbm()
        gbm.fit(design(feats, tr), tr["filled"].to_numpy(dtype=int))
        auc_gbm = fp._fill_auc(gbm.predict_proba(design(feats, te))[:, 1], y_te)
        try:
            names = list(cols) + ["latency_ms", "offset", "side"]
            imp = gbm.feature_importances_
            imps[sym] = sorted(zip(names, imp), key=lambda x: -x[1])[:8]
        except Exception:
            imps[sym] = []

        # LSTM on the SAME rows
        torch.manual_seed(0)
        np.random.seed(0)
        mk = lambda lab, sh: DataLoader(
            fp.FillDataset(feats, lab.reset_index(drop=True)),
            batch_size=fp.BATCH_SIZE, shuffle=sh, num_workers=0)
        model = fp.LSTMFillPredictor(n_features=len(cols))
        fp.train_model(model, mk(tr, True), mk(va, False), args.epochs, sym)
        auc_lstm = fp.evaluate_model(model, mk(te, False), sym)["auc"]

        rows.append({
            "asset": sym,
            "test_rows": len(te),
            "lookup": round(auc_lut, 4),
            "gbm": round(auc_gbm, 4),
            "lstm": round(auc_lstm, 4),
            "gbm_minus_lookup": round(auc_gbm - auc_lut, 4),
            "lstm_minus_lookup": round(auc_lstm - auc_lut, 4),
        })
        print(f"  lookup {auc_lut:.4f} | GBM {auc_gbm:.4f} | LSTM {auc_lstm:.4f}")

    if not rows:
        print("No assets processed.")
        return

    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/gbm_matched_results.csv", index=False)

    print("\n" + "=" * 78)
    print("MATCHED COMPARISON — AUC, grouped split, identical test rows")
    print("=" * 78)
    print(df.to_string(index=False))
    print("=" * 78)

    print("\nTop features by GBM importance:")
    for sym, top in imps.items():
        print(f"  {sym}: " + ", ".join(f"{n}({v:.3f})" for n, v in top[:6]))

    g = df["gbm_minus_lookup"]
    print("\n" + "=" * 78)
    if (g > 0.10).all():
        print("GBM beats the lookup table on ALL assets. The microstructure")
        print("signal is real and general. The LSTM's failure is architectural.")
    elif (g > 0.10).any():
        n = int((g > 0.10).sum())
        print(f"GBM beats the lookup table on {n} of {len(g)} assets.")
        print("Report per-asset. Do not make a pooled claim.")
    else:
        print("GBM does not replicate outside BTC. Investigate before claiming.")
    print("=" * 78)


if __name__ == "__main__":
    main()
