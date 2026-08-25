"""
gbm_price_test.py — is BTC's +0.166 microstructure, or price-level artefact?

gbm_matched.py produced an inverted pattern. The asset with the LARGEST GBM
advantage (BTC, +0.166) leans on raw level-2/3 prices, which have no
microstructure story. The asset with the most economically sensible features
(SOL: offset, spread, spread_bps) has a SMALLER advantage (+0.093).

A tree splitting on bid_price_2 is partitioning on absolute price level, which
in a temporally-ordered dataset proxies for time period, which proxies for
regime. That is not microstructure prediction; it is the model learning "in
August, fills behaved like this."

gbm_leakage_check.py Test 4 dropped RAW price columns but kept normalised ones
(bid_price_1_norm etc.), which can carry the same level information. This test
removes everything price-derived.

FEATURE SETS (each fitted per asset, grouped split, identical test rows):
  all       — all 109 features (reproduces gbm_matched)
  no_price  — every column with 'price' in the name removed, raw and normalised
  micro     — only spread, imbalance, OBI, depth, volume, volatility features
  cond_only — the 3 conditioning variables (floor)

READING (fixed before running):
  - BTC 'micro' stays near +0.166 over lookup -> real microstructure, report it
  - BTC 'micro' collapses toward SOL's +0.09  -> price-level artefact inflated
        BTC; report the consistent cross-asset effect, not the BTC number
  - all three converge on a similar gap under 'micro' -> that gap is the result

Run:  python gbm_price_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("src")))
import fill_predictor as fp

ASSETS = ["BTCUSD", "ETHUSD", "SOLUSD"]

MICRO_KEYS = ("spread", "imbalance", "obi", "depth", "vol", "qty", "size")


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


def design(feats, df, cols_idx):
    idx = df["snapshot_idx"].to_numpy(dtype=int)
    lob = feats[idx][:, cols_idx] if len(cols_idx) else np.empty((len(df), 0))
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
            eval_metric="logloss", n_jobs=-1, random_state=0)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=400, max_depth=6, learning_rate=0.05, random_state=0)


def grouped_split(labels):
    snaps = np.sort(labels["snapshot_idx"].unique())
    a, b = snaps[int(len(snaps) * .70)], snaps[int(len(snaps) * .85)]
    return labels[labels["snapshot_idx"] < a], labels[labels["snapshot_idx"] >= b]


rows = []
for sym in ASSETS:
    feats, labels, cols = load_asset(sym)
    if feats is None:
        print(f"[skip] {sym}")
        continue

    tr, te = grouped_split(labels)
    y_te = te["filled"].to_numpy(dtype=float)

    # lookup floor
    keys = ["offset", "latency_ms", "side"]
    cell = tr.groupby(keys)["filled"].mean()
    base = float(tr["filled"].mean())
    p = te.set_index(keys).index.map(cell).to_numpy(dtype=float)
    lut = fp._fill_auc(np.where(np.isnan(p), base, p), y_te)

    lower = [c.lower() for c in cols]
    sets = {
        "all":       list(range(len(cols))),
        "no_price":  [i for i, c in enumerate(lower) if "price" not in c],
        "micro":     [i for i, c in enumerate(lower)
                      if any(k in c for k in MICRO_KEYS) and "price" not in c],
        "cond_only": [],
    }

    print("\n" + "=" * 62)
    print(f"{sym}   lookup = {lut:.4f}   test rows = {len(te):,}")
    print("=" * 62)

    r = {"asset": sym, "lookup": round(lut, 4)}
    for name, idx in sets.items():
        m = make_gbm()
        m.fit(design(feats, tr, idx), tr["filled"].to_numpy(dtype=int))
        a = fp._fill_auc(m.predict_proba(design(feats, te, idx))[:, 1], y_te)
        r[name] = round(a, 4)
        r[f"{name}_gap"] = round(a - lut, 4)
        print(f"  {name:<12}{len(idx):>4} LOB feats   AUC {a:.4f}   "
              f"gap {a - lut:+.4f}")
    rows.append(r)

df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/gbm_price_test.csv", index=False)

print("\n" + "=" * 78)
print("GAP OVER LOOKUP TABLE, BY FEATURE SET")
print("=" * 78)
print(df[["asset", "lookup", "all_gap", "no_price_gap",
          "micro_gap", "cond_only_gap"]].to_string(index=False))
print("=" * 78)

btc = df[df.asset == "BTCUSD"]
if len(btc):
    a, m = float(btc["all_gap"].iloc[0]), float(btc["micro_gap"].iloc[0])
    print(f"\nBTC: all={a:+.4f}  micro={m:+.4f}  lost to price removal: {a - m:+.4f}")
    if m > 0.12:
        print("BTC advantage SURVIVES price removal. Microstructure is real")
        print("and the BTC number can be reported as-is.")
    elif m > 0.05:
        print("BTC advantage PARTIALLY survives. Report the 'micro' figures")
        print("across assets, not the inflated 'all' number.")
    else:
        print("BTC advantage was largely PRICE-LEVEL ARTEFACT. Report only the")
        print("micro feature set; the headline effect is much smaller.")

mg = df["micro_gap"]
print(f"\nmicro gap across assets: min {mg.min():+.4f}  max {mg.max():+.4f}  "
      f"mean {mg.mean():+.4f}")
print("=" * 78)
print("saved -> results/gbm_price_test.csv")
