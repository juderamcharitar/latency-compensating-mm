"""
regime_test.py — does the price channel carry information, or just time?

THE PROBLEM
-----------
sol_diagnostic.py showed train/test price-range overlap of 0.129 (BTC),
0.073 (ETH) and 0.251 (SOL). Under a single temporal split on a trending
70-day window, absolute price nearly identifies which partition a row is in.
So any gain from price-derived features is confounded: the trees may be
learning "this price level means test period" rather than anything about
the book.

THE TEST
--------
Re-run with a BLOCKED split. The timeline is cut into K contiguous blocks and
blocks are assigned to train/val/test in a repeating pattern, so all three
partitions span the whole price history and their price ranges overlap
heavily. Every block boundary carries the same 990 s embargo, so a training
label's 900 s forward window cannot reach into a test block.

Comparing the two split schemes separates the two explanations:

    price_gap collapses under blocking   -> it was a time proxy
    price_gap survives under blocking    -> price features carry real signal
    micro_gap survives under blocking    -> the microstructure result is not
                                            an artefact of split scheme

CAVEAT, stated up front: a blocked split is not a forecasting evaluation. It
allows training on data after the test period, which no live trader could do.
It answers "is there information here", not "could you have traded it". Both
splits are reported; neither replaces the other.

READING, fixed before running:
  - micro_gap holds within ~0.02 of the temporal result on BTC and ETH
        -> microstructure finding is robust to split scheme
  - price_gap drops toward zero under blocking
        -> confirms the time-proxy reading; price features stay excluded
  - price_gap holds under blocking
        -> price features carry real information and the exclusion was wrong

Run:  python regime_test.py --seeds 3 --blocks 40
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("src")))
import fill_predictor as fp

ASSETS = ["BTCUSD", "ETHUSD", "SOLUSD"]
MAX_HORIZON_S = 900
EMBARGO_S = MAX_HORIZON_S * 1.1
MICRO_KEYS = ("spread", "imbalance", "obi", "depth", "vol", "qty", "size")
MAX_LABELS = 200_000
LOOKBACK = 50


def load(sym):
    f = Path(f"data/features/{sym}_features.parquet")
    l = Path(f"data/features/{sym}_labels_v2.parquet")
    if not (f.exists() and l.exists()):
        return None
    fdf, ldf = pd.read_parquet(f), pd.read_parquet(l)
    cols = fp.get_feature_cols(fdf)
    feats = np.nan_to_num(fdf[cols].to_numpy(dtype=np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
    ep = pd.Timestamp("1970-01-01", tz="UTC")
    tsec = (pd.to_datetime(fdf["timestamp_utc"], utc=True) - ep
            ).dt.total_seconds().to_numpy()
    mid = fdf["mid_price"].to_numpy(float)
    ldf = ldf[ldf["snapshot_idx"] >= LOOKBACK]
    if len(ldf) > MAX_LABELS:
        ldf = ldf.sample(MAX_LABELS, random_state=0).sort_index()
    return feats, ldf.reset_index(drop=True), cols, tsec, mid


def temporal_split(labels, tsec):
    snaps = np.sort(labels["snapshot_idx"].unique())
    st = tsec[snaps]
    ta, tb = st[int(len(snaps) * .70)], st[int(len(snaps) * .85)]
    tr = snaps[st < ta - EMBARGO_S]
    te = snaps[st >= tb]
    return tr, te


def blocked_split(labels, tsec, k=150):
    """
    K contiguous time blocks assigned round-robin. Blocks 0-6 of each 10 go to
    train, 7-8 to val, 9 to test. Every boundary is embargoed by EMBARGO_S so
    a 900 s forward window cannot cross into another partition.

    k=150 was chosen by measuring price-range overlap on a simulated trending
    series: k=40 gave 0.748, k=100 gave 0.894, k=150 gives 0.924, at a cost of
    6.1% of snapshots dropped to boundary embargo. Below ~0.9 the blocking
    does not remove enough of the price separation to be worth running.
    """
    snaps = np.sort(labels["snapshot_idx"].unique())
    st = tsec[snaps]
    edges = np.linspace(st.min(), st.max(), k + 1)
    blk = np.clip(np.searchsorted(edges, st, side="right") - 1, 0, k - 1)

    role = np.where(blk % 10 < 7, 0, np.where(blk % 10 < 9, 1, 2))

    # embargo: drop snapshots within EMBARGO_S of any block edge
    dist = np.minimum(st - edges[blk], edges[blk + 1] - st)
    keep = dist > EMBARGO_S

    tr = snaps[(role == 0) & keep]
    te = snaps[(role == 2) & keep]
    return tr, te


def side_num(s):
    if s.dtype == object:
        return (s.astype(str).str.lower().str[0] == "a").astype(float).to_numpy()
    return s.to_numpy(float)


def cond_of(d):
    return np.column_stack([d["horizon_s"].to_numpy(float),
                            d["offset_bps"].to_numpy(float), side_num(d["side"])])


def make_gbt(seed):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", n_jobs=-1, random_state=seed)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier as H
        return H(max_iter=400, max_depth=6, learning_rate=0.05, random_state=seed)


def run(feats, labels, tr_s, te_s, idx, seeds):
    tr = labels[labels["snapshot_idx"].isin(tr_s)]
    te = labels[labels["snapshot_idx"].isin(te_s)]
    y = te["filled"].to_numpy(float)

    keys = ["offset_bps", "horizon_s", "side"]
    cell = tr.groupby(keys)["filled"].mean()
    p = te.set_index(keys).index.map(cell).to_numpy(float)
    lut = fp._fill_auc(np.where(np.isnan(p), tr["filled"].mean(), p), y)

    des = lambda d: (np.hstack([feats[d["snapshot_idx"].to_numpy(int)][:, idx],
                                cond_of(d)]).astype(np.float32) if len(idx)
                     else cond_of(d).astype(np.float32))
    a = []
    for s in range(seeds):
        m = make_gbt(s)
        m.fit(des(tr), tr["filled"].to_numpy(int))
        a.append(fp._fill_auc(m.predict_proba(des(te))[:, 1], y))
    return lut, float(np.mean(a)), float(np.std(a)), len(tr), len(te)


def overlap(mid, tr_s, te_s):
    a, b = mid[tr_s], mid[te_s]
    lo, hi = max(a.min(), b.min()), min(a.max(), b.max())
    full = max(a.max(), b.max()) - min(a.min(), b.min())
    return max(0.0, (hi - lo)) / full if full > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=150)
    args = ap.parse_args()

    rows = []
    for sym in ASSETS:
        got = load(sym)
        if got is None:
            print(f"[skip] {sym}")
            continue
        feats, labels, cols, tsec, mid = got
        low = [c.lower() for c in cols]
        groups = {
            "micro": [i for i, c in enumerate(low)
                      if any(k in c for k in MICRO_KEYS) and "price" not in c],
            "price": [i for i, c in enumerate(low) if "price" in c],
            "all":   list(range(len(cols))),
        }

        for scheme, fn in (("temporal", temporal_split),
                           ("blocked", lambda l, t: blocked_split(l, t, args.blocks))):
            tr_s, te_s = fn(labels, tsec)
            ov = overlap(mid, tr_s, te_s)
            print("\n" + "=" * 66)
            print(f"{sym}  [{scheme}]  price-range overlap {ov:.3f}  "
                  f"train {len(tr_s):,} / test {len(te_s):,} snapshots")
            print("=" * 66)
            for g, idx in groups.items():
                lut, mu, sd, ntr, nte = run(feats, labels, tr_s, te_s, idx, args.seeds)
                print(f"  {g:<6}{len(idx):>4} feats   lookup {lut:.4f}   "
                      f"gbt {mu:.4f} +/- {sd:.4f}   gap {mu - lut:+.4f}")
                rows.append({"asset": sym, "scheme": scheme, "group": g,
                             "overlap": round(ov, 3), "lookup": round(lut, 4),
                             "auc": round(mu, 4), "sd": round(sd, 4),
                             "gap": round(mu - lut, 4)})

    if not rows:
        return
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/regime_test.csv", index=False)

    piv = df.pivot_table(index=["asset", "group"], columns="scheme",
                         values="gap").reset_index()
    piv["delta"] = (piv["blocked"] - piv["temporal"]).round(4)
    ovp = df.pivot_table(index="asset", columns="scheme", values="overlap")

    print("\n" + "=" * 78)
    print("PRICE-RANGE OVERLAP BY SPLIT SCHEME")
    print("=" * 78)
    print(ovp.round(3).to_string())
    print("\nBlocked overlap should be near 1.0. If it is not, the blocking")
    print("did not achieve its purpose and the comparison below is void.")

    print("\n" + "=" * 78)
    print("GAP OVER LOOKUP: temporal vs blocked")
    print("=" * 78)
    print(piv.to_string(index=False))
    print("=" * 78)

    pr = piv[piv.group == "price"]
    mi = piv[piv.group == "micro"]
    print(f"\nprice gap  temporal {pr['temporal'].round(3).tolist()}  ->  "
          f"blocked {pr['blocked'].round(3).tolist()}")
    print(f"micro gap  temporal {mi['temporal'].round(3).tolist()}  ->  "
          f"blocked {mi['blocked'].round(3).tolist()}")

    print("\n" + "=" * 78)
    if (pr["blocked"] < pr["temporal"] - 0.02).sum() >= 2:
        print("Price gap SHRINKS under blocking on most assets: the price")
        print("channel was substantially a time proxy. Keep price features")
        print("excluded and report micro-only as the headline.")
    elif (pr["blocked"] > 0.02).all():
        print("Price gap SURVIVES blocking on every asset. Price-derived")
        print("features carry real information; excluding them was wrong.")
    else:
        print("Mixed. Report both schemes and do not lean on price features.")
    print("=" * 78)


if __name__ == "__main__":
    main()
