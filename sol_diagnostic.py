"""
sol_diagnostic.py — why does SOL invert?

In rerun_v3.py, SOL was the only asset where the microstructure feature subset
performed WORSE than the lookup table while the full feature set performed
better:

              GBT micro (63)   GBT all (109)   swing
    BTC          +0.0480          +0.0434      -0.005
    ETH          +0.0596          +0.0312      -0.028
    SOL          -0.0105          +0.0383      +0.049

On BTC and ETH the 46 price-derived columns HURT, consistent with the earlier
price-removal audit. On SOL they help by 0.049. Same pipeline, same features,
opposite sign. That needs an explanation before either SOL number is quotable.

DESIGN
------
Fit nested feature groups on every asset, 3 seeds, identical embargoed splits:

    cond      3 conditioning columns only            (should match lookup)
    micro     63 microstructure features
    price     46 price-derived features
    all       109 features

If SOL's gain lives in `price` alone, the trees are keying on something about
price levels that is absent or unhelpful on the other two assets. Feature
importances and a price-level drift check test that directly.

READING, fixed before running:
  - SOL price-only clearly above lookup, BTC/ETH price-only at or below it
        -> SOL-specific price-level effect; likeliest cause is trend/regime
           leakage through absolute price, and SOL's numbers need a caveat
  - price-only near lookup everywhere
        -> the swing is an interaction, not a main effect; report as
           unexplained rather than guessing
  - price-only above lookup on ALL assets
        -> the earlier price-removal audit was wrong and needs revisiting

Run:  python sol_diagnostic.py --seeds 3
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
    mid = fdf["mid_price"].to_numpy(float) if "mid_price" in fdf.columns else None
    ldf = ldf[ldf["snapshot_idx"] >= LOOKBACK]
    if len(ldf) > MAX_LABELS:
        ldf = ldf.sample(MAX_LABELS, random_state=0).sort_index()
    return feats, ldf.reset_index(drop=True), cols, tsec, mid


def split(labels, tsec):
    snaps = np.sort(labels["snapshot_idx"].unique())
    st = tsec[snaps]
    ta, tb = st[int(len(snaps) * .70)], st[int(len(snaps) * .85)]
    tr = snaps[st < ta - EMBARGO_S]
    te = snaps[st >= tb]
    pick = lambda s: labels[labels["snapshot_idx"].isin(s)]
    return pick(tr), pick(te)


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


def fit(feats, tr, te, idx, seed, want_imp=False):
    des = lambda d: (np.hstack([feats[d["snapshot_idx"].to_numpy(int)][:, idx],
                                cond_of(d)]).astype(np.float32) if len(idx)
                     else cond_of(d).astype(np.float32))
    m = make_gbt(seed)
    m.fit(des(tr), tr["filled"].to_numpy(int))
    a = fp._fill_auc(m.predict_proba(des(te))[:, 1], te["filled"].to_numpy(float))
    return (a, m) if want_imp else (a, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    rows, imps, drift = [], {}, []

    for sym in ASSETS:
        got = load(sym)
        if got is None:
            print(f"[skip] {sym}")
            continue
        feats, labels, cols, tsec, mid = got
        tr, te = split(labels, tsec)

        low = [c.lower() for c in cols]
        micro = [i for i, c in enumerate(low)
                 if any(k in c for k in MICRO_KEYS) and "price" not in c]
        price = [i for i, c in enumerate(low) if "price" in c]
        allf = list(range(len(cols)))

        keys = ["offset_bps", "horizon_s", "side"]
        cell = tr.groupby(keys)["filled"].mean()
        p = te.set_index(keys).index.map(cell).to_numpy(float)
        p = np.where(np.isnan(p), tr["filled"].mean(), p)
        lut = fp._fill_auc(p, te["filled"].to_numpy(float))

        print("\n" + "=" * 62)
        print(f"{sym}   lookup {lut:.4f}   micro {len(micro)}  price {len(price)}")
        print("=" * 62)

        r = {"asset": sym, "lookup": round(lut, 4)}
        for name, idx in (("cond", []), ("micro", micro),
                          ("price", price), ("all", allf)):
            aucs, model = [], None
            for s in range(args.seeds):
                a, m = fit(feats, tr, te, idx, s, want_imp=(name == "all" and s == 0))
                aucs.append(a)
                if m is not None:
                    model = m
            mu, sd = float(np.mean(aucs)), float(np.std(aucs))
            r[name] = round(mu, 4)
            r[f"{name}_gap"] = round(mu - lut, 4)
            print(f"  {name:<6}{len(idx):>4} feats   {mu:.4f} +/- {sd:.4f}"
                  f"   {mu - lut:+.4f}")
            if model is not None:
                names = list(cols) + ["horizon_s", "offset_bps", "side"]
                try:
                    imp = model.feature_importances_
                    top = sorted(zip(names, imp), key=lambda x: -x[1])[:8]
                    imps[sym] = top
                    pshare = sum(v for n, v in zip(names, imp) if "price" in n.lower())
                    r["price_importance_share"] = round(float(pshare), 4)
                except Exception:
                    pass
        rows.append(r)

        # does absolute price drift across the split? a tree can use price
        # level as a proxy for time period if it does
        if mid is not None:
            mtr = mid[tr["snapshot_idx"].to_numpy(int)]
            mte = mid[te["snapshot_idx"].to_numpy(int)]
            overlap = (min(mtr.max(), mte.max()) - max(mtr.min(), mte.min())) / \
                      (max(mtr.max(), mte.max()) - min(mtr.min(), mte.min()))
            drift.append({"asset": sym,
                          "train_mid_range": f"{mtr.min():.2f}-{mtr.max():.2f}",
                          "test_mid_range": f"{mte.min():.2f}-{mte.max():.2f}",
                          "range_overlap": round(float(overlap), 3)})

    if not rows:
        return
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/sol_diagnostic.csv", index=False)

    print("\n" + "=" * 80)
    print("GAP OVER LOOKUP BY FEATURE GROUP")
    print("=" * 80)
    print(df[["asset", "lookup", "cond_gap", "micro_gap",
              "price_gap", "all_gap"]].to_string(index=False))
    print("=" * 80)

    if drift:
        print("\nPRICE-LEVEL OVERLAP BETWEEN TRAIN AND TEST")
        print(pd.DataFrame(drift).to_string(index=False))
        print("Low overlap means absolute price separates the partitions, so a")
        print("tree can use price level as a proxy for time period.")

    print("\nTOP FEATURES (109-feature model, seed 0)")
    for s, t in imps.items():
        print(f"  {s}: " + ", ".join(f"{n}({v:.3f})" for n, v in t[:6]))

    pg = df.set_index("asset")["price_gap"]
    print("\n" + "=" * 80)
    if "SOLUSD" in pg and pg["SOLUSD"] > 0.02 and (pg.drop("SOLUSD") < 0.02).all():
        print("SOL-specific price effect. Its gain comes from price-derived")
        print("columns that do not help BTC or ETH. Check the overlap table:")
        print("if SOL's train/test price ranges barely overlap, the trees are")
        print("likely keying on price level as a time proxy, and SOL's numbers")
        print("need that caveat in the paper.")
    elif (pg > 0.02).all():
        print("Price features help everywhere. The earlier price-removal audit")
        print("was label-dependent and its conclusion does not carry over.")
    else:
        print("No clean price main effect. The SOL swing is an interaction")
        print("between feature groups. Report it as unexplained.")
    print("=" * 80)


if __name__ == "__main__":
    main()
