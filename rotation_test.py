"""
rotation_test.py — is the blocked-split result an artefact of one phase?

THE VULNERABILITY
-----------------
The blocked split was introduced after inspecting the temporal-split results,
in which SOL/USD showed a negative microstructure gap. A referee can reasonably
object that we saw a surprising result, designed an alternative partition, and
promoted the favourable alternative to the headline.

The block assignment uses one fixed phase: role = f(block_index % 10), with
indices 0-6 train, 7-8 validation, 9 test. That is one arbitrary choice out of
ten available rotations.

THE TEST
--------
Run all 10 phase rotations, holding everything else fixed: same 150 blocks,
same 990 s embargo, same 63-feature microstructure representation, same model
specification. Phase p assigns role = f((block_index + p) % 10).

Rotation is a legitimate robustness axis because the phase is arbitrary and
carries no information about outcomes. If the microstructure gap stays positive
across rotations on all three assets, the blocked result is not a lucky
partition. The spread across rotations also gives the partition variance that
the paper currently lists as unquantified.

PRE-REGISTERED READING, fixed before running:
  - gap positive in >= 8 of 10 rotations on all three assets
        -> robust; report mean and range across rotations as the headline
  - positive on BTC/ETH but mixed on SOL
        -> report per-asset; SOL's blocked result is phase-sensitive and the
           paper should say so rather than averaging it away
  - mixed on two or more assets
        -> the blocked result is not robust and should be demoted to a
           diagnostic, with the temporal split carrying the paper

VERIFICATION runs first and aborts on failure:
  V1  the 10 rotations produce genuinely different test sets
  V2  every rotation holds the 990 s embargo at all boundaries

Price-range overlap is NOT a gate. An earlier version of this script required
every rotation to exceed 0.7 overlap and aborted when BTC (min 0.603) and SOL
(min 0.534) failed. That would have selected rotations on a property plausibly
correlated with the outcome, which is the same defect the rotation test exists
to avoid. Overlap is instead measured per rotation and reported alongside the
gap, so its relationship to the result is visible rather than assumed.

SECOND PRE-REGISTERED READING, on that relationship:
  - gap positive across rotations regardless of overlap
        -> the effect does not depend on distributional overlap; stronger
           than the current blocked-split result
  - gap positive only in high-overlap rotations
        -> overlap drives the effect; SOL's temporal reversal and blocked
           recovery are both overlap artefacts and the paper must say so

Run:  python rotation_test.py --seeds 1
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
BLOCKS = 150
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
    micro = [i for i, c in enumerate(cols)
             if any(k in c.lower() for k in MICRO_KEYS)
             and "price" not in c.lower()]
    return feats, ldf.reset_index(drop=True), cols, tsec, mid, micro


def blocked_split(labels, tsec, phase, k=BLOCKS):
    """Blocked split with the role pattern rotated by `phase`."""
    snaps = np.sort(labels["snapshot_idx"].unique())
    st = tsec[snaps]
    edges = np.linspace(st.min(), st.max(), k + 1)
    blk = np.clip(np.searchsorted(edges, st, side="right") - 1, 0, k - 1)
    r = (blk + phase) % 10
    role = np.where(r < 7, 0, np.where(r < 9, 1, 2))
    dist = np.minimum(st - edges[blk], edges[blk + 1] - st)
    keep = dist > EMBARGO_S
    return snaps[(role == 0) & keep], snaps[(role == 2) & keep]


def overlap(mid, a, b):
    x, y = mid[a], mid[b]
    lo, hi = max(x.min(), y.min()), min(x.max(), y.max())
    full = max(x.max(), y.max()) - min(x.min(), y.min())
    return max(0.0, hi - lo) / full if full > 0 else 0.0


def min_gap(tsec, a, b):
    ta, tb = np.sort(tsec[a]), np.sort(tsec[b])
    i = np.searchsorted(tb, ta)
    g = np.inf
    for j in (i - 1, np.minimum(i, len(tb) - 1)):
        j = np.clip(j, 0, len(tb) - 1)
        g = min(g, np.abs(tb[j] - ta).min())
    return g


def verify(data):
    print("=" * 68)
    print("VERIFICATION (runs before the experiment)")
    print("=" * 68)
    ok = True
    for sym, (feats, labels, cols, tsec, mid, micro) in data.items():
        tests = {}
        for p in range(10):
            tr, te = blocked_split(labels, tsec, p)
            tests[p] = set(te.tolist())
        # V1 distinct test sets
        sizes = [len(tests[p] & tests[q]) / max(1, len(tests[p]))
                 for p in range(10) for q in range(10) if p < q]
        v1 = max(sizes) < 0.5
        # V2 embargo, V3 overlap
        gaps, ovs = [], []
        for p in range(10):
            tr, te = blocked_split(labels, tsec, p)
            gaps.append(min_gap(tsec, tr, te))
            ovs.append(overlap(mid, tr, te))
        v2 = min(gaps) >= EMBARGO_S
        ok &= v1 and v2
        print(f"\n  {sym}")
        print(f"    V1 rotations give distinct test sets   "
              f"{'PASS' if v1 else 'FAIL'}  (max overlap {max(sizes):.1%})")
        print(f"    V2 embargo >= {EMBARGO_S:.0f}s in all rotations   "
              f"{'PASS' if v2 else 'FAIL'}  (min {min(gaps):.0f}s)")
        print(f"    price overlap across rotations (reported, not gated): "
              f"{min(ovs):.3f}-{max(ovs):.3f}")
    print("\n" + "=" * 68)
    return ok


def make_gbt(seed):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", n_jobs=-1, random_state=seed)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier as H
        return H(max_iter=400, max_depth=6, learning_rate=0.05, random_state=seed)


def side_num(s):
    if s.dtype == object:
        return (s.astype(str).str.lower().str[0] == "a").astype(float).to_numpy()
    return s.to_numpy(float)


def cond_of(d):
    return np.column_stack([d["horizon_s"].to_numpy(float),
                            d["offset_bps"].to_numpy(float), side_num(d["side"])])


def evaluate(feats, labels, tr_s, te_s, idx, seeds):
    tr = labels[labels["snapshot_idx"].isin(tr_s)]
    te = labels[labels["snapshot_idx"].isin(te_s)]
    y = te["filled"].to_numpy(float)
    keys = ["offset_bps", "horizon_s", "side"]
    cell = tr.groupby(keys)["filled"].mean()
    p = te.set_index(keys).index.map(cell).to_numpy(float)
    lut = fp._fill_auc(np.where(np.isnan(p), tr["filled"].mean(), p), y)
    des = lambda d: np.hstack([feats[d["snapshot_idx"].to_numpy(int)][:, idx],
                               cond_of(d)]).astype(np.float32)
    a = []
    for s in range(seeds):
        m = make_gbt(s)
        m.fit(des(tr), tr["filled"].to_numpy(int))
        a.append(fp._fill_auc(m.predict_proba(des(te))[:, 1], y))
    return lut, float(np.mean(a)), len(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    data = {}
    for s in ASSETS:
        g = load(s)
        if g is None:
            print(f"[skip] {s}")
        else:
            data[s] = g
    if not data:
        print("nothing to run")
        return

    if not verify(data):
        print("VERIFICATION FAILED — not running the experiment.")
        sys.exit(1)
    print("All verification passed. Running rotations.\n")

    rows = []
    for sym, (feats, labels, cols, tsec, mid, micro) in data.items():
        print("=" * 68)
        print(f"{sym}   63-feature microstructure, 10 block phases")
        print("=" * 68)
        for p in range(10):
            tr_s, te_s = blocked_split(labels, tsec, p)
            ov = overlap(mid, tr_s, te_s)
            lut, auc, nte = evaluate(feats, labels, tr_s, te_s, micro, args.seeds)
            gap = auc - lut
            print(f"  phase {p}: overlap {ov:.3f}  lookup {lut:.4f}  "
                  f"gbt {auc:.4f}  gap {gap:+.4f}  ({nte:,} rows)")
            rows.append({"asset": sym, "phase": p, "overlap": round(ov, 3),
                         "lookup": round(lut, 4), "auc": round(auc, 4),
                         "gap": round(gap, 4), "test_rows": nte})

    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/rotation_test.csv", index=False)

    print("\n" + "=" * 74)
    print("MICROSTRUCTURE GAP ACROSS 10 BLOCK PHASES")
    print("=" * 74)
    s = df.groupby("asset")["gap"].agg(["mean", "std", "min", "max",
                                        lambda x: (x > 0).sum()])
    s.columns = ["mean", "sd", "min", "max", "n_positive"]
    print(s.round(4).to_string())
    print("=" * 74)

    print("\nOVERLAP VS GAP")
    print("=" * 74)
    for a in df["asset"].unique():
        d = df[df.asset == a]
        lo, hi = d[d.overlap < d.overlap.median()], d[d.overlap >= d.overlap.median()]
        r = np.corrcoef(d["overlap"], d["gap"])[0, 1] if d["overlap"].std() > 0 else float("nan")
        print(f"  {a}: overlap {d.overlap.min():.3f}-{d.overlap.max():.3f} | "
              f"gap low-overlap {lo.gap.mean():+.4f}  high-overlap {hi.gap.mean():+.4f} | "
              f"corr {r:+.2f}")
    print("  A gap that holds in low-overlap rotations does not depend on")
    print("  distributional overlap. A gap that tracks overlap does.")
    print("=" * 74)

    npos = s["n_positive"]
    print(f"\nphase 9 (the published one) is included above.")
    print(f"positive rotations: {dict(npos)}")
    if (npos >= 8).all():
        print("\nROBUST. The gap is positive in at least 8 of 10 rotations on")
        print("every asset. Report mean and range across phases; this also")
        print("quantifies partition variance, which was previously unmeasured.")
    elif (npos.drop("SOLUSD", errors="ignore") >= 8).all():
        print("\nBTC/ETH robust; SOL phase-sensitive. Report per-asset and say")
        print("SOL's blocked result depends on the partition phase.")
    else:
        print("\nNOT ROBUST. The blocked result depends on the arbitrary phase")
        print("and should be demoted to a diagnostic, with the temporal split")
        print("carrying the paper.")


if __name__ == "__main__":
    main()
