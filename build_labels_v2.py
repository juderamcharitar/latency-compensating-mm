"""
build_labels_v2.py — corrected label constructor for real Kraken data.

WHAT WAS WRONG
--------------
src/lob_features.py line 331 hardcodes

    interval_ms = SYNTHETIC_INTERVAL * 1000     # 100 ms per snapshot

and line 343 converts a latency window to a fixed snapshot count,

    n_steps = max(1, int(lat_ms / interval_ms))

That is correct for the synthetic generator, which emits a snapshot every
100 ms. It is wrong for Kraken, whose book channel is event-driven with an
18.2 s median gap. Measured on the real BTC/USD data, the labels were:

    "200 ms"  ->  2 snapshots ahead  ->    33.5 s median horizon  (167x)
    "500 ms"  ->  5 snapshots        ->   106.9 s                 (214x)
    "1000 ms" -> 10 snapshots        ->   237.2 s                 (237x)
    "2000 ms" -> 20 snapshots        ->   511.1 s                 (256x)

Line 327 also sets tick_size = spread / 4 rather than the venue tick, and the
fill test compared future MID prices rather than top-of-book.

WHAT THIS DOES INSTEAD
----------------------
1. Horizons are real elapsed seconds, resolved per row with searchsorted over
   the actual timestamps.
2. Tick size is inferred from observed price increments on the venue.
3. Fill condition is top-of-book, which is what a resting limit order needs:
       buy  at p fills iff  min(best_ask) <= p  within the horizon
       sell at p fills iff  max(best_bid) >= p  within the horizon
4. Rows with NO snapshot inside the horizon are UNOBSERVABLE and are dropped,
   not silently labelled "did not fill". This matters: at short horizons on a
   sparse feed, most rows have no observation, and calling them non-fills
   would manufacture the label.

Horizons default to 30/60/300/900 s, which the sampling rate can actually
resolve. They are no longer described as execution latencies.

Run:  python build_labels_v2.py --symbol BTCUSD
      python build_labels_v2.py --all
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Horizons in seconds. 30s and 60s are excluded: observability falls to
# 41-54% on ETH and SOL, and the observable subsample is biased toward active
# periods, which produced a non-monotonic fill rate (SOL 0.262 at 30s vs 0.259
# at 60s). 300s and 900s are >=88% observable on every asset.
HORIZONS_S = [300, 900]

# Quote offsets in BASIS POINTS of mid, not ticks. Venue ticks are not
# economically comparable across these assets: BTC's 0.1 tick on a ~65,000
# price is 0.00015% of mid, while SOL's 0.01 tick on ~150 is 0.0067%, roughly
# 40x larger in relative terms. With tick offsets the 1-5 range spanned a
# 0.176 fill-rate gradient on SOL but only 0.011 on BTC, so "offset" meant
# different things per asset. Basis points fix the comparison.
OFFSETS_BPS = [1, 2, 5, 10]
DATA_DIR   = Path("data/kraken/raw")
OUT_DIR    = Path("data/features")


# ── Tick size inference ───────────────────────────────────────────────────────

def infer_tick(prices: np.ndarray) -> float:
    """
    Smallest consistent price increment on the venue.

    Takes the modal positive difference between adjacent distinct quoted
    prices. Robust to occasional multi-tick jumps, unlike a naive minimum.
    """
    u = np.unique(prices[np.isfinite(prices)])
    if len(u) < 3:
        return 0.01
    d = np.diff(u)
    d = d[d > 0]
    if len(d) == 0:
        return 0.01
    # round to 8dp to collapse float noise, then take the mode
    vals, counts = np.unique(np.round(d, 8), return_counts=True)
    tick = float(vals[np.argmax(counts)])
    return tick if tick > 0 else 0.01


# ── Range min / max over variable windows ─────────────────────────────────────

def sparse_table(a: np.ndarray, op) -> tuple:
    """Build a sparse table for O(1) range queries."""
    n = len(a)
    k = max(1, int(np.floor(np.log2(n))) + 1)
    tab = np.empty((k, n), dtype=a.dtype)
    tab[0] = a
    for j in range(1, k):
        span = 1 << j
        half = span >> 1
        m = n - span + 1
        if m <= 0:
            tab[j, :] = tab[j - 1, :]
            continue
        tab[j, :m] = op(tab[j - 1, :m], tab[j - 1, half:half + m])
    return tab, k


def range_query(tab, k, lo: np.ndarray, hi: np.ndarray, op, fill):
    """Query op over [lo, hi) for arrays of bounds. Invalid ranges -> fill."""
    out = np.full(len(lo), fill, dtype=float)
    valid = hi > lo
    if not valid.any():
        return out
    l, h = lo[valid], hi[valid]
    length = h - l
    j = np.floor(np.log2(np.maximum(length, 1))).astype(int)
    j = np.minimum(j, k - 1)
    left = tab[j, l]
    right = tab[j, h - (1 << j)]
    out[valid] = op(left, right)
    return out


# ── Main label construction ───────────────────────────────────────────────────

def build(symbol: str) -> pd.DataFrame:
    files = sorted((DATA_DIR / symbol).rglob("*.parquet"))
    if not files:
        logger.error(f"no parquet files under {DATA_DIR / symbol}")
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
    df = df.reset_index(drop=True)

    # Seconds since epoch. Do NOT use .astype("int64")/1e9: parquet returns
    # datetime64[us] here, not [ns], which silently makes every interval
    # 1000x too small. Derive the unit instead of assuming it.
    _epoch = pd.Timestamp("1970-01-01", tz="UTC")
    t   = (df["timestamp_utc"] - _epoch).dt.total_seconds().to_numpy()
    bid = df["bid_price_1"].to_numpy(dtype=float)
    ask = df["ask_price_1"].to_numpy(dtype=float)
    mid = (bid + ask) / 2.0
    n   = len(df)

    tick = infer_tick(np.concatenate([bid, ask]))
    gaps = np.diff(t)
    span_days = (t[-1] - t[0]) / 86400.0
    logger.info(f"[{symbol}] {n:,} snapshots | tick={tick:g} | "
                f"median gap={np.median(gaps):.1f}s | mean gap={gaps.mean():.1f}s | "
                f"span={span_days:.1f}d")

    # Guard against timestamp-unit errors. An event-driven book feed cannot
    # plausibly deliver sub-second median gaps over a multi-week span, and a
    # unit slip of 1000x is exactly what silently produced 97% fill rates in
    # an earlier revision of this script.
    if np.median(gaps) < 1.0:
        raise ValueError(
            f"[{symbol}] median gap {np.median(gaps):.4f}s is implausible for "
            f"this feed. Check the timestamp unit conversion.")
    if not (1.0 < span_days < 400.0):
        raise ValueError(
            f"[{symbol}] dataset span {span_days:.2f} days is implausible. "
            f"Check the timestamp unit conversion.")

    # sparse tables for future top-of-book extremes
    tab_ask, k_ask = sparse_table(ask, np.minimum)
    tab_bid, k_bid = sparse_table(bid, np.maximum)

    idx = np.arange(n)
    rows = []

    for H in HORIZONS_S:
        # first index strictly after t[i], and first index after t[i] + H
        lo = np.searchsorted(t, t, side="right")
        hi = np.searchsorted(t, t + H, side="right")
        observable = hi > lo

        fut_ask = range_query(tab_ask, k_ask, lo, hi, np.minimum, np.nan)
        fut_bid = range_query(tab_bid, k_bid, lo, hi, np.maximum, np.nan)

        for off in OFFSETS_BPS:
            delta   = mid * (off / 10_000.0)
            # round to the venue tick so the quote is actually postable
            buy_px  = np.floor((mid - delta) / tick) * tick
            sell_px = np.ceil((mid + delta) / tick) * tick

            buy_fill  = observable & (fut_ask <= buy_px)
            sell_fill = observable & (fut_bid >= sell_px)

            for side, px, filled in (("bid", buy_px, buy_fill),
                                     ("ask", sell_px, sell_fill)):
                m = observable
                rows.append(pd.DataFrame({
                    "snapshot_idx": idx[m],
                    "side":         side,
                    "offset_bps":   off,
                    "horizon_s":    H,
                    "filled":       filled[m].astype(int),
                    "mid_at_submit": mid[m],
                    "quote_price":  px[m],
                    "n_obs_in_window": (hi - lo)[m],
                }))

        obs_rate = observable.mean()
        logger.info(f"[{symbol}] horizon {H:>4}s: {obs_rate:6.1%} of snapshots "
                    f"observable, median {int(np.median((hi-lo)[observable]))} "
                    f"snapshots in window")

    out = pd.concat(rows, ignore_index=True)
    return out, df, tick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    symbols = ["BTCUSD", "ETHUSD", "SOLUSD"] if args.all else [args.symbol]
    if symbols == [None]:
        ap.error("pass --symbol SYMBOL or --all")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for sym in symbols:
        res = build(sym)
        if isinstance(res, pd.DataFrame) and res.empty:
            continue
        labels, df, tick = res

        path = OUT_DIR / f"{sym}_labels_v2.parquet"
        labels.to_parquet(path, index=False)

        print("\n" + "=" * 64)
        print(f"{sym}  —  tick {tick:g}  —  {len(labels):,} label rows")
        print("=" * 64)
        piv = labels.pivot_table(index="horizon_s", columns="offset_bps",
                                 values="filled", aggfunc="mean")
        print("fill rate by horizon (s) and offset (bps):")
        print(piv.round(3).to_string())
        print(f"\noverall fill rate: {labels['filled'].mean():.3f}")

        # Fill rate must be non-decreasing in horizon: a longer window can only
        # add fill opportunities. A violation indicates observability bias.
        viol = []
        for c in piv.columns:
            col = piv[c].to_numpy()
            if np.any(np.diff(col) < -1e-9):
                viol.append(c)
        if viol:
            print(f"WARNING: fill rate decreases with horizon at offsets {viol} "
                  f"- check observability bias")
        else:
            print("monotonic in horizon: OK")

        # Offset gradient: the span the lookup reference has to work with.
        for h in piv.index:
            row = piv.loc[h].to_numpy()
            print(f"  horizon {h:>4}s: offset gradient "
                  f"{row[0]:.3f} -> {row[-1]:.3f}  (span {row[0]-row[-1]:+.3f})")
        print(f"saved -> {path}")

        summary.append({
            "asset": sym,
            "snapshots": len(df),
            "tick": tick,
            "label_rows": len(labels),
            "fill_rate": round(float(labels["filled"].mean()), 4),
        })

    if summary:
        s = pd.DataFrame(summary)
        Path("results").mkdir(exist_ok=True)
        s.to_csv("results/labels_v2_summary.csv", index=False)
        print("\n" + "=" * 64)
        print(s.to_string(index=False))
        print("=" * 64)
        print("\nNOTE: label rows are NOT snapshots x 32. Rows with no")
        print("observation inside the horizon are dropped as unobservable")
        print("rather than labelled as non-fills.")


if __name__ == "__main__":
    main()
