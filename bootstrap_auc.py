"""
Dependence-aware uncertainty for the paper's headline temporal-split result.

This script intentionally leaves rerun_v3.py untouched. It reproduces the
same sampled labels, grouped temporal split, lookup reference, microstructure
feature subset, and GBT specification, then saves row-level out-of-sample
predictions and bootstraps the paired AUC difference

    delta_AUC = AUC(GBT micro) - AUC(lookup)

by resampling contiguous time blocks. All label rows from a snapshot remain
together. The default 3600-second block is longer than the 900-second maximum
label horizon; sensitivity can be checked with --block-seconds.

Requires the same local files used by rerun_v3.py:
  data/features/BTCUSD_features.parquet
  data/features/BTCUSD_labels_v2.parquet
  ... ETHUSD / SOLUSD ...

Run:
  python bootstrap_auc.py --seeds 3 --n-bootstrap 2000 --block-seconds 3600

Optional sensitivity runs:
  python bootstrap_auc.py --seeds 3 --n-bootstrap 2000 --block-seconds 1800
  python bootstrap_auc.py --seeds 3 --n-bootstrap 2000 --block-seconds 7200

Outputs:
  results/bootstrap_predictions_<ASSET>.csv
  results/bootstrap_auc_summary.csv
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


def side_num(s):
    if s.dtype == object:
        return (s.astype(str).str.lower().str[0] == "a").astype(float).to_numpy()
    return s.to_numpy(dtype=float)


def cond_of(d):
    return np.column_stack([
        d["horizon_s"].to_numpy(float),
        d["offset_bps"].to_numpy(float),
        side_num(d["side"]),
    ])


def auc(y, p):
    return float(fp._fill_auc(np.asarray(p, float), np.asarray(y, float)))


def load(sym):
    fpath = Path(f"data/features/{sym}_features.parquet")
    lpath = Path(f"data/features/{sym}_labels_v2.parquet")
    if not fpath.exists() or not lpath.exists():
        raise FileNotFoundError(
            f"Missing {fpath} or {lpath}. Run this on the machine containing "
            "the paper's local feature/label parquet files."
        )

    fdf = pd.read_parquet(fpath)
    ldf = pd.read_parquet(lpath)
    cols = fp.get_feature_cols(fdf)
    feats = np.nan_to_num(
        fdf[cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if "timestamp_utc" not in fdf.columns:
        raise ValueError(f"{sym}: feature parquet lacks timestamp_utc")

    ts = pd.to_datetime(fdf["timestamp_utc"], utc=True)
    ep = pd.Timestamp("1970-01-01", tz="UTC")
    tsec = (ts - ep).dt.total_seconds().to_numpy()

    ldf = ldf[ldf["snapshot_idx"] >= LOOKBACK]
    if len(ldf) > MAX_LABELS:
        # Exactly matches rerun_v3.py.
        ldf = ldf.sample(MAX_LABELS, random_state=0).sort_index()
    ldf = ldf.reset_index(drop=True)

    micro = [
        i for i, c in enumerate(cols)
        if any(k in c.lower() for k in MICRO_KEYS) and "price" not in c.lower()
    ]
    return feats, ldf, tsec, micro


def split(labels, tsec):
    # Exactly matches rerun_v3.py's count-based, snapshot-grouped temporal split.
    snaps = np.sort(labels["snapshot_idx"].unique())
    st = tsec[snaps]
    t_a = st[int(len(snaps) * 0.70)]
    t_b = st[int(len(snaps) * 0.85)]

    tr_snaps = snaps[st < t_a - EMBARGO_S]
    va_snaps = snaps[(st >= t_a) & (st < t_b - EMBARGO_S)]
    te_snaps = snaps[st >= t_b]

    pick = lambda s: labels[labels["snapshot_idx"].isin(s)].copy()
    return pick(tr_snaps), pick(va_snaps), pick(te_snaps)


def lookup_predictions(tr, te):
    keys = ["offset_bps", "horizon_s", "side"]
    cell = tr.groupby(keys)["filled"].mean()
    p = te.set_index(keys).index.map(cell).to_numpy(float)
    p = np.where(np.isnan(p), tr["filled"].mean(), p)
    return p


def design(feats, d, idx):
    return np.hstack([
        feats[d["snapshot_idx"].to_numpy(int)][:, idx],
        cond_of(d),
    ]).astype(np.float32)


def gbt_predictions(feats, tr, te, idx, seed):
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=400,
            max_depth=6,
            learning_rate=0.05,
            random_state=seed,
        )

    xtr = design(feats, tr, idx)
    xte = design(feats, te, idx)
    model.fit(xtr, tr["filled"].to_numpy(int))
    return model.predict_proba(xte)[:, 1]


def make_time_blocks(pred, block_seconds):
    t0 = pred["timestamp_s"].min()
    block_id = np.floor((pred["timestamp_s"].to_numpy() - t0) / block_seconds).astype(int)
    pred = pred.copy()
    pred["block_id"] = block_id
    blocks = [g.index.to_numpy() for _, g in pred.groupby("block_id", sort=True)]
    if len(blocks) < 5:
        raise ValueError(
            f"Only {len(blocks)} time blocks at block_seconds={block_seconds}; "
            "choose a shorter block length."
        )
    return pred, blocks


def bootstrap_delta(pred, seed_cols, n_bootstrap, block_seconds, rng):
    pred, blocks = make_time_blocks(pred, block_seconds)
    y = pred["filled"].to_numpy(float)
    p_lookup = pred["lookup_pred"].to_numpy(float)
    p_gbt = pred[seed_cols].to_numpy(float)

    point_lookup = auc(y, p_lookup)
    point_seed_aucs = np.array([auc(y, p_gbt[:, j]) for j in range(p_gbt.shape[1])])
    point_delta = float(point_seed_aucs.mean() - point_lookup)

    draws = []
    n_blocks = len(blocks)
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        ix = np.concatenate([blocks[j] for j in chosen])
        yb = y[ix]

        # Rare resamples can contain one class only; AUC is then undefined.
        if np.unique(yb).size < 2:
            continue

        lookup_auc = auc(yb, p_lookup[ix])
        seed_aucs = np.array([auc(yb, p_gbt[ix, j]) for j in range(p_gbt.shape[1])])
        draws.append(float(seed_aucs.mean() - lookup_auc))

    if len(draws) < max(100, int(0.9 * n_bootstrap)):
        raise RuntimeError(
            f"Only {len(draws)} valid bootstrap draws out of {n_bootstrap}; "
            "inspect block length and class balance."
        )

    draws = np.asarray(draws)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "lookup_auc": point_lookup,
        "gbt_auc_mean_over_seeds": float(point_seed_aucs.mean()),
        "delta_auc": point_delta,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "n_blocks": n_blocks,
        "n_valid_bootstrap": len(draws),
    }


def run_asset(sym, seeds, n_bootstrap, block_seconds, rng):
    feats, labels, tsec, micro = load(sym)
    tr, _va, te = split(labels, tsec)
    te = te.reset_index(drop=True)

    print(f"\n{sym}: train rows={len(tr):,}, test rows={len(te):,}, micro features={len(micro)}")

    out = te[["snapshot_idx", "offset_bps", "horizon_s", "side", "filled"]].copy()
    out["timestamp_s"] = tsec[out["snapshot_idx"].to_numpy(int)]
    out["lookup_pred"] = lookup_predictions(tr, te)

    seed_cols = []
    for seed in range(seeds):
        col = f"gbt_pred_seed{seed}"
        print(f"  fitting GBT micro seed {seed}")
        out[col] = gbt_predictions(feats, tr, te, micro, seed)
        seed_cols.append(col)

    Path("results").mkdir(exist_ok=True)
    pred_path = Path(f"results/bootstrap_predictions_{sym}.csv")
    out.to_csv(pred_path, index=False)

    stats = bootstrap_delta(
        out,
        seed_cols=seed_cols,
        n_bootstrap=n_bootstrap,
        block_seconds=block_seconds,
        rng=rng,
    )
    stats.update({
        "asset": sym,
        "block_seconds": block_seconds,
        "n_test_rows": len(out),
        "n_test_snapshots": int(out["snapshot_idx"].nunique()),
        "n_seeds": seeds,
    })

    print(
        f"  delta AUC={stats['delta_auc']:+.4f}, "
        f"95% block-bootstrap CI [{stats['ci95_low']:+.4f}, {stats['ci95_high']:+.4f}], "
        f"blocks={stats['n_blocks']}"
    )
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--block-seconds", type=int, default=3600)
    ap.add_argument("--bootstrap-seed", type=int, default=20260904)
    args = ap.parse_args()

    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be >= 100")
    if args.block_seconds <= MAX_HORIZON_S:
        print(
            f"WARNING: block length {args.block_seconds}s is not longer than the "
            f"maximum label horizon ({MAX_HORIZON_S}s). Prefer > {MAX_HORIZON_S}s."
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    rows = [
        run_asset(sym, args.seeds, args.n_bootstrap, args.block_seconds, rng)
        for sym in ASSETS
    ]

    summary = pd.DataFrame(rows)
    cols = [
        "asset", "lookup_auc", "gbt_auc_mean_over_seeds", "delta_auc",
        "ci95_low", "ci95_high", "bootstrap_sd", "block_seconds",
        "n_blocks", "n_valid_bootstrap", "n_test_rows", "n_test_snapshots",
        "n_seeds",
    ]
    summary = summary[cols]
    summary.to_csv("results/bootstrap_auc_summary.csv", index=False)

    print("\nDEPENDENCE-AWARE DELTA-AUC SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved results/bootstrap_auc_summary.csv")


if __name__ == "__main__":
    main()
