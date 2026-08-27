"""
rerun_v3.py — the experiment. Run verify_v3.py first.

FIXES OVER rerun_v2.py
  1. Binary LSTM head with BCE. The v2 arms fed binary labels to a
     time-to-fill survival decoder with t_obs constant, which pinned loss at
     16.1181 and returned AUC exactly 0.5000 on all six arms.
  2. Count-based grouped split. v2 cut on the time range while snapshot
     density varies over the 70 days, producing 44/31/25 instead of 70/15/15.
  3. Multiple seeds, so SOL's -0.067 / +0.028 swing between feature sets can
     be read against variance rather than guessed at.

ARMS, identical embargoed test rows per asset:
  lookup      16-cell table on (offset_bps, horizon_s, side)
  gbt_micro   trees, 63 microstructure features
  gbt_all     trees, 109 features
  lstm_all    binary LSTM, 109 features, 50-snapshot window
  lstm_micro  binary LSTM, 63 features, 50-snapshot window

WHAT WOULD FALSIFY THE POSITIVE CLAIM, stated before running:
  If mean(gbt_gap) over seeds is below +0.02 on every asset, microstructure
  adds nothing beyond quote geometry at this frequency and the paper is a
  negative result. The v2 labels gave +0.105; the corrected single-seed run
  gave +0.005. This run decides it with error bars.

Run:  python rerun_v3.py --epochs 30 --seeds 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from loguru import logger

sys.path.insert(0, str(Path("src")))
import fill_predictor as fp
from verify_v3 import BinaryLSTM, train_binary, auc_of

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ASSETS = ["BTCUSD", "ETHUSD", "SOLUSD"]
MAX_HORIZON_S = 900
EMBARGO_S = MAX_HORIZON_S * 1.1
MICRO_KEYS = ("spread", "imbalance", "obi", "depth", "vol", "qty", "size")
MAX_LABELS = 200_000
LOOKBACK = 50


class LabelDS(Dataset):
    def __init__(self, feats, labels, idx=None):
        self.f, self.l, self.idx = feats, labels.reset_index(drop=True), idx

    def __len__(self):
        return len(self.l)

    def __getitem__(self, i):
        r = self.l.iloc[i]
        s = int(r["snapshot_idx"])
        x = self.f[s - LOOKBACK:s]
        if self.idx is not None:
            x = x[:, self.idx]
        sd = r["side"]
        sv = 0.0 if (isinstance(sd, str) and sd.lower().startswith("b")) else 1.0
        c = np.array([float(r["horizon_s"]) / MAX_HORIZON_S,
                      float(r["offset_bps"]) / 10.0, sv], dtype=np.float32)
        return (torch.tensor(x, dtype=torch.float32), torch.from_numpy(c),
                torch.tensor(float(r["filled"]), dtype=torch.float32))


def load(sym):
    fpath = Path(f"data/features/{sym}_features.parquet")
    lpath = Path(f"data/features/{sym}_labels_v2.parquet")
    if not fpath.exists() or not lpath.exists():
        logger.error(f"{sym}: missing features or v2 labels")
        return None
    fdf, ldf = pd.read_parquet(fpath), pd.read_parquet(lpath)
    cols = fp.get_feature_cols(fdf)
    feats = np.nan_to_num(fdf[cols].to_numpy(dtype=np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
    if "timestamp_utc" not in fdf.columns:
        logger.error(f"{sym}: features lack timestamp_utc")
        return None
    ep = pd.Timestamp("1970-01-01", tz="UTC")
    tsec = (pd.to_datetime(fdf["timestamp_utc"], utc=True) - ep
            ).dt.total_seconds().to_numpy()
    ldf = ldf[ldf["snapshot_idx"] >= LOOKBACK]
    if len(ldf) > MAX_LABELS:
        ldf = ldf.sample(MAX_LABELS, random_state=0).sort_index()
    ldf = ldf.reset_index(drop=True)
    micro = [i for i, c in enumerate(cols)
             if any(k in c.lower() for k in MICRO_KEYS) and "price" not in c.lower()]
    return feats, ldf, cols, tsec, micro


def split(labels, tsec):
    snaps = np.sort(labels["snapshot_idx"].unique())
    st = tsec[snaps]
    t_a, t_b = st[int(len(snaps) * .70)], st[int(len(snaps) * .85)]
    tr = snaps[st < t_a - EMBARGO_S]
    va = snaps[(st >= t_a) & (st < t_b - EMBARGO_S)]
    te = snaps[st >= t_b]
    pick = lambda s: labels[labels["snapshot_idx"].isin(s)]
    return pick(tr), pick(va), pick(te)


def side_num(s):
    if s.dtype == object:
        return (s.astype(str).str.lower().str[0] == "a").astype(float).to_numpy()
    return s.to_numpy(dtype=float)


def cond_of(d):
    return np.column_stack([d["horizon_s"].to_numpy(float),
                            d["offset_bps"].to_numpy(float), side_num(d["side"])])


def lookup(tr, te):
    k = ["offset_bps", "horizon_s", "side"]
    cell = tr.groupby(k)["filled"].mean()
    p = te.set_index(k).index.map(cell).to_numpy(float)
    p = np.where(np.isnan(p), tr["filled"].mean(), p)
    return fp._fill_auc(p, te["filled"].to_numpy(float)), len(cell)


def gbt(feats, tr, te, idx, seed):
    try:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", n_jobs=-1, random_state=seed)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier as H
        m = H(max_iter=400, max_depth=6, learning_rate=0.05, random_state=seed)
    des = lambda d: np.hstack([feats[d["snapshot_idx"].to_numpy(int)][:, idx],
                               cond_of(d)]).astype(np.float32)
    m.fit(des(tr), tr["filled"].to_numpy(int))
    return fp._fill_auc(m.predict_proba(des(te))[:, 1], te["filled"].to_numpy(float))


def lstm(feats, tr, va, te, nfeat, epochs, seed, idx=None):
    torch.manual_seed(seed); np.random.seed(seed)
    mk = lambda d, sh: DataLoader(LabelDS(feats, d, idx), batch_size=256,
                                  shuffle=sh, num_workers=0)
    model, hist = train_binary(BinaryLSTM(nfeat), mk(tr, True), mk(va, False),
                               epochs=epochs, quiet=True)
    trl = [h[0] for h in hist]
    moved = max(trl) - min(trl)
    if moved < 1e-4:
        logger.warning(f"  LSTM loss did not move (range {moved:.2e}) — "
                       f"treat this arm as broken, not as a negative result")
    return auc_of(model, mk(te, False)), moved, len(hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    rows = []
    for sym in ASSETS:
        got = load(sym)
        if got is None:
            continue
        feats, labels, cols, tsec, micro = got
        tr, va, te = split(labels, tsec)
        tot = len(tr) + len(va) + len(te)
        print("\n" + "=" * 70)
        print(f"{sym}  train {len(tr):,} ({len(tr)/tot:.0%})  "
              f"val {len(va):,} ({len(va)/tot:.0%})  "
              f"test {len(te):,} ({len(te)/tot:.0%})")
        print("=" * 70)

        lut, nc = lookup(tr, te)
        print(f"  lookup ({nc} cells)            {lut:.4f}")

        for name, idx, n in (("micro", micro, len(micro)),
                             ("all", list(range(len(cols))), len(cols))):
            gs = [gbt(feats, tr, te, idx, s) for s in range(a.seeds)]
            print(f"  GBT {name:<6} ({n:>3} feats)      "
                  f"{np.mean(gs):.4f} +/- {np.std(gs):.4f}   "
                  f"{np.mean(gs)-lut:+.4f}")
            rows.append({"asset": sym, "model": f"gbt_{name}", "n_feat": n,
                         "lookup": round(lut, 4), "auc": round(float(np.mean(gs)), 4),
                         "sd": round(float(np.std(gs)), 4),
                         "gap": round(float(np.mean(gs) - lut), 4)})

            ls, mv = [], []
            for s in range(a.seeds):
                a_, m_, ne = lstm(feats, tr, va, te, n, a.epochs, s, idx)
                ls.append(a_); mv.append(m_)
            print(f"  LSTM {name:<6} ({n:>3} feats)     "
                  f"{np.mean(ls):.4f} +/- {np.std(ls):.4f}   "
                  f"{np.mean(ls)-lut:+.4f}   loss moved {np.mean(mv):.4f}")
            rows.append({"asset": sym, "model": f"lstm_{name}", "n_feat": n,
                         "lookup": round(lut, 4), "auc": round(float(np.mean(ls)), 4),
                         "sd": round(float(np.std(ls)), 4),
                         "gap": round(float(np.mean(ls) - lut), 4),
                         "loss_moved": round(float(np.mean(mv)), 5)})

    if not rows:
        print("nothing ran"); return
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/rerun_v3_results.csv", index=False)

    print("\n" + "=" * 78)
    print(f"CORRECTED LABELS + BINARY LSTM — fill AUC, {a.seeds} seeds")
    print("=" * 78)
    print(df.to_string(index=False))
    print("=" * 78)

    broken = df[(df.model.str.startswith("lstm")) &
                (df.get("loss_moved", pd.Series(1, index=df.index)) < 1e-4)]
    if len(broken):
        print("WARNING: LSTM arms with no loss movement — broken, not negative:")
        print(broken[["asset", "model"]].to_string(index=False))

    g = df[df.model == "gbt_micro"]["gap"]
    print(f"\nGBT micro gap: {g.round(4).tolist()}  mean {g.mean():+.4f}")
    print("  v2 defective labels: +0.045 to +0.174 (mean +0.105)")
    print("  v2 corrected, 1 seed: +0.005")
    if (g > 0.02).all():
        print("\nMicrostructure adds signal on every asset. Positive result holds.")
    elif (g > 0.02).any():
        print(f"\nHolds on {int((g>0.02).sum())} of {len(g)} assets. Per-asset only.")
    else:
        print("\nMicrostructure adds nothing beyond quote geometry. The paper is")
        print("a negative result, and this time on correctly constructed labels.")


if __name__ == "__main__":
    main()
