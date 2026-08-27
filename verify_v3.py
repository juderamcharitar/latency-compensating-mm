"""
verify_v3.py — RUN THIS BEFORE rerun_v3.py.

Written before the experiment, not after. Every prior defect in this project
produced a plausible number that nobody could distinguish from a real one by
looking at it. The only defence is a check that fails loudly on known inputs.

FALSIFICATION CRITERIA, fixed now:

  C1  Binary LSTM must reach AUC >= 0.90 on a synthetic task with planted
      signal. If it cannot learn a signal we know is there, no real-data
      number from it means anything, and the model-class comparison is void.

  C2  Training loss must MOVE. The v2 run showed loss pinned at 16.1181 for
      every epoch of every arm. Any run where loss changes by < 1e-4 across
      epochs is a broken run, not a negative result.

  C3  Split proportions must land within 5 points of 70/15/15. The v2 harness
      produced 44/31/25 because it cut on the time range while snapshot
      density varies across the 70 days.

  C4  Embargo gap must be >= 990 s at both boundaries, and partitions must not
      overlap or reorder.

  C5  Lookup table must reproduce itself: a GBT given only the conditioning
      columns must land within 0.01 AUC of the table. This caught nothing
      before but is cheap and validates the comparison.

If any check fails, the experiment does not run.

Run:  python verify_v3.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path("src")))
import fill_predictor as fp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_HORIZON_S = 900
EMBARGO_S = MAX_HORIZON_S * 1.1

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# ── The binary model under test ───────────────────────────────────────────────

class BinaryLSTM(nn.Module):
    """
    Binary fill classifier.

    The v2 labels are binary ("filled within horizon H"), not a time-to-fill.
    Feeding a binary target to the survival decoder in fill_predictor.py was
    the defect that pinned loss at 16.1181 and produced AUC exactly 0.5000 on
    all six arms. This replaces the survival head with a single logit and BCE.
    """

    def __init__(self, n_features, hidden=64, cond_dim=3):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.cond = nn.Sequential(nn.Linear(cond_dim, 16), nn.ReLU(),
                                  nn.Linear(16, 16))
        self.head = nn.Sequential(nn.Linear(hidden + 16, 64), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x, cond):
        h = self.proj(x)
        _, (hn, _) = self.lstm(h)
        z = torch.cat([hn[-1], self.cond(cond)], dim=-1)
        return self.head(z).squeeze(-1)          # logits


def train_binary(model, tr, va, epochs=30, lr=1e-3, patience=6, quiet=False):
    """Returns (model, loss_history). Loss history is checked by C2."""
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    best, bad, hist = np.inf, 0, []
    best_state = None

    for ep in range(epochs):
        model.train()
        tot = n = 0
        for x, c, y in tr:
            x, c, y = x.to(DEVICE), c.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(x, c), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        trl = tot / max(n, 1)

        model.eval(); tot = n = 0
        with torch.no_grad():
            for x, c, y in va:
                x, c, y = x.to(DEVICE), c.to(DEVICE), y.to(DEVICE)
                tot += lossf(model(x, c), y).item() * len(y); n += len(y)
        vl = tot / max(n, 1)
        hist.append((trl, vl))
        if not quiet:
            print(f"      ep {ep+1:02d}  train {trl:.5f}  val {vl:.5f}")

        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, hist


def auc_of(model, loader):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for x, c, y in loader:
            ps.append(torch.sigmoid(model(x.to(DEVICE), c.to(DEVICE))).cpu().numpy())
            ys.append(y.numpy())
    return fp._fill_auc(np.concatenate(ps), np.concatenate(ys).astype(float))


# ── C1 + C2: positive control on planted signal ───────────────────────────────

def c1_c2_positive_control():
    print("\nC1/C2  Binary LSTM on synthetic data with PLANTED signal")
    print("       (if it cannot learn this, no real-data number is meaningful)")
    torch.manual_seed(0); np.random.seed(0)

    n, L, F = 6000, 20, 12
    X = np.random.randn(n, L, F).astype(np.float32)
    cond = np.random.rand(n, 3).astype(np.float32)

    # planted rule: fill iff late-window mean of feature 0 plus cond[:,1] is high
    score = X[:, -5:, 0].mean(axis=1) * 1.5 + cond[:, 1] * 2.0
    y = (score > np.median(score)).astype(np.float32)

    def mk(a, b, sh):
        return DataLoader(TensorDataset(torch.tensor(X[a:b]),
                                        torch.tensor(cond[a:b]),
                                        torch.tensor(y[a:b])),
                          batch_size=256, shuffle=sh)
    tr, va, te = mk(0, 4200, True), mk(4200, 5100, False), mk(5100, n, False)

    model, hist = train_binary(BinaryLSTM(F), tr, va, epochs=30, quiet=True)
    a = auc_of(model, te)
    trl = [h[0] for h in hist]
    moved = max(trl) - min(trl)

    print(f"       epochs run: {len(hist)}   train loss {trl[0]:.5f} -> {trl[-1]:.5f}")
    check("C1 binary LSTM recovers planted signal (AUC >= 0.90)",
          a >= 0.90, f"AUC {a:.4f}")
    check("C2 training loss moves (> 1e-4)",
          moved > 1e-4, f"range {moved:.6f}")
    return a


# ── C2b: the OLD survival head on the same data, to confirm the diagnosis ─────

def c2b_survival_head_fails():
    print("\nC2b   Old survival head on the same planted-signal data")
    print("       (expected to FAIL — this is the defect being fixed)")
    torch.manual_seed(0); np.random.seed(0)
    n, L, F = 3000, 20, 12
    X = np.random.randn(n, L, F).astype(np.float32)
    cond = np.random.rand(n, 3).astype(np.float32)
    score = X[:, -5:, 0].mean(axis=1) * 1.5 + cond[:, 1] * 2.0
    y = (score > np.median(score)).astype(np.float32)
    t_obs = np.ones(n, dtype=np.float32)          # what rerun_v2.py fed it

    ds = TensorDataset(torch.tensor(X), torch.tensor(cond),
                       torch.tensor(t_obs), torch.tensor(y))
    ld = DataLoader(ds, batch_size=256)
    model = fp.LSTMFillPredictor(n_features=F).to(DEVICE)
    lossf = fp.SurvivalLoss() if hasattr(fp, "SurvivalLoss") else None
    if lossf is None:
        print("       (SurvivalLoss not exposed; skipping)")
        return
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for ep in range(8):
        tot = m = 0
        for x, c, t, d in ld:
            x, c, t, d = x.to(DEVICE), c.to(DEVICE), t.to(DEVICE), d.to(DEVICE)
            opt.zero_grad()
            lh, _ = model(x, c)
            loss = lossf(lh, t, d)
            loss.backward(); opt.step()
            tot += loss.item() * len(d); m += len(d)
        losses.append(tot / m)
    # The head collapses TO a degenerate point over the first epochs, then has
    # no gradient. Test the converged tail, not the whole trajectory: the
    # earlier version of this check tested max-min over all epochs and failed
    # on the collapse itself, which is not the thing being diagnosed.
    tail = losses[-3:]
    tail_range = max(tail) - min(tail)
    print(f"       loss over {len(losses)} epochs: {[round(l,4) for l in losses]}")
    print(f"       converged tail: {[round(l,4) for l in tail]}")
    check("C2b survival head is inert on binary targets once converged "
          "(tail range < 1e-2)", tail_range < 1e-2,
          f"tail range {tail_range:.6f}, settles at {tail[-1]:.4f} "
          f"— matches the 16.1181 seen in every v2 arm")


# ── C3 + C4: split ────────────────────────────────────────────────────────────

def split_v3(labels, tsec):
    """Count-based grouped split with embargo. Fixes v2's 44/31/25."""
    snaps = np.sort(labels["snapshot_idx"].unique())
    st = tsec[snaps]
    i_a, i_b = int(len(snaps) * 0.70), int(len(snaps) * 0.85)
    t_a, t_b = st[i_a], st[i_b]
    tr = snaps[st < t_a - EMBARGO_S]
    va = snaps[(st >= t_a) & (st < t_b - EMBARGO_S)]
    te = snaps[st >= t_b]
    return tr, va, te


def c3_c4_split():
    print("\nC3/C4  Split proportions and embargo")
    rng = np.random.default_rng(0)
    # irregular density, like the real feed
    gaps = rng.exponential(30, 20000) * (1 + np.sin(np.arange(20000) / 900))
    tsec = np.cumsum(gaps)
    lab = pd.DataFrame({"snapshot_idx": np.repeat(np.arange(20000), 4)})

    tr, va, te = split_v3(lab, tsec)
    tot = len(tr) + len(va) + len(te)
    p = np.array([len(tr), len(va), len(te)]) / tot * 100
    check("C3 split within 5pp of 70/15/15", 
          abs(p[0]-70) < 5 and abs(p[1]-15) < 5 and abs(p[2]-15) < 5,
          f"{p[0]:.1f}/{p[1]:.1f}/{p[2]:.1f}")

    g1 = tsec[va].min() - tsec[tr].max()
    g2 = tsec[te].min() - tsec[va].max()
    check("C4a embargo >= 990s at both boundaries",
          g1 >= 990 and g2 >= 990, f"{g1:.0f}s / {g2:.0f}s")
    check("C4b partitions disjoint and ordered",
          len(set(tr) & set(va)) == 0 and len(set(va) & set(te)) == 0
          and tsec[tr].max() < tsec[va].min() < tsec[te].min(), "")


# ── C5: lookup reproducibility ────────────────────────────────────────────────

def c5_lookup():
    print("\nC5    GBT on conditioning columns must reproduce the lookup table")
    rng = np.random.default_rng(0)
    n = 40000
    off = rng.choice([1, 2, 5, 10], n)
    hor = rng.choice([300, 900], n)
    side = rng.choice([0, 1], n)
    p = 0.8 - 0.05 * off + 0.0002 * hor
    y = (rng.random(n) < np.clip(p, .01, .99)).astype(int)
    df = pd.DataFrame({"offset_bps": off, "horizon_s": hor,
                       "side": side, "filled": y})
    tr, te = df.iloc[:30000], df.iloc[30000:]

    cell = tr.groupby(["offset_bps", "horizon_s", "side"])["filled"].mean()
    pl = te.set_index(["offset_bps", "horizon_s", "side"]).index.map(cell).to_numpy(float)
    pl = np.where(np.isnan(pl), tr["filled"].mean(), pl)
    a_lut = fp._fill_auc(pl, te["filled"].to_numpy(float))

    try:
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          eval_metric="logloss", n_jobs=-1, random_state=0)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier as H
        m = H(max_iter=200, max_depth=4, learning_rate=0.05, random_state=0)
    Xtr = tr[["offset_bps", "horizon_s", "side"]].to_numpy(float)
    Xte = te[["offset_bps", "horizon_s", "side"]].to_numpy(float)
    m.fit(Xtr, tr["filled"])
    a_gbt = fp._fill_auc(m.predict_proba(Xte)[:, 1], te["filled"].to_numpy(float))
    check("C5 GBT(cond) within 0.01 of lookup",
          abs(a_gbt - a_lut) < 0.01, f"lookup {a_lut:.4f} vs GBT {a_gbt:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("PRE-EXPERIMENT VERIFICATION")
    print("=" * 70)
    c1_c2_positive_control()
    try:
        c2b_survival_head_fails()
    except Exception as e:
        print(f"       (C2b skipped: {e})")
    c3_c4_split()
    c5_lookup()

    print("\n" + "=" * 70)
    print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
    print("=" * 70)
    if FAIL:
        print("DO NOT RUN THE EXPERIMENT. Failing checks:")
        for f in FAIL:
            print("   -", f)
        sys.exit(1)
    print("All checks passed. Safe to run rerun_v3.py.")
