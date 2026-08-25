"""
gbm_leakage_check.py — is AUC 0.8264 real signal or leakage?

A jump from 0.637 (lookup table) to 0.826 (GBM) from a model change alone is
large. Before rewriting the paper around it, rule out the two paths by which a
tree model can reach that number without learning anything about microstructure.

  TEST 1  Feature importance. If raw mid_price / price levels dominate, the
          model is likely keying on price level or time index, not book state.

  TEST 2  Snapshot-grouped split. The original split cuts on LABEL rows, so all
          32 labels from one snapshot can straddle train and test. A tree can
          memorise a snapshot from its 109 features in train and recall its
          fill outcomes in test. Splitting by SNAPSHOT closes that path.

  TEST 3  Temporal vs shuffled. Financial data must be split temporally. If a
          shuffled split scores far higher than a temporal one, the model is
          exploiting adjacency rather than generalising.

  TEST 4  Price-free features. Drop raw price columns entirely, keep only
          normalised / relative quantities. Real microstructure signal should
          survive this; leakage on price level will not.

READING (fixed before running):
  - grouped-split AUC stays near 0.82  -> signal is real, paper needs rewriting
  - grouped-split AUC collapses to ~0.64 -> leakage, v2 null stands
  - anything between  -> partial leakage, report the grouped number only

Run:  python gbm_leakage_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("src")))
import fill_predictor as fp

print("=" * 72)
print("GBM LEAKAGE DIAGNOSTIC")
print("=" * 72)

features, labels, feat_cols = fp.load_data()
labels = labels[labels["snapshot_idx"] >= fp.LOOKBACK].reset_index(drop=True)
print(f"features : {features.shape}")
print(f"labels   : {len(labels):,}")


def side_num(s):
    if s.dtype == object:
        return (s.astype(str).str.lower().str[0] == "a").astype(float).to_numpy()
    return s.to_numpy(dtype=float)


def design(df, cols=None):
    idx = df["snapshot_idx"].to_numpy(dtype=int)
    lob = features[idx] if cols is None else features[idx][:, cols]
    cond = np.column_stack([
        df["latency_ms"].to_numpy(dtype=float),
        df["offset"].to_numpy(dtype=float),
        side_num(df["side"]),
    ])
    return np.hstack([lob, cond]).astype(np.float32)


def make_model():
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


def fit_auc(tr, te, cols=None):
    m = make_model()
    m.fit(design(tr, cols), tr["filled"].to_numpy(dtype=int))
    p = m.predict_proba(design(te, cols))[:, 1]
    return fp._fill_auc(p, te["filled"].to_numpy(dtype=float)), m


def lut_auc(tr, te):
    keys = ["offset", "latency_ms", "side"]
    cell = tr.groupby(keys)["filled"].mean()
    base = float(tr["filled"].mean())
    p = te.set_index(keys).index.map(cell).to_numpy(dtype=float)
    p = np.where(np.isnan(p), base, p)
    return fp._fill_auc(p, te["filled"].to_numpy(dtype=float))


results = {}

# ── Baseline: original row-wise temporal split (reproduces 0.8264) ───────────
n = len(labels)
tr_row, te_row = labels.iloc[:int(n * .70)], labels.iloc[int(n * .85):]
auc_row, model_row = fit_auc(tr_row, te_row)
results["row-split temporal (original)"] = auc_row
print(f"\n[1/5] row-split temporal      : {auc_row:.4f}")

# ── TEST 1: feature importance ───────────────────────────────────────────────
print("\n" + "-" * 72)
print("TEST 1 — top features by importance")
print("-" * 72)
names = list(feat_cols) + ["latency_ms", "offset", "side"]
try:
    imp = model_row.feature_importances_
    order = np.argsort(imp)[::-1][:15]
    for i in order:
        flag = ""
        nm = names[i].lower()
        if any(k in nm for k in ("mid_price", "price_") ) and "norm" not in nm:
            flag = "   <-- RAW PRICE"
        print(f"  {names[i]:<34}{imp[i]:>8.4f}{flag}")
    raw_price = sum(imp[i] for i, nm in enumerate(names)
                    if ("mid_price" in nm.lower() or "price_" in nm.lower())
                    and "norm" not in nm.lower())
    print(f"\n  raw-price share of total importance: {raw_price:.1%}")
except Exception as e:
    print(f"  (importances unavailable: {e})")

# ── TEST 2: snapshot-grouped temporal split ──────────────────────────────────
snaps = np.sort(labels["snapshot_idx"].unique())
cut_tr = snaps[int(len(snaps) * .70)]
cut_te = snaps[int(len(snaps) * .85)]
tr_g = labels[labels["snapshot_idx"] < cut_tr]
te_g = labels[labels["snapshot_idx"] >= cut_te]
auc_g, _ = fit_auc(tr_g, te_g)
lut_g = lut_auc(tr_g, te_g)
results["snapshot-grouped temporal"] = auc_g
print("\n" + "-" * 72)
print("TEST 2 — snapshot-grouped split (no snapshot spans train/test)")
print("-" * 72)
print(f"  train snapshots : {tr_g['snapshot_idx'].nunique():,}")
print(f"  test  snapshots : {te_g['snapshot_idx'].nunique():,}")
print(f"  GBM             : {auc_g:.4f}")
print(f"  lookup table    : {lut_g:.4f}")
print(f"  GBM - lookup    : {auc_g - lut_g:+.4f}")

# ── TEST 3: shuffled split, for contrast ─────────────────────────────────────
sh = labels.sample(frac=1.0, random_state=0).reset_index(drop=True)
auc_sh, _ = fit_auc(sh.iloc[:int(n * .70)], sh.iloc[int(n * .85):])
results["shuffled (invalid, contrast only)"] = auc_sh
print("\n" + "-" * 72)
print("TEST 3 — shuffled split (invalid for time series; contrast only)")
print("-" * 72)
print(f"  GBM shuffled    : {auc_sh:.4f}")
print(f"  vs row-split    : {auc_sh - auc_row:+.4f}")

# ── TEST 4: drop raw price columns ───────────────────────────────────────────
keep = [i for i, nm in enumerate(feat_cols)
        if not (("price" in nm.lower() or nm.lower() == "mid_price")
                and "norm" not in nm.lower() and "spread" not in nm.lower())]
auc_np, _ = fit_auc(tr_g, te_g, cols=keep)
results["grouped, no raw price"] = auc_np
print("\n" + "-" * 72)
print("TEST 4 — grouped split, raw price columns dropped")
print("-" * 72)
print(f"  features kept   : {len(keep)} of {len(feat_cols)}")
print(f"  GBM             : {auc_np:.4f}")

# ── Verdict ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
for k, v in results.items():
    print(f"  {k:<38}{v:>8.4f}")
print(f"  {'lookup table (grouped split)':<38}{lut_g:>8.4f}")
print("=" * 72)

g = auc_g - lut_g
if g > 0.10:
    print("Signal SURVIVES the grouped split. The LOB features carry real")
    print("predictive content the LSTM failed to extract. The paper's central")
    print("claim is wrong as written and Section 6 must be rewritten.")
elif g > 0.02:
    print("Signal partially survives. Report the GROUPED number, not 0.8264.")
    print("The claim becomes 'modest signal, architecture-dependent'.")
else:
    print("Signal DISAPPEARS under the grouped split. The 0.8264 was leakage")
    print("from snapshots straddling train and test. The v2 null stands, and")
    print("the paper should report this diagnostic as a guarded negative.")
print("=" * 72)

Path("results").mkdir(exist_ok=True)
out = dict(results)
out["lookup_grouped"] = lut_g
pd.DataFrame([out]).to_csv("results/gbm_leakage_check.csv", index=False)
print("\nsaved -> results/gbm_leakage_check.csv")
