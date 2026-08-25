"""
gbm_baseline.py — is the null architecture-specific, or does it hold for any model?

Section 9 of the paper concedes that no non-neural baseline was run. This is
that baseline. It compares three predictors on the IDENTICAL test rows:

  A) lookup table  — mean fill rate per (offset, latency, side), 32 cells
  B) gradient boosting — 109 LOB features + the 3 conditioning variables, flat
  C) LSTM          — reported from fill_predictor.py for reference

The LSTM sees a 50-snapshot window; the GBM sees only the current snapshot plus
conditioning. That favours the LSTM on temporal information, so if the GBM
matches it, the temporal window is not carrying anything either.

PRE-REGISTERED READING (fixed before running):
  GBM - lookup > +0.02  -> the LSTM specifically failed; a better model finds
                           signal, and the paper's claim must narrow to "this
                           architecture" rather than "these features"
  GBM - lookup within +/-0.02
                        -> the null is architecture-independent; quote geometry
                           is the ceiling for any model on this data
  GBM < lookup - 0.02   -> the features actively mislead flexible learners

Run:  python gbm_baseline.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("src")))
import fill_predictor as fp

print("=" * 70)
print("GRADIENT BOOSTING BASELINE")
print("=" * 70)

features, labels, feat_cols = fp.load_data()
print(f"features : {features.shape}")
print(f"labels   : {len(labels):,}")

# ── Match the LSTM's usable rows exactly ─────────────────────────────────────
labels = labels[labels["snapshot_idx"] >= fp.LOOKBACK].reset_index(drop=True)
n = len(labels)
tr, va = int(n * 0.70), int(n * 0.85)
train, test = labels.iloc[:tr], labels.iloc[va:]
print(f"train    : {len(train):,}")
print(f"test     : {len(test):,}")

y_train = train["filled"].to_numpy(dtype=int)
y_test = test["filled"].to_numpy(dtype=int)


def side_num(s):
    if s.dtype == object:
        return (s.astype(str).str.lower().str[0] == "a").astype(float).to_numpy()
    return s.to_numpy(dtype=float)


def design(df):
    """Current-snapshot LOB features + the 3 conditioning variables."""
    lob = features[df["snapshot_idx"].to_numpy(dtype=int)]      # (N, 109)
    cond = np.column_stack([
        df["latency_ms"].to_numpy(dtype=float),
        df["offset"].to_numpy(dtype=float),
        side_num(df["side"]),
    ])
    return np.hstack([lob, cond]).astype(np.float32)


X_train, X_test = design(train), design(test)
print(f"design   : {X_train.shape[1]} columns "
      f"({features.shape[1]} LOB + 3 conditioning)")

# ── A) Lookup table ──────────────────────────────────────────────────────────
keys = ["offset", "latency_ms", "side"]
cell = train.groupby(keys)["filled"].mean()
base = float(train["filled"].mean())
p_lut = test.set_index(keys).index.map(cell).to_numpy(dtype=float)
p_lut = np.where(np.isnan(p_lut), base, p_lut)
auc_lut = fp._fill_auc(p_lut, y_test.astype(float))

# ── B) Gradient boosting ─────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", n_jobs=-1, random_state=0,
    )
    name = "XGBoost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(
        max_iter=400, max_depth=6, learning_rate=0.05, random_state=0,
    )
    name = "HistGradientBoosting (sklearn)"

print(f"\nfitting {name} ...")
model.fit(X_train, y_train)
p_gbm = model.predict_proba(X_test)[:, 1]
auc_gbm = fp._fill_auc(p_gbm, y_test.astype(float))

# ── C) Conditioning-only GBM, to locate the signal ───────────────────────────
cond_cols = [X_train.shape[1] - 3, X_train.shape[1] - 2, X_train.shape[1] - 1]
try:
    from xgboost import XGBClassifier as _X
    m2 = _X(n_estimators=400, max_depth=6, learning_rate=0.05,
            eval_metric="logloss", n_jobs=-1, random_state=0)
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as _H
    m2 = _H(max_iter=400, max_depth=6, learning_rate=0.05, random_state=0)

m2.fit(X_train[:, cond_cols], y_train)
auc_cond = fp._fill_auc(m2.predict_proba(X_test[:, cond_cols])[:, 1],
                        y_test.astype(float))

# ── Report ───────────────────────────────────────────────────────────────────
rows = [
    ("Lookup table (32 cells)",              32,                  auc_lut),
    ("GBM, conditioning only (3 features)",  3,                   auc_cond),
    (f"GBM, full ({X_train.shape[1]} features)", X_train.shape[1], auc_gbm),
    ("LSTM (109 LOB feats, 50-snap window)", 90754,               0.6375),
]

print("\n" + "=" * 70)
print("AUC — identical test rows")
print("=" * 70)
print(f"{'Predictor':<42}{'Params/feats':>14}{'AUC':>10}")
print("-" * 70)
for label, k, a in rows:
    print(f"{label:<42}{k:>14,}{a:>10.4f}")
print("-" * 70)
print(f"{'GBM - lookup':<42}{'':>14}{auc_gbm - auc_lut:>+10.4f}")
print(f"{'GBM - GBM(cond only)':<42}{'':>14}{auc_gbm - auc_cond:>+10.4f}")
print("=" * 70)

gap = auc_gbm - auc_lut
if gap > 0.02:
    print("GBM beats the lookup table. The LOB features DO carry signal that")
    print("the LSTM failed to extract. The paper's claim must narrow to this")
    print("architecture rather than these features. Section 6 needs rewriting.")
elif gap > -0.02:
    print("GBM matches the lookup table. The null is architecture-independent:")
    print("quote geometry is the ceiling for a flexible learner too, not just")
    print("for the LSTM. This STRENGTHENS the paper's central claim.")
else:
    print("GBM underperforms the lookup table, so the 109 features actively")
    print("mislead a flexible learner on this data. Worth reporting as-is.")
print("=" * 70)

Path("results").mkdir(exist_ok=True)
pd.DataFrame(
    [{"predictor": l, "n_params_or_feats": k, "auc": round(a, 4)}
     for l, k, a in rows]
).to_csv("results/gbm_baseline_results.csv", index=False)
print("\nsaved -> results/gbm_baseline_results.csv")
