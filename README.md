# How Much Does Limit Order Book Microstructure Add Beyond Quote Geometry?

Code and results for the working paper of the same name (v4, August 2026).

**Author:** Jude Kriel Ramcharitar · Independent Researcher
**Paper:** [`paper_v4/main.pdf`](paper_v4/main.pdf)

---

## What this asks

A trader posting a limit order chooses three things for free: which side, how
far from mid to quote, and how long to leave the order resting. This paper asks
what limit order book microstructure adds *beyond* those choices, benchmarked
against a 16-cell lookup table holding only the mean historical crossing rate
per (offset, horizon, side) cell.

## What it finds

Gradient-boosted trees over 63 microstructure features beat that lookup
reference by **+0.048, +0.049 and +0.027 AUC** on BTC/USD, ETH/USD and SOL/USD
under a blocked split, mean +0.0415. Under a forecasting-ordered temporal split
the gain holds on BTC and ETH (+0.048, +0.060) and changes sign on SOL.

Model-class ordering is representation-dependent: trees lead on BTC and ETH
under both feature representations, while on SOL they lead by 0.0013 with all
109 features and trail by 0.0475 on the 63-feature subset. The paper draws no
general ordering.

The target is a **replay-defined crossing**, not a realised fill: it records
whether the opposite best quote reached or crossed the simulated limit price,
using book snapshots only and no trade data.

## Superseded work

Earlier versions of this project (v1–v3) were titled *Latency-Compensating
Market Making: An AI Framework for Retail Liquidity Provision* and framed
around 200 ms–2 s execution latency. **Those results should not be cited.**
Section 7 of the current paper documents five pipeline defects that invalidated
them, including a synthetic 100 ms clock applied to real data, which made the
labels described as "2000 ms" horizons a 511-second median window. The
frequently quoted "7× drawdown reduction" came from synthetic data with a
hand-specified fill rule and does not survive.

---

## Repository layout

```
src/
  collect_lob_data.py    Kraken WebSocket collector (systemd, Restart=always)
  lob_features.py        feature engineering, 109 features per snapshot
  fill_predictor.py      LSTM; binary head added in v4
build_labels_v2.py       corrected label builder (real timestamps, bps offsets)
verify_v3.py             pre-experiment checks; exits non-zero on failure
rerun_v3.py              main experiment: lookup vs GBT vs LSTM
regime_test.py           temporal vs blocked split comparison
rotation_test.py         10 block-phase rotations (phase robustness)
sol_diagnostic.py        feature-group decomposition
gbm_*.py                 earlier GBT comparisons and leakage audit
results/                 all result CSVs
paper_v4/                paper source, PDF, and locked number set
```

## Reproducing

```bash
pip install -r requirements.txt

# 1. features and labels
python src/lob_features.py --mode real --data-dir data/kraken/raw --symbol BTCUSD
python build_labels_v2.py --all

# 2. verification — must pass before anything else
python verify_v3.py

# 3. experiments
python rerun_v3.py --epochs 30 --seeds 3
python regime_test.py --seeds 3
python rotation_test.py --seeds 1
```

`verify_v3.py` runs a positive control (a binary LSTM must reach AUC ≥ 0.90 on
synthetic data with planted signal), asserts split proportions and embargo
gaps, and reproduces the defective survival head to confirm it is inert. It
exits non-zero on failure.

## Data

269,330 timestamped 10-level LOB snapshots across BTC/USD, ETH/USD and SOL/USD,
collected from Kraken over a 70.5-day calendar window (15 June – 24 August
2026), including a documented 14-day collector outage.

The raw dataset is not currently in this repository. Contact the author for
access pending deposit in a persistent archive.

## Citation

```
Ramcharitar, J.K. (2026). How Much Does Limit Order Book Microstructure Add
Beyond Quote Geometry? Evidence from 70 days of cryptocurrency data at
retail-accessible sampling frequency. Working Paper v4.
```

## AI assistance

Claude (Anthropic) was used for implementation, diagnosis and manuscript
drafting. Several defects catalogued in Section 7 originated in AI-generated
code, and several were identified through AI-assisted diagnosis. The author
directed the analysis, verified the results, and is responsible for all
content.
