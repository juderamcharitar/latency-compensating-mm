"""Single source of truth for every number in the paper."""

DATA = {
    "BTCUSD": dict(snaps=111002, labels=1760728, fill=0.5800, tick=0.1,
                   gap_med=11.5, gap_mean=54.9, obs300=98.4, obs900=99.9),
    "ETHUSD": dict(snaps=34380,  labels=515576,  fill=0.5340, tick=0.01,
                   gap_med=48.0, gap_mean=177.2, obs300=88.7, obs900=98.8),
    "SOLUSD": dict(snaps=123948, labels=1971128, fill=0.5337, tick=0.01,
                   gap_med=26.2, gap_mean=49.1, obs300=98.9, obs900=99.9),
}
SPAN_DAYS = 70.5

# rerun_v3.py, temporal split, 3 seeds
MAIN = {
    "BTCUSD": dict(lookup=0.6828, gbt_micro=(0.7307,0.0006), lstm_micro=(0.6857,0.0021),
                   gbt_all=(0.7262,0.0012), lstm_all=(0.6903,0.0043),
                   lstm_micro_moved=0.11887, lstm_all_moved=0.02635,
                   tr=140009, va=29827, te=29895),
    "ETHUSD": dict(lookup=0.6597, gbt_micro=(0.7193,0.0013), lstm_micro=(0.6983,0.0046),
                   gbt_all=(0.6908,0.0010), lstm_all=(0.6669,0.0036),
                   lstm_micro_moved=0.15928, lstm_all_moved=0.00779,
                   tr=140385, va=30102, te=29391),
    "SOLUSD": dict(lookup=0.6673, gbt_micro=(0.6568,0.0035), lstm_micro=(0.7043,0.0046),
                   gbt_all=(0.7056,0.0010), lstm_all=(0.7043,0.0074),
                   lstm_micro_moved=0.01393, lstm_all_moved=0.02126,
                   tr=140101, va=29632, te=29994),
}

# regime_test.py
REGIME = {
    # blocked-split micro arm: auc, sd  (from results/regime_test.csv)
    "_blocked_micro_sd": dict(BTCUSD=0.0004, ETHUSD=0.0006, SOLUSD=0.0003),
    "_blocked_micro_auc": dict(BTCUSD=0.7519, ETHUSD=0.7141, SOLUSD=0.6955),
    "BTCUSD": dict(ov_t=0.129, ov_b=0.815, lut_t=0.6828, lut_b=0.7040,
                   micro_t=0.0480, micro_b=0.0479,
                   price_t=0.0384, price_b=0.0383,
                   all_t=0.0434,  all_b=0.0456),
    "ETHUSD": dict(ov_t=0.073, ov_b=0.855, lut_t=0.6597, lut_b=0.6650,
                   micro_t=0.0596, micro_b=0.0491,
                   price_t=-0.0001, price_b=0.0370,
                   all_t=0.0312,  all_b=0.0513),
    "SOLUSD": dict(ov_t=0.251, ov_b=0.745, lut_t=0.6673, lut_b=0.6681,
                   micro_t=-0.0105, micro_b=0.0274,
                   price_t=0.0312, price_b=0.0115,
                   all_t=0.0383,  all_b=0.0295),
}

# fill rate by horizon and offset_bps, from build_labels_v2
MARGINALS = {
    "BTCUSD": {300:[0.697,0.622,0.433,0.237], 900:[0.815,0.766,0.626,0.440]},
    "ETHUSD": {300:[0.603,0.540,0.398,0.245], 900:[0.740,0.696,0.582,0.432]},
    "SOLUSD": {300:[0.577,0.517,0.387,0.242], 900:[0.753,0.712,0.612,0.466]},
}

DEFECTS = dict(
    cindex_pre=0.4749, cindex_post=0.6266,
    horizon_200=33.5, horizon_500=106.9, horizon_1000=237.2, horizon_2000=511.1,
    inert_loss=16.1181, unit_factor=1000,
    fingerprint_expected=111002*32, fingerprint_actual=3551392,
    c1_auc=0.9861,
)


# rotation_test.py — 10 block phases, 63-feature microstructure, 1 seed
ROTATION = {
    "BTCUSD": dict(mean=0.0473, sd=0.0052, lo=0.0378, hi=0.0529, npos=10,
                   ov_lo=0.603, ov_hi=0.961, corr=-0.17,
                   gap_lowov=0.0479, gap_highov=0.0467),
    "ETHUSD": dict(mean=0.0382, sd=0.0144, lo=0.0130, hi=0.0550, npos=10,
                   ov_lo=0.712, ov_hi=0.977, corr=+0.33,
                   gap_lowov=0.0387, gap_highov=0.0377),
    "SOLUSD": dict(mean=0.0346, sd=0.0103, lo=0.0133, hi=0.0463, npos=10,
                   ov_lo=0.534, ov_hi=0.787, corr=-0.14,
                   gap_lowov=0.0373, gap_highov=0.0320),
}
