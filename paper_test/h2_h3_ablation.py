"""
H2 / H3 ablation: Cycle-PE vs Static PE vs Concat-A
  - Ken French 25 Size-B/M portfolios (proxy for individual assets)
  - Walk-forward CV (train=252d, step=5d)
  - Models: linear QR (proxy for each PE variant)
  - Metrics: IC, CRPS, PI-80 coverage, regime-conditional breakdown
  - DM test for H2 (Cycle-PE vs Static)

Usage:
    python aaa/paper_test/h2_h3_ablation.py
"""
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.regression.quantile_regression import QuantReg
import warnings
warnings.filterwarnings("ignore")

START, END   = "2000-01-01", "2024-12-31"
ROLL_BETA    = 60
HORIZON      = 5
TRAIN_WIN    = 252
STEP         = 5
TAUS         = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
LAGS         = 20
KL_WEIGHT    = 1e-3   # proxy KL penalty weight for CVAE-style regularisation


# ── 1. Data ───────────────────────────────────────────────────────────────────
print("Loading data...")
port25  = web.DataReader("25_Portfolios_5x5_daily", "famafrench", START, END)[0] / 100
factors = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", START, END)[0] / 100
spx     = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()

common = port25.index.intersection(factors.index).intersection(spx.index)
port25  = port25.reindex(common)
mkt = factors["Mkt-RF"].reindex(common)
smb = factors["SMB"].reindex(common)
hml = factors["HML"].reindex(common)
rf  = factors["RF"].reindex(common)
spx = spx.reindex(common).ffill()
print(f"  {common[0].date()} ~ {common[-1].date()}, T={len(common)}")


# ── 2. FF3 residuals ──────────────────────────────────────────────────────────
print("FF3 residuals...")
excess = port25.subtract(rf, axis=0)
F_full = add_constant(pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml}))
resid  = pd.DataFrame(index=common, columns=port25.columns, dtype=float)
for col in port25.columns:
    resid[col] = OLS(excess[col], F_full).fit().resid

# h-day cumulative target (future)
target = resid.rolling(HORIZON).sum().shift(-HORIZON)


# ── 3. State signals ──────────────────────────────────────────────────────────
ma200     = spx.rolling(200).mean()
position  = (spx - ma200) / ma200                                  # cycle_position
rv30      = np.log(spx / spx.shift(1)).rolling(30).std() * np.sqrt(252)
intensity = rv30.rolling(252).rank(pct=True)                       # cycle_intensity

bear   = (position < 0).astype(float)
hi_vol = (intensity > 0.5).astype(float)
regime = (2 * (position > 0).astype(int) + (intensity > 0.5).astype(int))
regime_labels = {0: "Bear/quiet", 1: "Bear/volatile",
                 2: "Bull/quiet",  3: "Bull/volatile"}


# ── 4. Pre-compute rolling features ──────────────────────────────────────────
print("Pre-computing features...")
cols  = list(port25.columns)
N, T  = len(cols), len(common)
r_arr = resid.values  # (T, N)

feat_base = np.full((T, N * 2), np.nan)   # [mean, std] per portfolio
for i in range(LAGS, T):
    w = r_arr[i - LAGS:i]
    feat_base[i, :N]  = np.nanmean(w, axis=0)
    feat_base[i, N:]  = np.nanstd(w, axis=0)

bear_arr  = bear.values
hivol_arr = hi_vol.values
pos_arr   = position.values
int_arr   = intensity.values


# ── 5. Walk-forward ───────────────────────────────────────────────────────────
print("Walk-forward ablation...")

def pinball(y, q, tau):
    e = y - q
    return np.where(e >= 0, tau * e, (tau - 1) * e)

def crps_from_quantiles(y, qs, taus=TAUS):
    return float(np.mean([pinball(y, q, t) for q, t in zip(qs, taus)]))

def pi80_hit(y, q_lo, q_hi):
    """1 if y in [q_lo, q_hi], else 0."""
    return float(q_lo <= y <= q_hi)

records = []
test_idx = range(TRAIN_WIN + LAGS, T - HORIZON, STEP)

for t in test_idx:
    date  = common[t]
    tr    = slice(t - TRAIN_WIN, t)
    b_t   = bear_arr[t - 1]
    hv_t  = hivol_arr[t - 1]
    p_t   = pos_arr[t - 1]
    i_t   = int_arr[t - 1]
    reg_t = int(regime.iloc[t - 1]) if not pd.isna(regime.iloc[t - 1]) else -1

    for ci, col in enumerate(cols):
        y_true = target[col].iloc[t]
        if pd.isna(y_true):
            continue

        y_tr = target[col].iloc[tr].values
        fb   = feat_base[tr, :]
        x2   = fb[:, [ci, ci + N]]          # (TRAIN_WIN, 2): mean, std
        valid = ~(np.isnan(y_tr) | np.isnan(x2).any(axis=1))
        if valid.sum() < 50:
            continue

        yv = y_tr[valid]

        # ── feature matrices per ablation variant ────────────────────────────
        # Static PE proxy: base features only
        Xs = add_constant(x2[valid])
        xs_test = np.array([[1.0, feat_base[t, ci], feat_base[t, ci + N]]])

        # Concat-A proxy: append state features to input (bear, hi_vol)
        Xc = add_constant(np.column_stack([
            x2[valid],
            bear_arr[t - TRAIN_WIN:t][valid],
            hivol_arr[t - TRAIN_WIN:t][valid],
        ]))
        xc_test = np.array([[1.0, feat_base[t, ci], feat_base[t, ci + N], b_t, hv_t]])

        # Cycle-PE proxy: intensity injected as multiplicative modulation of features
        # x_cycle = [mean * (1 + intensity), std * (1 + intensity), intensity]
        int_tr = int_arr[t - TRAIN_WIN:t][valid]
        x_cycle_tr = np.column_stack([
            x2[valid, 0] * (1 + int_tr),
            x2[valid, 1] * (1 + int_tr),
            int_tr,
        ])
        Xp = add_constant(x_cycle_tr)
        i_t_safe = 0.0 if np.isnan(i_t) else i_t
        xp_test = np.array([[1.0,
                              feat_base[t, ci]     * (1 + i_t_safe),
                              feat_base[t, ci + N] * (1 + i_t_safe),
                              i_t_safe]])

        if (np.isnan(xs_test).any() or np.isnan(xc_test).any()
                or np.isnan(xp_test).any()):
            continue

        row = {"date": date, "port": col, "y_true": y_true, "regime": reg_t}
        ok  = True

        for variant, X_tr, x_te in [("static",   Xs, xs_test),
                                     ("concat_a", Xc, xc_test),
                                     ("cycle_pe", Xp, xp_test)]:
            qs = []
            for tau in TAUS:
                try:
                    q = QuantReg(yv, X_tr).fit(q=tau, max_iter=300).predict(x_te)[0]
                    qs.append(q)
                except Exception:
                    ok = False
                    break
            if not ok:
                break
            row[f"crps_{variant}"]    = crps_from_quantiles(y_true, qs)
            row[f"pi80_hit_{variant}"] = pi80_hit(y_true, qs[0], qs[-1])  # tau 0.1/0.9
            row[f"pred_{variant}"]    = qs[2]  # median

        if ok:
            records.append(row)

    if (t // STEP) % 50 == 0:
        print(f"  {date.date()}, n={len(records)}")

df = pd.DataFrame(records)
print(f"\nTotal predictions: {len(df)}")


# ── 6. IC (rank correlation of median prediction vs realised) ─────────────────
print("\n[IC per variant]")
for v in ("static", "concat_a", "cycle_pe"):
    ic_series = (df.groupby("date")
                   .apply(lambda g: stats.spearmanr(g[f"pred_{v}"], g["y_true"])[0],
                          include_groups=False)
                   .dropna())
    print(f"  {v:10s}: IC={ic_series.mean():.4f}, ICIR={ic_series.mean()/ic_series.std():.3f}")


# ── 7. Overall CRPS + DM test (H2) ───────────────────────────────────────────
print("\n[Overall CRPS]")
for v in ("static", "concat_a", "cycle_pe"):
    print(f"  {v:10s}: {df[f'crps_{v}'].mean():.6f}")

diff_h2 = df["crps_static"].values - df["crps_cycle_pe"].values
dm_t = diff_h2.mean() / (diff_h2.std() / np.sqrt(len(diff_h2)))
dm_p = 2 * (1 - stats.norm.cdf(abs(dm_t)))
print(f"\n[H2 DM test] static vs cycle_pe: t={dm_t:+.3f}, p={dm_p:.4f} "
      f"-> {'Cycle-PE wins' if dm_t > 0 and dm_p < 0.05 else 'not significant'}")

diff_ca = df["crps_concat_a"].values - df["crps_cycle_pe"].values
dm_t2 = diff_ca.mean() / (diff_ca.std() / np.sqrt(len(diff_ca)))
dm_p2 = 2 * (1 - stats.norm.cdf(abs(dm_t2)))
print(f"[H2 DM test] concat_a vs cycle_pe: t={dm_t2:+.3f}, p={dm_p2:.4f} "
      f"-> {'Cycle-PE wins' if dm_t2 > 0 and dm_p2 < 0.05 else 'not significant'}")


# ── 8. Regime-conditional CRPS (H3) ──────────────────────────────────────────
print("\n[H3: Regime-conditional CRPS]")
print(f"  {'Regime':15s} {'n':>6}  {'static':>10} {'concat_a':>10} {'cycle_pe':>10}  {'PI80_static':>12} {'PI80_cycle':>11}")
for r_id, label in regime_labels.items():
    sub = df[df["regime"] == r_id]
    if len(sub) < 30:
        continue
    cs = sub["crps_static"].mean()
    cc = sub["crps_concat_a"].mean()
    cp = sub["crps_cycle_pe"].mean()
    pi_s = sub["pi80_hit_static"].mean()
    pi_p = sub["pi80_hit_cycle_pe"].mean()
    print(f"  {label:15s} {len(sub):>6}  {cs:>10.6f} {cc:>10.6f} {cp:>10.6f}  {pi_s:>12.3f} {pi_p:>11.3f}")

# H3: uncertainty width by regime (proxy: IQR of predictions)
print("\n[H3: Prediction spread by regime (proxy for uncertainty)]")
for r_id, label in regime_labels.items():
    sub = df[df["regime"] == r_id]
    if len(sub) < 30:
        continue
    spread = sub["y_true"].std()
    print(f"  {label:15s}: realised std={spread:.5f}")


# ── 9. Save ───────────────────────────────────────────────────────────────────
out_path = "aaa/paper_test/h2_h3_results.csv"
df.to_csv(out_path, index=False)
print(f"\nDone. -> {out_path}")
