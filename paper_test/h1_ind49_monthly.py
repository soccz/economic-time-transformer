"""
H1 test with Ken French 49 Industry portfolios (monthly).
- Step 1: Fama-MacBeth on FF3 residuals to estimate lambda_1,t
- Step 2: lambda_1,t ~ cycle_position (+ controls)
- Monthly setup to reduce daily micro-noise.
"""

import warnings

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings("ignore")

START = "1990-01-01"
END = "2024-12-31"
SPX_START = "1985-01-01"

ROLL_BETA = 60
MA_MONTH_WINDOW = 10
INTENSITY_RANK_WINDOW = 24
HAC_LAGS_MONTHLY = 3


def to_month_end_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.PeriodIndex):
        out.index = out.index.to_timestamp("M")
    return out


def clean_percent_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([-99.99, -999.0, -999], np.nan)
    return out / 100.0


def last_value_percentile(x: np.ndarray) -> float:
    valid = x[~np.isnan(x)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid <= valid[-1]))


def load_data():
    ind49 = web.DataReader("49_Industry_Portfolios", "famafrench", START, END)[0]
    ff3 = web.DataReader("F-F_Research_Data_Factors", "famafrench", START, END)[0]
    wml_df = web.DataReader("F-F_Momentum_Factor", "famafrench", START, END)[0]

    ind49 = clean_percent_dataframe(to_month_end_index(ind49))
    ff3 = clean_percent_dataframe(to_month_end_index(ff3))
    wml_df = clean_percent_dataframe(to_month_end_index(wml_df))

    spx = yf.download(
        "^GSPC",
        start=SPX_START,
        end=END,
        auto_adjust=True,
        progress=False,
    )["Close"].squeeze()
    return ind49, ff3, wml_df, spx


def build_cycle_features(spx_daily: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    spx_m = spx_daily.resample("ME").last()
    ma10 = spx_m.rolling(MA_MONTH_WINDOW).mean()
    cycle_position = (spx_m - ma10) / ma10

    log_ret = np.log(spx_daily / spx_daily.shift(1))
    rv30 = log_ret.rolling(30).std() * np.sqrt(252)
    rv30_m = rv30.resample("ME").last()
    cycle_intensity = rv30_m.rolling(INTENSITY_RANK_WINDOW).apply(
        last_value_percentile, raw=True
    )

    regime = 2 * (cycle_position > 0).astype(int) + (cycle_intensity > 0.5).astype(int)
    return cycle_position, cycle_intensity, regime


print("Loading monthly datasets: Industry 49 + FF3 + WML + S&P500...")
ind49_ret, ff3, wml_df, spx_daily = load_data()

mkt = ff3["Mkt-RF"]
smb = ff3["SMB"]
hml = ff3["HML"]
rf = ff3["RF"]
wml_col = [c for c in wml_df.columns if "Mom" in c or "mom" in c][0]
wml_t = wml_df[wml_col]

cycle_position, cycle_intensity, regime = build_cycle_features(spx_daily)

common = (
    ind49_ret.index.intersection(ff3.index)
    .intersection(wml_t.index)
    .intersection(cycle_position.index)
    .intersection(cycle_intensity.index)
)

ind49_ret = ind49_ret.reindex(common)
mkt = mkt.reindex(common)
smb = smb.reindex(common)
hml = hml.reindex(common)
rf = rf.reindex(common)
wml_t = wml_t.reindex(common)
cycle_position = cycle_position.reindex(common)
cycle_intensity = cycle_intensity.reindex(common)
regime = regime.reindex(common)

print(f"Sample: {common[0].date()} ~ {common[-1].date()}, N={len(common)} months")


print("Estimating rolling monthly FF3 betas and residual targets...")
excess = ind49_ret.subtract(rf, axis=0)
F = pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml})

resid = pd.DataFrame(index=common, columns=ind49_ret.columns, dtype=float)
se_mkt_all = pd.DataFrame(index=common, columns=ind49_ret.columns, dtype=float)

for col in ind49_ret.columns:
    y = excess[col]
    for t in range(ROLL_BETA, len(common)):
        date = common[t]
        tr_idx = common[t - ROLL_BETA : t]
        y_tr = y.reindex(tr_idx)
        x_tr = add_constant(F.reindex(tr_idx))
        valid = y_tr.notna() & x_tr.notna().all(axis=1)
        if valid.sum() < 36:
            continue
        try:
            fit = OLS(y_tr[valid], x_tr.loc[valid]).fit()
            resid.loc[date, col] = (
                y.loc[date]
                - fit.params["MKT"] * mkt.loc[date]
                - fit.params["SMB"] * smb.loc[date]
                - fit.params["HML"] * hml.loc[date]
            )
            se_mkt_all.loc[date, col] = fit.bse.get("MKT", np.nan)
        except Exception:
            continue

resid = resid.dropna(how="all")
print(f"Residual matrix shape: {resid.shape}")


print("Building monthly momentum feature (12-1 months)...")
cum = (1 + ind49_ret).cumprod()
mom_12_1 = cum.shift(1) / cum.shift(12) - 1
mom_12_1 = mom_12_1.reindex(resid.index)


print("Running Fama-MacBeth Step 1...")
lam0_list, lam1_list, fm_dates = [], [], []
for date in resid.index:
    y_cs = resid.loc[date].dropna()
    x_cs = mom_12_1.loc[date].reindex(y_cs.index).dropna()
    idx = y_cs.index.intersection(x_cs.index)
    if len(idx) < 20:
        continue
    yv = y_cs[idx].values
    xv = stats.rankdata(x_cs[idx].values) / len(idx) - 0.5
    try:
        fit = OLS(yv, add_constant(xv)).fit()
        lam0_list.append(fit.params[0])
        lam1_list.append(fit.params[1])
        fm_dates.append(date)
    except Exception:
        continue

lam = pd.DataFrame(
    {"lambda_0": lam0_list, "lambda_1": lam1_list},
    index=pd.DatetimeIndex(fm_dates),
)
t_lam, p_lam = stats.ttest_1samp(lam["lambda_1"], 0.0)
print(
    f"lambda_1,t: n={len(lam)}, mean={lam['lambda_1'].mean():.4f}, "
    f"std={lam['lambda_1'].std():.4f}, t={t_lam:.3f}, p={p_lam:.4f}"
)


cp = cycle_position.reindex(lam.index).dropna()
wml_d = wml_t.reindex(cp.index).dropna()
idx_d = cp.index.intersection(wml_d.index)
rho_cp_wml, p_cp_wml = stats.spearmanr(cp[idx_d], wml_d[idx_d])
print(f"\n[Diag] Spearman(cycle_position, WML): rho={rho_cp_wml:.3f}, p={p_cp_wml:.4f}")
multicollinear = abs(rho_cp_wml) > 0.5
if multicollinear:
    print("  -> |rho|>0.5: potential multicollinearity. Skipping Step 2-C.")
else:
    print("  -> |rho|<=0.5: Step 2-C allowed.")

mom_abs_mean = mom_12_1.abs().mean(axis=1).reindex(lam.index)
se_mkt_mean = se_mkt_all.mean(axis=1).reindex(lam.index)
idx_e = mom_abs_mean.dropna().index.intersection(se_mkt_mean.dropna().index)
rho_mom_se, p_mom_se = stats.spearmanr(mom_abs_mean[idx_e], se_mkt_mean[idx_e])
print(f"[Diag] Spearman(|mom|, se(beta_MKT)): rho={rho_mom_se:.3f}, p={p_mom_se:.4f}")


reg = pd.DataFrame(
    {
        "lambda_1": lam["lambda_1"],
        "cycle_position": cycle_position.reindex(lam.index),
        "cycle_intensity": cycle_intensity.reindex(lam.index),
        "WML": wml_t.reindex(lam.index),
        "regime": regime.reindex(lam.index),
    }
).dropna()
print(f"\nStep 2 sample: {len(reg)} months")

r_a = OLS(reg["lambda_1"], add_constant(reg["cycle_position"])).fit(
    cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS_MONTHLY}
)
print(
    f"[Step 2-A] lambda_1 ~ cycle_position: "
    f"b={r_a.params['cycle_position']:.4f}, "
    f"t={r_a.tvalues['cycle_position']:.3f}, "
    f"p={r_a.pvalues['cycle_position']:.4f}"
)

r_b = OLS(
    reg["lambda_1"],
    add_constant(reg[["cycle_position", "cycle_intensity"]]),
).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS_MONTHLY})
print("[Step 2-B] lambda_1 ~ cycle_position + cycle_intensity:")
for var in ["cycle_position", "cycle_intensity"]:
    print(
        f"  {var}: b={r_b.params[var]:.4f}, "
        f"t={r_b.tvalues[var]:.3f}, p={r_b.pvalues[var]:.4f}"
    )

if not multicollinear:
    r_c = OLS(
        reg["lambda_1"],
        add_constant(reg[["cycle_position", "WML"]]),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS_MONTHLY})
    print("[Step 2-C] lambda_1 ~ cycle_position + WML:")
    for var in ["cycle_position", "WML"]:
        print(
            f"  {var}: b={r_c.params[var]:.4f}, "
            f"t={r_c.tvalues[var]:.3f}, p={r_c.pvalues[var]:.4f}"
        )

rho_lw, p_lw = stats.spearmanr(reg["lambda_1"], reg["WML"])
print(f"\n[Diag] Spearman(lambda_1,t, WML_t): rho={rho_lw:.3f}, p={p_lw:.4f}")

regime_labels = {
    0: "Bear/quiet",
    1: "Bear/volatile",
    2: "Bull/quiet",
    3: "Bull/volatile",
}
print("\n[Regime means of lambda_1,t]")
base = reg.loc[reg["regime"] == 0, "lambda_1"]
for code, label in regime_labels.items():
    grp = reg.loc[reg["regime"] == code, "lambda_1"]
    if len(grp) < 5:
        continue
    if code == 0:
        print(f"  {label}: mean={grp.mean():.4f}, std={grp.std():.4f}, n={len(grp)}")
    else:
        t_stat, p_val = stats.ttest_ind(grp, base, equal_var=False)
        print(
            f"  {label}: mean={grp.mean():.4f}, std={grp.std():.4f}, n={len(grp)}, "
            f"t vs Bear/quiet={t_stat:.3f}, p={p_val:.4f}"
        )


print("\n" + "=" * 60)
print("H1 decision (based on Step 2-A)")
if r_a.pvalues["cycle_position"] < 0.05:
    print(
        f"b={r_a.params['cycle_position']:.4f}, "
        f"p={r_a.pvalues['cycle_position']:.4f} -> H1 supported"
    )
    if r_a.params["cycle_position"] > 0:
        print("Sign: Bull -> stronger momentum premium")
    else:
        print("Sign: opposite to Bull-amplification prior")
else:
    print(
        f"b={r_a.params['cycle_position']:.4f}, "
        f"p={r_a.pvalues['cycle_position']:.4f} -> H1 not supported"
    )
print("=" * 60)
