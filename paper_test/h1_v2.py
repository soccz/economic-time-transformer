"""
H1 검증 v2 — 스펙 정렬
수정:
  1. Step 1 y: 당일 잔차 → 5일 누적 잔차 (스펙 타겟)
  2. Step 2 state: cycle_position_t → cycle_position_{t-1} (정보 집합 정합)
  3. endogeneity 진단: 날짜별 cross-sectional corr 후 시간 평균
  4. robustness: Industry 49 포트폴리오 동일 테스트
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings("ignore")

START, END = "1990-01-01", "2024-12-31"
ROLL_BETA = 60
MA_WINDOW = 200
RV_WINDOW = 30
RV_RANK_WINDOW = 252
HORIZON = 5


# ── 공통 함수 ────────────────────────────────────────────────────────────────

def build_regime(spx):
    ma200 = spx.rolling(MA_WINDOW).mean()
    cp = (spx - ma200) / ma200
    rv30 = np.log(spx / spx.shift(1)).rolling(RV_WINDOW).std() * np.sqrt(252)
    ci = rv30.rolling(RV_RANK_WINDOW).rank(pct=True)
    regime = (2*(cp > 0).astype(int) + (ci > 0.5).astype(int))
    return cp, ci, regime


def rolling_beta_resid(excess, F, mkt, smb, hml, common):
    """rolling β → FF3 잔차 + se(β_MKT)"""
    resid = pd.DataFrame(index=common, columns=excess.columns, dtype=float)
    se_mkt = pd.DataFrame(index=common, columns=excess.columns, dtype=float)
    for col in excess.columns:
        y = excess[col]
        for t in range(ROLL_BETA, len(common)):
            sl = slice(t - ROLL_BETA, t)
            yt = y.iloc[sl].values
            Xt = add_constant(F.iloc[sl].values)
            date = common[t]
            try:
                res = OLS(yt, Xt).fit()
                b = res.params
                resid.loc[date, col] = (y.loc[date]
                                        - b[1]*mkt.loc[date]
                                        - b[2]*smb.loc[date]
                                        - b[3]*hml.loc[date])
                se_mkt.loc[date, col] = res.bse[1]
            except Exception:
                pass
    return resid.astype(float), se_mkt.astype(float)


def build_y5(resid):
    """5일 누적 잔차: y_i,t = Σ_{h=1}^{5} ε̂_{i,t+h}"""
    return resid.shift(-1) + resid.shift(-2) + resid.shift(-3) \
           + resid.shift(-4) + resid.shift(-5)


def fama_macbeth_step1(y5, mom_12_1):
    """날짜별 cross-sectional OLS → λ_1,t"""
    lam1, dates = [], []
    for date in y5.index:
        yv_s = y5.loc[date].dropna()
        xv_s = mom_12_1.loc[date].reindex(yv_s.index).dropna()
        idx = yv_s.index.intersection(xv_s.index)
        if len(idx) < 10:
            continue
        yv = yv_s[idx].values
        xv = stats.rankdata(xv_s[idx].values) / len(idx) - 0.5
        try:
            res = OLS(yv, add_constant(xv)).fit()
            lam1.append(res.params[1])
            dates.append(date)
        except Exception:
            pass
    return pd.Series(lam1, index=pd.DatetimeIndex(dates), name="lambda_1")


def step2_regression(lam1, cp, ci, wml_t, label=""):
    """Step 2: λ_1,t ~ cp_{t-1} + ci_{t-1}, WML 통제 포함"""
    cp_lag = cp.shift(1)
    ci_lag = ci.shift(1)
    reg = pd.DataFrame({
        "lambda_1": lam1,
        "cp_lag": cp_lag.reindex(lam1.index),
        "ci_lag": ci_lag.reindex(lam1.index),
        "WML": wml_t.reindex(lam1.index),
    }).dropna()

    print(f"\n{'='*50}")
    print(f"[{label}] Step 2 샘플: {len(reg)}일")

    # (A) 단변량
    rA = OLS(reg["lambda_1"], add_constant(reg["cp_lag"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 10})
    print(f"  (A) λ_1 ~ cp_{{t-1}}: b={rA.params['cp_lag']:.5f}, "
          f"t={rA.tvalues['cp_lag']:.3f}, p={rA.pvalues['cp_lag']:.4f}")

    # (B) 다변량
    rB = OLS(reg["lambda_1"],
             add_constant(reg[["cp_lag", "ci_lag"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 10})
    for v in ["cp_lag", "ci_lag"]:
        print(f"  (B) {v}: b={rB.params[v]:.5f}, "
              f"t={rB.tvalues[v]:.3f}, p={rB.pvalues[v]:.4f}")

    # (C) WML 통제
    rho_cw, _ = stats.spearmanr(reg["cp_lag"], reg["WML"])
    if abs(rho_cw) <= 0.5:
        rC = OLS(reg["lambda_1"],
                 add_constant(reg[["cp_lag", "WML"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 10})
        for v in ["cp_lag", "WML"]:
            print(f"  (C) {v}: b={rC.params[v]:.5f}, "
                  f"t={rC.tvalues[v]:.3f}, p={rC.pvalues[v]:.4f}")
    else:
        print(f"  (C) 생략: Spearman(cp,WML)={rho_cw:.3f} > 0.5")

    # 체제별 λ_1,t
    regime = (2*(reg["cp_lag"] > 0).astype(int) +
              (reg["ci_lag"] > 0.5).astype(int))
    rl = {0:"Bear/quiet",1:"Bear/volatile",2:"Bull/quiet",3:"Bull/volatile"}
    base = reg.loc[regime == 0, "lambda_1"]
    print("  체제별 λ_1,t:")
    for r, name in rl.items():
        g = reg.loc[regime == r, "lambda_1"]
        if len(g) < 5:
            continue
        if r == 0:
            print(f"    {name}: mean={g.mean():.5f}, n={len(g)}")
        else:
            ts, ps = stats.ttest_ind(g, base, equal_var=False)
            print(f"    {name}: mean={g.mean():.5f}, n={len(g)}, "
                  f"t={ts:.3f}, p={ps:.4f}")

    p_main = rA.pvalues["cp_lag"]
    b_main = rA.params["cp_lag"]
    print(f"  → H1 판정: {'채택' if p_main < 0.05 else '기각'} "
          f"(b={b_main:.5f}, p={p_main:.4f})")
    return rA


def endogeneity_diag_cs(mom_12_1, se_mkt, label=""):
    """날짜별 cross-sectional corr(|mom_i|, se(β_i)) → 시간 평균"""
    rhos = []
    for date in mom_12_1.index:
        m = mom_12_1.loc[date].dropna().abs()
        s = se_mkt.loc[date].reindex(m.index).dropna()
        idx = m.index.intersection(s.index)
        if len(idx) < 5:
            continue
        r, _ = stats.spearmanr(m[idx], s[idx])
        rhos.append(r)
    rho_mean = np.nanmean(rhos)
    rho_t, rho_p = stats.ttest_1samp([r for r in rhos if not np.isnan(r)], 0)
    print(f"[{label}] endogeneity 진단 (단면 corr 시간평균): "
          f"mean ρ={rho_mean:.3f}, t={rho_t:.3f}, p={rho_p:.4f}")
    if abs(rho_mean) >= 0.1 and rho_p < 0.05:
        print("  → 유의: IV 접근 고려")
    else:
        print("  → 낮음: 편향 무시 가능")


# ── 데이터 로드 ──────────────────────────────────────────────────────────────

print("데이터 로드 중...")
factors  = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", START, END)[0] / 100
wml_df   = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", START, END)[0] / 100
port25   = web.DataReader("25_Portfolios_5x5_Daily", "famafrench", START, END)[0] / 100
port49   = web.DataReader("49_Industry_Portfolios_daily", "famafrench", START, END)[0] / 100
spx = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()

mkt = factors["Mkt-RF"]; smb = factors["SMB"]
hml = factors["HML"];    rf  = factors["RF"]
wml_col = [c for c in wml_df.columns if "Mom" in c or "mom" in c][0]
wml_t = wml_df[wml_col]

spx = spx.reindex(factors.index).ffill()
cp, ci, regime = build_regime(spx)


def run_h1(port_ret, label):
    common = port_ret.index.intersection(factors.index).intersection(spx.dropna().index)
    port_ret_ = port_ret.loc[common]
    # -999 (결측 코드) 처리
    port_ret_ = port_ret_.replace(-99.99, np.nan).replace(-999, np.nan) / 100 \
        if port_ret_.abs().max().max() > 10 else port_ret_.loc[common]

    mkt_ = mkt.reindex(common); smb_ = smb.reindex(common)
    hml_ = hml.reindex(common); rf_  = rf.reindex(common)
    wml_ = wml_t.reindex(common).ffill()
    cp_  = cp.reindex(common);  ci_  = ci.reindex(common)

    excess = port_ret_.subtract(rf_, axis=0)
    F = pd.DataFrame({"MKT": mkt_, "SMB": smb_, "HML": hml_})

    print(f"\n{'#'*55}")
    print(f"# {label}: {common[0].date()} ~ {common[-1].date()}, "
          f"N={len(common)}, ports={port_ret_.shape[1]}")

    print("rolling β 추정 중...")
    resid, se_mkt = rolling_beta_resid(excess, F, mkt_, smb_, hml_, common)

    cum = (1 + port_ret_).cumprod()
    mom_12_1 = (cum.shift(21) / cum.shift(252) - 1).reindex(resid.index)

    # 5일 누적 잔차 타겟
    y5 = build_y5(resid)
    # 마지막 5일은 미래 데이터 없으므로 제거
    y5 = y5.iloc[:-HORIZON]
    mom_12_1 = mom_12_1.reindex(y5.index)

    print("Fama-MacBeth Step 1 (5일 누적 잔차)...")
    lam1 = fama_macbeth_step1(y5, mom_12_1)
    print(f"λ_1,t: n={len(lam1)}, mean={lam1.mean():.5f}, std={lam1.std():.5f}")

    endogeneity_diag_cs(mom_12_1, se_mkt.reindex(y5.index), label)
    step2_regression(lam1, cp_, ci_, wml_, label)


# ── 실행 ─────────────────────────────────────────────────────────────────────

run_h1(port25, "25 Size-B/M")
run_h1(port49, "49 Industry")
