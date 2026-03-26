"""
H1 탐색: 어떤 state 변수가 λ_1,t를 가장 잘 설명하는가
- 타겟 변형: FF3 잔차 모멘텀 vs raw return 모멘텀
- state 변수 변형: position, intensity, WML, VIX proxy, 상호작용
- 결과: 유의한 조합만 요약 출력
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


# ── 1. 데이터 로드 ──────────────────────────────────────────────────────────

def load_data():
    port = web.DataReader("25_Portfolios_5x5_Daily", "famafrench", START, END)[0] / 100
    factors = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", START, END)[0] / 100
    wml = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", START, END)[0] / 100
    spx = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
    return port, factors, wml, spx

print("데이터 로드 중...")
port_ret, factors, wml_df, spx = load_data()

mkt = factors["Mkt-RF"]
smb = factors["SMB"]
hml = factors["HML"]
rf  = factors["RF"]
wml_col = [c for c in wml_df.columns if "Mom" in c or "mom" in c][0]
wml_t = wml_df[wml_col]

common = port_ret.index.intersection(factors.index).intersection(spx.index)
port_ret = port_ret.loc[common]
mkt = mkt.reindex(common); smb = smb.reindex(common)
hml = hml.reindex(common); rf  = rf.reindex(common)
wml_t = wml_t.reindex(common).ffill()
spx = spx.reindex(common).ffill()


# ── 2. state 신호 ───────────────────────────────────────────────────────────

ma200 = spx.rolling(MA_WINDOW).mean()
cycle_position = (spx - ma200) / ma200

rv30 = np.log(spx / spx.shift(1)).rolling(RV_WINDOW).std() * np.sqrt(252)
cycle_intensity = rv30.rolling(RV_RANK_WINDOW).rank(pct=True)

# VIX proxy: SPX 20d realized vol rank (VIX 대체)
vix_proxy = np.log(spx / spx.shift(1)).rolling(20).std().rolling(RV_RANK_WINDOW).rank(pct=True)

# 모멘텀 크래시 선행 신호: 직전 12개월 SPX 수익률
spx_mom12 = spx / spx.shift(252) - 1

bear = (cycle_position < 0).astype(float)
high_vol = (cycle_intensity > 0.7).astype(float)  # 상위 30% 변동성


# ── 3. rolling β → FF3 잔차 + raw return ────────────────────────────────────

excess = port_ret.subtract(rf, axis=0)
F = pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml})

resid = pd.DataFrame(index=common, columns=port_ret.columns, dtype=float)

print("rolling β 추정 중...")
for col in port_ret.columns:
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
        except Exception:
            pass

resid = resid.astype(float).dropna(how="all")


# ── 4. 모멘텀 피처 ──────────────────────────────────────────────────────────

cum = (1 + port_ret).cumprod()
mom_12_1 = cum.shift(21) / cum.shift(252) - 1

# raw return 기반 모멘텀도 계산
cum_raw = (1 + port_ret).cumprod()
mom_raw = cum_raw.shift(21) / cum_raw.shift(252) - 1


# ── 5. Fama-MacBeth Step 1 (두 타겟) ────────────────────────────────────────

def run_fmb(target_df, mom_df, label):
    lam1_list, dates_fm = [], []
    for date in target_df.index:
        y_cs = target_df.loc[date].dropna()
        x_cs = mom_df.loc[date].reindex(y_cs.index).dropna()
        idx = y_cs.index.intersection(x_cs.index)
        if len(idx) < 10:
            continue
        yv = y_cs[idx].values
        xv = stats.rankdata(x_cs[idx].values) / len(idx) - 0.5
        try:
            res = OLS(yv, add_constant(xv)).fit()
            lam1_list.append(res.params[1])
            dates_fm.append(date)
        except Exception:
            pass
    lam = pd.Series(lam1_list, index=pd.DatetimeIndex(dates_fm), name=f"lam1_{label}")
    print(f"  [{label}] n={len(lam)}, mean={lam.mean():.5f}, std={lam.std():.4f}")
    return lam

print("\nFama-MacBeth Step 1...")
lam_resid = run_fmb(resid, mom_12_1.reindex(resid.index), "FF3잔차")
lam_raw   = run_fmb(port_ret.reindex(resid.index), mom_raw.reindex(resid.index), "raw수익률")


# ── 6. Step 2 탐색: 모든 조합 ───────────────────────────────────────────────

def step2_search(lam_series, label):
    base = pd.DataFrame({
        "lam1":            lam_series,
        "position":        cycle_position.reindex(lam_series.index),
        "intensity":       cycle_intensity.reindex(lam_series.index),
        "vix_proxy":       vix_proxy.reindex(lam_series.index),
        "spx_mom12":       spx_mom12.reindex(lam_series.index),
        "WML":             wml_t.reindex(lam_series.index),
        "bear":            bear.reindex(lam_series.index),
        "high_vol":        high_vol.reindex(lam_series.index),
    }).dropna()

    base["bear_x_intensity"]  = base["bear"] * base["intensity"]
    base["bear_x_vix"]        = base["bear"] * base["vix_proxy"]
    base["intensity_sq"]      = base["intensity"] ** 2
    base["position_x_intensity"] = base["position"] * base["intensity"]
    base["WML_lag1"]          = base["WML"].shift(1)
    base = base.dropna()

    candidates = [
        ["intensity"],
        ["position"],
        ["vix_proxy"],
        ["WML"],
        ["WML_lag1"],
        ["spx_mom12"],
        ["intensity", "position"],
        ["intensity", "bear_x_intensity"],
        ["intensity", "position", "bear_x_intensity"],
        ["vix_proxy", "bear_x_vix"],
        ["intensity", "position", "position_x_intensity"],
        ["intensity_sq", "intensity"],
        ["WML", "intensity"],
        ["WML", "bear_x_intensity"],
        ["WML_lag1", "intensity"],
        ["WML_lag1", "bear_x_intensity"],
        ["spx_mom12", "intensity"],
        ["spx_mom12", "bear_x_intensity"],
    ]

    results = []
    for cols in candidates:
        try:
            r = OLS(base["lam1"], add_constant(base[cols])).fit(
                cov_type="HAC", cov_kwds={"maxlags": 10})
            # 모든 변수 중 최소 하나가 p<0.05인 조합만 기록
            min_p = min(r.pvalues[cols])
            results.append({
                "vars": "+".join(cols),
                "R2": r.rsquared,
                "min_p": min_p,
                "params": {v: (r.params[v], r.tvalues[v], r.pvalues[v]) for v in cols}
            })
        except Exception:
            pass

    results.sort(key=lambda x: x["min_p"])

    print(f"\n{'='*60}")
    print(f"[{label}] 유의한 조합 (min_p < 0.05, R² 기준 정렬)")
    print(f"{'='*60}")
    sig = [r for r in results if r["min_p"] < 0.05]
    if not sig:
        print("  → 유의한 조합 없음")
    for r in sig[:10]:
        print(f"\n  변수: {r['vars']}  |  R²={r['R2']:.4f}  |  min_p={r['min_p']:.4f}")
        for v, (b, t, p) in r["params"].items():
            sig_mark = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {v:30s}: b={b:+.5f}, t={t:+.3f}, p={p:.4f} {sig_mark}")

step2_search(lam_resid, "FF3잔차 모멘텀 프리미엄")
step2_search(lam_raw,   "raw수익률 모멘텀 프리미엄")

print("\n완료.")
