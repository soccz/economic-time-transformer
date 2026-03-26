"""
H1 데이터 문제 진단
1. λ_1,t 스케일 및 신호 강도 확인
2. raw 수익률 타겟 vs FF3 잔차 타겟 비교
3. 모멘텀 피처 유효성 (포트폴리오 내 분산)
4. 월별 집계 시 결과 변화 (일별 노이즈 제거)
5. 선행 연구 재현: Fama-MacBeth on raw returns (모멘텀 프리미엄 존재 확인)
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

# ── 데이터 로드 (h1_test.py와 동일) ────────────────────────────────────────
print("데이터 로드 중...")
port_ret = web.DataReader("25_Portfolios_5x5_Daily", "famafrench", START, END)[0] / 100
factors  = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", START, END)[0] / 100
wml_df   = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", START, END)[0] / 100
spx = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()

mkt = factors["Mkt-RF"]; smb = factors["SMB"]
hml = factors["HML"];    rf  = factors["RF"]
wml_col = [c for c in wml_df.columns if "Mom" in c or "mom" in c][0]
wml_t = wml_df[wml_col]

common = port_ret.index.intersection(factors.index).intersection(spx.index)
port_ret = port_ret.loc[common]
mkt = mkt.reindex(common); smb = smb.reindex(common)
hml = hml.reindex(common); rf  = rf.reindex(common)
wml_t = wml_t.reindex(common).ffill()
spx   = spx.reindex(common).ffill()

excess = port_ret.subtract(rf, axis=0)
F = pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml})

ma200 = spx.rolling(MA_WINDOW).mean()
cycle_position = (spx - ma200) / ma200
rv30 = np.log(spx / spx.shift(1)).rolling(RV_WINDOW).std() * np.sqrt(252)
cycle_intensity = rv30.rolling(RV_RANK_WINDOW).rank(pct=True)

# ── 진단 1: 모멘텀 피처 유효성 ─────────────────────────────────────────────
print("\n[진단 1] 포트폴리오 모멘텀 피처 분산")
cum = (1 + port_ret).cumprod()
mom_12_1 = cum.shift(21) / cum.shift(252) - 1

# 날짜별 cross-sectional std (신호 분산이 충분한지)
mom_cs_std = mom_12_1.std(axis=1).dropna()
print(f"  mom_12_1 cross-sectional std: mean={mom_cs_std.mean():.4f}, "
      f"min={mom_cs_std.min():.4f}, max={mom_cs_std.max():.4f}")
print(f"  → 포트폴리오 간 모멘텀 분산이 충분한가: "
      f"{'충분 (>0.05)' if mom_cs_std.mean() > 0.05 else '부족 — 집계 희석 의심'}")

# ── 진단 2: raw 수익률로 Fama-MacBeth (모멘텀 프리미엄 존재 확인) ──────────
print("\n[진단 2] raw 수익률 타겟으로 Fama-MacBeth (선행 연구 재현)")
lam1_raw, dates_raw = [], []
for date in excess.index[252:]:  # mom 계산 가능 시점부터
    y_cs = excess.loc[date].dropna()
    x_cs = mom_12_1.loc[date].reindex(y_cs.index).dropna()
    idx = y_cs.index.intersection(x_cs.index)
    if len(idx) < 10:
        continue
    yv = y_cs[idx].values
    xv = stats.rankdata(x_cs[idx].values) / len(idx) - 0.5
    try:
        res = OLS(yv, add_constant(xv)).fit()
        lam1_raw.append(res.params[1])
        dates_raw.append(date)
    except Exception:
        pass

lam_raw = pd.Series(lam1_raw, index=pd.DatetimeIndex(dates_raw))
t_raw, p_raw = stats.ttest_1samp(lam_raw.dropna(), 0)
print(f"  λ_1,t(raw): mean={lam_raw.mean():.4f}, std={lam_raw.std():.4f}")
print(f"  t-test vs 0: t={t_raw:.3f}, p={p_raw:.4f}")
print(f"  → 모멘텀 프리미엄 존재: {'YES' if p_raw < 0.05 and lam_raw.mean() > 0 else 'NO/WEAK'}")

# ── 진단 3: FF3 잔차 타겟 λ_1,t 스케일 비교 ───────────────────────────────
print("\n[진단 3] FF3 잔차 타겟 λ_1,t 스케일")
# rolling β (간소화: 전체 기간 단일 β 사용해서 빠르게 확인)
betas_full = {}
for col in port_ret.columns:
    y = excess[col].dropna()
    Xt = add_constant(F.reindex(y.index).dropna())
    y = y.reindex(Xt.index)
    try:
        res = OLS(y, Xt).fit()
        betas_full[col] = res.params[1:]  # [MKT, SMB, HML]
    except Exception:
        betas_full[col] = np.array([1.0, 0.0, 0.0])

resid_full = pd.DataFrame(index=common, columns=port_ret.columns, dtype=float)
for col, b in betas_full.items():
    resid_full[col] = (excess[col]
                       - b[0]*mkt - b[1]*smb - b[2]*hml)

lam1_resid, dates_resid = [], []
for date in resid_full.index[252:]:
    y_cs = resid_full.loc[date].dropna()
    x_cs = mom_12_1.loc[date].reindex(y_cs.index).dropna()
    idx = y_cs.index.intersection(x_cs.index)
    if len(idx) < 10:
        continue
    yv = y_cs[idx].values
    xv = stats.rankdata(x_cs[idx].values) / len(idx) - 0.5
    try:
        res = OLS(yv, add_constant(xv)).fit()
        lam1_resid.append(res.params[1])
        dates_resid.append(date)
    except Exception:
        pass

lam_resid = pd.Series(lam1_resid, index=pd.DatetimeIndex(dates_resid))
t_res, p_res = stats.ttest_1samp(lam_resid.dropna(), 0)
print(f"  λ_1,t(FF3 잔차, 전체기간 β): mean={lam_resid.mean():.4f}, std={lam_resid.std():.4f}")
print(f"  t-test vs 0: t={t_res:.3f}, p={p_res:.4f}")
print(f"  raw 대비 신호 감소율: {(1 - abs(lam_resid.mean())/max(abs(lam_raw.mean()),1e-8)):.1%}")

# ── 진단 4: 월별 집계 후 Step 2 ────────────────────────────────────────────
print("\n[진단 4] 월별 집계 후 Step 2 (일별 노이즈 제거)")
lam_raw_m = lam_raw.resample("ME").mean().dropna()
cp_m = cycle_position.resample("ME").last().reindex(lam_raw_m.index).dropna()
wml_m = wml_t.resample("ME").sum().reindex(cp_m.index).dropna()
idx_m = lam_raw_m.index.intersection(cp_m.index).intersection(wml_m.index)

reg_m = pd.DataFrame({
    "lambda_1": lam_raw_m[idx_m],
    "cycle_position": cp_m[idx_m],
    "WML": wml_m[idx_m],
}).dropna()

rM = OLS(reg_m["lambda_1"],
         add_constant(reg_m["cycle_position"])).fit(
    cov_type="HAC", cov_kwds={"maxlags": 3})
print(f"  [월별, raw 타겟] λ_1 ~ cycle_position: "
      f"b={rM.params['cycle_position']:.4f}, "
      f"t={rM.tvalues['cycle_position']:.3f}, "
      f"p={rM.pvalues['cycle_position']:.4f}")

# ── 진단 5: 체제별 WML 팩터 수익률 자체 확인 ──────────────────────────────
print("\n[진단 5] 체제별 WML 팩터 수익률 (기준선)")
regime = (2*(cycle_position > 0).astype(int) + (cycle_intensity > 0.5).astype(int))
regime_labels = {0:"Bear/quiet",1:"Bear/volatile",2:"Bull/quiet",3:"Bull/volatile"}
wml_reg = pd.DataFrame({"WML": wml_t, "regime": regime}).dropna()
base_wml = wml_reg.loc[wml_reg["regime"]==0, "WML"]
for r, label in regime_labels.items():
    grp = wml_reg.loc[wml_reg["regime"]==r, "WML"]
    if r == 0:
        print(f"  {label}: mean={grp.mean()*252:.3f}(연율), n={len(grp)}")
    else:
        t_s, p_s = stats.ttest_ind(grp, base_wml, equal_var=False)
        print(f"  {label}: mean={grp.mean()*252:.3f}(연율), n={len(grp)}, "
              f"t vs Bear/quiet={t_s:.3f}, p={p_s:.4f}")

print("\n[요약]")
print(f"  포트폴리오 모멘텀 분산: {mom_cs_std.mean():.4f}")
print(f"  raw λ_1,t 유의성: p={p_raw:.4f}")
print(f"  FF3잔차 λ_1,t 유의성: p={p_res:.4f}")
print(f"  월별 집계 후 cycle_position 유의성: p={rM.pvalues['cycle_position']:.4f}")
