"""
확인 1: WML 변동성이 MKT/SMB/HML보다 cycle state에 더 민감한가?
확인 2: WML 변동성의 cycle state 의존성이 현재 state만으로 충분한가,
        아니면 직전 경로(path)가 추가 정보를 주는가?
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

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
print("데이터 로드 중...")
factors = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", START, END)[0] / 100
wml_df  = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", START, END)[0] / 100
spx = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()

wml_col = [c for c in wml_df.columns if "Mom" in c or "mom" in c][0]
wml_t = wml_df[wml_col]

common = factors.index.intersection(spx.index).intersection(wml_t.index)
mkt = factors["Mkt-RF"].reindex(common)
smb = factors["SMB"].reindex(common)
hml = factors["HML"].reindex(common)
wml = wml_t.reindex(common).ffill()
spx = spx.reindex(common).ffill()

# ── state 신호 ───────────────────────────────────────────────────────────────
ma200     = spx.rolling(200).mean()
position  = (spx - ma200) / ma200
rv30      = np.log(spx / spx.shift(1)).rolling(30).std() * np.sqrt(252)
intensity = rv30.rolling(252).rank(pct=True)
bear      = (position < 0).astype(float)
bxi       = bear * intensity

# ── 팩터별 21일 rolling 변동성 ───────────────────────────────────────────────
vol_wml = wml.rolling(21).std()
vol_mkt = mkt.rolling(21).std()
vol_smb = smb.rolling(21).std()
vol_hml = hml.rolling(21).std()

def hac(y, Xdf, maxlags=10):
    df = pd.concat([y.rename("y"), Xdf], axis=1).dropna()
    r = OLS(df["y"], add_constant(df[Xdf.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return r

# ════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("확인 1: WML vs MKT/SMB/HML — cycle state 민감도 비교")
print("  각 팩터 변동성 ~ pos_lag1 + int_lag1 + bear×int_lag1")
print("="*65)

Xdf = pd.DataFrame({
    "pos_lag1": position.shift(1),
    "int_lag1": intensity.shift(1),
    "bxi_lag1": bxi.shift(1),
})

factor_vols = {"WML": vol_wml, "MKT": vol_mkt, "SMB": vol_smb, "HML": vol_hml}
results_c1 = {}

for fname, fvol in factor_vols.items():
    r = hac(fvol, Xdf)
    results_c1[fname] = r
    sig = {v: (r.params[v], r.tvalues[v], r.pvalues[v]) for v in Xdf.columns}
    print(f"\n  [{fname} vol]  R²={r.rsquared:.4f}")
    for v, (b, t, p) in sig.items():
        mk = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        print(f"    {v:15s}: b={b:+.5f}, t={t:+.3f}, p={p:.4f} {mk}")

# WML이 다른 팩터보다 R² 높은지, bear×int 계수가 더 큰지 비교
print("\n  [R² 비교 요약]")
for fname, r in results_c1.items():
    bxi_p = r.pvalues.get("bxi_lag1", 1.0)
    bxi_b = r.params.get("bxi_lag1", 0.0)
    print(f"    {fname}: R²={r.rsquared:.4f}, bear×int b={bxi_b:+.5f}, p={bxi_p:.4f}")

# WML이 특이적인지: WML vol을 다른 팩터 vol로 통제한 후 state 설명력 잔존하는가
print("\n  [WML 특이성 검정: 다른 팩터 vol 통제 후]")
Xdf2 = pd.DataFrame({
    "pos_lag1": position.shift(1),
    "int_lag1": intensity.shift(1),
    "bxi_lag1": bxi.shift(1),
    "mkt_vol":  vol_mkt,
    "smb_vol":  vol_smb,
    "hml_vol":  vol_hml,
})
r_ctrl = hac(vol_wml, Xdf2)
print(f"  R²={r_ctrl.rsquared:.4f}")
for v in ["pos_lag1", "int_lag1", "bxi_lag1"]:
    b, t, p = r_ctrl.params[v], r_ctrl.tvalues[v], r_ctrl.pvalues[v]
    mk = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"    {v:15s}: b={b:+.5f}, t={t:+.3f}, p={p:.4f} {mk}")

# ════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("확인 2: 현재 state만으로 충분한가, path가 추가 정보를 주는가?")
print("  타겟: WML vol21d")
print("="*65)

# 현재 state만 (scalar model)
Xdf_scalar = pd.DataFrame({
    "pos_lag1": position.shift(1),
    "int_lag1": intensity.shift(1),
    "bxi_lag1": bxi.shift(1),
})
r_scalar = hac(vol_wml, Xdf_scalar)
print(f"\n  [Scalar state only]  R²={r_scalar.rsquared:.4f}")

# path 추가: 직전 5/10/21일 WML 수익률 경로 특성
wml_path_mean5  = wml.rolling(5).mean().shift(1)   # 직전 5일 평균 수익률
wml_path_mean21 = wml.rolling(21).mean().shift(1)  # 직전 21일 평균 수익률
wml_path_skew21 = wml.rolling(21).skew().shift(1)  # 직전 21일 왜도 (크래시 전 음의 왜도)
wml_drawdown    = (wml.rolling(63).max() - wml).shift(1)  # 직전 고점 대비 낙폭

# state + WML 자체 경로
Xdf_path1 = pd.DataFrame({
    "pos_lag1":       position.shift(1),
    "int_lag1":       intensity.shift(1),
    "bxi_lag1":       bxi.shift(1),
    "wml_mean5":      wml_path_mean5,
    "wml_mean21":     wml_path_mean21,
    "wml_skew21":     wml_path_skew21,
    "wml_drawdown":   wml_drawdown,
})
r_path1 = hac(vol_wml, Xdf_path1)
print(f"\n  [State + WML path features]  R²={r_path1.rsquared:.4f}")
for v in ["wml_mean5", "wml_mean21", "wml_skew21", "wml_drawdown"]:
    b, t, p = r_path1.params[v], r_path1.tvalues[v], r_path1.pvalues[v]
    mk = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"    {v:20s}: b={b:+.6f}, t={t:+.3f}, p={p:.4f} {mk}")

# state + SPX 경로 (시장 경로)
spx_ret = np.log(spx / spx.shift(1))
spx_path_mean21  = spx_ret.rolling(21).mean().shift(1)
spx_path_skew21  = spx_ret.rolling(21).skew().shift(1)
spx_drawdown     = (spx.rolling(63).max() - spx).shift(1) / spx.shift(1)
spx_above_ma50   = ((spx / spx.rolling(50).mean() - 1)).shift(1)  # 단기 추세

Xdf_path2 = pd.DataFrame({
    "pos_lag1":        position.shift(1),
    "int_lag1":        intensity.shift(1),
    "bxi_lag1":        bxi.shift(1),
    "spx_mean21":      spx_path_mean21,
    "spx_skew21":      spx_path_skew21,
    "spx_drawdown":    spx_drawdown,
    "spx_above_ma50":  spx_above_ma50,
})
r_path2 = hac(vol_wml, Xdf_path2)
print(f"\n  [State + SPX path features]  R²={r_path2.rsquared:.4f}")
for v in ["spx_mean21", "spx_skew21", "spx_drawdown", "spx_above_ma50"]:
    b, t, p = r_path2.params[v], r_path2.tvalues[v], r_path2.pvalues[v]
    mk = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"    {v:20s}: b={b:+.6f}, t={t:+.3f}, p={p:.4f} {mk}")

# F-test: path features가 scalar model 대비 유의한 추가 설명력을 주는가
from statsmodels.stats.anova import anova_lm
print("\n  [F-test: path features의 추가 설명력]")
for label, r_full, extra_vars in [
    ("WML path", r_path1, ["wml_mean5","wml_mean21","wml_skew21","wml_drawdown"]),
    ("SPX path", r_path2, ["spx_mean21","spx_skew21","spx_drawdown","spx_above_ma50"]),
]:
    # 수동 F-test
    r2_full = r_full.rsquared
    r2_base = r_scalar.rsquared
    n = r_full.nobs
    k_full = len(r_full.params)
    k_base = len(r_scalar.params)
    q = k_full - k_base
    F = ((r2_full - r2_base) / q) / ((1 - r2_full) / (n - k_full))
    p_F = 1 - stats.f.cdf(F, q, n - k_full)
    print(f"    {label}: ΔR²={r2_full-r2_base:.4f}, F={F:.3f}, p={p_F:.4f}"
          + (" → path 유의" if p_F < 0.05 else " → path 불필요"))

# ════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("최종 판정")
print("="*65)
wml_r2 = results_c1["WML"].rsquared
mkt_r2 = results_c1["MKT"].rsquared
smb_r2 = results_c1["SMB"].rsquared
hml_r2 = results_c1["HML"].rsquared

wml_bxi_p = results_c1["WML"].pvalues["bxi_lag1"]
mkt_bxi_p = results_c1["MKT"].pvalues["bxi_lag1"]

ctrl_bxi_p = r_ctrl.pvalues["bxi_lag1"]

print(f"\n확인 1 — WML 특이성:")
print(f"  WML R²={wml_r2:.4f} vs MKT={mkt_r2:.4f}, SMB={smb_r2:.4f}, HML={hml_r2:.4f}")
if wml_r2 > max(mkt_r2, smb_r2, hml_r2) * 1.2:
    print("  → WML이 다른 팩터보다 cycle state에 유의하게 더 민감 ✓")
else:
    print("  → WML이 다른 팩터와 유사한 수준 — 모멘텀 특이성 약함 ✗")

print(f"\n  다른 팩터 vol 통제 후 bear×int p={ctrl_bxi_p:.4f}")
if ctrl_bxi_p < 0.05:
    print("  → WML 특이적 cycle state 민감도 존재 ✓")
else:
    print("  → 통제 후 사라짐 — 일반 변동성 효과일 가능성 ✗")
