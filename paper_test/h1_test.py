"""
H1 빠른 검증: 체제가 단면 모멘텀 프리미엄(λ_1,t)을 조절하는가
- Ken French 25 Size-B/M 포트폴리오 (CRSP 대체)
- Step 1: Fama-MacBeth → λ_1,t
- Step 2: λ_1,t ~ cycle_position + cycle_intensity (연속)
- 진단: cycle_position-WML 다중공선성, β 오차-mom 상관
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

print(f"공통 날짜: {common[0].date()} ~ {common[-1].date()}, N={len(common)}")


# ── 2. 체제 신호 ────────────────────────────────────────────────────────────

ma200 = spx.rolling(MA_WINDOW).mean()
cycle_position = (spx - ma200) / ma200

rv30 = np.log(spx / spx.shift(1)).rolling(RV_WINDOW).std() * np.sqrt(252)
cycle_intensity = rv30.rolling(RV_RANK_WINDOW).rank(pct=True)

regime = (2 * (cycle_position > 0).astype(int) +
          (cycle_intensity > 0.5).astype(int))
regime_labels = {0: "Bear/quiet", 1: "Bear/volatile",
                 2: "Bull/quiet",  3: "Bull/volatile"}


# ── 3. rolling β → FF3 잔차 ─────────────────────────────────────────────────

excess = port_ret.subtract(rf, axis=0)
F = pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml})

resid = pd.DataFrame(index=common, columns=port_ret.columns, dtype=float)
se_mkt_all = pd.DataFrame(index=common, columns=port_ret.columns, dtype=float)

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
            b = res.params  # [const, MKT, SMB, HML]
            resid.loc[date, col] = (y.loc[date]
                                    - b[1]*mkt.loc[date]
                                    - b[2]*smb.loc[date]
                                    - b[3]*hml.loc[date])
            se_mkt_all.loc[date, col] = res.bse[1]
        except Exception:
            pass

resid = resid.astype(float).dropna(how="all")
se_mkt_all = se_mkt_all.astype(float)
print(f"잔차 계산 완료: {resid.shape}")


# ── 4. 모멘텀 피처 (12-1개월) ───────────────────────────────────────────────

cum = (1 + port_ret).cumprod()
mom_12_1 = cum.shift(21) / cum.shift(252) - 1
mom_12_1 = mom_12_1.reindex(resid.index)


# ── 5. Fama-MacBeth Step 1 ──────────────────────────────────────────────────

print("Fama-MacBeth Step 1 실행 중...")
lam0_list, lam1_list, dates_fm = [], [], []

for date in resid.index:
    y_cs = resid.loc[date].dropna()
    x_cs = mom_12_1.loc[date].reindex(y_cs.index).dropna()
    idx = y_cs.index.intersection(x_cs.index)
    if len(idx) < 10:
        continue
    yv = y_cs[idx].values
    xv = stats.rankdata(x_cs[idx].values) / len(idx) - 0.5  # cross-sectional rank
    try:
        res = OLS(yv, add_constant(xv)).fit()
        lam0_list.append(res.params[0])
        lam1_list.append(res.params[1])
        dates_fm.append(date)
    except Exception:
        pass

lam = pd.DataFrame({"lambda_0": lam0_list, "lambda_1": lam1_list},
                   index=pd.DatetimeIndex(dates_fm))
print(f"λ_1,t: n={len(lam)}, mean={lam['lambda_1'].mean():.4f}, "
      f"std={lam['lambda_1'].std():.4f}")


# ── 6. 진단: cycle_position-WML 다중공선성 ─────────────────────────────────

cp = cycle_position.reindex(lam.index).dropna()
wml_d = wml_t.reindex(cp.index).dropna()
idx_d = cp.index.intersection(wml_d.index)
rho_cp_wml, p_cp_wml = stats.spearmanr(cp[idx_d], wml_d[idx_d])
print(f"\n[진단] Spearman(cycle_position, WML): ρ={rho_cp_wml:.3f}, p={p_cp_wml:.4f}")
multicollinear = abs(rho_cp_wml) > 0.5
if multicollinear:
    print("  → |ρ|>0.5: 다중공선성 위험. Step 2-C 생략, 단변량만 사용.")
else:
    print("  → |ρ|≤0.5: 다변량 회귀 진행 가능.")


# ── 7. 진단: β 추정 오차-mom 상관 (endogeneity 진단) ──────────────────────

mom_abs = mom_12_1.abs()
se_mean = se_mkt_all.reindex(resid.index)

# 날짜별 cross-sectional 평균으로 집계 후 시계열 상관
mom_cs_mean = mom_abs.mean(axis=1).reindex(lam.index).dropna()
se_cs_mean  = se_mean.mean(axis=1).reindex(mom_cs_mean.index).dropna()
idx_e = mom_cs_mean.index.intersection(se_cs_mean.index)
rho_mom_se, p_mom_se = stats.spearmanr(mom_cs_mean[idx_e], se_cs_mean[idx_e])
print(f"[진단] Spearman(|mom|, se(β_MKT)): ρ={rho_mom_se:.3f}, p={p_mom_se:.4f}")
if abs(rho_mom_se) >= 0.1 and p_mom_se < 0.05:
    print("  → 유의: endogeneity 가능. IV 접근(mom_t-2) 고려 필요.")
else:
    print("  → 낮음: 편향 무시 가능.")


# ── 8. Step 2 회귀 ──────────────────────────────────────────────────────────

reg = pd.DataFrame({
    "lambda_1":        lam["lambda_1"],
    "cycle_position":  cycle_position.reindex(lam.index),
    "cycle_intensity": cycle_intensity.reindex(lam.index),
    "WML":             wml_t.reindex(lam.index),
    "regime":          regime.reindex(lam.index),
}).dropna()

print(f"\nStep 2 샘플: {len(reg)}일")

# (A) 단변량
rA = OLS(reg["lambda_1"],
         add_constant(reg["cycle_position"])).fit(
    cov_type="HAC", cov_kwds={"maxlags": 10})
b_A = rA.params["cycle_position"]
t_A = rA.tvalues["cycle_position"]
p_A = rA.pvalues["cycle_position"]
print(f"\n[Step 2-A] λ_1 ~ cycle_position: b={b_A:.4f}, t={t_A:.3f}, p={p_A:.4f}")

# (B) 다변량 (연속 신호)
rB = OLS(reg["lambda_1"],
         add_constant(reg[["cycle_position", "cycle_intensity"]])).fit(
    cov_type="HAC", cov_kwds={"maxlags": 10})
print("[Step 2-B] λ_1 ~ cycle_position + cycle_intensity:")
for v in ["cycle_position", "cycle_intensity"]:
    print(f"  {v}: b={rB.params[v]:.4f}, t={rB.tvalues[v]:.3f}, p={rB.pvalues[v]:.4f}")

# (C) WML 통제 (다중공선성 낮을 때만)
if not multicollinear:
    rC = OLS(reg["lambda_1"],
             add_constant(reg[["cycle_position", "WML"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 10})
    print("[Step 2-C] λ_1 ~ cycle_position + WML:")
    for v in ["cycle_position", "WML"]:
        print(f"  {v}: b={rC.params[v]:.4f}, t={rC.tvalues[v]:.3f}, p={rC.pvalues[v]:.4f}")

# (D) H1-Vol 핵심: intensity + Bear×intensity 상호작용
reg["bear_x_intensity"] = (reg["cycle_position"] < 0).astype(float) * reg["cycle_intensity"]
rD = OLS(reg["lambda_1"],
         add_constant(reg[["cycle_intensity", "cycle_position", "bear_x_intensity"]])).fit(
    cov_type="HAC", cov_kwds={"maxlags": 10})
print("[Step 2-D] λ_1 ~ intensity + position + Bear×intensity (H1-Vol):")
for v in ["cycle_intensity", "cycle_position", "bear_x_intensity"]:
    print(f"  {v}: b={rD.params[v]:.4f}, t={rD.tvalues[v]:.3f}, p={rD.pvalues[v]:.4f}")
d_coef = rD.params["bear_x_intensity"]
d_pval = rD.pvalues["bear_x_intensity"]
print(f"  → H1-Vol 판정: d={'<0 (Bear/volatile에서 모멘텀 크래시)' if d_coef < 0 else '>0 (예상 반대)'},"
      f" p={d_pval:.4f} {'→ 유의' if d_pval < 0.05 else '→ 비유의'}")

# (E) 비선형 확장: intensity² 추가
reg["intensity_sq"] = reg["cycle_intensity"] ** 2
rE = OLS(reg["lambda_1"],
         add_constant(reg[["cycle_intensity", "intensity_sq", "cycle_position", "bear_x_intensity"]])).fit(
    cov_type="HAC", cov_kwds={"maxlags": 10})
print("[Step 2-E] + intensity² (비선형 꼬리 테스트):")
for v in ["cycle_intensity", "intensity_sq", "bear_x_intensity"]:
    print(f"  {v}: b={rE.params[v]:.4f}, t={rE.tvalues[v]:.3f}, p={rE.pvalues[v]:.4f}")


# ── 9. λ_1,t vs WML_t 시계열 상관 ─────────────────────────────────────────

rho_lw, p_lw = stats.spearmanr(reg["lambda_1"], reg["WML"])
print(f"\n[진단] Spearman(λ_1,t, WML_t): ρ={rho_lw:.3f}, p={p_lw:.4f}")


# ── 10. 체제별 λ_1,t 평균 비교 ────────────────────────────────────────────

print("\n[체제별 λ_1,t]")
base = reg.loc[reg["regime"] == 0, "lambda_1"]
for r, label in regime_labels.items():
    grp = reg.loc[reg["regime"] == r, "lambda_1"]
    if len(grp) < 5:
        continue
    if r == 0:
        print(f"  {label}: mean={grp.mean():.4f}, std={grp.std():.4f}, n={len(grp)}")
    else:
        t_s, p_s = stats.ttest_ind(grp, base, equal_var=False)
        print(f"  {label}: mean={grp.mean():.4f}, std={grp.std():.4f}, "
              f"n={len(grp)}, t vs Bear/quiet={t_s:.3f}, p={p_s:.4f}")


# ── 11. H1 판정 ────────────────────────────────────────────────────────────

print("\n" + "="*50)
print("H1 판정 요약")
print(f"  H1-Pos (Step 2-A): b={b_A:.4f}, p={p_A:.4f} → {'채택' if p_A < 0.05 else '기각'}")
print(f"  H1-Vol (Step 2-D, bear_x_intensity): b={d_coef:.4f}, p={d_pval:.4f} → {'채택 (d<0 확인)' if d_pval < 0.05 and d_coef < 0 else '기각 또는 부호 불일치'}")
print("="*50)
