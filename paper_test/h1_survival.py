"""
H1 생존 가능성 전수 탐색
틀: "S&P cycle state(position/intensity)가 모멘텀 관련 예측력을 조절한다"

후보 H1 형태:
  A. cycle state → λ_1,t (Fama-MacBeth 모멘텀 프리미엄) [기존, 실패]
  B. cycle state → WML 예측 (모멘텀 팩터 수익률 타이밍)
  C. cycle state → WML 변동성/꼬리 (불확실성 조절)
  D. cycle state × WML → λ_1,t (WML 통제 후 잔여 조절)
  E. cycle state → IC_t (모멘텀 신호의 예측력 자체가 체제별로 다른가)
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

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
print("데이터 로드 중...")
port_ret = web.DataReader("25_Portfolios_5x5_Daily", "famafrench", START, END)[0] / 100
factors  = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", START, END)[0] / 100
wml_df   = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", START, END)[0] / 100
spx = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()

mkt = factors["Mkt-RF"]; smb = factors["SMB"]; hml = factors["HML"]; rf = factors["RF"]
wml_col = [c for c in wml_df.columns if "Mom" in c or "mom" in c][0]
wml_t = wml_df[wml_col]

common = port_ret.index.intersection(factors.index).intersection(spx.index)
port_ret = port_ret.loc[common]
mkt = mkt.reindex(common); smb = smb.reindex(common)
hml = hml.reindex(common); rf  = rf.reindex(common)
wml_t = wml_t.reindex(common).ffill()
spx = spx.reindex(common).ffill()

# ── state 신호 ───────────────────────────────────────────────────────────────
ma200 = spx.rolling(MA_WINDOW).mean()
position  = (spx - ma200) / ma200
rv30      = np.log(spx / spx.shift(1)).rolling(RV_WINDOW).std() * np.sqrt(252)
intensity = rv30.rolling(RV_RANK_WINDOW).rank(pct=True)
bear      = (position < 0).astype(float)
high_vol  = (intensity > 0.7).astype(float)
regime    = (2*(position>0).astype(int) + (intensity>0.5).astype(int))

# ── rolling β → FF3 잔차 ─────────────────────────────────────────────────────
excess = port_ret.subtract(rf, axis=0)
F = pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml})
resid = pd.DataFrame(index=common, columns=port_ret.columns, dtype=float)

print("rolling β 추정 중...")
for col in port_ret.columns:
    y = excess[col]
    for t in range(ROLL_BETA, len(common)):
        sl = slice(t - ROLL_BETA, t)
        try:
            res = OLS(y.iloc[sl].values, add_constant(F.iloc[sl].values)).fit()
            b = res.params
            resid.loc[common[t], col] = (y.loc[common[t]]
                - b[1]*mkt.loc[common[t]] - b[2]*smb.loc[common[t]] - b[3]*hml.loc[common[t]])
        except Exception:
            pass
resid = resid.astype(float).dropna(how="all")

# ── 모멘텀 피처 ──────────────────────────────────────────────────────────────
cum = (1 + port_ret).cumprod()
mom_12_1 = cum.shift(21) / cum.shift(252) - 1

# ── Fama-MacBeth Step 1 → λ_1,t ──────────────────────────────────────────────
print("Fama-MacBeth Step 1...")
lam1_list, dates_fm = [], []
for date in resid.index:
    y_cs = resid.loc[date].dropna()
    x_cs = mom_12_1.loc[date].reindex(y_cs.index).dropna()
    idx = y_cs.index.intersection(x_cs.index)
    if len(idx) < 10: continue
    xv = stats.rankdata(x_cs[idx].values) / len(idx) - 0.5
    try:
        res = OLS(y_cs[idx].values, add_constant(xv)).fit()
        lam1_list.append(res.params[1])
        dates_fm.append(date)
    except Exception:
        pass
lam1 = pd.Series(lam1_list, index=pd.DatetimeIndex(dates_fm))

# ── 날짜별 IC_t (모멘텀 신호의 단면 예측력) ──────────────────────────────────
print("날짜별 IC_t 계산...")
ic_list, ic_dates = [], []
for date in resid.index:
    y_cs = resid.loc[date].dropna()
    x_cs = mom_12_1.loc[date].reindex(y_cs.index).dropna()
    idx = y_cs.index.intersection(x_cs.index)
    if len(idx) < 10: continue
    rho, _ = stats.spearmanr(x_cs[idx].values, y_cs[idx].values)
    ic_list.append(rho)
    ic_dates.append(date)
ic_t = pd.Series(ic_list, index=pd.DatetimeIndex(ic_dates))

def hac_reg(y, X_df, label, maxlags=10):
    df = pd.concat([y.rename("y"), X_df], axis=1).dropna()
    r = OLS(df["y"], add_constant(df[X_df.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    sig_vars = {v: (r.params[v], r.tvalues[v], r.pvalues[v])
                for v in X_df.columns if r.pvalues[v] < 0.05}
    return r, sig_vars

def print_result(label, r, sig_vars, note=""):
    print(f"\n  [{label}] R²={r.rsquared:.4f} {note}")
    if sig_vars:
        for v, (b, t, p) in sig_vars.items():
            mk = "***" if p<0.001 else "**" if p<0.01 else "*"
            print(f"    {v:35s}: b={b:+.5f}, t={t:+.3f}, p={p:.4f} {mk}")
    else:
        print("    → 유의한 변수 없음")

print("\n" + "="*65)
print("H1 후보 A: cycle state → λ_1,t (기존 접근, 확인용)")
print("="*65)
base = pd.DataFrame({"pos": position, "int": intensity,
                     "bear_x_int": bear*intensity}).reindex(lam1.index)
r, sv = hac_reg(lam1, base, "A")
print_result("A: pos+int+bear×int → λ_1,t", r, sv, "(기존 실패 확인)")

print("\n" + "="*65)
print("H1 후보 B: cycle state → WML_t+1 예측 (모멘텀 팩터 타이밍)")
print("="*65)
# WML을 1일 앞으로 당겨서 예측 타겟으로
wml_fwd = wml_t.shift(-1)  # t+1 WML
# 5일 누적 WML도 테스트
wml_fwd5 = wml_t.rolling(5).sum().shift(-5)

for wml_target, tag in [(wml_fwd, "WML_t+1"), (wml_fwd5, "WML_5d누적")]:
    Xdf = pd.DataFrame({
        "pos_lag1":       position.shift(1),
        "int_lag1":       intensity.shift(1),
        "bear_x_int_lag1": bear.shift(1)*intensity.shift(1),
    })
    r, sv = hac_reg(wml_target, Xdf, f"B_{tag}")
    print_result(f"B: pos+int+bear×int → {tag}", r, sv)

print("\n" + "="*65)
print("H1 후보 C: cycle state → |WML| 또는 WML 변동성 (불확실성 조절)")
print("="*65)
wml_abs  = wml_t.abs()
wml_vol  = wml_t.rolling(21).std()  # 21일 rolling std
for target, tag in [(wml_abs, "|WML|"), (wml_vol, "WML_vol21d")]:
    Xdf = pd.DataFrame({
        "int_lag1":       intensity.shift(1),
        "pos_lag1":       position.shift(1),
        "bear_x_int_lag1": bear.shift(1)*intensity.shift(1),
    })
    r, sv = hac_reg(target, Xdf, f"C_{tag}")
    print_result(f"C: state → {tag}", r, sv)

print("\n" + "="*65)
print("H1 후보 D: WML 통제 후 cycle state의 잔여 조절력")
print("  λ_1,t ~ WML + state  (WML 흡수 후 state 유의한가)")
print("="*65)
Xdf = pd.DataFrame({
    "WML":            wml_t,
    "int":            intensity,
    "pos":            position,
    "bear_x_int":     bear*intensity,
    "WML_x_int":      wml_t*intensity,   # WML × intensity 상호작용
    "WML_x_bear":     wml_t*bear,        # WML × bear 상호작용
})
r, sv = hac_reg(lam1, Xdf, "D_full")
print_result("D: WML+state+WML×state → λ_1,t", r, sv,
             "(WML 통제 후 state 잔여 조절)")

# WML×intensity만 추가한 버전
Xdf2 = pd.DataFrame({"WML": wml_t, "WML_x_int": wml_t*intensity, "WML_x_bear": wml_t*bear})
r2, sv2 = hac_reg(lam1, Xdf2, "D_interact")
print_result("D2: WML + WML×int + WML×bear → λ_1,t", r2, sv2)

print("\n" + "="*65)
print("H1 후보 E: cycle state → IC_t (모멘텀 신호 예측력 자체의 체제 의존성)")
print("  IC_t = Spearman(mom_rank, resid_return) 날짜별")
print("="*65)
Xdf = pd.DataFrame({
    "int_lag1":       intensity.shift(1),
    "pos_lag1":       position.shift(1),
    "bear_x_int_lag1": bear.shift(1)*intensity.shift(1),
})
r, sv = hac_reg(ic_t, Xdf, "E")
print_result("E: state → IC_t", r, sv)

# 체제별 IC_t 평균
ic_reg = pd.DataFrame({"ic": ic_t, "regime": regime.reindex(ic_t.index)}).dropna()
regime_labels = {0:"Bear/quiet",1:"Bear/volatile",2:"Bull/quiet",3:"Bull/volatile"}
print("\n  체제별 IC_t 평균:")
base_grp = ic_reg.loc[ic_reg["regime"]==0, "ic"]
for r_id, label in regime_labels.items():
    grp = ic_reg.loc[ic_reg["regime"]==r_id, "ic"]
    if len(grp) < 5: continue
    t_s, p_s = stats.ttest_ind(grp, base_grp, equal_var=False)
    print(f"    {label:15s}: mean={grp.mean():+.5f}, n={len(grp):4d}"
          + (f", t={t_s:.3f}, p={p_s:.4f}" if r_id != 0 else ""))

print("\n" + "="*65)
print("H1 후보 F: cycle state → λ_1,t (월별 집계 — 일별 노이즈 제거)")
print("="*65)
lam1_m    = lam1.resample("ME").mean()
pos_m     = position.resample("ME").last()
int_m     = intensity.resample("ME").last()
bear_m    = bear.resample("ME").last()
bxi_m     = (bear_m * int_m)
Xdf = pd.DataFrame({"int": int_m, "pos": pos_m, "bear_x_int": bxi_m}).reindex(lam1_m.index)
r, sv = hac_reg(lam1_m, Xdf, "F_monthly", maxlags=3)
print_result("F: state → λ_1,t (월별)", r, sv)

# 월별 체제별 평균
lam1_m_reg = pd.DataFrame({"lam1": lam1_m,
    "regime": regime.resample("ME").last().reindex(lam1_m.index)}).dropna()
print("\n  체제별 λ_1,t 월평균:")
base_grp = lam1_m_reg.loc[lam1_m_reg["regime"]==0, "lam1"]
for r_id, label in regime_labels.items():
    grp = lam1_m_reg.loc[lam1_m_reg["regime"]==r_id, "lam1"]
    if len(grp) < 5: continue
    t_s, p_s = stats.ttest_ind(grp, base_grp, equal_var=False)
    print(f"    {label:15s}: mean={grp.mean():+.6f}, n={len(grp):3d}"
          + (f", t={t_s:.3f}, p={p_s:.4f}" if r_id != 0 else ""))

print("\n" + "="*65)
print("최종 요약: 틀 안에서 H1 생존 가능성")
print("="*65)
