"""
H1-Vol 분포 검정
선형 회귀가 아니라 체제별 λ_1,t 분포 자체가 다른가를 검정
- KS test (분포 형태 차이)
- Permutation test (평균 차이, 분포 무관)
- Tail 분석 (Bear/volatile에서 음수 꼬리가 두꺼운가)
- raw return 기반 λ_1,t도 병행 (generated target 방어)
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

START, END = "1990-01-01", "2024-12-31"
ROLL_BETA = 60
np.random.seed(42)

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
print("데이터 로드 중...")
port_ret = web.DataReader("25_Portfolios_5x5_Daily", "famafrench", START, END)[0] / 100
factors  = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", START, END)[0] / 100
spx = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()

mkt = factors["Mkt-RF"]; smb = factors["SMB"]; hml = factors["HML"]; rf = factors["RF"]
common = port_ret.index.intersection(factors.index).intersection(spx.index)
port_ret = port_ret.loc[common]
mkt=mkt.reindex(common); smb=smb.reindex(common)
hml=hml.reindex(common); rf=rf.reindex(common)
spx = spx.reindex(common).ffill()

# ── state 신호 ───────────────────────────────────────────────────────────────
ma200     = spx.rolling(200).mean()
position  = (spx - ma200) / ma200
intensity = np.log(spx/spx.shift(1)).rolling(30).std().mul(np.sqrt(252)).rolling(252).rank(pct=True)
regime    = (2*(position>0).astype(int) + (intensity>0.5).astype(int))
regime_labels = {0:"Bear/quiet", 1:"Bear/volatile", 2:"Bull/quiet", 3:"Bull/volatile"}

# ── rolling β → FF3 잔차 ─────────────────────────────────────────────────────
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

excess = port_ret.subtract(rf, axis=0)
F = pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml})
resid = pd.DataFrame(index=common, columns=port_ret.columns, dtype=float)

print("rolling β 추정 중...")
for col in port_ret.columns:
    y = excess[col]
    for t in range(ROLL_BETA, len(common)):
        sl = slice(t-ROLL_BETA, t)
        try:
            res = OLS(y.iloc[sl].values, add_constant(F.iloc[sl].values)).fit()
            b = res.params
            resid.loc[common[t], col] = (y.loc[common[t]]
                - b[1]*mkt.loc[common[t]] - b[2]*smb.loc[common[t]] - b[3]*hml.loc[common[t]])
        except: pass
resid = resid.astype(float).dropna(how="all")

# ── 모멘텀 피처 ──────────────────────────────────────────────────────────────
cum = (1 + port_ret).cumprod()
mom_12_1 = cum.shift(21) / cum.shift(252) - 1

# ── FMB Step 1: FF3 잔차 기반 λ_1,t ─────────────────────────────────────────
def run_fmb(target_df, mom_df):
    lam1, dates = [], []
    for date in target_df.index:
        y_cs = target_df.loc[date].dropna()
        x_cs = mom_df.loc[date].reindex(y_cs.index).dropna()
        idx = y_cs.index.intersection(x_cs.index)
        if len(idx) < 10: continue
        xv = stats.rankdata(x_cs[idx].values) / len(idx) - 0.5
        try:
            res = OLS(y_cs[idx].values, add_constant(xv)).fit()
            lam1.append(res.params[1]); dates.append(date)
        except: pass
    return pd.Series(lam1, index=pd.DatetimeIndex(dates))

print("FMB Step 1...")
lam_resid = run_fmb(resid, mom_12_1.reindex(resid.index))
lam_raw   = run_fmb(port_ret.reindex(resid.index), mom_12_1.reindex(resid.index))

def permutation_test(a, b, n_perm=5000):
    """평균 차이의 permutation test"""
    obs = np.mean(a) - np.mean(b)
    combined = np.concatenate([a, b])
    count = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        count += abs(combined[:len(a)].mean() - combined[len(a):].mean()) >= abs(obs)
    return obs, count / n_perm

def analyze_regime_dist(lam_series, label):
    reg_df = pd.DataFrame({
        "lam1":   lam_series,
        "regime": regime.reindex(lam_series.index)
    }).dropna()

    print(f"\n{'='*60}")
    print(f"[{label}] 체제별 λ_1,t 분포 검정")
    print(f"{'='*60}")

    groups = {r: reg_df.loc[reg_df["regime"]==r, "lam1"].values
              for r in range(4) if (reg_df["regime"]==r).sum() >= 30}

    # 기술통계
    print(f"\n  기술통계:")
    for r, g in groups.items():
        print(f"  {regime_labels[r]:15s}: n={len(g):4d}, "
              f"mean={g.mean():+.5f}, std={g.std():.5f}, "
              f"skew={stats.skew(g):+.3f}, "
              f"p10={np.percentile(g,10):+.5f}")

    # KS test: Bear/quiet vs 나머지
    base = groups.get(0)
    if base is None: return
    print(f"\n  KS test (vs Bear/quiet):")
    for r, g in groups.items():
        if r == 0: continue
        ks, p = stats.ks_2samp(base, g)
        print(f"  {regime_labels[r]:15s}: KS={ks:.4f}, p={p:.4f}"
              + (" ***" if p<0.001 else " **" if p<0.01 else " *" if p<0.05 else ""))

    # Permutation test: Bear/quiet vs Bear/volatile (핵심 비교)
    bv = groups.get(1)
    if bv is not None:
        obs_diff, p_perm = permutation_test(base, bv)
        print(f"\n  Permutation test (Bear/quiet vs Bear/volatile):")
        print(f"  mean diff={obs_diff:+.5f}, p={p_perm:.4f}"
              + (" → 유의" if p_perm < 0.05 else " → 비유의"))

    # Tail 분석: 음수 꼬리 (λ_1,t < 0 비율, 하위 10% 평균)
    print(f"\n  Tail 분석 (모멘텀 크래시 지표):")
    for r, g in groups.items():
        neg_rate = (g < 0).mean()
        tail10   = np.percentile(g, 10)
        print(f"  {regime_labels[r]:15s}: P(λ<0)={neg_rate:.3f}, "
              f"10th pct={tail10:+.5f}")

    # Bear/quiet vs Bear/volatile tail 차이
    if bv is not None:
        bq_neg = (base < 0).mean()
        bv_neg = (bv < 0).mean()
        print(f"\n  Bear/quiet P(λ<0)={bq_neg:.3f} vs Bear/volatile P(λ<0)={bv_neg:.3f}")
        print(f"  → 차이={bv_neg-bq_neg:+.3f} "
              + ("(Bear/volatile에서 음수 비율 증가 ✓)" if bv_neg > bq_neg else "(예상 반대 ✗)"))

    # ANOVA: 4개 체제 평균이 동일한가
    f_stat, p_anova = stats.f_oneway(*groups.values())
    print(f"\n  ANOVA (4체제 평균 동일 귀무): F={f_stat:.3f}, p={p_anova:.4f}"
          + (" → 체제별 평균 유의하게 다름 ✓" if p_anova < 0.05 else " → 유의하지 않음"))

    # Kruskal-Wallis (비모수)
    kw, p_kw = stats.kruskal(*groups.values())
    print(f"  Kruskal-Wallis:              H={kw:.3f},  p={p_kw:.4f}"
          + (" → 체제별 분포 유의하게 다름 ✓" if p_kw < 0.05 else " → 유의하지 않음"))

analyze_regime_dist(lam_resid, "FF3 잔차 모멘텀 프리미엄")
analyze_regime_dist(lam_raw,   "raw 수익률 모멘텀 프리미엄")

print("\n완료.")
