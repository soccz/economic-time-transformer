"""
선택 1: 개별 종목 기반 H1 재검정
- S&P 500 구성종목 수익률 (yfinance, 생존편향 있음 - 탐색용)
- 단면 N ~400으로 Fama-MacBeth λ_1,t 재추정
- H1-Vol: state → λ_1,t 분포/평균/tail 검정
- Robustness: W=60/120, OLS/Ridge β 4가지 조합
주의: 생존편향으로 인해 결과는 탐색적으로만 해석
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

START, END = "2005-01-01", "2024-12-31"
np.random.seed(42)

# ── 1. S&P 500 현재 구성종목 ──────────────────────────────────────────────────
print("S&P 500 구성종목 수집 중...")
try:
    import requests
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers)
    sp500_table = pd.read_html(r.text)[0]
    tickers = sp500_table["Symbol"].str.replace(".", "-", regex=False).tolist()
except Exception:
    tickers = [
        "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B","UNH","LLY",
        "JPM","V","XOM","AVGO","PG","MA","HD","COST","MRK","ABBV",
        "CVX","KO","PEP","ADBE","WMT","BAC","CRM","TMO","ACN","MCD",
        "CSCO","ABT","NFLX","LIN","DHR","AMD","TXN","NEE","PM","ORCL",
        "AMGN","QCOM","UPS","HON","IBM","CAT","GE","SPGI","INTU","AMAT",
        "DE","GS","BLK","AXP","SYK","BKNG","GILD","MDT","ADP","VRTX",
        "ISRG","REGN","PLD","CI","CB","MMC","ZTS","MO","SO","DUK",
        "CL","TGT","CME","AON","ITW","EMR","FDX","NSC","PNC","USB",
        "WFC","C","MS","GD","RTX","LMT","NOC","BA","MMM","DOW",
        "DD","NEM","FCX","SLB","EOG","PXD","MPC","VLO","PSX","OXY"
    ]
print(f"  총 {len(tickers)}개 종목")

# ── 2. 수익률 다운로드 ────────────────────────────────────────────────────────
print("수익률 다운로드 중...")
raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)["Close"]
ret = raw.pct_change().dropna(how="all")
min_obs = int(len(ret) * 0.7)
ret = ret.loc[:, ret.count() >= min_obs]
print(f"  유효 종목: {ret.shape[1]}개, 기간: {ret.index[0].date()}~{ret.index[-1].date()}")

# ── 3. 팩터 데이터 ───────────────────────────────────────────────────────────
print("팩터 데이터 로드...")
factors = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", START, END)[0] / 100
spx = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()

common_dates = ret.index.intersection(factors.index).intersection(spx.index)
ret  = ret.reindex(common_dates)
mkt  = factors["Mkt-RF"].reindex(common_dates)
smb  = factors["SMB"].reindex(common_dates)
hml  = factors["HML"].reindex(common_dates)
rf   = factors["RF"].reindex(common_dates)
spx  = spx.reindex(common_dates).ffill()
print(f"  공통 날짜: {common_dates[0].date()}~{common_dates[-1].date()}, N={len(common_dates)}")

# ── 4. state 신호 ────────────────────────────────────────────────────────────
ma200    = spx.rolling(200).mean()
position = (spx - ma200) / ma200
intensity = np.log(spx/spx.shift(1)).rolling(30).std().mul(np.sqrt(252)).rolling(252).rank(pct=True)
regime   = (2*(position>0).astype(int) + (intensity>0.5).astype(int))
regime_labels = {0:"Bear/quiet",1:"Bear/volatile",2:"Bull/quiet",3:"Bull/volatile"}

# ── 5. 모멘텀 피처 (12-1개월) ────────────────────────────────────────────────
print("모멘텀 피처 계산...")
cum = (1 + ret.fillna(0)).cumprod()
mom_12_1 = cum.shift(21) / cum.shift(252) - 1

excess  = ret.subtract(rf, axis=0)
F_arr   = pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml}).values.astype(np.float64)
exc_arr = excess.values.astype(np.float64)
T, N    = exc_arr.shape

# ── 6. rolling β 함수 (W, ridge_alpha 파라미터) ──────────────────────────────
def compute_resid(W, ridge_alpha=0.0):
    X_full  = np.hstack([np.ones((T, 1)), F_arr])          # (T, 4)
    win_idx = np.arange(W)[None, :] + np.arange(T-W)[:, None]
    X_wins  = X_full[win_idx]                               # (T-W, W, 4)
    CHUNK   = 50

    def _chunk(col_slice):
        y       = exc_arr[:, col_slice]
        y_wins  = y[win_idx]                                # (T-W, W, c)
        nan_frac = np.isnan(y_wins).mean(axis=1)            # (T-W, c)
        c       = len(col_slice)
        beta    = np.full((T-W, 4, c), np.nan)
        ridge_I = ridge_alpha * np.eye(4)

        for ci in range(c):
            yc      = y_wins[:, :, ci]
            valid_t = nan_frac[:, ci] <= 0.3
            if valid_t.sum() == 0: continue
            Xv, yv  = X_wins[valid_t], yc[valid_t]
            n_v     = valid_t.sum()
            XtX_v   = np.zeros((n_v, 4, 4))
            Xty_v   = np.zeros((n_v, 4))
            for i, (xi, yi) in enumerate(zip(Xv, yv)):
                ok = ~np.isnan(yi)
                if ok.sum() < 20: XtX_v[i] = np.nan; continue
                XtX_v[i] = xi[ok].T @ xi[ok] + ridge_I
                Xty_v[i] = xi[ok].T @ yi[ok]
            good = ~np.isnan(XtX_v[:, 0, 0])
            if good.sum() == 0: continue
            try:    b = np.linalg.solve(XtX_v[good], Xty_v[good])
            except: b = np.linalg.lstsq(XtX_v[good], Xty_v[good], rcond=None)[0]
            beta[np.where(valid_t)[0][good], :, ci] = b

        X_today   = X_full[W:]
        exc_today = exc_arr[W:, col_slice]
        pred      = np.einsum('tf,tfc->tc', X_today, beta)
        rc        = exc_today - pred
        return np.where(~np.isnan(exc_today) & ~np.isnan(pred), rc, np.nan)

    chunks  = [np.arange(i, min(i+CHUNK, N)) for i in range(0, N, CHUNK)]
    results = Parallel(n_jobs=12, prefer="threads")(delayed(_chunk)(ch) for ch in chunks)
    arr     = np.concatenate(results, axis=1)
    df      = pd.DataFrame(arr, index=common_dates[W:], columns=excess.columns)
    return df.loc[df.notna().sum(axis=1) >= 50]

# ── 7. FM Step 1 함수 ────────────────────────────────────────────────────────
def fama_macbeth(resid):
    lam1_list, dates_fm = [], []
    for date in resid.index:
        y_cs = resid.loc[date].dropna()
        x_cs = mom_12_1.loc[date].reindex(y_cs.index).dropna()
        idx  = y_cs.index.intersection(x_cs.index)
        if len(idx) < 50: continue
        xv = stats.rankdata(x_cs[idx].values) / len(idx) - 0.5
        try:
            res = OLS(y_cs[idx].values, add_constant(xv)).fit()
            lam1_list.append(res.params[1]); dates_fm.append(date)
        except: pass
    return pd.Series(lam1_list, index=pd.DatetimeIndex(dates_fm))

# ── 8. Step 2 검정 함수 ──────────────────────────────────────────────────────
from statsmodels.regression.linear_model import OLS as smOLS

def run_step2(lam1, label):
    print(f"\n{'='*60}")
    print(f"[{label}]  λ_1,t: n={len(lam1)}, mean={lam1.mean():.5f}, std={lam1.std():.4f}")
    reg = pd.DataFrame({
        "lam1":     lam1,
        "pos_lag1": position.shift(1).reindex(lam1.index),
        "int_lag1": intensity.shift(1).reindex(lam1.index),
        "regime":   regime.reindex(lam1.index),
    }).dropna()
    reg["bxi_lag1"]  = (reg["pos_lag1"] < 0).astype(float) * reg["int_lag1"]
    reg["int2_lag1"] = reg["int_lag1"] ** 2
    reg["int_hi"]    = (reg["int_lag1"] > reg["int_lag1"].quantile(0.8)).astype(float)
    reg["bxi_hi"]    = (reg["pos_lag1"] < 0).astype(float) * reg["int_hi"]
    hac = dict(cov_type="HAC", cov_kwds={"maxlags": 10})

    def _show(title, res, vars_):
        print(f"  {title}")
        for v in vars_:
            b, t, p = res.params[v], res.tvalues[v], res.pvalues[v]
            mk = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
            print(f"    {v:14s}: b={b:+.6f}, t={t:+.3f}, p={p:.4f} {mk}")

    _show("M1: int+pos+Bear×int",
          smOLS(reg["lam1"], add_constant(reg[["int_lag1","pos_lag1","bxi_lag1"]])).fit(**hac),
          ["int_lag1","pos_lag1","bxi_lag1"])
    _show("M2: +int²",
          smOLS(reg["lam1"], add_constant(reg[["int_lag1","int2_lag1","pos_lag1","bxi_lag1"]])).fit(**hac),
          ["int_lag1","int2_lag1","pos_lag1","bxi_lag1"])
    _show("M3: Bear×HighVol dummy",
          smOLS(reg["lam1"], add_constant(reg[["int_lag1","pos_lag1","int_hi","bxi_hi"]])).fit(**hac),
          ["int_lag1","pos_lag1","int_hi","bxi_hi"])

    print("  [체제별 λ_1,t]")
    base = reg.loc[reg["regime"]==0, "lam1"].values
    for r_id, rlabel in regime_labels.items():
        grp = reg.loc[reg["regime"]==r_id, "lam1"].values
        if len(grp) < 30: continue
        ks, p_ks = stats.ks_2samp(base, grp) if r_id != 0 else (0, 1)
        _, p_t   = stats.ttest_ind(grp, base, equal_var=False) if r_id != 0 else (0, 1)
        print(f"    {rlabel:15s}: n={len(grp):4d}, mean={grp.mean():+.6f}, "
              f"std={grp.std():.5f}, p5={np.percentile(grp,5):+.6f}, p95={np.percentile(grp,95):+.6f}"
              + (f", KS p={p_ks:.4f}, t p={p_t:.4f}" if r_id != 0 else ""))

    groups  = [reg.loc[reg["regime"]==r_id,"lam1"].values for r_id in range(4)
               if (reg["regime"]==r_id).sum()>=30]
    f_stat, p_anova = stats.f_oneway(*groups)
    kw, p_kw        = stats.kruskal(*groups)
    print(f"  ANOVA F={f_stat:.3f} p={p_anova:.4f} | KW H={kw:.3f} p={p_kw:.4f}")

# ── 9. 4가지 조합 실행 ───────────────────────────────────────────────────────
CONFIGS = [
    (60,  0.0,    "W=60  OLS"),
    (120, 0.0,    "W=120 OLS"),
    (60,  1e-4,   "W=60  Ridge(α=1e-4)"),
    (120, 1e-4,   "W=120 Ridge(α=1e-4)"),
]

for W, alpha, label in CONFIGS:
    print(f"\n>>> rolling β: {label}")
    resid = compute_resid(W, alpha)
    print(f"    잔차: {resid.shape}")
    lam1  = fama_macbeth(resid)
    run_step2(lam1, label)

print("\n\n=== 완료 ===")


import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings("ignore")

START, END = "2005-01-01", "2024-12-31"  # 2005~: 종목 수 충분
ROLL_BETA = 60
np.random.seed(42)

# ── 1. S&P 500 현재 구성종목 (생존편향 있음, 탐색용) ─────────────────────────
print("S&P 500 구성종목 수집 중...")
# yfinance로 S&P 500 구성종목 티커 수집
try:
    import requests
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers)
    sp500_table = pd.read_html(r.text)[0]
    tickers = sp500_table["Symbol"].str.replace(".", "-", regex=False).tolist()
except Exception:
    # fallback: 대형주 위주 수동 리스트 (100개)
    tickers = [
        "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B","UNH","LLY",
        "JPM","V","XOM","AVGO","PG","MA","HD","COST","MRK","ABBV",
        "CVX","KO","PEP","ADBE","WMT","BAC","CRM","TMO","ACN","MCD",
        "CSCO","ABT","NFLX","LIN","DHR","AMD","TXN","NEE","PM","ORCL",
        "AMGN","QCOM","UPS","HON","IBM","CAT","GE","SPGI","INTU","AMAT",
        "DE","GS","BLK","AXP","SYK","BKNG","GILD","MDT","ADP","VRTX",
        "ISRG","REGN","PLD","CI","CB","MMC","ZTS","MO","SO","DUK",
        "CL","TGT","CME","AON","ITW","EMR","FDX","NSC","PNC","USB",
        "WFC","C","MS","GD","RTX","LMT","NOC","BA","MMM","DOW",
        "DD","NEM","FCX","SLB","EOG","PXD","MPC","VLO","PSX","OXY"
    ]
print(f"  총 {len(tickers)}개 종목")

# ── 2. 일별 수익률 다운로드 ──────────────────────────────────────────────────
print("수익률 다운로드 중 (시간 소요)...")
raw = yf.download(tickers, start=START, end=END,
                  auto_adjust=True, progress=False)["Close"]
ret = raw.pct_change().dropna(how="all")
# 데이터 충분한 종목만 (전체 기간 70% 이상)
min_obs = int(len(ret) * 0.7)
ret = ret.loc[:, ret.count() >= min_obs]
print(f"  유효 종목: {ret.shape[1]}개, 기간: {ret.index[0].date()}~{ret.index[-1].date()}")

# ── 3. 팩터 데이터 ───────────────────────────────────────────────────────────
print("팩터 데이터 로드...")
factors = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", START, END)[0] / 100
spx = yf.download("^GSPC", start=START, end=END,
                  auto_adjust=True, progress=False)["Close"].squeeze()

common_dates = ret.index.intersection(factors.index).intersection(spx.index)
ret     = ret.reindex(common_dates)
mkt     = factors["Mkt-RF"].reindex(common_dates)
smb     = factors["SMB"].reindex(common_dates)
hml     = factors["HML"].reindex(common_dates)
rf      = factors["RF"].reindex(common_dates)
spx     = spx.reindex(common_dates).ffill()
print(f"  공통 날짜: {common_dates[0].date()}~{common_dates[-1].date()}, N={len(common_dates)}")

# ── 4. state 신호 ────────────────────────────────────────────────────────────
ma200     = spx.rolling(200).mean()
position  = (spx - ma200) / ma200
intensity = np.log(spx/spx.shift(1)).rolling(30).std().mul(np.sqrt(252)).rolling(252).rank(pct=True)
bear      = (position < 0).astype(float)
regime    = (2*(position>0).astype(int) + (intensity>0.5).astype(int))
regime_labels = {0:"Bear/quiet",1:"Bear/volatile",2:"Bull/quiet",3:"Bull/volatile"}

# ── 5. 모멘텀 피처 (12-1개월) ────────────────────────────────────────────────
print("모멘텀 피처 계산...")
cum = (1 + ret.fillna(0)).cumprod()
mom_12_1 = cum.shift(21) / cum.shift(252) - 1

# ── 6. rolling β → FF3 잔차 (masked rolling OLS + 병렬) ────────────────────
print("rolling β 추정 중 (masked + parallel)...")
from joblib import Parallel, delayed

excess = ret.subtract(rf, axis=0)           # (T, N)
F = pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml})

W = ROLL_BETA
T, N = excess.shape
X_full = np.hstack([np.ones((T, 1)), F.values.astype(np.float64)])  # (T, 4)
exc_arr = excess.values.astype(np.float64)  # (T, N)

# rolling window 인덱스: (T-W, W)
win_idx = np.arange(W)[None, :] + np.arange(T - W)[:, None]  # (T-W, W)
X_wins = X_full[win_idx]   # (T-W, W, 4)  — 팩터는 NaN 없음

# 종목 청크별로 masked XtX/Xty 계산 → β → 잔차
CHUNK = 50  # 청크 크기 (메모리 vs 병렬 균형)

def _beta_chunk(col_slice):
    """col_slice: 종목 인덱스 배열. 반환: resid_chunk (T-W, len(col_slice))"""
    y = exc_arr[:, col_slice]               # (T, c)
    y_wins = y[win_idx]                     # (T-W, W, c)
    nan_mask = np.isnan(y_wins)             # (T-W, W, c)  True=결측
    nan_frac = nan_mask.mean(axis=1)        # (T-W, c)

    c = len(col_slice)
    beta = np.full((T - W, 4, c), np.nan)

    # 종목별로 masked OLS (팩터는 NaN 없으므로 X_wins 공유)
    for ci in range(c):
        yc = y_wins[:, :, ci]               # (T-W, W)
        valid_t = nan_frac[:, ci] <= 0.3    # (T-W,)
        if valid_t.sum() == 0:
            continue
        # 유효한 t에 대해서만 계산
        Xv = X_wins[valid_t]                # (n_valid, W, 4)
        yv = yc[valid_t]                    # (n_valid, W)
        # 각 t마다 NaN 행 제거 후 OLS
        XtX_v = np.zeros((valid_t.sum(), 4, 4))
        Xty_v = np.zeros((valid_t.sum(), 4))
        for i, (xi, yi) in enumerate(zip(Xv, yv)):
            ok = ~np.isnan(yi)
            if ok.sum() < 20:
                XtX_v[i] = np.nan
                continue
            XtX_v[i] = xi[ok].T @ xi[ok]
            Xty_v[i] = xi[ok].T @ yi[ok]
        # 배치 solve
        good = ~np.isnan(XtX_v[:, 0, 0])
        if good.sum() == 0:
            continue
        try:
            b = np.linalg.solve(XtX_v[good], Xty_v[good])  # (n_good, 4)
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(XtX_v[good], Xty_v[good], rcond=None)[0]
        t_idx_valid = np.where(valid_t)[0][good]
        beta[t_idx_valid, :, ci] = b

    # 잔차 = excess_today - X_today @ β
    X_today = X_full[W:]                    # (T-W, 4)
    exc_today = exc_arr[W:, col_slice]      # (T-W, c)
    pred = np.einsum('tf,tfc->tc', X_today, beta)  # (T-W, c)
    resid_chunk = exc_today - pred
    # 오늘 NaN이거나 β 추정 실패면 NaN
    resid_chunk = np.where(~np.isnan(exc_today) & ~np.isnan(pred), resid_chunk, np.nan)
    return resid_chunk

col_chunks = [np.arange(i, min(i + CHUNK, N)) for i in range(0, N, CHUNK)]
results = Parallel(n_jobs=12, prefer="threads")(
    delayed(_beta_chunk)(ch) for ch in col_chunks
)

resid_arr = np.concatenate(results, axis=1)  # (T-W, N)
resid = pd.DataFrame(resid_arr, index=common_dates[W:], columns=excess.columns)
resid = resid.loc[resid.notna().sum(axis=1) >= 50]
print(f"잔차 계산 완료: {resid.shape}")

# ── 7. Fama-MacBeth Step 1 ───────────────────────────────────────────────────
print("Fama-MacBeth Step 1...")
lam1_list, dates_fm = [], []

for date in resid.index:
    y_cs = resid.loc[date].dropna()
    x_cs = mom_12_1.loc[date].reindex(y_cs.index).dropna()
    idx = y_cs.index.intersection(x_cs.index)
    if len(idx) < 50:  # 최소 50개 종목
        continue
    xv = stats.rankdata(x_cs[idx].values) / len(idx) - 0.5
    try:
        res = OLS(y_cs[idx].values, add_constant(xv)).fit()
        lam1_list.append(res.params[1])
        dates_fm.append(date)
    except:
        pass

lam1 = pd.Series(lam1_list, index=pd.DatetimeIndex(dates_fm))
print(f"λ_1,t: n={len(lam1)}, mean={lam1.mean():.5f}, std={lam1.std():.4f}")

# ── 8. H1-Vol 검정 ───────────────────────────────────────────────────────────
from statsmodels.regression.linear_model import OLS as smOLS

reg = pd.DataFrame({
    "lam1":      lam1,
    "pos_lag1":  position.shift(1).reindex(lam1.index),
    "int_lag1":  intensity.shift(1).reindex(lam1.index),
    "regime":    regime.reindex(lam1.index),
}).dropna()
reg["bxi_lag1"]  = (reg["pos_lag1"] < 0).astype(float) * reg["int_lag1"]
reg["int2_lag1"] = reg["int_lag1"] ** 2
reg["int_hi"]    = (reg["int_lag1"] > reg["int_lag1"].quantile(0.8)).astype(float)  # 상위 20% 더미
reg["bxi_hi"]    = (reg["pos_lag1"] < 0).astype(float) * reg["int_hi"]

print(f"\nStep 2 샘플: {len(reg)}일")

def _show(label, res, vars_):
    print(f"\n[{label}]")
    for v in vars_:
        b, t, p = res.params[v], res.tvalues[v], res.pvalues[v]
        mk = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        print(f"  {v:14s}: b={b:+.6f}, t={t:+.3f}, p={p:.4f} {mk}")

hac = dict(cov_type="HAC", cov_kwds={"maxlags": 10})

# M1: 선형 (기존)
r1 = smOLS(reg["lam1"], add_constant(reg[["int_lag1","pos_lag1","bxi_lag1"]])).fit(**hac)
_show("M1: linear  int+pos+Bear×int", r1, ["int_lag1","pos_lag1","bxi_lag1"])

# M2: 비선형 — intensity²
r2 = smOLS(reg["lam1"], add_constant(reg[["int_lag1","int2_lag1","pos_lag1","bxi_lag1"]])).fit(**hac)
_show("M2: +int²", r2, ["int_lag1","int2_lag1","pos_lag1","bxi_lag1"])

# M3: 꼬리 더미 — Bear × (intensity > 80th pct)
r3 = smOLS(reg["lam1"], add_constant(reg[["int_lag1","pos_lag1","int_hi","bxi_hi"]])).fit(**hac)
_show("M3: Bear×HighVol dummy (top-20%)", r3, ["int_lag1","pos_lag1","int_hi","bxi_hi"])

# 체제별 분포
print("\n[체제별 λ_1,t]")
base = reg.loc[reg["regime"]==0, "lam1"].values
for r_id, label in regime_labels.items():
    grp = reg.loc[reg["regime"]==r_id, "lam1"].values
    if len(grp) < 30: continue
    ks, p_ks = stats.ks_2samp(base, grp) if r_id != 0 else (0, 1)
    t_s, p_t = stats.ttest_ind(grp, base, equal_var=False) if r_id != 0 else (0, 1)
    print(f"  {label:15s}: n={len(grp):4d}, mean={grp.mean():+.6f}, "
          f"std={grp.std():.5f}, p10={np.percentile(grp,10):+.6f}"
          + (f", KS p={p_ks:.4f}, t-test p={p_t:.4f}" if r_id != 0 else ""))

# ANOVA
groups = [reg.loc[reg["regime"]==r_id,"lam1"].values
          for r_id in range(4) if (reg["regime"]==r_id).sum()>=30]
f_stat, p_anova = stats.f_oneway(*groups)
kw, p_kw = stats.kruskal(*groups)
print(f"\nANOVA: F={f_stat:.3f}, p={p_anova:.4f}")
print(f"Kruskal-Wallis: H={kw:.3f}, p={p_kw:.4f}")

# λ₁ 분산의 state 의존성 (H3 동기)
print("\n[체제별 λ_1,t 분산 — H3 동기]")
for r_id, label in regime_labels.items():
    grp = reg.loc[reg["regime"]==r_id, "lam1"].values
    if len(grp) < 30: continue
    print(f"  {label:15s}: std={grp.std():.6f}, IQR={np.percentile(grp,75)-np.percentile(grp,25):.6f}, "
          f"p5={np.percentile(grp,5):+.6f}, p95={np.percentile(grp,95):+.6f}")

print("\n완료.")
