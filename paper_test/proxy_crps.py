"""
Proxy: State-blind vs State-aware CRPS (Ken French 25 portfolios)
- 벡터화: numpy pinball loss + closed-form quantile (linear QR via gradient)
- Walk-forward: train=252, step=5 (속도)
- 체제별 CRPS + DM test
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

START, END  = "2000-01-01", "2024-12-31"
LAGS        = 20
HORIZON     = 5
TRAIN_WIN   = 252
STEP        = 5          # 5일마다 예측 (속도)
TAUS        = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

# ── 1. 데이터 ─────────────────────────────────────────────────────────────────
print("데이터 로드...")
port25  = web.DataReader("25_Portfolios_5x5_daily", "famafrench", START, END)[0] / 100
factors = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", START, END)[0] / 100
spx     = yf.download("^GSPC", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()

common  = port25.index.intersection(factors.index).intersection(spx.index)
port25  = port25.reindex(common)
mkt     = factors["Mkt-RF"].reindex(common)
smb     = factors["SMB"].reindex(common)
hml     = factors["HML"].reindex(common)
rf      = factors["RF"].reindex(common)
spx     = spx.reindex(common).ffill()
print(f"  {common[0].date()}~{common[-1].date()}, T={len(common)}")

# ── 2. FF3 잔차 ───────────────────────────────────────────────────────────────
print("FF3 잔차...")
excess = port25.subtract(rf, axis=0)
F_full = add_constant(pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml}))
resid  = pd.DataFrame(index=common, columns=port25.columns, dtype=float)
for col in port25.columns:
    resid[col] = OLS(excess[col], F_full).fit().resid

# 5일 누적 타겟 (미래)
target = resid.rolling(HORIZON).sum().shift(-HORIZON)

# ── 3. state ──────────────────────────────────────────────────────────────────
ma200     = spx.rolling(200).mean()
position  = (spx - ma200) / ma200
rv30      = np.log(spx / spx.shift(1)).rolling(30).std() * np.sqrt(252)
intensity = rv30.rolling(252).rank(pct=True)
bear      = (position < 0).astype(float)
hi_vol    = (intensity > 0.5).astype(float)
regime    = (2*(position>0).astype(int) + (intensity>0.5).astype(int))
regime_labels = {0:"Bear/quiet",1:"Bear/volatile",2:"Bull/quiet",3:"Bull/volatile"}

# ── 4. 피처 행렬 사전 계산 ────────────────────────────────────────────────────
# 각 시점 t에서: [mean(resid[t-L:t]), std(resid[t-L:t])] per portfolio
# shape: (T, 25*2) for blind, (T, 25*2+2) for aware
print("피처 계산...")
cols = list(port25.columns)
N    = len(cols)
T    = len(common)

# rolling mean/std per portfolio: (T, N*2)
r_arr = resid.values  # (T, N)
feat_blind = np.full((T, N * 2), np.nan)
for i in range(LAGS, T):
    w = r_arr[i-LAGS:i]  # (LAGS, N)
    feat_blind[i, :N]  = np.nanmean(w, axis=0)
    feat_blind[i, N:]  = np.nanstd(w, axis=0)

bear_arr  = bear.values
hivol_arr = hi_vol.values

# ── 5. Walk-forward ───────────────────────────────────────────────────────────
print("Walk-forward 예측...")

def pinball(y, q, tau):
    e = y - q
    return np.where(e >= 0, tau * e, (tau - 1) * e)

def crps_approx(y, quantiles, taus):
    """quantiles: (n_tau,), y: scalar"""
    return np.mean([pinball(y, q, tau) for q, tau in zip(quantiles, taus)])

records = []
test_indices = range(TRAIN_WIN + LAGS, T - HORIZON, STEP)

for t in test_indices:
    date = common[t]
    tr   = slice(t - TRAIN_WIN, t)

    # state at t-1
    b_t   = bear_arr[t-1]
    hv_t  = hivol_arr[t-1]
    reg_t = int(regime.iloc[t-1]) if not pd.isna(regime.iloc[t-1]) else -1

    for ci, col in enumerate(cols):
        y_true = target[col].iloc[t]
        if pd.isna(y_true): continue

        # 학습 데이터
        y_tr = target[col].iloc[tr].values
        fb   = feat_blind[tr, :]  # (TRAIN_WIN, N*2)

        # 해당 포트폴리오 피처만 (mean, std)
        x_col_tr = fb[:, [ci, ci+N]]  # (TRAIN_WIN, 2)
        valid    = ~(np.isnan(y_tr) | np.isnan(x_col_tr).any(axis=1))
        if valid.sum() < 50: continue

        Xb = add_constant(x_col_tr[valid])
        Xa = add_constant(np.column_stack([
            x_col_tr[valid],
            bear_arr[t-TRAIN_WIN:t][valid],
            hivol_arr[t-TRAIN_WIN:t][valid]
        ]))
        yv = y_tr[valid]

        # 테스트 피처
        xb_test = np.array([[1.0, feat_blind[t, ci], feat_blind[t, ci+N]]])
        xa_test = np.array([[1.0, feat_blind[t, ci], feat_blind[t, ci+N], b_t, hv_t]])

        if np.isnan(xb_test).any() or np.isnan(xa_test).any(): continue

        q_blind, q_aware = [], []
        ok = True
        for tau in TAUS:
            try:
                qb = QuantReg(yv, Xb).fit(q=tau, max_iter=300).predict(xb_test)[0]
                qa = QuantReg(yv, Xa).fit(q=tau, max_iter=300).predict(xa_test)[0]
                q_blind.append(qb); q_aware.append(qa)
            except:
                ok = False; break
        if not ok: continue

        cb = crps_approx(y_true, q_blind, TAUS)
        ca = crps_approx(y_true, q_aware, TAUS)
        records.append({"date": date, "port": col, "y_true": y_true,
                        "crps_blind": cb, "crps_aware": ca, "regime": reg_t})

    if (t // STEP) % 50 == 0:
        print(f"  {date.date()}, n={len(records)}")

df = pd.DataFrame(records)
print(f"\n총 예측: {len(df)}")

# ── 6. 결과 ───────────────────────────────────────────────────────────────────
cb_all = df["crps_blind"].mean()
ca_all = df["crps_aware"].mean()
print(f"\n[전체 CRPS]")
print(f"  blind : {cb_all:.6f}")
print(f"  aware : {ca_all:.6f}")
print(f"  개선  : {(cb_all-ca_all)/cb_all*100:+.2f}%")

diff = df["crps_blind"].values - df["crps_aware"].values
dm_t = diff.mean() / (diff.std() / np.sqrt(len(diff)))
dm_p = 2*(1 - stats.norm.cdf(abs(dm_t)))
print(f"  DM    : t={dm_t:+.3f}, p={dm_p:.4f}")

print(f"\n[체제별 CRPS]")
for r_id, label in regime_labels.items():
    sub = df[df["regime"]==r_id]
    if len(sub) < 30: continue
    cb = sub["crps_blind"].mean(); ca = sub["crps_aware"].mean()
    d  = sub["crps_blind"].values - sub["crps_aware"].values
    t_s = d.mean() / (d.std() / np.sqrt(len(d)))
    p_s = 2*(1 - stats.norm.cdf(abs(t_s)))
    print(f"  {label:15s}: n={len(sub):5d}, blind={cb:.6f}, aware={ca:.6f}, "
          f"개선={( cb-ca)/cb*100:+.2f}%, DM p={p_s:.4f}")

df.to_csv("proxy_crps_results.csv", index=False)
print("\n완료. → proxy_crps_results.csv")
