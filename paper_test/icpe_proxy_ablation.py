"""
IC-PE proxy ablation

Purpose:
  Fast, paper-specific proxy runner for the new index-conditioned PE thesis.

Main comparisons:
  - static features only
  - concat conditioning
  - IC-PE-style modulated features

Diagnostics:
  - CRPS / IC / PI-80
  - simple state-swap sensitivity for the IC-PE proxy

Notes:
  - This is a proxy experiment, not the final neural model.
  - It uses linear quantile regression to test whether the conditioning story is
    directionally alive before heavier model work.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools import add_constant
from statsmodels.tools.sm_exceptions import IterationLimitWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=IterationLimitWarning)


TAUS = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IC-PE proxy ablation")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--index-symbol", default="^GSPC")
    parser.add_argument("--target", choices=("residual", "raw"), default="residual")
    parser.add_argument("--roll-beta", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--lags", type=int, default=20)
    parser.add_argument("--train-win", type=int, default=252)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default="paper/index_conditioned_pe/results",
        help="Directory for csv outputs",
    )
    return parser.parse_args()


def pinball(y: float, q: float, tau: float) -> float:
    err = y - q
    return float(tau * err if err >= 0 else (tau - 1.0) * err)


def crps_from_quantiles(y: float, qs: list[float]) -> float:
    return float(np.mean([pinball(y, q, tau) for q, tau in zip(qs, TAUS)]))


def pi80_hit(y: float, q_lo: float, q_hi: float) -> float:
    return float(q_lo <= y <= q_hi)


def safe_ic(group: pd.DataFrame, pred_col: str) -> float:
    corr = stats.spearmanr(group[pred_col], group["y_true"])[0]
    return float(corr) if not np.isnan(corr) else np.nan


def load_data(start: str, end: str, index_symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    print(f"[load] Ken French 25 portfolios + FF3 + {index_symbol}", flush=True)
    port25 = web.DataReader("25_Portfolios_5x5_daily", "famafrench", start, end)[0] / 100
    factors = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start, end)[0] / 100
    index_close = yf.download(index_symbol, start=start, end=end, auto_adjust=True, progress=False)["Close"].squeeze()

    common = port25.index.intersection(factors.index).intersection(index_close.index)
    port25 = port25.reindex(common)
    factors = factors.reindex(common)
    index_close = index_close.reindex(common).ffill()

    print(f"[load] {common[0].date()} ~ {common[-1].date()} | T={len(common)}", flush=True)
    return port25, factors, index_close


def build_target(
    port25: pd.DataFrame,
    factors: pd.DataFrame,
    target_kind: str,
    roll_beta: int,
    horizon: int,
) -> pd.DataFrame:
    rf = factors["RF"]
    excess = port25.subtract(rf, axis=0)

    if target_kind == "raw":
        return excess.rolling(horizon).sum().shift(-horizon)

    print(f"[target] rolling FF3 residual target | beta window={roll_beta}", flush=True)
    resid = pd.DataFrame(index=port25.index, columns=port25.columns, dtype=float)
    factor_cols = pd.DataFrame(
        {
            "MKT": factors["Mkt-RF"],
            "SMB": factors["SMB"],
            "HML": factors["HML"],
        },
        index=port25.index,
    )

    for col in port25.columns:
        y = excess[col]
        fitted = pd.Series(index=port25.index, dtype=float)
        for end_idx in range(roll_beta, len(port25.index)):
            sl = slice(end_idx - roll_beta, end_idx)
            x_win = add_constant(factor_cols.iloc[sl])
            y_win = y.iloc[sl]
            if y_win.isna().any() or x_win.isna().any().any():
                continue
            fit = OLS(y_win, x_win).fit()
            x_t = add_constant(factor_cols.iloc[[end_idx]], has_constant="add")
            fitted.iloc[end_idx] = fit.predict(x_t).iloc[0]
        resid[col] = y - fitted

    return resid.rolling(horizon).sum().shift(-horizon)


def build_state(index_close: pd.Series) -> pd.DataFrame:
    ma200 = index_close.rolling(200).mean()
    position = (index_close - ma200) / ma200
    rv30 = np.log(index_close / index_close.shift(1)).rolling(30).std() * np.sqrt(252)
    intensity = rv30.rolling(252).rank(pct=True)

    regime = 2 * (position > 0).astype(int) + (intensity > 0.5).astype(int)
    return pd.DataFrame(
        {
            "position": position,
            "intensity": intensity,
            "regime": regime,
        },
        index=index_close.index,
    )


def build_base_features(values: np.ndarray, lags: int) -> np.ndarray:
    t_size, n_assets = values.shape
    feats = np.full((t_size, n_assets * 2), np.nan)
    for idx in range(lags, t_size):
        window = values[idx - lags:idx]
        feats[idx, :n_assets] = np.nanmean(window, axis=0)
        feats[idx, n_assets:] = np.nanstd(window, axis=0)
    return feats


def fit_quantiles(X: np.ndarray, y: np.ndarray, x_test: np.ndarray) -> list[float] | None:
    quantiles: list[float] = []
    for tau in TAUS:
        try:
            fitted = QuantReg(y, X).fit(q=tau, max_iter=300)
            quantiles.append(float(fitted.predict(x_test)[0]))
        except Exception:
            return None
    return quantiles


def run_proxy(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    port25, factors, index_close = load_data(args.start, args.end, args.index_symbol)
    target = build_target(port25, factors, args.target, args.roll_beta, args.horizon)
    state = build_state(index_close)

    cols = list(port25.columns)
    t_size = len(port25.index)
    n_assets = len(cols)
    base = build_base_features(target.values, args.lags)

    records: list[dict[str, float | int | str | pd.Timestamp]] = []
    test_range = range(args.train_win + args.lags, t_size - args.horizon, args.step)
    print(f"[walk] {len(list(test_range))} test dates | assets={n_assets}", flush=True)

    for step_idx, t in enumerate(test_range, start=1):
        date = port25.index[t]
        tr = slice(t - args.train_win, t)
        pos_t = float(state["position"].iloc[t - 1]) if not pd.isna(state["position"].iloc[t - 1]) else 0.0
        int_t = float(state["intensity"].iloc[t - 1]) if not pd.isna(state["intensity"].iloc[t - 1]) else 0.0
        reg_t = int(state["regime"].iloc[t - 1]) if not pd.isna(state["regime"].iloc[t - 1]) else -1

        for asset_idx, col in enumerate(cols):
            y_true = target[col].iloc[t]
            if pd.isna(y_true):
                continue

            y_train = target[col].iloc[tr].values
            feat_train = base[tr, :]
            x2 = feat_train[:, [asset_idx, asset_idx + n_assets]]
            valid = ~(np.isnan(y_train) | np.isnan(x2).any(axis=1))
            if valid.sum() < 50:
                continue

            yv = y_train[valid]
            pos_tr = state["position"].iloc[t - args.train_win:t].values[valid]
            int_tr = state["intensity"].iloc[t - args.train_win:t].values[valid]

            xs = add_constant(x2[valid])
            xc = add_constant(np.column_stack([x2[valid], pos_tr, int_tr]))

            mod_tr = 1.0 + int_tr
            xp = add_constant(
                np.column_stack(
                    [
                        x2[valid, 0] * mod_tr,
                        x2[valid, 1] * mod_tr,
                        pos_tr,
                        int_tr,
                    ]
                )
            )

            feat_now_mean = base[t, asset_idx]
            feat_now_std = base[t, asset_idx + n_assets]
            if np.isnan([feat_now_mean, feat_now_std]).any():
                continue

            xs_test = np.array([[1.0, feat_now_mean, feat_now_std]])
            xc_test = np.array([[1.0, feat_now_mean, feat_now_std, pos_t, int_t]])
            xp_test = np.array([[1.0, feat_now_mean * (1.0 + int_t), feat_now_std * (1.0 + int_t), pos_t, int_t]])

            q_static = fit_quantiles(xs, yv, xs_test)
            q_concat = fit_quantiles(xc, yv, xc_test)
            q_icpe = fit_quantiles(xp, yv, xp_test)
            if q_static is None or q_concat is None or q_icpe is None:
                continue

            swap_pos = -pos_t
            swap_int = 1.0 - int_t
            xp_swap = np.array(
                [[1.0, feat_now_mean * (1.0 + swap_int), feat_now_std * (1.0 + swap_int), swap_pos, swap_int]]
            )
            q_icpe_swap = fit_quantiles(xp, yv, xp_swap)
            if q_icpe_swap is None:
                continue

            records.append(
                {
                    "date": date,
                    "asset": col,
                    "y_true": float(y_true),
                    "regime": reg_t,
                    "position": pos_t,
                    "intensity": int_t,
                    "crps_static": crps_from_quantiles(y_true, q_static),
                    "crps_concat": crps_from_quantiles(y_true, q_concat),
                    "crps_icpe": crps_from_quantiles(y_true, q_icpe),
                    "pi80_static": pi80_hit(y_true, q_static[0], q_static[-1]),
                    "pi80_concat": pi80_hit(y_true, q_concat[0], q_concat[-1]),
                    "pi80_icpe": pi80_hit(y_true, q_icpe[0], q_icpe[-1]),
                    "pred_static": q_static[2],
                    "pred_concat": q_concat[2],
                    "pred_icpe": q_icpe[2],
                    "pred_icpe_swap": q_icpe_swap[2],
                    "state_swap_delta": abs(q_icpe[2] - q_icpe_swap[2]),
                }
            )

        if step_idx % 20 == 0:
            print(f"[walk] {date.date()} | predictions={len(records)}", flush=True)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No predictions generated.")

    ic_rows = []
    for variant in ("static", "concat", "icpe"):
        ic_series = (
            df.groupby("date")
            .apply(lambda grp: safe_ic(grp, f"pred_{variant}"), include_groups=False)
            .dropna()
        )
        ic_mean = float(ic_series.mean()) if len(ic_series) else np.nan
        icir = float(ic_mean / ic_series.std()) if len(ic_series) and ic_series.std() not in (0.0, np.nan) else np.nan
        ic_rows.append({"variant": variant, "ic": ic_mean, "icir": icir})

    summary = pd.DataFrame(ic_rows)
    summary["mean_crps"] = [df["crps_static"].mean(), df["crps_concat"].mean(), df["crps_icpe"].mean()]
    summary["pi80"] = [df["pi80_static"].mean(), df["pi80_concat"].mean(), df["pi80_icpe"].mean()]
    return df, summary


def print_summary(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("\n[overall summary]", flush=True)
    print(summary.to_string(index=False), flush=True)

    diff_s = df["crps_static"].values - df["crps_icpe"].values
    diff_c = df["crps_concat"].values - df["crps_icpe"].values
    dm_s = diff_s.mean() / (diff_s.std() / np.sqrt(len(diff_s)))
    dm_c = diff_c.mean() / (diff_c.std() / np.sqrt(len(diff_c)))
    p_s = 2.0 * (1.0 - stats.norm.cdf(abs(dm_s)))
    p_c = 2.0 * (1.0 - stats.norm.cdf(abs(dm_c)))

    def dm_label(t_stat: float, p_val: float, left: str, right: str) -> str:
        if p_val >= 0.05:
            return "not significant"
        if t_stat > 0:
            return f"{right} wins"
        return f"{left} wins"

    print(
        f"\n[dm] static vs icpe: t={dm_s:+.3f}, p={p_s:.4f} | "
        f"{dm_label(dm_s, p_s, 'static', 'IC-PE')}",
        flush=True,
    )
    print(
        f"[dm] concat vs icpe: t={dm_c:+.3f}, p={p_c:.4f} | "
        f"{dm_label(dm_c, p_c, 'concat', 'IC-PE')}",
        flush=True,
    )

    print(
        f"\n[diagnostic] mean state-swap delta={df['state_swap_delta'].mean():.6f} | "
        f"median={df['state_swap_delta'].median():.6f}",
        flush=True,
    )

    regime_labels = {0: "Bear/quiet", 1: "Bear/volatile", 2: "Bull/quiet", 3: "Bull/volatile"}
    print("\n[regime summary]", flush=True)
    for reg_id, label in regime_labels.items():
        sub = df[df["regime"] == reg_id]
        if len(sub) < 30:
            continue
        print(
            f"{label:15s} n={len(sub):5d} | "
            f"crps_static={sub['crps_static'].mean():.6f} | "
            f"crps_concat={sub['crps_concat'].mean():.6f} | "
            f"crps_icpe={sub['crps_icpe'].mean():.6f} | "
            f"swap_delta={sub['state_swap_delta'].mean():.6f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, summary = run_proxy(args)
    print_summary(df, summary)

    stem = f"icpe_proxy_{args.index_symbol.replace('^', '').lower()}_{args.target}"
    df_path = out_dir / f"{stem}_predictions.csv"
    summary_path = out_dir / f"{stem}_summary.csv"
    df.to_csv(df_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\n[save] {df_path}", flush=True)
    print(f"[save] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
