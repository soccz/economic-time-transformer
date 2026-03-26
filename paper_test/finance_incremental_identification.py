from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from scipy import stats
import statsmodels.api as sm
import yfinance as yf


MODEL_SPECS: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        ("model0", []),
        ("model1", ["bear", "intensity"]),
        ("model2", ["bear", "intensity", "bear_x_intensity"]),
        ("model3", ["position", "intensity", "position_x_intensity"]),
        ("model4", ["position", "intensity", "position_x_intensity", "indexret", "intensity_x_indexret"]),
        ("model2_vix", ["bear", "vix_proxy", "bear_x_vix"]),
    ]
)


MODEL_LABELS = {
    "model0": "Model 0",
    "model1": "Model 1",
    "model2": "Model 2",
    "model3": "Model 3",
    "model4": "Model 4",
    "model2_vix": "Model 2V",
}

INDEXRET_SPECS = ("ret1", "ret5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="First-pass finance incremental-identification table for future WML returns"
    )
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--eval-start", default="2020-01-01")
    parser.add_argument("--eval-end", default="2024-12-31")
    parser.add_argument("--index-symbol", default="^GSPC")
    parser.add_argument("--horizon", type=int, default=1, help="1 = WML_t+1, k>1 = k-day summed future WML")
    parser.add_argument(
        "--indexret-spec",
        choices=INDEXRET_SPECS,
        default="ret1",
        help="ret1 = 1-day log index return, ret5 = 5-day mean log index return",
    )
    parser.add_argument("--hac-lag", type=int, default=10)
    parser.add_argument(
        "--oos-hac-lag",
        type=int,
        default=None,
        help="Newey-West lag for OOS daily loss-difference tests; default=max(4, horizon-1)",
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=0,
        help="0 means expanding window; otherwise use a rolling window of this many observations",
    )
    parser.add_argument("--min-train", type=int, default=504)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        default="paper/economic_time/results/finance_incremental_identification",
    )
    return parser.parse_args()


def _extract_close(frame: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(frame, pd.Series):
        return frame.rename("Close")
    if isinstance(frame.columns, pd.MultiIndex):
        if ("Close", "") in frame.columns:
            return frame[("Close", "")].rename("Close")
        close_cols = [col for col in frame.columns if col[0] == "Close"]
        if close_cols:
            return frame[close_cols[0]].rename("Close")
    if "Close" in frame.columns:
        return frame["Close"].rename("Close")
    return frame.iloc[:, 0].rename("Close")


def load_data(start: str, end: str, index_symbol: str) -> tuple[pd.Series, pd.Series]:
    wml_df = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", start, end)[0] / 100.0
    wml_col = [col for col in wml_df.columns if "Mom" in col or "mom" in col][0]
    wml = wml_df[wml_col].rename("wml")

    index_raw = yf.download(index_symbol, start=start, end=end, auto_adjust=True, progress=False)
    index_close = _extract_close(index_raw).rename("index_close")

    common = wml.index.intersection(index_close.index)
    wml = wml.reindex(common).ffill()
    index_close = index_close.reindex(common).ffill()
    return wml, index_close


def build_target(wml: pd.Series, horizon: int) -> pd.Series:
    if horizon <= 1:
        return wml.shift(-1).rename("target")
    return wml.rolling(horizon).sum().shift(-horizon).rename("target")


def build_indexret(log_ret: pd.Series, indexret_spec: str) -> pd.Series:
    if indexret_spec == "ret1":
        return log_ret.rename("indexret")
    if indexret_spec == "ret5":
        return log_ret.rolling(5).mean().rename("indexret")
    raise ValueError(f"Unsupported indexret_spec: {indexret_spec}")


def build_proxy_frame(
    wml: pd.Series,
    index_close: pd.Series,
    horizon: int,
    indexret_spec: str,
) -> pd.DataFrame:
    log_ret = np.log(index_close / index_close.shift(1))
    ma200 = index_close.rolling(200).mean()
    position = (index_close - ma200) / ma200
    rv30 = log_ret.rolling(30).std() * np.sqrt(252.0)
    intensity = rv30.rolling(252).rank(pct=True)
    bear = (position < 0).astype(float)
    high_vol = (intensity > 0.5).astype(float)
    vix_proxy = log_ret.rolling(20).std().rolling(252).rank(pct=True)
    indexret = build_indexret(log_ret, indexret_spec)
    target = build_target(wml, horizon)

    df = pd.DataFrame(
        {
            "target": target,
            "wml": wml,
            "index_close": index_close,
            "position": position,
            "intensity": intensity,
            "bear": bear,
            "high_vol": high_vol,
            "vix_proxy": vix_proxy,
            "indexret": indexret,
        }
    )
    df["bear_x_intensity"] = df["bear"] * df["intensity"]
    df["bear_x_vix"] = df["bear"] * df["vix_proxy"]
    df["position_x_intensity"] = df["position"] * df["intensity"]
    df["intensity_x_indexret"] = df["intensity"] * df["indexret"]
    df["regime"] = (2 * (df["position"] > 0).astype(int) + df["high_vol"].astype(int)).astype(int)
    return df.dropna().copy()


def design_matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if cols:
        return sm.add_constant(frame[cols], has_constant="add")
    return pd.DataFrame({"const": np.ones(len(frame), dtype=float)}, index=frame.index)


def fit_hac(sample: pd.DataFrame, cols: list[str], hac_lag: int):
    y = sample["target"].astype(float)
    x = design_matrix(sample, cols).astype(float)
    return sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag})


def f_test_zero(result, terms: list[str]) -> dict[str, float | int | str]:
    restriction = ", ".join(f"{term} = 0" for term in terms)
    test = result.f_test(restriction)
    fvalue = float(np.asarray(test.fvalue).squeeze())
    pvalue = float(np.asarray(test.pvalue).squeeze())
    df_denom = getattr(test, "df_denom", np.nan)
    return {
        "restriction": restriction,
        "f_stat": fvalue,
        "p_value": pvalue,
        "df_num": len(terms),
        "df_denom": float(df_denom) if df_denom is not None else np.nan,
    }


def build_in_sample_outputs(
    sample: pd.DataFrame, hac_lag: int
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results: dict[str, object] = {}
    model_rows: list[dict[str, float | str]] = []
    coef_rows: list[dict[str, float | str]] = []

    for model_name, cols in MODEL_SPECS.items():
        result = fit_hac(sample, cols, hac_lag)
        results[model_name] = result
        model_rows.append(
            {
                "model_kind": model_name,
                "model_label": MODEL_LABELS[model_name],
                "nobs": int(result.nobs),
                "rsquared": float(result.rsquared),
                "rsquared_adj": float(result.rsquared_adj),
                "aic": float(result.aic),
                "bic": float(result.bic),
            }
        )
        for term in result.params.index:
            coef_rows.append(
                {
                    "model_kind": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "term": term,
                    "coef": float(result.params[term]),
                    "t_value": float(result.tvalues[term]),
                    "p_value": float(result.pvalues[term]),
                }
            )

    nested_rows: list[dict[str, float | str]] = []
    nested_rows.append(
        {
            "comparison": "model2_vs_model1",
            "comparison_label": "Model 2 > Model 1",
            **f_test_zero(results["model2"], ["bear_x_intensity"]),
        }
    )
    nested_rows.append(
        {
            "comparison": "model4_vs_model3",
            "comparison_label": "Model 4 > Model 3",
            **f_test_zero(results["model4"], ["indexret", "intensity_x_indexret"]),
        }
    )
    nested_rows.append(
        {
            "comparison": "model4_interaction_only",
            "comparison_label": "Model 4 interaction only",
            **f_test_zero(results["model4"], ["intensity_x_indexret"]),
        }
    )

    benchmark_rows: list[dict[str, float | str]] = []
    benchmark_rows.append(
        {
            "comparison": "model4_vs_model2_adj_r2",
            "comparison_label": "Model 4 vs Model 2",
            "delta_rsquared_adj": float(results["model4"].rsquared_adj - results["model2"].rsquared_adj),
        }
    )
    benchmark_rows.append(
        {
            "comparison": "model4_vs_model2_vix_adj_r2",
            "comparison_label": "Model 4 vs Model 2V",
            "delta_rsquared_adj": float(results["model4"].rsquared_adj - results["model2_vix"].rsquared_adj),
        }
    )

    return (
        results,
        pd.DataFrame(model_rows),
        pd.DataFrame(coef_rows),
        pd.DataFrame(nested_rows + benchmark_rows),
    )


def build_oos_predictions(
    sample: pd.DataFrame,
    eval_start: str,
    eval_end: str,
    train_window: int,
    min_train: int,
) -> pd.DataFrame:
    eval_start_ts = pd.Timestamp(eval_start)
    eval_end_ts = pd.Timestamp(eval_end)

    records: list[dict[str, float | str | pd.Timestamp | int]] = []
    for idx in range(len(sample)):
        date = sample.index[idx]
        if date < eval_start_ts or date > eval_end_ts:
            continue

        train_end = idx
        if train_end < min_train:
            continue
        train_start = 0 if train_window <= 0 else max(0, train_end - train_window)
        train = sample.iloc[train_start:train_end]
        if len(train) < min_train:
            continue

        row = sample.iloc[idx : idx + 1]
        y_true = float(row["target"].iloc[0])
        base_info = {
            "date": date,
            "y_true": y_true,
            "bear": float(row["bear"].iloc[0]),
            "high_vol": float(row["high_vol"].iloc[0]),
            "regime": int(row["regime"].iloc[0]),
        }

        for model_name, cols in MODEL_SPECS.items():
            x_train = design_matrix(train, cols).astype(float)
            y_train = train["target"].astype(float)
            fit = sm.OLS(y_train, x_train).fit()
            x_now = design_matrix(row, cols).astype(float)
            pred = float(fit.predict(x_now).iloc[0])
            abs_err = abs(pred - y_true)
            sq_err = (pred - y_true) ** 2
            records.append(
                {
                    **base_info,
                    "model_kind": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "pred": pred,
                    "abs_err": abs_err,
                    "sq_err": sq_err,
                    "train_start": train.index[0],
                    "train_end": train.index[-1],
                    "train_nobs": int(len(train)),
                }
            )

    if not records:
        raise RuntimeError("No OOS predictions generated. Check eval dates and training window.")
    return pd.DataFrame(records)


def nw_one_sided_positive(diff: pd.Series, nlags: int) -> dict[str, float | int]:
    diff = diff.dropna().astype(float)
    n = len(diff)
    if n < 10:
        return {"mean": np.nan, "t_nw": np.nan, "p_one_sided": np.nan, "n": n}

    x = diff.to_numpy()
    mean = float(x.mean())
    ols = sm.OLS(x, np.ones((n, 1), dtype=float)).fit()
    cov = sm.stats.sandwich_covariance.cov_hac(ols, nlags=nlags)
    se = math.sqrt(float(cov[0, 0])) if np.isfinite(cov[0, 0]) and cov[0, 0] > 0 else np.nan
    t_stat = mean / se if se and np.isfinite(se) else np.nan
    p_value = 1.0 - stats.norm.cdf(t_stat) if np.isfinite(t_stat) else np.nan
    return {"mean": mean, "t_nw": t_stat, "p_one_sided": p_value, "n": n}


def summarize_oos(pred_df: pd.DataFrame, oos_hac_lag: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, float | str | int]] = []
    baseline_sse = float(pred_df.loc[pred_df["model_kind"] == "model0", "sq_err"].sum())

    for model_name in MODEL_SPECS:
        sub = pred_df[pred_df["model_kind"] == model_name].copy()
        corr = sub[["pred", "y_true"]].corr().iloc[0, 1] if len(sub) >= 2 else np.nan
        metric_rows.append(
            {
                "model_kind": model_name,
                "model_label": MODEL_LABELS[model_name],
                "n_obs": int(len(sub)),
                "mae": float(sub["abs_err"].mean()),
                "rmse": float(np.sqrt(sub["sq_err"].mean())),
                "pred_y_corr": float(corr) if np.isfinite(corr) else np.nan,
                "direction_acc": float((np.sign(sub["pred"]) == np.sign(sub["y_true"])).mean()),
                "oos_r2_vs_model0": 1.0 - float(sub["sq_err"].sum()) / baseline_sse if baseline_sse > 0 else np.nan,
            }
        )

    metric_df = pd.DataFrame(metric_rows)

    pivot_abs = pred_df.pivot(index="date", columns="model_kind", values="abs_err")
    pivot_sq = pred_df.pivot(index="date", columns="model_kind", values="sq_err")

    compare_rows: list[dict[str, float | str | int]] = []
    compare_rows.append(
        {
            "comparison": "model4_vs_model3_abs_err",
            "comparison_label": "Model 4 vs Model 3 | abs err",
            **nw_one_sided_positive(pivot_abs["model3"] - pivot_abs["model4"], oos_hac_lag),
        }
    )
    compare_rows.append(
        {
            "comparison": "model4_vs_model3_sq_err",
            "comparison_label": "Model 4 vs Model 3 | sq err",
            **nw_one_sided_positive(pivot_sq["model3"] - pivot_sq["model4"], oos_hac_lag),
        }
    )
    compare_rows.append(
        {
            "comparison": "model4_vs_model2_abs_err",
            "comparison_label": "Model 4 vs Model 2 | abs err",
            **nw_one_sided_positive(pivot_abs["model2"] - pivot_abs["model4"], oos_hac_lag),
        }
    )
    compare_rows.append(
        {
            "comparison": "model4_vs_model2_vix_abs_err",
            "comparison_label": "Model 4 vs Model 2V | abs err",
            **nw_one_sided_positive(pivot_abs["model2_vix"] - pivot_abs["model4"], oos_hac_lag),
        }
    )

    regime_rows: list[dict[str, float | str | int]] = []
    for (model_name, regime), sub in pred_df.groupby(["model_kind", "regime"]):
        regime_rows.append(
            {
                "model_kind": model_name,
                "model_label": MODEL_LABELS[model_name],
                "regime": int(regime),
                "n_obs": int(len(sub)),
                "mae": float(sub["abs_err"].mean()),
                "rmse": float(np.sqrt(sub["sq_err"].mean())),
                "mean_y_true": float(sub["y_true"].mean()),
                "mean_pred": float(sub["pred"].mean()),
            }
        )

    return metric_df, pd.DataFrame(compare_rows), pd.DataFrame(regime_rows)


def build_flag_table(
    nested_df: pd.DataFrame,
    oos_compare_df: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    nested_map = nested_df.set_index("comparison")
    oos_map = oos_compare_df.set_index("comparison")

    interaction_p = float(nested_map.loc["model4_interaction_only", "p_value"])
    model4_joint_p = float(nested_map.loc["model4_vs_model3", "p_value"])
    model4_joint_f = float(nested_map.loc["model4_vs_model3", "f_stat"])
    oos_abs_mean = float(oos_map.loc["model4_vs_model3_abs_err", "mean"])
    oos_abs_p = float(oos_map.loc["model4_vs_model3_abs_err", "p_one_sided"])
    oos_sq_mean = float(oos_map.loc["model4_vs_model3_sq_err", "mean"])
    oos_sq_p = float(oos_map.loc["model4_vs_model3_sq_err", "p_one_sided"])
    oos_bench_mean = float(oos_map.loc["model4_vs_model2_abs_err", "mean"])
    oos_bench_p = float(oos_map.loc["model4_vs_model2_abs_err", "p_one_sided"])

    flags = [
        {
            "flag": "model4_joint_terms_significant",
            "value": bool(model4_joint_p < alpha),
            "detail": f"Model 4 vs Model 3 added terms F={model4_joint_f:.4f}, p={model4_joint_p:.4g}",
        },
        {
            "flag": "interaction_term_significant_in_model4",
            "value": bool(interaction_p < alpha),
            "detail": f"intensity_x_indexret in Model 4 p={interaction_p:.4g}",
        },
        {
            "flag": "model4_beats_model3_oos_abs_err",
            "value": bool(oos_abs_mean > 0 and oos_abs_p < alpha),
            "detail": f"delta abs err={oos_abs_mean:.6f}, one-sided p={oos_abs_p:.4g}",
        },
        {
            "flag": "model4_beats_model3_oos_sq_err",
            "value": bool(oos_sq_mean > 0 and oos_sq_p < alpha),
            "detail": f"delta sq err={oos_sq_mean:.6f}, one-sided p={oos_sq_p:.4g}",
        },
        {
            "flag": "model4_beats_model2_oos_abs_err",
            "value": bool(oos_bench_mean > 0 and oos_bench_p < alpha),
            "detail": f"delta abs err={oos_bench_mean:.6f}, one-sided p={oos_bench_p:.4g}",
        },
    ]
    return pd.DataFrame(flags)


def _add_metadata(df: pd.DataFrame, index_symbol: str, indexret_spec: str, horizon: int) -> pd.DataFrame:
    annotated = df.copy()
    annotated["index_symbol"] = index_symbol
    annotated["indexret_spec"] = indexret_spec
    annotated["horizon"] = horizon
    return annotated


def run_analysis(
    *,
    wml: pd.Series,
    index_close: pd.Series,
    index_symbol: str,
    horizon: int,
    indexret_spec: str,
    eval_start: str,
    eval_end: str,
    hac_lag: int,
    oos_hac_lag: int,
    train_window: int,
    min_train: int,
    alpha: float,
) -> dict[str, pd.DataFrame]:
    sample = build_proxy_frame(wml, index_close, horizon, indexret_spec)
    _in_sample_results, model_df, coef_df, nested_df = build_in_sample_outputs(sample, hac_lag)
    pred_df = build_oos_predictions(
        sample=sample,
        eval_start=eval_start,
        eval_end=eval_end,
        train_window=train_window,
        min_train=min_train,
    )
    oos_metric_df, oos_compare_df, regime_df = summarize_oos(pred_df, oos_hac_lag)
    flags_df = build_flag_table(nested_df, oos_compare_df, alpha)

    return {
        "dataset": _add_metadata(sample.reset_index(names="date"), index_symbol, indexret_spec, horizon),
        "model_summary": _add_metadata(model_df, index_symbol, indexret_spec, horizon),
        "coefficients": _add_metadata(coef_df, index_symbol, indexret_spec, horizon),
        "tests": _add_metadata(nested_df, index_symbol, indexret_spec, horizon),
        "oos_predictions": _add_metadata(pred_df, index_symbol, indexret_spec, horizon),
        "oos_summary": _add_metadata(oos_metric_df, index_symbol, indexret_spec, horizon),
        "oos_tests": _add_metadata(oos_compare_df, index_symbol, indexret_spec, horizon),
        "regime_summary": _add_metadata(regime_df, index_symbol, indexret_spec, horizon),
        "flags": _add_metadata(flags_df, index_symbol, indexret_spec, horizon),
    }


def save_outputs(results: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / "finance_incremental_dataset.csv"
    model_path = out_dir / "finance_incremental_model_summary.csv"
    coef_path = out_dir / "finance_incremental_coefficients.csv"
    nested_path = out_dir / "finance_incremental_tests.csv"
    pred_path = out_dir / "finance_incremental_oos_predictions.csv"
    oos_metric_path = out_dir / "finance_incremental_oos_summary.csv"
    oos_compare_path = out_dir / "finance_incremental_oos_tests.csv"
    regime_path = out_dir / "finance_incremental_regime_summary.csv"
    flag_path = out_dir / "finance_incremental_flags.csv"

    results["dataset"].to_csv(dataset_path, index=False)
    results["model_summary"].to_csv(model_path, index=False)
    results["coefficients"].to_csv(coef_path, index=False)
    results["tests"].to_csv(nested_path, index=False)
    results["oos_predictions"].to_csv(pred_path, index=False)
    results["oos_summary"].to_csv(oos_metric_path, index=False)
    results["oos_tests"].to_csv(oos_compare_path, index=False)
    results["regime_summary"].to_csv(regime_path, index=False)
    results["flags"].to_csv(flag_path, index=False)

    return {
        "dataset": dataset_path,
        "model_summary": model_path,
        "coefficients": coef_path,
        "tests": nested_path,
        "oos_predictions": pred_path,
        "oos_summary": oos_metric_path,
        "oos_tests": oos_compare_path,
        "regime_summary": regime_path,
        "flags": flag_path,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    oos_hac_lag = args.oos_hac_lag if args.oos_hac_lag is not None else max(4, args.horizon - 1)
    wml, index_close = load_data(args.start, args.end, args.index_symbol)
    results = run_analysis(
        wml=wml,
        index_close=index_close,
        index_symbol=args.index_symbol,
        horizon=args.horizon,
        indexret_spec=args.indexret_spec,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        hac_lag=args.hac_lag,
        oos_hac_lag=oos_hac_lag,
        train_window=args.train_window,
        min_train=args.min_train,
        alpha=args.alpha,
    )
    paths = save_outputs(results, out_dir)

    print("\n[in-sample model summary]", flush=True)
    print(results["model_summary"].to_string(index=False), flush=True)
    print("\n[confirmatory tests]", flush=True)
    print(results["tests"].to_string(index=False), flush=True)
    print("\n[oos summary]", flush=True)
    print(results["oos_summary"].to_string(index=False), flush=True)
    print("\n[oos loss tests]", flush=True)
    print(results["oos_tests"].to_string(index=False), flush=True)
    print("\n[flags]", flush=True)
    print(results["flags"].to_string(index=False), flush=True)
    print(f"\n[save] {paths['dataset']}", flush=True)
    print(f"[save] {paths['model_summary']}", flush=True)
    print(f"[save] {paths['coefficients']}", flush=True)
    print(f"[save] {paths['tests']}", flush=True)
    print(f"[save] {paths['oos_predictions']}", flush=True)
    print(f"[save] {paths['oos_summary']}", flush=True)
    print(f"[save] {paths['oos_tests']}", flush=True)
    print(f"[save] {paths['regime_summary']}", flush=True)
    print(f"[save] {paths['flags']}", flush=True)


if __name__ == "__main__":
    main()
