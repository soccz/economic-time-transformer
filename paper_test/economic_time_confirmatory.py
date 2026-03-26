from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac


PRED_RE = re.compile(
    r"economic_time_(?P<market>[a-z0-9]+)_.*?_(?P<model>concat_a|tau_rope|static)_predictions\.csv$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confirmatory pooled test for path-2 preregistered hypotheses")
    parser.add_argument(
        "--prediction-dirs",
        required=True,
        help="comma-separated directories containing *_predictions.csv files",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nw-lag", type=int, default=4)
    return parser.parse_args()


def load_predictions(pred_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for pred_dir in pred_dirs:
        for path in sorted(pred_dir.glob("*_predictions.csv")):
            m = PRED_RE.match(path.name)
            if not m:
                continue
            market = m.group("market").upper()
            model = m.group("model")
            df = pd.read_csv(path)
            df["market"] = market
            df["model_kind"] = model
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No matching prediction files found")
    return pd.concat(frames, ignore_index=True)


def daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (market, model_kind, date), sub in df.groupby(["market", "model_kind", "date"]):
        ic = stats.spearmanr(sub["pred"], sub["y_true"])[0]
        mae = np.mean(np.abs(sub["pred"] - sub["y_true"]))
        intensity = sub["regime"].map({0: 0, 1: 1, 2: 0, 3: 1}).mean()
        rows.append(
            {
                "market": market,
                "model_kind": model_kind,
                "date": pd.to_datetime(date),
                "ic": ic,
                "mae": mae,
                "high_vol": int(intensity > 0.5),
            }
        )
    return pd.DataFrame(rows)


def nw_one_sided_positive(diff: pd.Series, nlags: int) -> dict:
    diff = diff.dropna()
    n = len(diff)
    if n < 10:
        return {"mean": np.nan, "t_nw": np.nan, "p_one_sided": np.nan, "n": n}

    x = diff.to_numpy(dtype=float)
    mean = float(x.mean())
    var = float(np.var(x, ddof=1))
    if not np.isfinite(var):
        return {"mean": mean, "t_nw": np.nan, "p_one_sided": np.nan, "n": n}

    # OLS on a constant with HAC covariance gives the Newey-West t-stat for the mean.
    X = np.ones((n, 1), dtype=float)
    res = sm.OLS(x, X).fit()
    cov = cov_hac(res, nlags=nlags)
    se = math.sqrt(float(cov[0, 0])) if np.isfinite(cov[0, 0]) and cov[0, 0] > 0 else np.nan
    t_stat = mean / se if se and np.isfinite(se) else np.nan
    p = 1.0 - stats.norm.cdf(t_stat) if np.isfinite(t_stat) else np.nan
    return {"mean": mean, "t_nw": t_stat, "p_one_sided": p, "n": n}


def main():
    args = parse_args()
    pred_dirs = [Path(p.strip()) for p in args.prediction_dirs.split(",") if p.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = load_predictions(pred_dirs)
    preds_path = out_dir / "confirmatory_predictions_combined.csv"
    preds.to_csv(preds_path, index=False)

    daily = daily_metrics(preds)
    daily_path = out_dir / "confirmatory_daily_metrics.csv"
    daily.to_csv(daily_path, index=False)

    hv = daily[daily["high_vol"] == 1].copy()
    pivot_ic = hv.pivot_table(index=["market", "date"], columns="model_kind", values="ic")
    pivot_mae = hv.pivot_table(index=["market", "date"], columns="model_kind", values="mae")

    delta_ic = pivot_ic["tau_rope"] - pivot_ic["concat_a"]
    delta_mae = pivot_mae["concat_a"] - pivot_mae["tau_rope"]

    h1 = nw_one_sided_positive(delta_ic, nlags=args.nw_lag)
    h2 = nw_one_sided_positive(delta_mae, nlags=args.nw_lag)

    result = pd.DataFrame(
        [
            {"hypothesis": "H1_high_vol_IC", **h1},
            {"hypothesis": "H2_high_vol_MAE", **h2},
        ]
    )
    result_path = out_dir / "confirmatory_hypothesis_tests.csv"
    result.to_csv(result_path, index=False)

    market_rows = []
    for market, sub in hv.groupby("market"):
        sub_ic = sub.pivot_table(index="date", columns="model_kind", values="ic")
        sub_mae = sub.pivot_table(index="date", columns="model_kind", values="mae")
        market_rows.append({"market": market, "metric": "delta_ic", **nw_one_sided_positive(sub_ic["tau_rope"] - sub_ic["concat_a"], nlags=args.nw_lag)})
        market_rows.append({"market": market, "metric": "delta_mae", **nw_one_sided_positive(sub_mae["concat_a"] - sub_mae["tau_rope"], nlags=args.nw_lag)})
    market_df = pd.DataFrame(market_rows)
    market_path = out_dir / "confirmatory_market_breakdown.csv"
    market_df.to_csv(market_path, index=False)

    print("\n[confirmatory tests]", flush=True)
    print(result.to_string(index=False), flush=True)
    print("\n[market breakdown]", flush=True)
    print(market_df.to_string(index=False), flush=True)
    print(f"\n[save] {preds_path}", flush=True)
    print(f"[save] {daily_path}", flush=True)
    print(f"[save] {result_path}", flush=True)
    print(f"[save] {market_path}", flush=True)


if __name__ == "__main__":
    main()
