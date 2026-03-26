from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REGIME_LABELS = {
    0: "Bear/quiet",
    1: "Bear/volatile",
    2: "Bull/quiet",
    3: "Bull/volatile",
}


PRED_RE = re.compile(
    r"economic_time_(?P<market>[a-z0-9]+)_.*?_(?P<model>static|concat_a|tau_rope)_predictions\.csv$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regime-split report for economic-time prediction files")
    parser.add_argument(
        "--prediction-dirs",
        required=True,
        help="comma-separated directories containing *_predictions.csv files",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def daily_ic(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby("date")
        .apply(lambda g: stats.spearmanr(g["pred"], g["y_true"])[0], include_groups=False)
        .dropna()
    )


def summarize_file(path: Path) -> list[dict]:
    m = PRED_RE.match(path.name)
    if not m:
        return []
    market = m.group("market").upper()
    model = m.group("model")
    df = pd.read_csv(path)
    if "regime" not in df.columns:
        return []
    rows = []
    for regime, sub in df[df["regime"] >= 0].groupby("regime"):
        ic = daily_ic(sub)
        rows.append(
            {
                "market": market,
                "model_kind": model,
                "regime": int(regime),
                "regime_label": REGIME_LABELS.get(int(regime), f"regime_{regime}"),
                "n_obs": int(len(sub)),
                "n_dates": int(sub["date"].nunique()),
                "ic_mean": float(ic.mean()) if len(ic) else np.nan,
                "icir": float(ic.mean() / ic.std()) if len(ic) and ic.std() > 0 else np.nan,
                "mae_mean": float(np.mean(np.abs(sub["pred"] - sub["y_true"]))),
            }
        )
    return rows


def main():
    args = parse_args()
    pred_dirs = [Path(p.strip()) for p in args.prediction_dirs.split(",") if p.strip()]
    rows: list[dict] = []
    for pred_dir in pred_dirs:
        for path in sorted(pred_dir.glob("*_predictions.csv")):
            rows.extend(summarize_file(path))

    report = pd.DataFrame(rows).sort_values(["market", "regime", "model_kind"]).reset_index(drop=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    print(report.to_string(index=False), flush=True)
    print(f"\n[save] {out_path}", flush=True)


if __name__ == "__main__":
    main()
