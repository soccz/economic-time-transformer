from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from finance_incremental_identification import (
    INDEXRET_SPECS,
    load_data,
    run_analysis,
    save_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep index-return definitions and future-WML horizons for finance incremental identification"
    )
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--eval-start", default="2020-01-01")
    parser.add_argument("--eval-end", default="2024-12-31")
    parser.add_argument("--index-symbol", default="^GSPC")
    parser.add_argument("--horizons", default="1,5", help="comma-separated future-WML horizons")
    parser.add_argument(
        "--indexret-specs",
        default="ret1,ret5",
        help=f"comma-separated indexret specs from: {','.join(INDEXRET_SPECS)}",
    )
    parser.add_argument("--hac-lag", type=int, default=10)
    parser.add_argument(
        "--oos-hac-lag",
        type=int,
        default=None,
        help="optional fixed Newey-West lag for OOS tests; default=max(4, horizon-1)",
    )
    parser.add_argument("--train-window", type=int, default=0)
    parser.add_argument("--min-train", type=int, default=504)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        default="paper/economic_time/results/finance_incremental_sweep",
    )
    return parser.parse_args()


def _parse_horizons(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("No horizons provided")
    return values


def _parse_specs(raw: str) -> list[str]:
    specs = [token.strip() for token in raw.split(",") if token.strip()]
    if not specs:
        raise ValueError("No indexret specs provided")
    unknown = [spec for spec in specs if spec not in INDEXRET_SPECS]
    if unknown:
        raise ValueError(f"Unknown indexret specs: {unknown}")
    return specs


def build_summary_row(results: dict[str, pd.DataFrame], spec_dir: Path) -> dict[str, object]:
    tests = results["tests"].set_index("comparison")
    oos_summary = results["oos_summary"].set_index("model_kind")
    oos_tests = results["oos_tests"].set_index("comparison")
    coeffs = results["coefficients"]
    flags = results["flags"].set_index("flag")

    model4_interaction = coeffs[
        (coeffs["model_kind"] == "model4") & (coeffs["term"] == "intensity_x_indexret")
    ].iloc[0]

    return {
        "indexret_spec": results["tests"]["indexret_spec"].iloc[0],
        "horizon": int(results["tests"]["horizon"].iloc[0]),
        "output_dir": str(spec_dir),
        "model2_vs_model1_p": float(tests.loc["model2_vs_model1", "p_value"]),
        "model4_vs_model3_p": float(tests.loc["model4_vs_model3", "p_value"]),
        "model4_interaction_p": float(tests.loc["model4_interaction_only", "p_value"]),
        "model4_interaction_coef": float(model4_interaction["coef"]),
        "model4_oos_r2": float(oos_summary.loc["model4", "oos_r2_vs_model0"]),
        "model3_oos_r2": float(oos_summary.loc["model3", "oos_r2_vs_model0"]),
        "model4_vs_model3_abs_p": float(oos_tests.loc["model4_vs_model3_abs_err", "p_one_sided"]),
        "model4_vs_model3_abs_mean": float(oos_tests.loc["model4_vs_model3_abs_err", "mean"]),
        "model4_vs_model2_abs_p": float(oos_tests.loc["model4_vs_model2_abs_err", "p_one_sided"]),
        "model4_joint_flag": bool(flags.loc["model4_joint_terms_significant", "value"]),
        "model4_interaction_flag": bool(flags.loc["interaction_term_significant_in_model4", "value"]),
        "model4_oos_flag": bool(flags.loc["model4_beats_model3_oos_abs_err", "value"]),
    }


def main() -> None:
    args = parse_args()
    horizons = _parse_horizons(args.horizons)
    specs = _parse_specs(args.indexret_specs)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    wml, index_close = load_data(args.start, args.end, args.index_symbol)

    summary_rows: list[dict[str, object]] = []
    flag_frames: list[pd.DataFrame] = []

    for horizon in horizons:
        oos_hac_lag = args.oos_hac_lag if args.oos_hac_lag is not None else max(4, horizon - 1)
        for spec in specs:
            spec_dir = out_root / f"h{horizon}_{spec}"
            results = run_analysis(
                wml=wml,
                index_close=index_close,
                index_symbol=args.index_symbol,
                horizon=horizon,
                indexret_spec=spec,
                eval_start=args.eval_start,
                eval_end=args.eval_end,
                hac_lag=args.hac_lag,
                oos_hac_lag=oos_hac_lag,
                train_window=args.train_window,
                min_train=args.min_train,
                alpha=args.alpha,
            )
            save_outputs(results, spec_dir)
            summary_rows.append(build_summary_row(results, spec_dir))
            flag_frames.append(results["flags"].assign(output_dir=str(spec_dir)))
            print(
                f"[spec] h={horizon} spec={spec} | "
                f"joint_p={summary_rows[-1]['model4_vs_model3_p']:.4f} | "
                f"interaction_p={summary_rows[-1]['model4_interaction_p']:.4f} | "
                f"oos_r2={summary_rows[-1]['model4_oos_r2']:.6f}",
                flush=True,
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["model4_vs_model3_p", "model4_oos_r2"],
        ascending=[True, False],
    ).reset_index(drop=True)
    flags_df = pd.concat(flag_frames, ignore_index=True)

    summary_path = out_root / "finance_incremental_sweep_summary.csv"
    flags_path = out_root / "finance_incremental_sweep_flags.csv"
    summary_df.to_csv(summary_path, index=False)
    flags_df.to_csv(flags_path, index=False)

    print("\n[sweep summary]", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"\n[save] {summary_path}", flush=True)
    print(f"[save] {flags_path}", flush=True)


if __name__ == "__main__":
    main()
