"""
Noise Injection Experiment on Real Financial Data.

Measures the SNR crossover point: at what noise level does FiLM-A start
beating Concat-A?  This directly probes the effective signal-to-noise ratio
of the financial conditioning signals (position, intensity).

Protocol:
  1. Load GSPC 2022-2024, build FF3 residual target, build splits (same
     pipeline as economic_time_supervised).
  2. For each noise_level in [0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
       - Add N(0, noise_level) to the context arrays (position & intensity
         sequences) in train / val / test.
       - Train concat_a and film_a for 3 epochs, seed 7.
       - Evaluate on test set -> IC, ICIR, MAE.
  3. Save all results to noise_injection_results.csv.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AAA_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
for p in (AAA_ROOT, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from paper_test.economic_time_supervised import (  # noqa: E402
    EconSplitData,
    build_loader,
    build_market_features,
    build_model,
    build_split,
    evaluate_model,
    train_model,
)
from paper_test.icpe_hybrid_supervised import (  # noqa: E402
    build_state,
    build_target,
    load_data,
    set_seed,
)


def inject_noise(split: EconSplitData, noise_level: float, rng: np.random.Generator) -> EconSplitData:
    """Return a copy of *split* with Gaussian noise added to context arrays."""
    if noise_level == 0.0:
        return split
    noisy_ctx = split.context + rng.normal(0.0, noise_level, size=split.context.shape).astype(np.float32)
    return EconSplitData(
        asset_src=split.asset_src,
        market_seq=split.market_seq,
        context=noisy_ctx,
        y=split.y,
        dates=split.dates,
        regime=split.regime,
    )


def make_args(**overrides) -> argparse.Namespace:
    """Build a default Namespace matching economic_time_supervised expectations."""
    defaults = dict(
        start="2022-01-01",
        end="2024-12-31",
        index_symbol="^GSPC",
        target="residual",
        seq_len=30,
        horizon=5,
        roll_beta=60,
        epochs=3,
        batch_size=512,
        lr=1e-3,
        weight_decay=1e-4,
        d_model=32,
        heads=4,
        layers=2,
        device="cpu",
        seed=7,
        relative_bias_mode="relu",
        relative_bias_gamma=1.0,
        fixed_bias_slopes=False,
        tau_align_lambda=0.1,
        tau_geom_lambda=0.0,
        tau_geom_warmup_epochs=1,
        tau_ord_lambda=0.0,
        tau_ord_margin=1e-3,
        tau_ord_sigmas="0.1,0.2,0.4",
        model_kinds="concat_a,film_a",
        output_dir=str(AAA_ROOT / "experiments"),
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def main():
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[noise_injection] device={device}", flush=True)

    args = make_args(device=device)
    set_seed(args.seed)

    # ---- Load data & build splits (once) ----
    print("[noise_injection] loading data ...", flush=True)
    port25, factors, index_close = load_data(args.start, args.end, args.index_symbol)
    source, target = build_target(port25, factors, args.target, args.roll_beta, args.horizon)
    position, intensity, regime = build_state(index_close)

    dates = source.index
    train_split, val_split, test_split = build_split(
        dates, source, target, position, intensity, regime, index_close, args.seq_len,
    )

    # Report context statistics for reference
    ctx_std = train_split.context.std(axis=0)
    ctx_mean = train_split.context.mean(axis=0)
    print(f"[noise_injection] context shape={train_split.context.shape}", flush=True)
    print(f"[noise_injection] context mean={ctx_mean}, std={ctx_std}", flush=True)
    print(f"[noise_injection] train={len(train_split.y)}, val={len(val_split.y)}, test={len(test_split.y)}", flush=True)

    noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    model_kinds = ["concat_a", "film_a"]
    results = []

    for noise_level in noise_levels:
        rng = np.random.default_rng(args.seed)
        noisy_train = inject_noise(train_split, noise_level, rng)
        noisy_val = inject_noise(val_split, noise_level, rng)
        noisy_test = inject_noise(test_split, noise_level, rng)

        for model_kind in model_kinds:
            set_seed(args.seed)
            print(f"\n{'='*60}", flush=True)
            print(f"[noise_injection] noise={noise_level:.1f}  model={model_kind}", flush=True)
            print(f"{'='*60}", flush=True)

            model = train_model(args, model_kind, noisy_train, noisy_val)
            test_loader = build_loader(noisy_test, args.batch_size, shuffle=False)
            metrics, _ = evaluate_model(model, model_kind, test_loader, args.device)

            row = {
                "noise_level": noise_level,
                "model_kind": model_kind,
                "test_ic": metrics["ic"],
                "test_icir": metrics["icir"],
                "test_mae": metrics["mae"],
                "param_count": metrics["param_count"],
            }
            results.append(row)
            print(
                f"[noise_injection] noise={noise_level:.1f} {model_kind:10s} "
                f"IC={metrics['ic']:.4f}  ICIR={metrics['icir']:.4f}  MAE={metrics['mae']:.6f}",
                flush=True,
            )

            # Free GPU memory between runs
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---- Save results ----
    df = pd.DataFrame(results)
    out_path = Path(args.output_dir) / "noise_injection_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[noise_injection] results saved to {out_path}", flush=True)

    # ---- Print summary table ----
    print("\n" + "=" * 70, flush=True)
    print("NOISE INJECTION EXPERIMENT SUMMARY", flush=True)
    print("=" * 70, flush=True)
    pivot_ic = df.pivot(index="noise_level", columns="model_kind", values="test_ic")
    pivot_ic["film_minus_concat"] = pivot_ic["film_a"] - pivot_ic["concat_a"]
    print("\nTest IC by noise level:", flush=True)
    print(pivot_ic.to_string(float_format="%.4f"), flush=True)

    pivot_icir = df.pivot(index="noise_level", columns="model_kind", values="test_icir")
    pivot_icir["film_minus_concat"] = pivot_icir["film_a"] - pivot_icir["concat_a"]
    print("\nTest ICIR by noise level:", flush=True)
    print(pivot_icir.to_string(float_format="%.4f"), flush=True)

    # Find crossover
    crossover = pivot_ic[pivot_ic["film_minus_concat"] > 0]
    if len(crossover) > 0:
        first_cross = crossover.index[0]
        print(f"\nSNR CROSSOVER: FiLM-A first beats Concat-A at noise_level={first_cross}", flush=True)
        print(f"  Context std ~ {ctx_std.mean():.4f}, so SNR ~ {ctx_std.mean() / first_cross:.2f}", flush=True)
    else:
        print("\nNo crossover found: Concat-A dominates at all tested noise levels.", flush=True)


if __name__ == "__main__":
    main()
