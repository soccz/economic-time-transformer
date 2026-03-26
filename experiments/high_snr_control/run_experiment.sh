#!/bin/bash
# High-SNR control experiment: Does FiLM beat concat with discrete conditioning?
# Tests Paper 1's SNR theory: FiLM should excel when conditioning signals are clean/discrete.
#
# Model kinds tested:
#   static                          - no conditioning baseline
#   concat_a:binned_intensity_only  - concat with 16-bin discretized intensity (1 channel)
#   film_a:binned_intensity_only    - FiLM with 16-bin discretized intensity (1 channel)
#   concat_a:binned_all             - concat with 4-bin discretized position+intensity (2 channels)
#   film_a:binned_all               - FiLM with 4-bin discretized position+intensity (2 channels)
#   concat_a                        - concat with raw continuous context (control)
#   film_a:intensity_only           - FiLM with raw continuous intensity (control)

PYTHON=/mnt/20t/main/gan_t/.venv_local/bin/python3
SCRIPT=/mnt/20t/main/gan_t/aaa/paper_test/economic_time_supervised.py
OUTDIR=/mnt/20t/main/gan_t/aaa/experiments/high_snr_control

MODELS="static,concat_a,concat_a:binned_intensity_only,film_a:binned_intensity_only,concat_a:binned_all,film_a:binned_all,film_a:intensity_only"

for SEED in 7 17 27; do
    echo "========== SEED=$SEED =========="
    $PYTHON $SCRIPT \
        --start 2022-01-01 \
        --end 2024-12-31 \
        --index-symbol "^GSPC" \
        --epochs 3 \
        --seed $SEED \
        --device cuda \
        --model-kinds "$MODELS" \
        --output-dir "$OUTDIR/seed_${SEED}"
done

echo ""
echo "========== ALL SEEDS COMPLETE =========="
echo "Aggregating results..."

$PYTHON -c "
import pandas as pd
import numpy as np
from pathlib import Path

outdir = Path('$OUTDIR')
all_rows = []
for seed_dir in sorted(outdir.glob('seed_*')):
    seed = int(seed_dir.name.split('_')[1])
    for csv in seed_dir.glob('*_summary.csv'):
        df = pd.read_csv(csv)
        df['seed'] = seed
        all_rows.append(df)

if not all_rows:
    print('No results found!')
    exit(1)

combined = pd.concat(all_rows, ignore_index=True)
combined.to_csv(outdir / 'all_seeds_raw.csv', index=False)

agg = combined.groupby('model_kind').agg(
    ic_mean=('ic', 'mean'),
    ic_std=('ic', 'std'),
    icir_mean=('icir', 'mean'),
    icir_std=('icir', 'std'),
    mae_mean=('mae', 'mean'),
    seeds=('seed', 'count'),
).sort_values('ic_mean', ascending=False).reset_index()

print()
print('=== HIGH-SNR CONTROL: FiLM vs Concat with Discrete Conditioning ===')
print()
print(agg.to_string(index=False))
print()

# Key comparison
film_binned = agg[agg['model_kind'] == 'film_a:binned_intensity_only']
concat_binned = agg[agg['model_kind'] == 'concat_a:binned_intensity_only']
film_all = agg[agg['model_kind'] == 'film_a:binned_all']
concat_all = agg[agg['model_kind'] == 'concat_a:binned_all']

print('--- Key Comparisons ---')
if len(film_binned) and len(concat_binned):
    delta = film_binned['ic_mean'].values[0] - concat_binned['ic_mean'].values[0]
    print(f'FiLM vs Concat (binned intensity): IC delta = {delta:+.4f}  {\"FiLM WINS\" if delta > 0 else \"Concat WINS\"}')
if len(film_all) and len(concat_all):
    delta = film_all['ic_mean'].values[0] - concat_all['ic_mean'].values[0]
    print(f'FiLM vs Concat (binned all):       IC delta = {delta:+.4f}  {\"FiLM WINS\" if delta > 0 else \"Concat WINS\"}')

agg.to_csv(outdir / 'aggregated_results.csv', index=False)
print()
print(f'Saved: {outdir}/aggregated_results.csv')
print(f'Saved: {outdir}/all_seeds_raw.csv')
"
