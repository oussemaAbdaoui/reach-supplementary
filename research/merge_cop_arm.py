"""merge_cop_arm.py — fold a re-run COP arm back into its model's grid CSV.

WHY THIS EXISTS
---------------
The COP arm was re-fitted after the example-oracle sampler was corrected (see
valid_oracle_samples in run_comparison_nn.py). Re-running one arm has to go to
its own file, because run_comparison_nn.py resumes at INSTANCE granularity:
pointed at the main CSV it would find all 15 instances present and skip the
work. So the arm is run into <dataset>_nn_comparison_cop_<tag>.csv and merged
here, leaving one structured result set per model rather than a scatter of
per-arm files.

The superseded rows are not discarded: the previous main CSV is copied to
_superseded_<name> first, so the U-fitted and V-fitted COP arms remain
comparable.

    python research/merge_cop_arm.py cancer cogito671b
"""
from __future__ import annotations

import os
import shutil
import sys

import pandas as pd

RESULTS = os.path.join('research', 'results')
ARM_ORDER = ['cop', 'cop_k', 'cop_best', 'lmpaxp', 'anchors', 'shap', 'lime']


def merge(dataset: str, tag: str, arm: str = 'cop') -> int:
    main = os.path.join(RESULTS, f'{dataset}_nn_comparison_{tag}.csv')
    part = os.path.join(RESULTS, f'{dataset}_nn_comparison_{arm}_{tag}.csv')
    for p in (main, part):
        if not os.path.exists(p):
            print(f'  [missing] {p}')
            return 1

    dm, dp = pd.read_csv(main), pd.read_csv(part)
    dp = dp[dp['arm'] == arm]
    if dp.empty:
        print(f'  [empty] no {arm!r} rows in {part}')
        return 1

    backup = os.path.join(RESULTS, f'_superseded_{dataset}_nn_comparison_{tag}_Ufit{arm}.csv')
    shutil.copyfile(main, backup)

    old = dm[dm['arm'] == arm]
    keep = dm[dm['arm'] != arm]
    merged = pd.concat([keep, dp], ignore_index=True)
    merged['_o'] = merged['arm'].map({a: i for i, a in enumerate(ARM_ORDER)})
    merged = (merged.sort_values(['instance', '_o'])
                    .drop(columns='_o')
                    .reset_index(drop=True))
    merged.to_csv(main, index=False)

    print(f'  {dataset}/{tag}: replaced {len(old)} {arm} rows with {len(dp)}')
    if len(old) and 'baseline_precision' in dm.columns:
        print(f'    baseline {old.baseline_precision.mean():.2f} -> '
              f'{dp.baseline_precision.mean():.2f} '
              f'({dp.baseline_precision.mean() - old.baseline_precision.mean():+.2f})')
        for s in ('score_v4', 'score_v6'):
            if s in dm.columns:
                print(f'    {s:9} {old[s].mean():.2f} -> {dp[s].mean():.2f} '
                      f'({dp[s].mean() - old[s].mean():+.2f})')
    print(f'    superseded copy -> {backup}')
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('usage: merge_cop_arm.py <dataset> <tag> [arm]')
    sys.exit(merge(sys.argv[1], sys.argv[2],
                   sys.argv[3] if len(sys.argv) > 3 else 'cop'))
