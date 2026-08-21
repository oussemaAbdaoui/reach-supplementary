"""cop_sampler_check.py — is COP's weak tail its guarantee type, or its sampler?

THE QUESTION
------------
`cop_explainer.fit_cop_rule` draws its example oracle from D = Uniform({0,1}^d)
(cop_explainer.py:101), verbatim [Koriche §2/§4.1]. But d here is a K-bins
ONE-HOT encoding, so a uniform draw satisfies one-hot on all m attributes with
probability (3/8)^m -- effectively never. COP therefore MINIMISES ITS LOSS on
points that encode no possible instance, while the paper EVALUATES it under the
structurally valid sampler V. That is a train/eval mismatch, and it sits
underneath the paper's claim that the in-expectation regime "drops below chance
on individual instances": some of that tail may be the mismatch rather than the
guarantee type.

WHAT THIS SCRIPT DOES
---------------------
For each instance it fits the SAME COP twice -- identical solver, k, m, seed,
timeout -- changing only the distribution the samples come from:

  U  Uniform({0,1}^d)                     what the code does today
  V  Uniform over one-hot-VALID points    the structural correction

and scores BOTH rules under V, which is the distribution the paper reports. If
the tail is a guarantee-type property, the two score alike. If it is the
sampler, V-fitted rules dominate.

No LLM calls: fitting and scoring are local.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from bb_metrics import precision_hat          # noqa: E402
from cop_explainer import fit_cop_rule        # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'chk', os.path.join('research', 'check_nn_feasible_sampling.py'))
chk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chk)

RANDOM_SEED = 42
N_EVAL = 1000
DELTA = 0.05
M_SAMPLES = 5000
K = 7


def valid_samples(groups, n, rng):
    """n rows, each with exactly one active bin per one-hot group."""
    d = sum(len(g) for g in groups)
    Z = np.zeros((n, d), dtype=np.int8)
    for g in groups:
        pick = rng.integers(0, len(g), size=n)
        Z[np.arange(n), np.asarray(g)[pick]] = 1
    return Z


def precision_valid(predict, x_bin, S, groups, rng):
    """Conditional precision under V (fix S to x_S, complete validly)."""
    Z = chk.sample_onehot_valid(x_bin, S, groups, N_EVAL, rng)
    pred = int(np.asarray(predict(x_bin.reshape(1, -1))).ravel()[0])
    return 100.0 * float(np.mean(np.asarray(predict(Z)).ravel() == pred))


def main(dataset='cancer', n_instances=8):
    predict, Xtr_bin, Xte_bin, groups = chk.build_tier(dataset)
    import random
    random.seed(RANDOM_SEED)
    positions = random.sample(range(len(Xte_bin)),
                              min(n_instances, len(Xte_bin)))

    rows = []
    for it, pos in enumerate(positions, 1):
        x = Xte_bin[pos]
        rng = np.random.default_rng(RANDOM_SEED + pos)

        # --- fit on U (current behaviour): let fit_cop_rule draw its own ---
        r_u = fit_cop_rule(predict, x, k=K, m=M_SAMPLES, timeout_s=60,
                           seed=RANDOM_SEED + pos)
        # --- fit on V: same solver, structurally valid oracle samples ---
        Zv = valid_samples(groups, M_SAMPLES, rng)
        yv = np.asarray(predict(Zv)).ravel()
        r_v = fit_cop_rule(predict, x, k=K, m=M_SAMPLES, timeout_s=60,
                           seed=RANDOM_SEED + pos, samples=Zv, labels=yv)

        pu = precision_valid(predict, x, r_u.S, groups,
                             np.random.default_rng(RANDOM_SEED + 31 * pos))
        pv = precision_valid(predict, x, r_v.S, groups,
                             np.random.default_rng(RANDOM_SEED + 31 * pos))
        rows.append({'instance': pos, 'len_U': len(r_u.S), 'len_V': len(r_v.S),
                     'V_prec_of_U_fit': round(pu, 2),
                     'V_prec_of_V_fit': round(pv, 2),
                     'delta': round(pv - pu, 2),
                     'loss_U': round(r_u.empirical_loss, 4),
                     'loss_V': round(r_v.empirical_loss, 4),
                     'base_rate_U': round(r_u.base_rate_same, 3),
                     'base_rate_V': round(r_v.base_rate_same, 3),
                     'status_U': r_u.status, 'status_V': r_v.status})
        print(f'[{it}/{len(positions)}] inst {pos}: '
              f'U-fit {pu:.1f} | V-fit {pv:.1f} | delta {pv - pu:+.1f}',
              flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join('research', 'results', f'cop_sampler_check_{dataset}.csv')
    df.to_csv(out, index=False)
    print(f'\nsaved -> {out}\n')
    print('=== COP rule precision under V, by the distribution it was FIT on ===')
    print(f'  fit on U (current) : {df.V_prec_of_U_fit.mean():.2f}  '
          f'(min {df.V_prec_of_U_fit.min():.1f})')
    print(f'  fit on V (fixed)   : {df.V_prec_of_V_fit.mean():.2f}  '
          f'(min {df.V_prec_of_V_fit.min():.1f})')
    print(f'  mean improvement   : {df.delta.mean():+.2f} points')
    print(f'  instances below 50 : U-fit {(df.V_prec_of_U_fit < 50).sum()}'
          f' / V-fit {(df.V_prec_of_V_fit < 50).sum()}  of {len(df)}')
    print(f'\n  same-class base rate: U {df.base_rate_U.mean():.3f} | '
          f'V {df.base_rate_V.mean():.3f}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cancer',
         int(sys.argv[2]) if len(sys.argv) > 2 else 8)
