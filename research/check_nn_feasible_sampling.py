"""
check_nn_feasible_sampling.py — is the NN tier's erosion an artifact of
sampling infeasible points?

THE QUESTION
------------
The NN tier estimates conditional precision under D = Uniform({0,1}^d),
inherited from [Koriche §4.1]. But d is a K-bins ONE-HOT encoding: three
mutually exclusive bins per attribute. A uniform draw over {0,1}^d satisfies
one-hot on a single attribute with probability 3/8, and on all m attributes
with probability (3/8)^m -- effectively never. So the reported precision is
dominated by points that encode no possible instance, where the MLP's
behaviour is unconstrained by anything it was trained on.

Re-scoring the BT tier exposed how much this can matter: there a sufficient
reason -- which must score 100 by definition -- scored 52 (chance) under
uniform D and 92 under a feasible D. This script asks whether the NN tier's
headline erosion survives the same correction.

THREE DISTRIBUTIONS, SAME ESTIMATOR
-----------------------------------
  U  Uniform({0,1}^d)                 the tier's current choice; reproduced
                                      here to verify the rebuild is faithful
  V  Uniform over ONE-HOT-VALID points every attribute keeps exactly one
                                      active bin; the structural correction
  E  Empirical binarized training set  feasible by construction

All three fix S to x_S by construction and query the same MLP; only the
completion distribution differs. U and V use n=1000 (Hoeffding half-width
0.043 at delta=0.05); E uses whatever the training set supplies, so no bound
is claimed for it.

REPRODUCIBILITY
---------------
The model and split are rebuilt exactly as `run_comparison_nn.py` builds them
(seed 42, quantile 3-bin one-hot, MLPClassifier defaults). The script asserts
that its recomputed U-precision matches the value stored in the CSV; if that
gate fails, the rebuild diverged and no other number here can be trusted.

The length-matched `cop_k` arm lives in a separate sweep CSV and is re-scored
the same way, into its own output file; it is what the COP-vs-LmPAXp
head-to-head reads, so it must be available under V like everything else.

USAGE
-----
    python research/check_nn_feasible_sampling.py            # all 5 datasets
    python research/check_nn_feasible_sampling.py heart
Run from the project root.
"""

from __future__ import annotations

import ast
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
sys.path.insert(0, ROOT)

from sklearn.neural_network import MLPClassifier  # noqa: E402
from run_calibration_nn import load_dataset, binarize  # noqa: E402
from bb_metrics import precision_hat, precision_training  # noqa: E402

RESULTS = os.path.join(ROOT, 'research', 'results')
RANDOM_SEED = 42          # identical to run_comparison_nn.py
N_SAMPLES = 1000
DELTA = 0.05
T_WEAK = 0.95             # WeakPAXp threshold, identical to the NN tier
DATASETS = ['cancer', 'compas', 'diabetes', 'employee', 'heart']
GATE_TOL = 0.5            # percentage points; U-recompute vs stored value


def build_tier(dataset: str):
    """Rebuild the NN tier exactly as run_comparison_nn.py does."""
    X_train, X_test, y_train, y_test = load_dataset(dataset)
    Xtr_bin, Xte_bin, _names, kbins = binarize(X_train, X_test)
    mlp = MLPClassifier(random_state=RANDOM_SEED).fit(Xtr_bin.astype(float), y_train)

    def predict(Z):
        return mlp.predict(np.atleast_2d(np.asarray(Z)).astype(float))

    # One-hot groups: consecutive index blocks, one block per original column.
    groups, j = [], 0
    for edges in kbins.bin_edges_:
        nb = len(edges) - 1
        groups.append(np.arange(j, j + nb))
        j += nb
    assert j == Xtr_bin.shape[1], 'group indices do not cover the binary space'
    return predict, Xtr_bin, Xte_bin, groups


def sample_onehot_valid(x_bin, S, groups, n, rng):
    """
    n draws from Uniform({one-hot-valid points} | z_S = x_S).

    Per attribute group: if S fixes an active bin, the group is determined; if
    S only fixes bins to zero, the active bin is drawn uniformly from those the
    constraint still permits. Every returned row is a valid encoding AND agrees
    with x on S, so all n samples are effective.
    """
    d = x_bin.shape[0]
    Sset = set(int(j) for j in S)
    Z = np.zeros((n, d), dtype=np.int8)
    for g in groups:
        fixed = [j for j in g if j in Sset]
        on = [j for j in fixed if x_bin[j] == 1]
        if on:                                  # active bin pinned by S
            Z[:, on[0]] = 1
            continue
        banned = {j for j in fixed if x_bin[j] == 0}
        allowed = [j for j in g if j not in banned]
        if not allowed:                         # unreachable for a valid x
            continue
        pick = rng.choice(allowed, size=n)
        Z[np.arange(n), pick] = 1
    return Z


def precision_valid(predict, x_bin, S, groups, n, seed):
    rng = np.random.default_rng(seed)
    pred = int(np.asarray(predict(x_bin.reshape(1, -1))).ravel()[0])
    Z = sample_onehot_valid(x_bin, S, groups, n, rng)
    labels = np.asarray(predict(Z)).ravel()
    return 100.0 * float(np.mean(labels == pred))


def parse_S(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    try:
        v = ast.literal_eval(str(cell))
    except (ValueError, SyntaxError):
        return None
    return [int(i) for i in v] if isinstance(v, (list, tuple)) else None


def run_dataset(dataset: str, suffix: str = ''):
    csv = os.path.join(RESULTS, f'{dataset}_nn_comparison{suffix}.csv')
    if not os.path.exists(csv):
        print(f'  [skip] {csv} missing')
        return None, None
    df = pd.read_csv(csv)
    predict, Xtr_bin, Xte_bin, groups = build_tier(dataset)

    rows, gate = [], []
    for _, r in df.iterrows():
        pos = int(r['instance'])
        if not (0 <= pos < len(Xte_bin)):
            continue
        x = Xte_bin[pos]
        rec = dict(dataset=dataset, instance=pos, arm=str(r['arm']))
        if 'cop_variant' in r:
            rec['cop_variant'] = r['cop_variant']
        if 'len_S_arm' in r:
            rec['len_S_arm'] = r['len_S_arm']
        for tag, col, stored in (('arm', 'S_arm', 'baseline_precision'),
                                 ('v4', 'S_v4', 'score_v4'),
                                 ('v5', 'S_v5', 'score_v5'),
                                 ('v6', 'S_v6', 'score_v6')):
            S = parse_S(r.get(col))
            if S is None:
                for t in 'UVE':
                    rec[f'{t}_{tag}'] = rec[f'{t}wp_{tag}'] = None
                continue
            seed = RANDOM_SEED + 31 * pos
            u = 100.0 * precision_hat(predict, x, S, n=N_SAMPLES,
                                      delta=DELTA, seed=seed).precision
            v = precision_valid(predict, x, S, groups, N_SAMPLES, seed)
            e, n_nb = precision_training(predict, x, S, Xtr_bin)
            e = None if e is None else 100.0 * (1 - e)
            rec[f'U_{tag}'], rec[f'V_{tag}'] = round(u, 2), round(v, 2)
            rec[f'E_{tag}'] = None if e is None else round(e, 2)
            # WeakPAXp at T, tested on the point estimate, per distribution.
            rec[f'Uwp_{tag}'] = float(u >= 100 * T_WEAK)
            rec[f'Vwp_{tag}'] = float(v >= 100 * T_WEAK)
            rec[f'Ewp_{tag}'] = None if e is None else float(e >= 100 * T_WEAK)
            rec[f'nnb_{tag}'] = n_nb
            if stored in r and pd.notna(r[stored]):
                gate.append(abs(u - float(r[stored])))
        rec['v5_rounds'] = r.get('v5_rounds')
        rows.append(rec)

    out = pd.DataFrame(rows)
    gate_max = max(gate) if gate else float('nan')
    return out, gate_max


def report(allr):
    """Per-arm means under each distribution, plus erosion and retention."""
    for tag, label in (('U', 'Uniform {0,1}^d  (current choice; mostly INFEASIBLE points)'),
                       ('V', 'One-hot-VALID uniform  (structural correction)'),
                       ('E', 'Empirical training distribution')):
        cols = [f'{tag}_arm', f'{tag}_v4', f'{tag}_v5', f'{tag}_v6']
        if not set(cols).issubset(allr.columns):
            continue
        g = allr.groupby('arm')[cols].mean().round(2)
        g['erosion_v4'] = (g[f'{tag}_arm'] - g[f'{tag}_v4']).round(2)
        g['retained_v6'] = (100 * g[f'{tag}_v6'] / g[f'{tag}_arm']).round(1)
        wp = [f'{tag}wp_arm', f'{tag}wp_v4', f'{tag}wp_v6']
        if set(wp).issubset(allr.columns):
            g = g.join(allr.groupby('arm')[wp].mean().round(3))
        print(f'\n=== D = {label} ===')
        print(g.to_string())
    if 'v5_rounds' in allr.columns:
        print('\nmean v5_rounds by arm:')
        print(allr.groupby('arm')['v5_rounds'].mean().round(2).to_string())


def run_sweep(datasets, suffix, out_name):
    """Re-score one sweep (main 5-arm, or the length-matched cop_k) under U/V/E."""
    frames, gates = [], {}
    for ds in datasets:
        print(f'[{ds}{suffix}] rebuilding tier and re-estimating under U / V / E ...',
              flush=True)
        out, gate_max = run_dataset(ds, suffix)
        if out is None:
            continue
        frames.append(out)
        gates[ds] = gate_max
        print(f'  {len(out)} rows | reproduction gate: max |U_recomputed - '
              f'stored| = {gate_max:.2f} pts', flush=True)

    if not frames:
        print(f'  nothing to do for suffix {suffix!r}')
        return None
    allr = pd.concat(frames, ignore_index=True)
    dest = os.path.join(RESULTS, out_name)
    allr.to_csv(dest, index=False)
    print(f'\nwrote {dest}')

    bad = {k: v for k, v in gates.items() if not (v == v) or v > GATE_TOL}
    if bad:
        print(f'\n*** REPRODUCTION GATE FAILED for {bad} ***')
        print('The rebuilt model does not match the one that produced the CSVs; '
              'the V and E columns below cannot be compared to the paper.')
    report(allr)
    return allr


def main(argv):
    datasets = argv[1:] or DATASETS
    main_sweep = run_sweep(datasets, '', 'nn_feasible_sampling_check.csv')
    # The length-matched COP arm: same estimator, separate sweep file, so the
    # COP-vs-LmPAXp head-to-head is not stuck on the invalid uniform sampler.
    run_sweep(datasets, '_cop_k', 'nn_feasible_sampling_cop_k.csv')
    return 0 if main_sweep is not None else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
