"""
run_cop_sweep.py — replication of [Koriche Fig. 2] on our data, no LLM.

Precision as a function of explanation SIZE k, on the NN-tier black box
(MLP on K-bins features), for:
  cop@k    : loss-minimal rule at EXACT size k (k = 1..K_MAX), m=5000
  cop      : floating-K Table-1 optimum (the loss-optimal size), m=5000
  lmpaxp   : the certified explanation at its natural length (reference point)

This is simultaneously:
  - the paper-protocol comparison the LLM grid lacked (equal size budget), and
  - the size/precision trade-off curve of the report's hypothesis (§4),
    obtained from [K]'s own experimental design.

Same 15 seed-42 instances as the LLM run -> rows join with
<ds>_nn_comparison.csv on `instance`.

Usage: python run_cop_sweep.py [dataset]   (default diabetes)
       XAI_N_INSTANCES / XAI_KMAX to override (defaults 15 / 9).
Output: research/results/<ds>_cop_sweep.csv + printed pivot.
"""

from __future__ import annotations

import os
import random
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier

from bb_metrics import precision_hat
from cop_explainer import fit_cop_rule, theorem1_m
from lmpaxp_explainer import fit_lmpaxp
from run_calibration_nn import load_dataset, binarize

# The V estimator comes from the script that produced the paper's V columns, so
# a precision_V here is directly comparable with one there rather than being a
# second implementation of the same idea.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research'))
from check_nn_feasible_sampling import precision_valid  # noqa: E402

DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else 'diabetes'
N_INSTANCES  = int(os.getenv('XAI_N_INSTANCES', '15'))
K_MAX        = int(os.getenv('XAI_KMAX', '9'))
RANDOM_SEED  = 42
M_LEARN      = 5000        # oracle queries are free (local MLP); Thm-1 tighter
N_EVAL       = 1000
T            = 0.95
N_WORKERS    = int(os.getenv('XAI_WORKERS', '1'))   # 1 = deterministic CP-SAT
CSV_TAG      = os.getenv('XAI_CSV_TAG', '')         # e.g. '_det' -> separate CSV

X_train, X_test, y_train, y_test = load_dataset(DATASET_NAME)
Xtr_bin, Xte_bin, bin_names, _kbins = binarize(X_train, X_test)
d = Xtr_bin.shape[1]

# One-hot groups, one consecutive block per original column.
_ONEHOT_GROUPS, _j = [], 0
for _col, _edges in zip(X_train.columns, _kbins.bin_edges_):
    _nb = len(_edges) - 1
    _ONEHOT_GROUPS.append(np.arange(_j, _j + _nb))
    _j += _nb


def valid_oracle_samples(m, rng):
    """m draws with exactly one active bin per attribute.

    [K §2/§4.1] draws the example oracle from Uniform({0,1}^d). Here d is a
    one-hot encoding, so a uniform draw is realisable with probability
    (3/8)^#attributes -- effectively never -- and the learned rule would
    minimise loss on impossible points while being scored on realisable ones.
    Measured cost of that mismatch on cancer: 8.5 precision points on average
    and the whole below-chance tail (18.9 -> 58.7 worst case). Solver, k, m,
    seed and timeout are unchanged; only where the samples come from.
    """
    Z = np.zeros((m, d), dtype=np.int8)
    for g in _ONEHOT_GROUPS:
        Z[np.arange(m), g[rng.integers(0, len(g), size=m)]] = 1
    return Z

mlp = MLPClassifier(random_state=RANDOM_SEED).fit(Xtr_bin.astype(float), y_train)
predict = lambda Z: mlp.predict(np.atleast_2d(np.asarray(Z)).astype(float))
print(f'{DATASET_NAME}: d={d} | MLP test acc '
      f'{mlp.score(Xte_bin.astype(float), y_test):.4f} | m_learn={M_LEARN} '
      f'(Thm-1 m_ref for eps=0.1: {theorem1_m(d, K_MAX)})', flush=True)

random.seed(RANDOM_SEED)
positions = random.sample(range(len(Xte_bin)), min(N_INSTANCES, len(Xte_bin)))

# Resume: instances already complete in the CSV are skipped (kill-safe).
OUT = f'research/results/{DATASET_NAME}_cop_sweep{CSV_TAG}.csv'
rows, done = [], set()
if os.path.exists(OUT):
    _prev = pd.read_csv(OUT)
    per_inst = _prev.groupby('instance').size()
    done = set(per_inst[per_inst >= K_MAX + 2].index)   # lmpaxp + cop_free + K_MAX
    rows = _prev[_prev.instance.isin(done)].to_dict('records')
    print(f'resume: {len(done)} instances already complete, skipping', flush=True)

for it, pos in enumerate(positions):
    if pos in done:
        continue
    x = Xte_bin[pos]
    seed = RANDOM_SEED + pos
    rng = np.random.default_rng(seed)
    Z = valid_oracle_samples(M_LEARN, rng)      # one draw / instance, realisable
    zl = predict(Z)

    def add(method, k_req, S, extra=None):
        # TWO ESTIMATORS, BOTH STORED. The rules are FITTED on one-hot-valid
        # oracle samples, but `precision` scores them under Uniform({0,1}^d) --
        # the sampler §5.5 rejects, whose invalid points produce the entire
        # below-chance tail. Selecting a length on that curve and then reporting
        # the winner under V would optimise one metric and report another, so
        # the valid-sampler estimate is stored alongside it and is what
        # _cop_best_k selects on. `precision` is kept unchanged so every stored
        # sweep stays comparable with the ones written before this column.
        est = precision_hat(predict, x, S, n=N_EVAL, seed=RANDOM_SEED + 31 * pos)
        pv = precision_valid(predict, x, S, _ONEHOT_GROUPS, N_EVAL,
                             RANDOM_SEED + 31 * pos)
        rows.append({'dataset': DATASET_NAME, 'instance': pos, 'method': method,
                     'k_requested': k_req, 'len_S': len(S),
                     'precision': round(est.precision * 100, 2),
                     'precision_V': round(pv, 2),
                     'weak_paxp': est.precision >= T,
                     'weak_paxp_V': pv >= 100 * T,
                     # S itself: without it a stored sweep cannot be re-scored
                     # under a different distribution without re-running the
                     # solver (4h35m for cancer at k=18).
                     'S': str(list(S)), **(extra or {})})
        return est

    lmp = fit_lmpaxp(predict, x, T=T, seed=seed)
    add('lmpaxp', None, lmp.S, {'precision_cert': round(lmp.precision_cert*100, 2)})

    r_free = fit_cop_rule(predict, x, k=K_MAX, m=M_LEARN, timeout_s=60,
                          seed=seed, samples=Z, labels=zl, n_workers=N_WORKERS)
    add('cop_free', K_MAX, r_free.S,
        {'cop_loss': round(r_free.empirical_loss, 4),
         'base_rate_same': round(r_free.base_rate_same, 3)})

    for k in range(1, K_MAX + 1):
        try:
            r = fit_cop_rule(predict, x, k=k, m=M_LEARN, timeout_s=60, seed=seed,
                             samples=Z, labels=zl, exact_k=True, n_workers=N_WORKERS)
        except RuntimeError:
            # solver returned UNKNOWN at 60 s — one retry with a longer budget,
            # then skip this (instance, k) cell rather than kill the sweep
            try:
                r = fit_cop_rule(predict, x, k=k, m=M_LEARN, timeout_s=180,
                                 seed=seed, samples=Z, labels=zl, exact_k=True, n_workers=N_WORKERS)
            except RuntimeError as e:
                print(f'  !! inst {pos} k={k}: {e} — cell skipped', flush=True)
                rows.append({'dataset': DATASET_NAME, 'instance': pos,
                             'method': 'cop@k', 'k_requested': k,
                             'len_S': None, 'precision': None,
                             'precision_V': None, 'weak_paxp': None,
                             'weak_paxp_V': None, 'S': None,
                             'cop_status': 'UNKNOWN'})
                continue
        add('cop@k', k, r.S, {'cop_loss': round(r.empirical_loss, 4),
                              'cop_status': r.status})

    lp = [r for r in rows if r['instance'] == pos]
    lmp_p = next(r['precision'] for r in lp if r['method'] == 'lmpaxp')
    ck = {r['k_requested']: r['precision'] for r in lp
          if r['method'] == 'cop@k' and r['precision'] is not None}
    print(f'[{it+1:02d}/{len(positions)}] inst {pos} | lmpaxp {lmp_p:.1f}/{len(lmp.S)} '
          f'| cop_free {lp[1]["precision"]:.1f}/{len(r_free.S)} | cop@k: '
          + ' '.join(f'{k}:{ck[k]:.0f}' for k in sorted(ck)), flush=True)
    # incremental save — a killed run keeps every completed instance
    os.makedirs('research/results', exist_ok=True)
    pd.DataFrame(rows).to_csv(
        f'research/results/{DATASET_NAME}_cop_sweep{CSV_TAG}.csv', index=False)

df = pd.DataFrame(rows)
out = f'research/results/{DATASET_NAME}_cop_sweep{CSV_TAG}.csv'
df.to_csv(out, index=False)
print(f'\nsaved -> {out}')

print('\n=== [K Fig. 2] style: mean precision by size ===')
sw = df[df.method == 'cop@k'].groupby('k_requested')['precision'].agg(['mean', 'std'])
print(sw.round(2).to_string())
fr = df[df.method == 'cop_free']
lm = df[df.method == 'lmpaxp']
print(f'\ncop_free : precision {fr.precision.mean():.2f} at mean len {fr.len_S.mean():.2f} '
      f'(loss {fr.cop_loss.mean():.3f}, base_rate_same {fr.base_rate_same.mean():.3f})')
print(f'lmpaxp   : precision {lm.precision.mean():.2f} at mean len {lm.len_S.mean():.2f} '
      f'(certified reference)')
k_near = int(round(lm.len_S.mean()))
if k_near in sw.index:
    print(f'\nequal-budget check: cop@{k_near} = {sw.loc[k_near, "mean"]:.2f} '
          f'vs lmpaxp = {lm.precision.mean():.2f} at ~same length')
