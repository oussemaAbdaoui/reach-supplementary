"""
analyze_judge.py — does a text-reading verifier detect guarantee erosion?

Joins the judge scores (`*_nn_judge.csv`, from run_judge.py) to the measured
conditional precision under the structurally valid sampler
(`nn_feasible_sampling_check.csv`) and evaluates the four predictions
registered before the judge was run.

  J4  GATE  test-retest |pass1 - pass2| small (< ~8 pts). Check FIRST: a judge
            that disagrees with itself makes a flat curve uninformative rather
            than a finding.
  J1        judge scores flat across v4 -> v6, the manipulation that moves
            LmPAXp fidelity ~+14 pts.
  J2        per-arm correlation between judge FAITHFULNESS and measured
            fidelity is near zero.
  J3        the judge does not reproduce the arm ordering fidelity gives.

READING J1 CORRECTLY
--------------------
A v4 -> v6 rise in judge scores does NOT by itself mean the judge sees
fidelity. v6 is structured per-condition generation: its explanations are
longer and enumerate conditions explicitly, and a judge rewards that form.
The discriminating question is whether the judge rises *in proportion to
fidelity*. Fidelity recovers for the rule arms and plateaus for the
attribution arms; a judge tracking form rises for both alike. So J2 --
per-arm correlation -- decides the claim, and J1 alone cannot.

USAGE
    python research/analyze_judge.py            # all datasets found
Run from the project root.
"""

from __future__ import annotations

import os
import glob
import sys

import pandas as pd

RESULTS = os.path.join('research', 'results')
DIMS = ['faithfulness', 'coherence', 'completeness', 'halluc_risk']
STRATS = ['v4', 'v5', 'v6']


def load_judge():
    frames = []
    for f in sorted(os.listdir(RESULTS)):
        if f.endswith('_nn_judge.csv'):
            frames.append(pd.read_csv(os.path.join(RESULTS, f)))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_fidelity():
    """V-sampler precision, melted to one row per (dataset, instance, arm, strategy)."""
    p = os.path.join(RESULTS, 'nn_feasible_sampling_check.csv')
    if not os.path.exists(p):
        return None
    f = pd.read_csv(p)
    rows = []
    for st in STRATS:
        col = f'V_{st}'
        if col not in f.columns:
            continue
        sub = f[['dataset', 'instance', 'arm', 'V_arm', col]].copy()
        sub['strategy'] = st
        sub = sub.rename(columns={col: 'fidelity'})
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


KEY = ['dataset', 'instance', 'arm', 'strategy']


def gate_j4(j):
    """Self-consistency across passes, and what averaging buys.

    The gate is reported two ways on purpose. The per-judgement spread is what
    was pre-registered and is the honest headline. But every claim is made on
    averages, so the standard error of the averaged estimate is what actually
    governs whether an effect is readable -- and averaging N passes shrinks it
    by sqrt(N). Reporting only the first would overstate the problem; reporting
    only the second would quietly move the goalposts.
    """
    npass = j.groupby(KEY)['pass'].nunique()
    multi = npass[npass > 1].index
    if not len(multi):
        print('J4  GATE : no repeated judgements -- cannot assess '
              'self-consistency. Treat every result below as provisional.')
        return None
    rep = j.set_index(KEY).loc[multi].reset_index()
    n_rep = len(multi)
    print(f'J4  GATE : {n_rep} explanations judged {rep.groupby(KEY)["pass"].nunique().max()}x')
    worst = 0.0
    for d in DIMS:
        g = rep.groupby(KEY)[d]
        spread = (g.max() - g.min()).mean()          # per-judgement disagreement
        sem = (g.std() / g.count().pow(0.5)).mean()  # SE of the averaged score
        worst = max(worst, spread)
        print(f'           {d:14s} mean spread {spread:5.2f} | '
              f'SE of mean {sem:5.2f}')
    verdict = 'PASS' if worst < 8 else 'FAIL'
    print(f'           worst per-judgement spread {worst:.2f} -> gate {verdict}')
    if verdict == 'FAIL':
        print('           Pre-registered gate FAILS on single judgements. Read '
              'effects against the SE of the mean above, and treat any effect '
              'smaller than ~3x that SE as unreadable.')
    return worst


def average_passes(j):
    """Collapse passes to one averaged score per explanation."""
    return j.groupby(KEY, as_index=False)[DIMS].mean()


def main():
    if not glob.glob(os.path.join('research', 'results', '*_nn_comparison_*.csv')):
        sys.exit('research/results/ is empty: no results ship with this bundle. '
                 'Generate them first (README, "Reproducing the study").')
    j = load_judge()
    if j is None:
        print('no *_nn_judge.csv found -- run run_judge.py first')
        return 1
    j = j[j['parse_ok'] == 1]
    print(f'judge = {j["judge_model"].iloc[0]} | {len(j)} judgements '
          f'across {j["dataset"].nunique()} dataset(s) | '
          f'parse ok {100*load_judge()["parse_ok"].mean():.1f}%\n')

    gate_j4(j)
    first = average_passes(j)          # every claim below is on pass-averages

    print('\nJ1       : judge score by strategy (is it flat?)')
    print(first.groupby('strategy')[DIMS].mean().round(2).to_string())

    fid = load_fidelity()
    if fid is None:
        print('\n[no nn_feasible_sampling_check.csv -- skipping J2/J3]')
        return 0
    m = first.merge(fid, on=['dataset', 'instance', 'arm', 'strategy'], how='inner')
    print(f'\njoined {len(m)} rows with V-sampler fidelity')

    print('\nJ1/J2    : per-arm v4 -> v6 movement, judge vs measured fidelity')
    g = m[m.strategy.isin(['v4', 'v6'])]
    piv = g.pivot_table(index='arm', columns='strategy',
                        values=['faithfulness', 'fidelity'])
    out = pd.DataFrame({
        'judge_v4': piv[('faithfulness', 'v4')].round(1),
        'judge_v6': piv[('faithfulness', 'v6')].round(1),
        'judge_d':  (piv[('faithfulness', 'v6')] - piv[('faithfulness', 'v4')]).round(1),
        'fid_v4':   piv[('fidelity', 'v4')].round(1),
        'fid_v6':   piv[('fidelity', 'v6')].round(1),
        'fid_d':    (piv[('fidelity', 'v6')] - piv[('fidelity', 'v4')]).round(1),
    })
    print(out.to_string())
    print('\n  If judge_d is large where fid_d is small (the attribution arms),')
    print('  the judge is tracking FORM, not fidelity.')

    print('\nJ2       : correlation(judge dimension, measured fidelity), per arm')
    rows = {}
    for arm, sub in m.groupby('arm'):
        rows[arm] = {d: round(sub[d].corr(sub['fidelity']), 3) for d in DIMS}
    rows['ALL'] = {d: round(m[d].corr(m['fidelity']), 3) for d in DIMS}
    print(pd.DataFrame(rows).T.to_string())

    print('\nJ3       : arm ordering, judge faithfulness vs measured fidelity')
    o_j = m.groupby('arm')['faithfulness'].mean().sort_values(ascending=False)
    o_f = m.groupby('arm')['fidelity'].mean().sort_values(ascending=False)
    print('  judge   :', ' > '.join(o_j.index))
    print('  fidelity:', ' > '.join(o_f.index))
    print('  identical ordering:', list(o_j.index) == list(o_f.index))
    return 0


if __name__ == '__main__':
    sys.exit(main())
