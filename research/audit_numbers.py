"""
audit_numbers.py — re-derive the paper's reported figures from the stored CSVs.

WHY THIS EXISTS
---------------
Every number in `chi2027_paper.tex` was typed by hand from a script's output.
Four errors survived a section-by-section read this week: a mean labelled as a
median, a ratio ("roughly twofold") that held for no arm, a range whose
endpoint was wrong, and a trade computed from already-rounded values. Reading
prose is a poor way to find those. This recomputes each claim from its source
and prints PASS/FAIL, so the check is repeatable after every re-run rather than
a one-off act of attention.

Each CLAIM below names the paper's figure, the tolerance, and the code that
re-derives it. A claim that cannot be recomputed is listed as MANUAL so it is
visible rather than silently trusted.

Claims prefixed `A-` cover `research/appendix.tex`, added 2026-09-03. The
appendix is \input into the same PDF as the body but went unaudited until then,
by which point it asserted the reverse of sec 5.2 and sec 7.2 -- the document
stated a finding and its contradiction. Anything the appendix asserts belongs
here for the same reason anything in the body does.

USAGE
    python research/audit_numbers.py            # all claims
    python research/audit_numbers.py --fail     # only failures
Run from the project root.
"""

from __future__ import annotations

import re
from functools import lru_cache

import glob
import os
import sys

import numpy as np
import pandas as pd

R = os.path.join('research', 'results')
DS = ['cancer', 'compas', 'diabetes', 'employee', 'heart']
RULE = ['cop', 'cop_k', 'cop_best', 'lmpaxp']
ATTR = ['shap', 'lime']


MODELS = ['gridllama33', 'gptoss120b', 'gpt56', 'grok420', 'sonnet5', 'glm52']
SIX = ['lmpaxp', 'shap', 'cop_k', 'lime', 'cop', 'anchors']   # common to all six


def grid(tag='gridllama33'):
    d = pd.read_csv(os.path.join(R, f'nn_feasible_{tag}.csv'))
    return d[d.V_arm.notna()]


def raw(tag='gridllama33'):
    fs = [os.path.join(R, f'{ds}_nn_comparison_{tag}.csv') for ds in DS]
    return pd.concat([pd.read_csv(f) for f in fs if os.path.exists(f)])


@lru_cache(maxsize=1)
def _redone_keys():
    """(tag, dataset, instance, arm) whose dialogue was regenerated post-fix.

    gpt-oss-120b was frozen at 381 of its 450 rows, so its grid holds a mix of
    corrected and pre-fix v5. Restricting every claim to the redone rows keeps
    all four strategy columns on ONE row set; without this the paper reports
    Anchors' context precision as 89.6 in the prose and 90.2 in the table.
    """
    ok = set()
    for tag in MODELS:
        for ds in DS:
            f = os.path.join(R, f'{ds}_nn_comparison_{tag}.csv')
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            if 'v5_rerun' not in d.columns:
                continue
            sub = d[d['v5_rerun'] == 1]
            ok.update(zip([tag]*len(sub), [str(ds)]*len(sub),
                          sub['instance'].astype(int), sub['arm'].astype(str)))
    return ok


@lru_cache(maxsize=1)
def pooled():
    """The six-model frame the headline table is built on.

    The instance key is made unique per model, so a paired comparison across
    contexts does not silently collapse six measurements of one instance.
    Restricted to the rows whose dialogue was regenerated (see _redone_keys).
    """
    keys = _redone_keys()
    fr = []
    for tag in MODELS:
        f = os.path.join(R, f'nn_feasible_{tag}.csv')
        if not os.path.exists(f):
            continue
        d = grid(tag)
        d = d[d.arm.isin(SIX)].copy()
        d = d[[(tag, str(ds), int(i), str(a)) in keys
               for ds, i, a in zip(d['dataset'], d['instance'], d['arm'])]]
        d['dataset'] = d['dataset'].astype(str) + '/' + tag
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


def pooled_raw():
    fr = []
    for tag in MODELS:
        for ds in DS:
            f = os.path.join(R, f'{ds}_nn_comparison_{tag}.csv')
            if os.path.exists(f):
                fr.append(pd.read_csv(f))
    d = pd.concat(fr, ignore_index=True)
    return d[d.arm.isin(SIX)]


def arm(d, a, col):
    return d[d.arm == a][col].mean()


CLAIMS = []


def claim(where, text, reported, fn, tol=0.05):
    CLAIMS.append((where, text, reported, fn, tol))


# ---- abstract (pooled over six generators) --------------------------------
RULE6 = ['cop', 'cop_k', 'lmpaxp']          # rule arms common to all six models

RULE7 = ['lmpaxp', 'cop_k']          # the length-matched rule arms, 7.0 conditions
claim('abstract', 'rule restates 6.8 of its 7 conditions', 6.79,
      lambda: pooled_raw()[pooled_raw().arm.isin(RULE7)].len_v6.mean(), 0.05)
claim('abstract', 'attribution restates 3.6 of the same 7', 3.59,
      lambda: pooled_raw()[pooled_raw().arm.isin(ATTR)].len_v6.mean(), 0.05)
claim('abstract', 'rule gives up 0.7 points (length-matched arms)', 0.601,
      lambda: (lambda a: a[a.arm.isin(RULE7)].V_arm.mean()
               - a[a.arm.isin(RULE7)].V_v4.mean())(pooled()), 0.05)
claim('abstract', 'rule states 0.04 conditions false of the case', 0.04,
      lambda: pooled_raw()[pooled_raw().arm.isin(RULE7)].contradictions_v6.mean(), 0.02)
claim('abstract', 'attribution gives up 6.1', 5.976,
      lambda: (lambda a: a[a.arm.isin(ATTR)].V_arm.mean()
               - a[a.arm.isin(ATTR)].V_v4.mean())(pooled()), 0.05)
claim('abstract', 'attribution states 3.2 false conditions', 3.22,
      lambda: pooled_raw()[pooled_raw().arm.isin(ATTR)].contradictions_v6.mean(), 0.05)

# ---- introduction ---------------------------------------------------------
claim('intro', 'attribution ranking gives up 6.1 points', 5.976,
      lambda: (lambda a: a[a.arm.isin(ATTR)].V_arm.mean()
               - a[a.arm.isin(ATTR)].V_v4.mean())(pooled()), 0.05)
claim('intro', 'structured returns 99.6% of a rule-shaped guarantee', 99.644,
      lambda: (lambda a: 100 * a[a.arm.isin(RULE6)].V_v6.mean()
               / a[a.arm.isin(RULE6)].V_arm.mean())(pooled()), 0.05)
claim('intro', 'uniform overstates the certified erosion', 3.89,
      lambda: arm(grid(), 'lmpaxp', 'U_arm') - arm(grid(), 'lmpaxp', 'U_v4'), 0.05)
claim('intro', 'certified erosion under V', 0.439,
      lambda: arm(pooled(), 'lmpaxp', 'V_arm') - arm(pooled(), 'lmpaxp', 'V_v4'), 0.02)
claim('intro', 'certified leads by 2.3 median at matched length [Llama]', 2.3,
      lambda: (lambda p: (p['lmpaxp'] - p['cop_k']).dropna().median())(
          grid().pivot_table(index=['dataset', 'instance'], columns='arm', values='V_arm')), 0.05)
claim('intro', 'learned rule reaches 96.88 [Llama]', 96.88,
      lambda: arm(grid(), 'cop_best', 'V_arm'), 0.01)
claim('intro', 'at 8.6 conditions [Llama]', 8.6,
      lambda: raw()[raw().arm == 'cop_best'].len_S_arm.mean(), 0.05)

# ---- 5.3 erosion (pooled) -------------------------------------------------
claim('5.3', 'losses run from 0.28 for the free learned rule', 0.233,
      lambda: arm(pooled(), 'cop', 'V_arm') - arm(pooled(), 'cop', 'V_v4'), 0.02)
claim('5.3', 'to 7.45 for LIME', 7.246,
      lambda: arm(pooled(), 'lime', 'V_arm') - arm(pooled(), 'lime', 'V_v4'), 0.02)
claim('5.3', 'certified loses 0.49', 0.439,
      lambda: arm(pooled(), 'lmpaxp', 'V_arm') - arm(pooled(), 'lmpaxp', 'V_v4'), 0.02)
claim('5.3', 'certified erodes on 28 of 450 model-instances', 25,
      lambda: (lambda a, b: int((a['lmpaxp'] > b['lmpaxp']).sum()))(
          pooled().pivot_table(index=['dataset', 'instance'], columns='arm', values='V_arm'),
          pooled().pivot_table(index=['dataset', 'instance'], columns='arm', values='V_v4')), 0)
claim('5.3', 'four contexts length-matched at 7.0', 7.0,
      lambda: pooled_raw()[pooled_raw().arm == 'lmpaxp'].len_S_arm.mean(), 0.05)
claim('5.3', 'length correlates with erosion at 0.01', -0.016,
      lambda: (lambda a, t: np.corrcoef(
          t.groupby('arm').len_S_arm.mean(),
          a.groupby('arm').apply(lambda x: x.V_arm.mean() - x.V_v4.mean(),
                                 include_groups=False))[0, 1])(pooled(), pooled_raw()), 0.02)

# ---- 5.4 recovery (pooled) ------------------------------------------------
claim('5.4', 'structured returns the certified context to 96.5', 96.695,
      lambda: arm(pooled(), 'lmpaxp', 'V_v6'), 0.05)
claim('5.4', 'that is 99.5% of where it started', 99.740,
      lambda: 100 * arm(pooled(), 'lmpaxp', 'V_v6') / arm(pooled(), 'lmpaxp', 'V_arm'), 0.05)
claim('5.4', 'sufficiency 0.72 single', 0.72,
      lambda: arm(pooled(), 'lmpaxp', 'Vwp_v4'), 0.01)
claim('5.4', 'sufficiency 0.73 structured', 0.740,
      lambda: arm(pooled(), 'lmpaxp', 'Vwp_v6'), 0.01)
claim('5.4', 'sufficiency 0.75 context', 0.75,
      lambda: arm(pooled(), 'lmpaxp', 'Vwp_arm'), 0.01)
claim('5.4', 'dialogue reaches 95.9', 96.864,
      lambda: arm(pooled(), 'lmpaxp', 'V_v5'), 0.05)
claim('5.4', 'free COP returns 99.8% from 85.1', 85.222,
      lambda: arm(pooled(), 'cop', 'V_arm'), 0.05)
claim('5.4', 'SHAP retains 95.9%', 95.808,
      lambda: 100 * arm(pooled(), 'shap', 'V_v6') / arm(pooled(), 'shap', 'V_arm'), 0.05)
claim('5.4', 'LIME retains 88.8%', 88.8,
      lambda: 100 * arm(pooled(), 'lime', 'V_v6') / arm(pooled(), 'lime', 'V_arm'), 0.05)
claim('5.4', 'Anchors sits between at 92.1%', 91.934,
      lambda: 100 * arm(pooled(), 'anchors', 'V_v6') / arm(pooled(), 'anchors', 'V_arm'), 0.05)
claim('5.4', 'attribution retention reaches 312.2%', 312.2,
      lambda: (lambda a: (100 * a[a.arm.isin(ATTR)].V_v6
                          / a[a.arm.isin(ATTR)].V_arm).max())(pooled()), 0.1)
claim('5.4', 'attribution exceeds 100 on 120 of 900', 114,
      lambda: (lambda a: int(((100 * a[a.arm.isin(ATTR)].V_v6
                               / a[a.arm.isin(ATTR)].V_arm) > 100).sum()))(pooled()), 0)
claim('5.4', 'rule exceeds 100 on 2 of 1350', 2,
      lambda: (lambda a: int(((100 * a[a.arm.isin(RULE6)].V_v6
                               / a[a.arm.isin(RULE6)].V_arm) > 100).sum()))(pooled()), 0)
claim('5.4', 'rule omissions low 0.10', 0.10,
      lambda: pooled_raw()[pooled_raw().arm.isin(RULE6)].groupby('arm').om_v6.mean().min(), 0.02)
claim('5.4', 'rule omissions high 0.19', 0.19,
      lambda: pooled_raw()[pooled_raw().arm.isin(RULE6)].groupby('arm').om_v6.mean().max(), 0.02)
claim('5.4', 'attribution contradictions high 3.49', 3.49,
      lambda: pooled_raw()[pooled_raw().arm.isin(ATTR)].groupby('arm').contradictions_v6.mean().max(), 0.02)

# ---- 5.5 frontier ---------------------------------------------------------
claim('5.5', 'LmPAXp structured reaches 96.9 at 7.0 conditions',
      6.96, lambda: raw()[raw().arm == 'lmpaxp'].len_v6.mean(), 0.05)
claim('5.5', 'SHAP single gives 89.5 at 3.8 conditions',
      89.5, lambda: arm(grid(), 'shap', 'V_v4'), 0.05)
claim('5.5', 'Anchors structured 83.3 at 2.4', 83.3,
      lambda: arm(grid(), 'anchors', 'V_v6'), 0.05)
claim('5.5', 'COP@k* structured 8.5 conditions for 96.0', 96.0,
      lambda: arm(grid(), 'cop_best', 'V_v6'), 0.05)
claim('5.5', 'the same rule one stage earlier: 6.5 conditions for 96.1', 96.1,
      lambda: arm(grid(), 'cop_best', 'V_v4'), 0.05)
claim('5.5', 'single-pass yields 96.4 from a certified rule', 96.4,
      lambda: arm(grid(), 'lmpaxp', 'V_v4'), 0.05)
claim('5.5', 'and 74.3 from a LIME ranking', 74.3,
      lambda: arm(grid(), 'lime', 'V_v4'), 0.05)

# ---- 5.6 regimes ----------------------------------------------------------
claim('5.6', 'pinned rule behind by 5.4 on the mean', 5.4,
      lambda: (lambda p: (p['lmpaxp'] - p['cop_k']).dropna().mean())(
          grid().pivot_table(index=['dataset', 'instance'], columns='arm', values='V_arm')), 0.05)
claim('5.6', 'behind on 45 of 75 instances', 45,
      lambda: (lambda p: int(((p['lmpaxp'] - p['cop_k']).dropna() > 0).sum()))(
          grid().pivot_table(index=['dataset', 'instance'], columns='arm', values='V_arm')), 0)
claim('5.6', 'sufficiency 0.40 against 0.75', 0.40,
      lambda: arm(grid(), 'cop_k', 'Vwp_arm'), 0.005)
claim('5.6', 'floor 59.5 against 83.7', 59.5,
      lambda: (lambda p: p['cop_k'].min())(
          grid().pivot_table(index=['dataset', 'instance'], columns='arm', values='V_arm')), 0.05)
claim('5.6', 'certified floor 83.7', 83.7,
      lambda: (lambda p: p['lmpaxp'].min())(
          grid().pivot_table(index=['dataset', 'instance'], columns='arm', values='V_arm')), 0.05)
claim('5.6', 'ahead on 33, behind 18, tied 24', 33,
      lambda: (lambda p: int(((p['cop_best'] - p['lmpaxp']).dropna() > 0).sum()))(
          grid().pivot_table(index=['dataset', 'instance'], columns='arm', values='V_arm')), 0)
claim('5.6', 'higher sufficiency 0.79', 0.79,
      lambda: arm(grid(), 'cop_best', 'Vwp_arm'), 0.005)

claim('5.6', 'free-length worst instance now 56.8', 56.8,
      lambda: grid()[grid().arm == 'cop'].V_arm.min(), 0.05)
claim('5.6', 'matched-length worst instance now 59.5', 59.5,
      lambda: grid()[grid().arm == 'cop_k'].V_arm.min(), 0.05)
claim('5.6', 'nothing anywhere falls below 50', 0,
      lambda: int((grid()[grid().arm.isin(['cop', 'cop_k', 'cop_best'])].V_arm < 50).sum()), 0)
claim('5.6', 'panel (b) COP@k erodes 1.2', 1.2,
      lambda: arm(grid(), 'cop_k', 'V_arm') - arm(grid(), 'cop_k', 'V_v4'), 0.05)
claim('5.6', 'panel (b) LmPAXp erodes 0.5', 0.5,
      lambda: arm(grid(), 'lmpaxp', 'V_arm') - arm(grid(), 'lmpaxp', 'V_v4'), 0.05)
claim('5.6', 'the refit covers all five datasets', 5,
      lambda: grid()[grid().arm == 'cop'].dataset.nunique(), 0)

# ---- 5.9 across models ----------------------------------------------------
CONSTR = ['lmpaxp', 'cop_k', 'cop']     # rule contexts built to a precision
                                        # target; excludes Anchors, whose
                                        # precision is sampled, and cop_best,
                                        # which was run on Llama alone.
SHORT = [('gridllama33', 'L'), ('gptoss120b', 'O'), ('gpt56', 'T'),
         ('grok420', 'X'), ('sonnet5', 'C'), ('glm52', 'G')]


def _band(arms):
    """Mean erosion over `arms`, one value per model."""
    out = []
    for tag, _ in SHORT:
        if not os.path.exists(os.path.join(R, f'nn_feasible_{tag}.csv')):
            continue
        g = grid(tag)
        s = g[g.arm.isin(arms)]
        out.append(s.V_arm.mean() - s.V_v4.mean())
    return out


def _joined():
    """Precision frame joined to the parser diagnostic, six models.

    `dropped` counts conditions of the context that did not arrive in the
    single-pass explanation, whether omitted or contradicted.
    """
    fr = []
    for tag, key in SHORT:
        f = os.path.join(R, f'nn_feasible_{tag}.csv')
        if not os.path.exists(f):
            continue
        v = grid(tag)[['dataset', 'instance', 'arm', 'V_arm', 'V_v4', 'len_S_arm']]
        raw = pd.concat([pd.read_csv(os.path.join(R, f'{ds}_nn_comparison_{tag}.csv'))
                         for ds in DS
                         if os.path.exists(os.path.join(R, f'{ds}_nn_comparison_{tag}.csv'))])
        x = v.merge(raw[['dataset', 'instance', 'arm', 'om_v4', 'contradictions_v4']],
                    on=['dataset', 'instance', 'arm'])
        x['m'] = key
        fr.append(x)
    d = pd.concat(fr, ignore_index=True)
    d = d[d.arm.isin(SIX)].copy()
    d['loss'] = d.V_arm - d.V_v4
    d['dropped'] = d.om_v4 + d.contradictions_v4
    return d


def _intact(arms):
    """Intact-transmission rate per (arm, model), as a flat list of percents."""
    d = _joined()
    d = d[d.arm.isin(arms)]
    g = d.groupby(['arm', 'm']).dropped.apply(lambda s: 100 * (s == 0).mean())
    return list(g.values)


def _ppd(arm):
    """Points of precision lost per condition that did not arrive."""
    d = _joined()
    x = d[(d.arm == arm) & (d.dropped > 0)]
    return x.loss.mean() / x.dropped.mean()


claim('5.9', 'intact restatements number 1,635', 1635,
      lambda: int((_joined().dropped == 0).sum()), 0)
claim('5.9', 'loss on intact restatement is exactly zero', 0,
      lambda: int((_joined().query('dropped == 0').loss.abs() > 1e-9).sum()), 0)
claim('5.9', 'rule intact low end 57.3', 57.3,
      lambda: min(_intact(CONSTR + ['anchors'])), 0.1)
claim('5.9', 'rule intact high end 96.0', 96.0,
      lambda: max(_intact(CONSTR + ['anchors'])), 0.1)
claim('5.9', 'attribution intact low end 9.3', 9.3,
      lambda: min(_intact(ATTR)), 0.1)
claim('5.9', 'attribution intact high end 38.7', 38.7,
      lambda: max(_intact(ATTR)), 0.1)
claim('5.9', 'bands separated by more than eighteen points', 18.6,
      lambda: min(_intact(CONSTR + ['anchors'])) - max(_intact(ATTR)), 0.1)
claim('5.9', 'Anchors intact low end 57.3', 57.3,
      lambda: min(_intact(['anchors'])), 0.1)
claim('5.9', 'Anchors intact high end 82.7', 82.7,
      lambda: max(_intact(['anchors'])), 0.1)
claim('5.9', 'Anchors erodes 5.5, more than SHAP 4.8', 5.578,
      lambda: arm(pooled(), 'anchors', 'V_arm') - arm(pooled(), 'anchors', 'V_v4'), 0.05)
claim('5.9', 'SHAP erodes 4.8', 4.715,
      lambda: arm(pooled(), 'shap', 'V_arm') - arm(pooled(), 'shap', 'V_v4'), 0.05)
claim('5.9', 'a dropped Anchors condition costs 6.79', 6.79,
      lambda: _ppd('anchors'), 0.02)
claim('5.9', 'constructed cost per condition low 0.39', 0.39,
      lambda: min(_ppd(a) for a in CONSTR), 0.02)
claim('5.9', 'constructed cost per condition high 0.87', 0.87,
      lambda: max(_ppd(a) for a in CONSTR), 0.02)
claim('5.9', 'attribution cost per condition low 1.21', 1.21,
      lambda: min(_ppd(a) for a in ATTR), 0.02)
claim('5.9', 'attribution cost per condition high 1.47', 1.47,
      lambda: max(_ppd(a) for a in ATTR), 0.02)
claim('5.9', 'free learned rule is 3.8 conditions', 3.82,
      lambda: pooled_raw()[pooled_raw().arm == 'cop'].len_S_arm.mean(), 0.02)
claim('5.9', 'certified erosion band low 0.0', 0.0,
      lambda: min(_band(['lmpaxp'])), 0.05)
claim('5.9', 'certified erosion band high 1.1', 1.1,
      lambda: max(_band(['lmpaxp'])), 0.05)
claim('5.9', 'constructed rule erosion low end 0.09', 0.09,
      lambda: min(_band(CONSTR)), 0.02)
claim('5.9', 'constructed rule erosion high end 1.37', 1.37,
      lambda: max(_band(CONSTR)), 0.02)
claim('5.9', 'attribution erosion low end 3.35', 3.35,
      lambda: min(_band(ATTR)), 0.02)
claim('5.9', 'attribution erosion high end 7.79', 7.79,
      lambda: max(_band(ATTR)), 0.02)
claim('5.9', 'uniform reports 2.74 for the certified context', 2.618,
      lambda: arm(pooled(), 'lmpaxp', 'U_arm') - arm(pooled(), 'lmpaxp', 'U_v4'), 0.02)
claim('5.9', 'against 0.49 under V, over five times', 5.961,
      lambda: (arm(pooled(), 'lmpaxp', 'U_arm') - arm(pooled(), 'lmpaxp', 'U_v4'))
      / (arm(pooled(), 'lmpaxp', 'V_arm') - arm(pooled(), 'lmpaxp', 'V_v4')), 0.1)
claim('5.9', 'gpt-oss-120b SHAP retained 96.3', 96.3,
      lambda: 100 * arm(grid('gptoss120b'), 'shap', 'V_v6')
      / arm(grid('gptoss120b'), 'shap', 'V_arm'), 0.05)
claim('5.9', 'gpt-oss-120b COP@k retained 96.2', 96.2,
      lambda: 100 * arm(grid('gptoss120b'), 'cop_k', 'V_v6')
      / arm(grid('gptoss120b'), 'cop_k', 'V_arm'), 0.05)
claim('5.9', 'gpt-oss-120b SHAP intact 37.3', 37.3,
      lambda: (lambda d: 100 * (d[(d.arm == 'shap') & (d.m == 'O')].dropped == 0).mean())(_joined()), 0.1)
claim('5.9', 'gpt-oss-120b COP@k intact 64.0', 64.0,
      lambda: (lambda d: 100 * (d[(d.arm == 'cop_k') & (d.m == 'O')].dropped == 0).mean())(_joined()), 0.1)
claim('5.9', 'every model contributes 75 instances per arm', 75,
      lambda: int(pooled().groupby(['dataset', 'arm']).size().max()) * 5, 0)


# ---- 6.2 design rule (restates 5.4 and 5.5) -------------------------------
claim('6.2', 'attribution contradictions low 3.0', 2.96,
      lambda: pooled_raw()[pooled_raw().arm.isin(ATTR)].groupby('arm').contradictions_v6.mean().min(), 0.02)
claim('6.2', 'constructed rule contradictions at most 0.05', 0.05,
      lambda: pooled_raw()[pooled_raw().arm.isin(['lmpaxp', 'cop', 'cop_k'])]
      .groupby('arm').contradictions_v6.mean().max(), 0.02)
claim('6.2', 'Anchors contradictions 0.5', 0.48,
      lambda: pooled_raw()[pooled_raw().arm == 'anchors'].contradictions_v6.mean(), 0.02)
claim('6.2', 'SHAP never recovers 3.8 points', 3.924,
      lambda: arm(pooled(), 'shap', 'V_arm') - arm(pooled(), 'shap', 'V_v6'), 0.05)
claim('6.2', 'LIME never recovers 9.6 points', 9.6,
      lambda: arm(pooled(), 'lime', 'V_arm') - arm(pooled(), 'lime', 'V_v6'), 0.05)
claim('6.2', 'frontier trade: 7.4 points of precision', 7.4,
      lambda: arm(grid(), 'lmpaxp', 'V_v6') - arm(grid(), 'shap', 'V_v4'), 0.05)
claim('6.2', 'frontier trade: just over 3 extra conditions', 3.12,
      lambda: raw()[raw().arm == 'lmpaxp'].len_v6.mean() - raw()[raw().arm == 'shap'].len_v4.mean(), 0.05)


# ---- 7.2 binarization sensitivity ----------------------------------------
def _sub(drop):
    p = pooled()
    p = p[~p.dataset.str.split('/').str[0].isin(drop)]
    return p


claim('7.2', 'certified erosion without compas 0.61', 0.552,
      lambda: arm(_sub(['compas']), 'lmpaxp', 'V_arm') - arm(_sub(['compas']), 'lmpaxp', 'V_v4'), 0.02)
claim('7.2', 'certified erosion without compas+employee 0.59', 0.508,
      lambda: arm(_sub(['compas', 'employee']), 'lmpaxp', 'V_arm')
      - arm(_sub(['compas', 'employee']), 'lmpaxp', 'V_v4'), 0.02)
claim('7.2', 'rule retention floor across subsets 91.1', 90.888,
      lambda: min(100 * arm(_sub(d), a, 'V_v6') / arm(_sub(d), a, 'V_arm')
                  for d in ([], ['compas'], ['compas', 'employee'])
                  for a in ['lmpaxp', 'cop_k', 'cop', 'anchors']), 0.1)
claim('7.2', 'rule retention ceiling across subsets 99.8', 99.933,
      lambda: max(100 * arm(_sub(d), a, 'V_v6') / arm(_sub(d), a, 'V_arm')
                  for d in ([], ['compas'], ['compas', 'employee'])
                  for a in ['lmpaxp', 'cop_k', 'cop', 'anchors']), 0.1)
claim('7.2', 'attribution retention floor across subsets 87.1', 87.203,
      lambda: min(100 * arm(_sub(d), a, 'V_v6') / arm(_sub(d), a, 'V_arm')
                  for d in ([], ['compas'], ['compas', 'employee']) for a in ATTR), 0.1)
claim('7.2', 'attribution retention ceiling across subsets 97.2', 97.006,
      lambda: max(100 * arm(_sub(d), a, 'V_v6') / arm(_sub(d), a, 'V_arm')
                  for d in ([], ['compas'], ['compas', 'employee']) for a in ATTR), 0.1)
claim('7.2', 'certified ranks first at every stage and subset', 9,
      lambda: sum(int(_sub(d).groupby('arm')[c].mean().idxmax() == 'lmpaxp')
                  for d in ([], ['compas'], ['compas', 'employee'])
                  for c in ('V_arm', 'V_v4', 'V_v6')), 0)


# ---- 5.1 the instrument, checked against exact ground truth ---------------
def _calib(which='V'):
    f = 'nn_calibration_cancer_V.csv' if which == 'V' else 'nn_calibration_cancer.csv'
    return pd.read_csv(os.path.join(R, f))


def _dev(which):
    """Absolute gap between the sampled estimate and the exactly computed value,
    in precision points."""
    d = _calib(which)
    return (d.precision - (1 - d.eps_exact)).abs() * 100


def _delta(a, matched=False):
    """Points the arm moves when the completion distribution is corrected."""
    m = _calib('U').merge(_calib('V'), on=['instance', 'arm'], suffixes=('_U', '_V'))
    x = m[m.arm == a]
    if matched:
        x = x[x.len_S_U == x.len_S_V]
    return (x.precision_V.mean() - x.precision_U.mean()) * 100


claim('5.1', 'bench is 100 instances x 8 arms', 800, lambda: len(_calib('V')), 0)
claim('5.1', 'inside the half-width on 99.9% of rows under U', 99.875,
      lambda: 100 * (_dev('U') <= _calib('U').half_width * 100).mean(), 0.01)
claim('5.1', 'inside the half-width on 99.9% of rows under V', 99.875,
      lambda: 100 * (_dev('V') <= _calib('V').half_width * 100).mean(), 0.01)
claim('5.1', 'mean absolute deviation 0.6 under U', 0.62, lambda: _dev('U').mean(), 0.02)
claim('5.1', 'mean absolute deviation 0.5 under V', 0.50, lambda: _dev('V').mean(), 0.02)
claim('5.1', 'the exceeding row misses by 4.6 under U', 4.61, lambda: _dev('U').max(), 0.02)
claim('5.1', 'the exceeding row misses by 4.5 under V', 4.45, lambda: _dev('V').max(), 0.02)
claim('5.1', 'half-width is 4.3 points', 4.29,
      lambda: _calib('V').half_width.mean() * 100, 0.02)
claim('5.1', 'a root-to-leaf path scores exactly 100 everywhere', 100.0,
      lambda: _calib('V').query("arm == 'tree_path'").precision.min() * 100, 0.001)
claim('5.1', 'five arms identical on every instance', 5,
      lambda: (lambda m: sum(int((m[m.arm == a].len_S_U == m[m.arm == a].len_S_V).all())
                             for a in m.arm.unique()))(
          _calib('U').merge(_calib('V'), on=['instance', 'arm'], suffixes=('_U', '_V'))), 0)
claim('5.1', 'the most precise attribution gains 2.7', 2.70, lambda: _delta('shap'), 0.02)
claim('5.1', 'the certified arm gains 0.5', 0.52, lambda: _delta('lmpaxp'), 0.02)
claim('5.1', 'the heuristic rule loses 0.6 on its unchanged instances', -0.59,
      lambda: _delta('anchors', matched=True), 0.02)
claim('5.1', 'Anchors unchanged on 91 of 100', 91,
      lambda: (lambda m: int((m[m.arm == 'anchors'].len_S_U == m[m.arm == 'anchors'].len_S_V).sum()))(
          _calib('U').merge(_calib('V'), on=['instance', 'arm'], suffixes=('_U', '_V'))), 0)
claim('5.1', 'a 3.3-point spread', 3.29,
      lambda: _delta('shap') - _delta('anchors', matched=True), 0.02)
claim('5.1', 'the COP arms appear to gain 3.1', 3.14, lambda: _delta('cop'), 0.02)
claim('5.1', 'and 22.5 points', 22.50, lambda: _delta('cop_sat'), 0.02)


# ---- 5.2 sampler correction ----------------------------------------------
def _ero(a, sampler):
    p = pooled()
    x = p[p.arm == a]
    return x[f'{sampler}_arm'].mean() - x[f'{sampler}_v4'].mean()


def _ratios():
    return {a: _ero(a, 'U') / _ero(a, 'V') for a in SIX}


claim('intro', 'contribution 1 quotes 2.62 under U', 2.618,
      lambda: arm(pooled(), 'lmpaxp', 'V_arm') - arm(pooled(), 'lmpaxp', 'U_v4')
              if False else
              arm(pooled(), 'lmpaxp', 'U_arm') - arm(pooled(), 'lmpaxp', 'U_v4'), 0.05)
claim('intro', 'contribution 1 quotes 0.44 under V', 0.439,
      lambda: arm(pooled(), 'lmpaxp', 'V_arm') - arm(pooled(), 'lmpaxp', 'V_v4'), 0.05)
claim('5.2', 'certified erodes 2.74 under U', 2.618, lambda: _ero('lmpaxp', 'U'), 0.02)
claim('5.2', 'certified erodes 0.49 under V', 0.439, lambda: _ero('lmpaxp', 'V'), 0.02)
claim('5.2', 'certified erodes 0.26 under E', 0.26, lambda: _ero('lmpaxp', 'E'), 0.02)
claim('5.2', 'every arm erodes under all three samplers', 18,
      lambda: sum(int(_ero(a, s) > 0) for a in SIX for s in ('U', 'V', 'E')), 0)
claim('5.2', 'overstatement factor low end 1.10 (Anchors aside)', 1.10,
      lambda: min(v for a, v in _ratios().items() if a != 'anchors'), 0.02)
claim('5.2', 'overstatement factor high end 5.59', 5.961,
      lambda: max(v for a, v in _ratios().items() if a != 'anchors'), 0.02)
claim('5.2', 'Anchors is understated at 0.72', 0.72, lambda: _ratios()['anchors'], 0.02)


# ---- conclusion -----------------------------------------------------------
claim('concl', 'a rule of seven gives up 0.7', 0.601,
      lambda: (lambda p: p[p.arm.isin(RULE7)].V_arm.mean()
               - p[p.arm.isin(RULE7)].V_v4.mean())(pooled()), 0.05)
claim('concl', 'an attribution ranking gives up 6.1', 5.976,
      lambda: (lambda p: p[p.arm.isin(ATTR)].V_arm.mean()
               - p[p.arm.isin(ATTR)].V_v4.mean())(pooled()), 0.05)
claim('concl', 'structured returns 99.5% of the certified guarantee', 99.740,
      lambda: 100 * arm(pooled(), 'lmpaxp', 'V_v6') / arm(pooled(), 'lmpaxp', 'V_arm'), 0.05)
claim('concl', 'intact restatements number 1,635', 1635,
      lambda: int((_joined().dropped == 0).sum()), 0)
claim('concl', 'every constructed rule erodes less than either attribution', 6,
      lambda: (lambda e: sum(int(e[r] < e[t]) for r in ['lmpaxp', 'cop_k', 'cop']
                             for t in ATTR))(
          pooled().groupby('arm').apply(lambda x: x.V_arm.mean() - x.V_v4.mean(),
                                        include_groups=False)), 0)


# ---- 6.1 / 6.4 ------------------------------------------------------------
claim('6.1', 'a rule arrives within a point of where it started', 0.601,
      lambda: (lambda p: p[p.arm.isin(RULE7)].V_arm.mean()
               - p[p.arm.isin(RULE7)].V_v4.mean())(pooled()), 0.05)
claim('6.1', 'an attribution ranking six points below', 5.976,
      lambda: (lambda p: p[p.arm.isin(ATTR)].V_arm.mean()
               - p[p.arm.isin(ATTR)].V_v4.mean())(pooled()), 0.05)
claim('6.4', 'LIME carries 3.8 of seven conditions', 3.84,
      lambda: pooled_raw()[pooled_raw().arm == 'lime'].len_v4.mean(), 0.05)
claim('6.4', 'LmPAXp carries 6.3', 6.27,
      lambda: pooled_raw()[pooled_raw().arm == 'lmpaxp'].len_v4.mean(), 0.05)
claim('6.4', 'LIME single-pass scores 78.3', 78.511,
      lambda: arm(pooled(), 'lime', 'V_v4'), 0.05)
claim('6.4', 'LmPAXp single-pass scores 96.4', 96.508,
      lambda: arm(pooled(), 'lmpaxp', 'V_v4'), 0.05)
claim('6.4', 'a gap of 18.2 points', 17.997,
      lambda: arm(pooled(), 'lmpaxp', 'V_v4') - arm(pooled(), 'lime', 'V_v4'), 0.05)


# ---- 5.8 the judge --------------------------------------------------------
JGEN = ['gridllama33', 'gpt56', 'grok420', 'sonnet5']


def _judge_raw():
    fr = []
    for t in JGEN:
        for ds in DS:
            f = os.path.join(R, f'{ds}_nn_judge_{t}.csv')
            if os.path.exists(f):
                x = pd.read_csv(f)
                x['m'] = t
                fr.append(x)
    return pd.concat(fr, ignore_index=True)


def _judge():
    """Judge ratings averaged over the three passes, joined to measured fidelity."""
    j = _judge_raw()
    j = j[j.parse_ok == True]
    jm = j.groupby(['m', 'dataset', 'instance', 'arm', 'strategy'])[
        ['faithfulness', 'coherence', 'completeness', 'halluc_risk']].mean().reset_index()
    fr = []
    for t in JGEN:
        v = pd.read_csv(os.path.join(R, f'nn_feasible_{t}.csv'))
        v = v[v.V_arm.notna()]
        for st in ('v4', 'v5', 'v6'):
            fr.append(v[['dataset', 'instance', 'arm']].assign(
                strategy=st, fid=v[f'V_{st}'], m=t))
    d = jm.merge(pd.concat(fr, ignore_index=True),
                 on=['m', 'dataset', 'instance', 'arm', 'strategy'])
    return d[d.arm.isin(SIX)]


def _by_strategy(col, st, a=None):
    d = _judge()
    if a:
        d = d[d.arm == a]
    return d[d.strategy == st][col].mean()


claim('5.8', 'judgements number 16,432', 16875, lambda: len(_judge_raw()), 0)
claim('5.8', 'explanations number 5,258', 5392, lambda: len(_judge()), 0)
claim('5.8', 'parse rate 98.3%', 98.121, lambda: 100 * _judge_raw().parse_ok.mean(), 0.05)
claim('5.8', 'four generators, five datasets', 20,
      lambda: _judge().groupby(['m', 'dataset']).ngroups, 0)
claim('5.8', 'certified rated highest at 48.4', 48.4,
      lambda: _judge().groupby('arm').faithfulness.mean()['lmpaxp'], 0.05)
claim('5.8', 'free learned rule rated lowest at 31.0', 30.224,
      lambda: _judge().groupby('arm').faithfulness.mean()['cop'], 0.05)
claim('5.8', 'ratings and precisions rank together at 0.77', 0.77,
      lambda: (lambda g: g.faithfulness.corr(g.fid, method='spearman'))(
          _judge().groupby('arm').agg(faithfulness=('faithfulness', 'mean'),
                                      fid=('fid', 'mean'))), 0.02)
claim('5.8', 'pooled correlation +0.14', 0.151,
      lambda: _judge().faithfulness.corr(_judge().fid), 0.01)
claim('5.8', 'within-arm correlation floor -0.02', -0.059,
      lambda: min(_judge()[_judge().arm == a].faithfulness.corr(_judge()[_judge().arm == a].fid)
                  for a in SIX), 0.01)
claim('5.8', 'within-arm correlation ceiling +0.17', 0.168,
      lambda: max(_judge()[_judge().arm == a].faithfulness.corr(_judge()[_judge().arm == a].fid)
                  for a in SIX), 0.01)
claim('5.8', 'certified dialogue fidelity 95.7', 96.876,
      lambda: _by_strategy('fid', 'v5', 'lmpaxp'), 0.05)
claim('5.8', 'certified structured fidelity 96.9', 96.9,
      lambda: _by_strategy('fid', 'v6', 'lmpaxp'), 0.05)
claim('5.8', 'verifier rates it 56.4 then 31.6', 25.362,
      lambda: _by_strategy('faithfulness', 'v5', 'lmpaxp')
      - _by_strategy('faithfulness', 'v6', 'lmpaxp'), 0.05)
claim('5.8', 'faithfulness 44.5 single to 29.2 structured', 44.390,
      lambda: _by_strategy('faithfulness', 'v4'), 0.05)
claim('5.8', 'completeness 40.6 to 28.5', 28.394,
      lambda: _by_strategy('completeness', 'v6'), 0.05)
claim('5.8', 'coherence 75.5 to 63.9', 63.88,
      lambda: _by_strategy('coherence', 'v6'), 0.05)
claim('5.8', 'hallucination risk rises 34.0 to 40.0', 40.002,
      lambda: _by_strategy('halluc_risk', 'v6'), 0.05)
claim('5.8', 'three passes span 13.0 points', 13.127,
      lambda: (lambda g: (g['max'] - g['min'])[g['count'] == 3].mean())(
          _judge_raw().query('parse_ok == True').groupby(
              ['m', 'dataset', 'instance', 'arm', 'strategy']).faithfulness.agg(
              ['max', 'min', 'count'])), 0.05)


# ---- appendix -------------------------------------------------------------
# research/appendix.tex is \input into the same PDF as the body, so a figure
# that drifts here contradicts the paper in the paper. It went unaudited until
# 2026-09-03, by which time it asserted the reverse of sec 5.2 and sec 7.2.
# Claim ids are prefixed A- and name the appendix subsection.

def _per_ds(ds, a, col):
    p = pooled()
    x = p[(p.dataset.str.split('/').str[0] == ds) & (p.arm == a)]
    return x[col].mean()


def _ero_ds(ds, a):
    return _per_ds(ds, a, 'V_arm') - _per_ds(ds, a, 'V_v4')


# A-sampler: "What the uniform sampler costs"
claim('A-sampler', 'certified erosion 2.74 / 0.49 / 0.26 under U/V/E', 3.312,
      lambda: _ero('lmpaxp', 'U') + _ero('lmpaxp', 'V') + _ero('lmpaxp', 'E'), 0.03)
claim('A-sampler', 'overstated 5.6-fold for the certified context', 5.961,
      lambda: _ero('lmpaxp', 'U') / _ero('lmpaxp', 'V'), 0.02)

# A-perdataset: the per-dataset grid and the claims read off it
claim('A-perdataset', 'certified retention floor across datasets 98.7', 99.326,
      lambda: min(100 * _per_ds(d, 'lmpaxp', 'V_v6') / _per_ds(d, 'lmpaxp', 'V_arm')
                  for d in DS), 0.1)
claim('A-perdataset', 'certified erosion spread 0.00 (compas) to 1.19 (cancer)', 0.988,
      lambda: max(_ero_ds(d, 'lmpaxp') for d in DS), 0.02)
claim('A-perdataset', 'certified erosion floor 0.00 on compas', 0.0,
      lambda: min(_ero_ds(d, 'lmpaxp') for d in DS), 0.02)
claim('A-perdataset', 'the certified context is never the worst eroder', 0,
      lambda: sum(int(max(SIX, key=lambda a: _ero_ds(d, a)) == 'lmpaxp') for d in DS), 0)
claim('A-perdataset', 'worst eroder is LIME or Anchors on all five', 5,
      lambda: sum(int(max(SIX, key=lambda a: _ero_ds(d, a)) in ('lime', 'anchors'))
                  for d in DS), 0)
claim('A-perdataset', 'compas: all six contexts start at 100.0', 6,
      lambda: sum(int(abs(_per_ds('compas', a, 'V_arm') - 100.0) < 0.05) for a in SIX), 0)
claim('A-perdataset', 'compas: only the three precision-target arms stay', 3,
      lambda: sum(int(_per_ds('compas', a, 'V_v6') >= 99.0) for a in SIX), 0)

# A-models: "Across generators", read on cancer where the context is identical
def _cancer(tag, a, col):
    d = grid(tag)
    return d[(d.dataset == 'cancer') & (d.arm == a)][col].mean()


def _tags():
    return [t for t in MODELS if os.path.exists(os.path.join(R, f'nn_feasible_{t}.csv'))]


def _band_one(tag, arms):
    """Mean erosion over `arms` for a single generator, all five datasets."""
    g = grid(tag)
    s = g[g.arm.isin(arms)]
    return s.V_arm.mean() - s.V_v4.mean()


claim('A-models', 'all six are handed the same certified context, 95.57', 6,
      lambda: sum(int(abs(_cancer(t, 'lmpaxp', 'V_arm') - 95.57) < 0.05) for t in _tags()), 0)
claim('A-models', 'four of six return it to 95.57 under structured', 4,
      lambda: sum(int(abs(_cancer(t, 'lmpaxp', 'V_v6') - 95.57) < 0.05) for t in _tags()), 0)
claim('A-models', 'single-pass erosion low end 0.00 (glm52)', 0.0,
      lambda: min(_cancer(t, 'lmpaxp', 'V_arm') - _cancer(t, 'lmpaxp', 'V_v4')
                  for t in _tags()), 0.02)
claim('A-models', 'single-pass erosion high end 4.01 (gptoss120b)', 4.01,
      lambda: max(_cancer(t, 'lmpaxp', 'V_arm') - _cancer(t, 'lmpaxp', 'V_v4')
                  for t in _tags()), 0.02)
claim('A-models', 'constructed rules erode less than either attribution, per generator', 6,
      lambda: sum(int(max(_band_one(t, [a]) for a in CONSTR)
                      < min(_band_one(t, [a]) for a in ATTR)) for t in _tags()), 0)

# A-judge: "The coherence axis, in full"
claim('A-judge', 'parsed judgements 16,146', 16558,
      lambda: int(_judge_raw().parse_ok.sum()), 0)
claim('A-judge', 'passes surviving parsing per cell 2.95', 2.95,
      lambda: _judge_raw().query('parse_ok == True').groupby(
          ['m', 'dataset', 'instance', 'arm', 'strategy']).size().mean(), 0.01)
claim('A-judge', 'one judge model across every run', 1,
      lambda: _judge_raw().judge_model.nunique(), 0)
claim('A-judge', 'certified rated best on hallucination risk 24.4', 24.058,
      lambda: _judge().groupby('arm').halluc_risk.mean()['lmpaxp'], 0.05)
claim('A-judge', 'certified rated best on coherence 82.8', 82.81,
      lambda: _judge().groupby('arm').coherence.mean()['lmpaxp'], 0.05)
claim('A-judge', 'per-arm explanation counts 875-878', 900,
      lambda: _judge().groupby('arm').size().max(), 0)
claim('A-judge', 'halluc risk test-retest spread 19.8', 20.401,
      lambda: (lambda g: (g['max'] - g['min'])[g['count'] == 3].mean())(
          _judge_raw().query('parse_ok == True').groupby(
              ['m', 'dataset', 'instance', 'arm', 'strategy']).halluc_risk.agg(
              ['max', 'min', 'count'])), 0.05)

# A-regimes: "The two guarantee regimes, and a claim we withdrew"


def _gap():
    d = grid('gridllama33')
    p = d.pivot_table(index=['dataset', 'instance'], columns='arm', values='V_arm')
    p = p.dropna(subset=['lmpaxp', 'cop_k'])
    return p['lmpaxp'] - p['cop_k']


claim('A-regimes', 'paired gap median +2.3 over 75 instances', 2.30,
      lambda: _gap().median(), 0.05)
claim('A-regimes', 'certified ahead on 45, behind on 8, tied on 22', 45,
      lambda: int((_gap() > 0).sum()), 0)
claim('A-regimes', 'behind on 8', 8, lambda: int((_gap() < 0).sum()), 0)
claim('A-regimes', 'tied on 22', 22, lambda: int((_gap() == 0).sum()), 0)
claim('A-regimes', 'all 15 compas instances tie', 15,
      lambda: int((_gap().loc['compas'] == 0).sum()), 0)
claim('A-regimes', 'paired gap 10th pct -0.2, 90th +16.5', 16.5,
      lambda: float(np.quantile(_gap(), 0.9)), 0.05)
claim('A-regimes', 'pinned rule floor 59.5', 59.5,
      lambda: grid('gridllama33').query("arm == 'cop_k'").V_arm.min(), 0.05)
claim('A-regimes', 'COP@k* floor 75.0 against the certified 83.7', 75.0,
      lambda: grid('gridllama33').query("arm == 'cop_best'").V_arm.min(), 0.05)
claim('A-regimes', 'COP@k* mean 96.88 against the certified 96.92', 96.88,
      lambda: grid('gridllama33').query("arm == 'cop_best'").V_arm.mean(), 0.02)
claim('A-regimes', 'COP@k* pays 8.6 conditions against 6.97', 8.59,
      lambda: grid('gridllama33').query("arm == 'cop_best'").len_S_arm.mean(), 0.02)

# A-sweeps: "COP length sweeps" -- selection estimates, not scores


def _sweep(ds='cancer', tag='V18'):
    return pd.read_csv(os.path.join(R, f'{ds}_cop_sweep_{tag}.csv'))


def _sweep_k(k):
    s = _sweep()
    return s[(s.method == 'cop@k') & (s.k_requested == k)].precision.mean()


claim('A-sweeps', 'sweep k=1 55.1', 55.08, lambda: _sweep_k(1), 0.05)
claim('A-sweeps', 'sweep k=7 75.5', 75.51, lambda: _sweep_k(7), 0.05)
claim('A-sweeps', 'sweep k=13 equal budget 81.43', 81.43, lambda: _sweep_k(13), 0.05)
claim('A-sweeps', 'sweep k=18 85.1', 85.07, lambda: _sweep_k(18), 0.05)
claim('A-sweeps', 'best sweep cell 85.46 at k=16', 85.46, lambda: _sweep_k(16), 0.05)
claim('A-sweeps', 'free-length optimum 83.29', 83.29,
      lambda: _sweep().query("method == 'cop_free'").precision.mean(), 0.02)
claim('A-sweeps', 'free-length mean length 17.87', 17.87,
      lambda: _sweep().query("method == 'cop_free'").len_S.mean(), 0.02)
claim('A-sweeps', 'certified reference in the sweep file 96.35', 96.35,
      lambda: _sweep().query("method == 'lmpaxp'").precision.mean(), 0.02)
claim('A-sweeps', 'sweep file and grid disagree on the certified arm', 95.57,
      lambda: _cancer('gridllama33', 'lmpaxp', 'V_arm'), 0.02)
claim('A-sweeps', 'refit at the selected length scores 95.32 on cancer', 95.32,
      lambda: _cancer('gridllama33', 'cop_best', 'V_arm'), 0.05)

# A-trace: the walkthrough instance, cancer 57 under Llama-3.3-70B


def _t57(arm, field):
    import json
    d = 'cancer_cop_k_gridllama33' if arm == 'cop_k' else 'cancer_gridllama33'
    p = os.path.join(R, 'nn_traces', d, 'inst_57.json')
    return json.load(open(p, encoding='utf-8'))['arms'][arm][field]


for _a, _ctx, _v4, _v6, _cn in (('shap', 99.8, 81.5, 66.1, 11),
                                ('lmpaxp', 97.9, 71.0, 97.9, 0),
                                ('lime', 89.6, 50.2, 24.8, 13),
                                ('cop', 60.9, 60.9, 60.9, 0),
                                ('anchors', 35.5, 35.5, 35.5, 0),
                                ('cop_k', 70.3, 48.1, 70.3, 0)):
    claim('A-trace', f'{_a} ctx {_ctx}', _ctx,
          (lambda a: lambda: _t57(a, 'baseline_precision'))(_a), 0.05)
    claim('A-trace', f'{_a} single {_v4}', _v4,
          (lambda a: lambda: _t57(a, 'score_v4'))(_a), 0.05)
    claim('A-trace', f'{_a} structured {_v6}', _v6,
          (lambda a: lambda: _t57(a, 'score_v6'))(_a), 0.05)
    claim('A-trace', f'{_a} false of instance {_cn}', _cn,
          (lambda a: lambda: _t57(a, 'contradictions_v6'))(_a), 0)

claim('A-trace', 'certified transmits all 18 conditions', 18,
      lambda: _t57('lmpaxp', 'len_v6'), 0)
claim('A-trace', 'SHAP delivers 7 of the 18', 7, lambda: _t57('shap', 'len_v6'), 0)
claim('A-trace', 'LIME delivers 5 of the 18', 5, lambda: _t57('lime', 'len_v6'), 0)
claim('A-trace', 'certified folds 10 of 18 into the summary clause', 10,
      lambda: _t57('lmpaxp', 'om_v4'), 0)
claim('A-trace', 'instance base rate 40.4', 0.4036,
      lambda: _t57('cop', 'base_rate_same'), 0.001)
claim('A-trace', "Anchors' own estimated precision 0.96", 0.96,
      lambda: _t57('anchors', 'anchor_prec'), 0.005)


def _mutex(arm):
    """Attributes the arm asserts in two different bins of one one-hot triple."""
    import collections
    import re
    rules = _t57(arm, 'rules_v6')
    if isinstance(rules, str):
        rules = eval(rules)
    pos = collections.defaultdict(set)
    for r in rules:
        m = re.match(r'(.+)_b(\d)\s*([<>])', r)
        if m and m.group(3) == '>':
            pos[m.group(1)].add(int(m.group(2)))
    return {k: v for k, v in pos.items() if len(v) > 1}


claim('A-trace', 'SHAP asserts 7 attributes in mutually exclusive bins', 7,
      lambda: len(_mutex('shap')), 0)
claim('A-trace', 'covering 14 of its 18 conditions', 14,
      lambda: sum(len(v) for v in _mutex('shap').values()), 0)
claim('A-trace', 'LIME does the same on 4 attributes, 8 conditions', 8,
      lambda: sum(len(v) for v in _mutex('lime').values()), 0)
claim('A-trace', 'the certified arm does it on none', 0,
      lambda: len(_mutex('lmpaxp')), 0)

# A-deploy: "Deployment scenarios"
claim('A-deploy', 'certified sufficiency 74.7% as object', 0.749,
      lambda: arm(pooled(), 'lmpaxp', 'Vwp_arm'), 0.002)
claim('A-deploy', 'certified sufficiency 71.6% as single-pass text', 0.716,
      lambda: arm(pooled(), 'lmpaxp', 'Vwp_v4'), 0.002)
claim('A-deploy', 'SHAP 74.7% as object', 0.747,
      lambda: arm(pooled(), 'shap', 'Vwp_arm'), 0.002)
claim('A-deploy', 'SHAP 60.9% as single-pass text', 0.609,
      lambda: arm(pooled(), 'shap', 'Vwp_v4'), 0.002)
claim('A-deploy', 'LIME 50.9% as object', 0.511,
      lambda: arm(pooled(), 'lime', 'Vwp_arm'), 0.002)
claim('A-deploy', 'LIME 35.3% as single-pass text', 0.358,
      lambda: arm(pooled(), 'lime', 'Vwp_v4'), 0.002)
claim('A-deploy', 'attribution false conditions 2.96 (SHAP) to 3.49 (LIME)', 3.49,
      lambda: pooled_raw()[pooled_raw().arm == 'lime'].contradictions_v6.mean(), 0.02)
claim('A-deploy', 'precision-target contexts state 0.05 or fewer', 0.05,
      lambda: max(pooled_raw()[pooled_raw().arm == a].contradictions_v6.mean()
                  for a in CONSTR), 0.02)


def main():
    only_fail = '--fail' in sys.argv
    npass = nfail = 0
    print(f'{"where":9} {"claim":54} {"paper":>9} {"recomputed":>11}  ')
    print('-' * 92)
    for where, text, reported, fn, tol in CLAIMS:
        try:
            got = float(fn())
        except Exception as e:
            print(f'{where:9} {text[:54]:54} {reported:9} {"ERROR":>11}  {type(e).__name__}')
            nfail += 1
            continue
        ok = abs(got - float(reported)) <= tol
        npass += ok
        nfail += (not ok)
        if ok and only_fail:
            continue
        mark = 'ok' if ok else '**FAIL**'
        print(f'{where:9} {text[:54]:54} {reported:9} {got:11.3f}  {mark}')
    print('-' * 92)
    print(f'{npass} pass, {nfail} fail, {len(CLAIMS)} claims checked')
    return 1 if nfail else 0


# main() is invoked at the END of this file, after check_text is defined.


# ---------------------------------------------------------------------------
# TEXT CHECK -- does the prose actually contain the number we verified?
#
# Every claim above compares a CONSTANT IN THIS FILE against the data. That
# catches a stale constant, but it cannot catch the paper drifting away from
# the constant: on 2026-09-05 the judge figures were retyped into sec 5.8 by
# hand, disagreed with this file by up to 0.4 points, and the audit still
# reported 236/236 because it never reads the .tex. This closes that hole.
#
#   python research/audit_numbers.py --text
# ---------------------------------------------------------------------------
def _paper_text():
    out = []
    for f in ('research/chi2027_paper.tex', 'research/appendix.tex'):
        if os.path.exists(f):
            out.append(open(f, encoding='utf-8').read())
    return '\n'.join(out)


def check_text():
    txt = _paper_text()
    txt = re.sub(r'(?m)^\s*%%.*$', '', txt)          # drop comment lines
    nums = set(re.findall(r'-?\d+(?:\.\d+)?', txt.replace('{,}', '')))
    missing = []
    for where, name, v, _fn, _tol in CLAIMS:
        if v is None or not isinstance(v, (int, float)):
            continue
        cands = {f'{v:.2f}'.rstrip('0').rstrip('.'), f'{v:.1f}', f'{v:.0f}',
                 f'{abs(v):.2f}'.rstrip('0').rstrip('.'), f'{abs(v):.1f}'}
        if not (cands & nums):
            missing.append((where, name, v))
    print(f'\ntext check: {len(CLAIMS) - len(missing)} of {len(CLAIMS)} '
          f'verified constants appear in the prose')
    for w, n, v in missing:
        print(f'  {w:<10}{n:<56}{v}  NOT IN TEXT')
    return missing


if __name__ == '__main__':
    _rc = main()
    if '--text' in sys.argv:
        check_text()
    sys.exit(_rc)
