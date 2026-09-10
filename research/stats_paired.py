"""
stats_paired.py -- paired inferential statistics for the CHI 2027 erosion tables.

WHY THIS EXISTS
---------------
Tables 3 (`tab:erosion`) and 6 (`tab:models`) reported arm means only. The
experimental unit is the *instance*, and every arm is measured on the same
instances, so the observations are paired and the means discard the pairing
that makes the comparison powerful. This script adds, for every quantity the
paper asserts:

    * a percentile bootstrap 95% CI on the paired mean difference,
    * a paired Wilcoxon signed-rank test (exact where scipy can),
    * a matched-pairs rank-biserial effect size (and Cohen's d_z alongside),
    * Holm correction across the arms tested within one family.

WHAT IT DOES *NOT* DO
---------------------
It does not re-score anything. It reads the V-rescored grids produced by
`check_nn_feasible_sampling.py` and does arithmetic on them, so it cannot
change any number already in the paper -- only qualify it.

TWO CHOICES THAT ARE NOT COSMETIC
---------------------------------
1. **Degenerate arms get no p-value.** Structured generation reproduces the
   rule-shaped contexts exactly: `V_v6 - V_arm` is 0.00 on every instance for
   several arms. Wilcoxon on an all-zero difference vector is undefined (scipy
   returns p=1.0, which reads as "no evidence" when the truth is "identical on
   15/15"). Those cells report the tie count instead. An exact statement beats
   a degenerate test.

2. **Instances are nested in datasets.** For the pooled 5-dataset grid the
   headline statistic is the *dataset-balanced* mean of per-instance
   differences, and the CI is a two-stage cluster bootstrap (resample datasets,
   then instances within them). A flat bootstrap over 75 pooled instances
   understates the uncertainty that matters, because it treats five datasets as
   if they were seventy-five. Both are printed so the difference is visible.

USAGE
-----
    python research/stats_paired.py            # report + LaTeX fragments
    python research/stats_paired.py --quiet    # write files, print nothing

Outputs `results/stats_paired.md` (full report) and the LaTeX table bodies the
paper pastes into `tab:erosion` and `tab:models`.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

SEED = 20270101          # fixed so the CIs in the paper are reproducible
B_BOOT = 20000
B_PERM = 50000

# Context shape. This is the paper's independent variable: whether the context
# hands the model a conjunction bound to the instance (rule) or a ranking over
# features (attribution). See sec:contexts.
SHAPE = {'lmpaxp': 'rule', 'anchors': 'rule', 'cop': 'rule', 'cop_k': 'rule',
         'cop_best': 'rule', 'shap': 'attr', 'lime': 'attr'}

# Display names, in the order the paper's tables use.
LABEL = {'lmpaxp': 'LmPAXp', 'shap': 'SHAP', 'anchors': 'Anchors',
         'lime': 'LIME', 'cop': 'COP', 'cop_k': r'COP@$k$',
         'cop_best': r'COP@$k^\ast$'}


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def boot_ci(x, rng, B=B_BOOT, alpha=0.05):
    """Percentile bootstrap CI on the mean of x."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    draws = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
    return tuple(np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def cluster_boot_ci(x, strata, rng, B=B_BOOT, alpha=0.05):
    """Two-stage cluster bootstrap: resample datasets, then instances within.

    Respects the nesting of instances inside datasets. Wider than boot_ci
    whenever the effect varies across datasets -- which is exactly when the
    flat interval is lying.
    """
    x = np.asarray(x, float)
    strata = np.asarray(strata)
    groups = [x[strata == s] for s in pd.unique(strata)]
    k = len(groups)
    draws = np.empty(B)
    for b in range(B):
        picked = [groups[i] for i in rng.integers(0, k, k)]
        # dataset-balanced: mean of per-dataset means, each resampled
        draws[b] = np.mean([g[rng.integers(0, len(g), len(g))].mean()
                            for g in picked])
    return tuple(np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation: the effect size that belongs
    with Wilcoxon. In [-1, 1]; -1 means every non-tied pair moved down."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def cohen_dz(d):
    d = np.asarray(d, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else np.nan


def paired(d, strata=None, rng=None):
    """Full paired summary of a difference vector.

    Returns a dict. `p` is None when the vector is all zeros -- see the module
    docstring; callers must render that as a tie count, never as p=1.
    """
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    n_tie = int((d == 0).sum())
    out = {'n': n, 'n_tie': n_tie, 'mean': float(d.mean()) if n else np.nan,
           'median': float(np.median(d)) if n else np.nan,
           'sd': float(d.std(ddof=1)) if n > 1 else np.nan,
           'rbc': rank_biserial(d), 'dz': cohen_dz(d), 'degenerate': n_tie == n}
    out['ci'] = boot_ci(d, rng)
    out['p'] = None if out['degenerate'] else float(wilcoxon(d).pvalue)
    if strata is not None:
        out['ci_cluster'] = cluster_boot_ci(d, strata, rng)
        s = np.asarray(strata)
        out['balanced_mean'] = float(np.mean([d[s == v].mean() for v in pd.unique(s)]))
        out['by_stratum'] = {str(v): float(d[s == v].mean()) for v in pd.unique(s)}
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order.
    None entries (degenerate tests) pass through untouched and do not consume
    a comparison -- they are not tests."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    adj = list(pvals)
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)          # enforce monotonicity
        adj[i] = running
    return adj


def perm_shape(diff, strata, rng, B=B_PERM):
    """Sign-flip permutation on the rule-minus-attribution contrast.

    The statistic is the dataset-balanced mean, so no dataset dominates by
    contributing more instances. Sign-flipping is valid pair-wise regardless of
    nesting; balancing the statistic is what makes the nesting harmless.
    """
    diff = np.asarray(diff, float)
    s = np.asarray(strata)
    levels = pd.unique(s)
    masks = [s == v for v in levels]

    def stat(v):
        return np.mean([v[..., m].mean(axis=-1) for m in masks], axis=0)

    obs = stat(diff)
    flips = rng.choice([-1.0, 1.0], (B, len(diff)))
    null = stat(diff * flips)
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    return obs, max(p, 1.0 / B)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(path):
    d = pd.read_csv(path)
    d['shape'] = d['arm'].map(SHAPE)
    if d['shape'].isna().any():
        bad = sorted(d.loc[d['shape'].isna(), 'arm'].unique())
        raise SystemExit(f'{path}: unmapped arm(s) {bad} -- add them to SHAPE')
    return d


def fmt_p(p):
    if p is None:
        return '---'
    if p < 1e-4:
        return '$<$0.0001'
    return f'{p:.4f}'


def fmt_ci(ci):
    return f'[{ci[0]:.1f}, {ci[1]:.1f}]'


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------

def arm_table(d, rng, arms, label):
    """Erosion (v4 - ctx) and recovery (v6 - ctx) per arm, with Holm."""
    rows = []
    for arm in arms:
        a = d[d.arm == arm].dropna(subset=['V_arm', 'V_v4', 'V_v6'])
        if len(a) < 5:
            continue
        rows.append({
            'arm': arm, 'shape': SHAPE[arm], 'label': LABEL[arm],
            'ctx': a.V_arm.mean(), 'v4': a.V_v4.mean(), 'v6': a.V_v6.mean(),
            'ctx_sd': a.V_arm.std(ddof=1), 'v4_sd': a.V_v4.std(ddof=1),
            'erosion': paired((a.V_v4 - a.V_arm).values, rng=rng),
            'recovery': paired((a.V_v6 - a.V_arm).values, rng=rng),
        })
    for fam in ('erosion', 'recovery'):
        adj = holm([r[fam]['p'] for r in rows])
        for r, p in zip(rows, adj):
            r[fam]['p_holm'] = p
    # instance ids repeat across datasets, so the unit is the (dataset,
    # instance) pair -- nunique() on `instance` alone undercounts.
    n_inst = len(d[['dataset', 'instance']].drop_duplicates())
    return {'label': label, 'rows': rows, 'n_inst': n_inst}


def shape_contrast(d, rng, strat_col='dataset'):
    """Rule-shaped vs attribution-shaped, paired within instance.

    Each instance contributes the mean over its rule arms and the mean over its
    attribution arms, so the pairing is exact and no arm is counted twice.
    """
    out = {}
    for col, tag in [('V_v4', 'single-pass (v4)'), ('V_v6', 'structured (v6)')]:
        piv = d.pivot_table(index=[strat_col, 'instance'], columns='shape',
                            values=col, aggfunc='mean').dropna()
        diff = (piv['rule'] - piv['attr']).values
        strata = piv.index.get_level_values(0).values
        res = paired(diff, strata=strata, rng=rng)
        res['obs_balanced'], res['p_perm'] = perm_shape(diff, strata, rng)
        res['n_rule_gt'] = int((diff > 0).sum())
        out[tag] = res
    return out


def retention_reliability(d):
    """Retention = v6/ctx. Reported as a *spread*, not a mean.

    The ratio is not comparable across shapes -- for a rule context the
    denominator is a guarantee to preserve; for an attribution context it is an
    artefact of scoring a ranking as if it were a rule, and can be exceeded.
    So we report the range and the worst case, which is the claim that holds:
    rule-shaped is reliable, not merely higher.
    """
    d = d.copy()
    d['ret'] = d.V_v6 / d.V_arm * 100
    g = d.groupby(['shape', 'arm'])['ret']
    per_arm = pd.DataFrame({'min': g.min(), 'median': g.median(),
                            'mean': g.mean(), 'max': g.max(),
                            'n_over_100': g.apply(lambda x: int((x > 100.01).sum())),
                            'n': g.size()})
    by_shape = d.groupby('shape')['ret'].agg(['min', 'median', 'mean', 'max'])
    return per_arm, by_shape


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def md_arm_table(res):
    L = [f"### {res['label']}  ({res['n_inst']} instances, paired)", '',
         '| Context | Shape | ctx | v4 | Erosion | 95% CI | Wilcoxon p | Holm p | r_rb | v6 | Recovery (v6-ctx) |',
         '|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|']
    for r in res['rows']:
        e, v = r['erosion'], r['recovery']
        rec = (f"identical on {v['n_tie']}/{v['n']}" if v['degenerate']
               else f"{v['mean']:+.2f}, p={fmt_p(v['p_holm'])}")
        L.append(f"| {r['label']} | {r['shape']} | {r['ctx']:.1f} | {r['v4']:.1f} | "
                 f"{e['mean']:+.2f} | {fmt_ci(e['ci'])} | {fmt_p(e['p'])} | "
                 f"{fmt_p(e['p_holm'])} | {e['rbc']:+.2f} | {r['v6']:.1f} | {rec} |")
    return '\n'.join(L) + '\n'


def md_shape(sc, title):
    L = [f'### {title}', '',
         '| Comparison | rule - attr | flat 95% CI | cluster 95% CI | perm p | r_rb | d_z | rule>attr |',
         '|---|---:|---|---|---:|---:|---:|---:|']
    for tag, r in sc.items():
        L.append(f"| {tag} | {r['mean']:+.2f} | {fmt_ci(r['ci'])} | "
                 f"{fmt_ci(r.get('ci_cluster', (np.nan, np.nan)))} | {fmt_p(r['p_perm'])} | "
                 f"{r['rbc']:+.2f} | {r['dz']:+.2f} | {r['n_rule_gt']}/{r['n']} |")
    L.append('')
    for tag, r in sc.items():
        # a single-dataset grid has nothing to break down
        if len(r.get('by_stratum', {})) > 1:
            L.append(f"- {tag} per dataset: " +
                     ', '.join(f'{k} {v:+.1f}' for k, v in r['by_stratum'].items()))
    return '\n'.join(L) + '\n'


def stars(p):
    """Holm-adjusted significance marker. Kept to markers rather than a p
    column because acmart's single-column review format has no room for both a
    CI and a p-value per row."""
    if p is None:
        return ''
    for thr, mark in ((0.001, r'$^{***}$'), (0.01, r'$^{**}$'), (0.05, r'$^{*}$')):
        if p < thr:
            return mark
    return r'$^{\textrm{n.s.}}$'


def tex_ci(ci):
    """CI in LaTeX, with proper minus signs."""
    def n(v):
        return f'{v:.1f}'.replace('-', '$-$')
    return f'[{n(ci[0])}, {n(ci[1])}]'


def tex_num(v, dec=1):
    return f'{v:+.{dec}f}'.replace('-', '$-$').replace('+', '')


def tex_recovery(v):
    """Recovery cell: an exact tie count where the arm reproduced its context
    on every instance, otherwise the signed shortfall with its marker."""
    if v['degenerate']:
        return r'\textit{exact} (%d/%d)' % (v['n_tie'], v['n'])
    return f"{tex_num(v['mean'])}{stars(v['p_holm'])}"


def tex_erosion(res):
    """Body of tab:erosion: erosion now carries its CI and a Holm marker, and
    the retained column is followed by whether the return to ctx is exact."""
    L = []
    for r in sorted(res['rows'], key=lambda x: -x['ctx']):
        e, v = r['erosion'], r['recovery']
        ret = r['v6'] / r['ctx'] * 100
        L.append(f"{r['label']:8} & {'rule' if r['shape']=='rule' else 'attr.'} & "
                 f"{r['ctx']:.1f} & {r['v4']:.1f} & "
                 f"{tex_num(e['mean'])}{stars(e['p_holm'])}~{tex_ci(e['ci'])} & "
                 f"{r['v6']:.1f} & {ret:.1f}\\% & {tex_recovery(v)} \\\\")
    return '\n'.join(L)


def tex_models(per_model):
    """Body of tab:models in long format -- one row per (context, model).

    The paired L/C column layout cannot also carry two CIs per row without
    overrunning the text block, and the long form is what point 4 of the review
    actually asks for: the reader scans down one context and sees whether the
    finding reproduces.
    """
    arms = ['shap', 'lmpaxp', 'cop_k', 'lime', 'cop', 'anchors']
    idx = {m['label']: {r['arm']: r for r in m['rows']} for m in per_model}
    # Short model keys for the table's second column. Taking the first letter
    # collided once the grid grew: gpt-oss-120b, gpt-5.6-terra and grok-4.20
    # all render as "g". These are set explicitly, and any model not listed
    # falls back to its position so a new one is visible rather than silently
    # duplicating an existing key.
    _SHORT = {'Llama-3.3-70B': 'L', 'gpt-oss-120b': 'O', 'gpt-5.6-terra': 'T',
              'grok-4.20': 'X', 'Claude Sonnet 5': 'C', 'GLM-5.2': 'G'}
    short = {n: _SHORT.get(n.split(' (')[0], f'M{i+1}') for i, n in enumerate(idx)}
    L = []
    for arm in arms:
        if not all(arm in idx[n] for n in idx):
            continue
        for j, n in enumerate(idx):
            r = idx[n][arm]
            e, v = r['erosion'], r['recovery']
            ret = r['v6'] / r['ctx'] * 100
            head = (f"{LABEL[arm]} & {'rule' if SHAPE[arm]=='rule' else 'attr.'}"
                    if j == 0 else ' & ')
            L.append(f"{head} & {short[n]} & {r['ctx']:.1f} & {r['v4']:.1f} & "
                     f"{tex_num(e['mean'])}{stars(e['p_holm'])}~{tex_ci(e['ci'])} & "
                     f"{ret:.1f}\\% & {tex_recovery(v)} \\\\")
        L.append(r'\addlinespace[1pt]')
    return '\n'.join(L)


# --------------------------------------------------------------------------

def main():
    quiet = '--quiet' in sys.argv
    rng = np.random.default_rng(SEED)

    # ALL GRIDS ARE THE READER-FACING RUN OF 2026-08-30, on the reseeded
    # pipeline, so every model was handed identical contexts. The earlier
    # prompt generation is not read here: mixing the two would compare prompts
    # rather than models. Cogito-671B is excluded for the same reason -- it ran
    # before the explainer reseeding fix, and three of its six arms disagree
    # with the current contexts on nearly every instance.
    MODELS = [
        ('gridllama33', 'Llama-3.3-70B'),
        ('gptoss120b',  'gpt-oss-120b'),
        ('gpt56',       'gpt-5.6-terra'),
        ('grok420',     'grok-4.20'),
        ('sonnet5',     'Claude Sonnet 5'),
        ('glm52',       'GLM-5.2'),
    ]
    ARMS = ['lmpaxp', 'shap', 'cop_k', 'lime', 'cop', 'anchors']

    # THE HEADLINE TABLE IS POOLED OVER ALL SIX MODELS. Reporting one model as
    # the default would read as though the findings belonged to that generator
    # and the others were a robustness check; they do not, and the pooled table
    # says so on its face. The experimental unit becomes the (model, instance)
    # pair, 450 per arm, and the pairing across contexts is preserved within it.
    # COP@k* is absent because it was run once, on Llama: its context precision
    # involves no language model, so repeating it per generator would measure
    # nothing. Sections that need it say which grid they are on.
    pooled = load(os.path.join(RES, 'nn_feasible_gridllama33.csv'))
    frames, per_model = [], []
    for tag, label in MODELS:
        f = os.path.join(RES, f'nn_feasible_{tag}.csv')
        if not os.path.exists(f):
            if not quiet:
                print(f'  [skip] {os.path.basename(f)} missing')
            continue
        d = load(f)
        frames.append((label, d))
        per_model.append(arm_table(d, rng, ARMS,
                                   f'{label} (5 datasets, reader-facing prompts)'))
    if not quiet:
        print(f'{len(frames)} model grids read')

    # Pooled frame: (model, instance) is the unit, so the instance key is made
    # unique per model before concatenating, or the paired tests would collapse
    # six measurements of one instance into one.
    six = []
    for tag, label in MODELS:
        f = os.path.join(RES, f'nn_feasible_{tag}.csv')
        if not os.path.exists(f):
            continue
        d = load(f)
        d = d[d.arm.isin(ARMS)].copy()
        d['dataset'] = d['dataset'].astype(str) + '/' + tag
        six.append(d)
    six = pd.concat(six, ignore_index=True)
    pooled_six = arm_table(six, rng, ARMS,
                           'Pooled over six generators (450 model-instances per arm)')

    five = arm_table(pooled, rng, ARMS,
                     'Table 3 grid -- 5 datasets, Llama-3.3-70B, reader-facing prompts')

    sc_pooled = shape_contrast(six, rng)
    sc_models = {label: shape_contrast(d, rng, strat_col='dataset')
                 for label, d in frames}
    per_arm_ret, by_shape_ret = retention_reliability(pooled)

    parts = ['# Paired statistics for the erosion tables', '',
             f'Generated by `research/stats_paired.py` (seed {SEED}, '
             f'{B_BOOT} bootstrap draws, {B_PERM} permutations). '
             'All figures under the structurally valid sampler $\\mathcal{V}$.', '',
             '`Erosion` is v4 - ctx; `Recovery` is v6 - ctx. Both are paired '
             'per instance. Arms whose recovery is exactly zero on every '
             'instance report the tie count instead of a p-value: Wilcoxon is '
             'undefined on an all-zero difference vector.', '',
             '## 1. Per-arm erosion and recovery', '',
             md_arm_table(pooled_six), md_arm_table(five)]
    for m in per_model:
        parts.append(md_arm_table(m))
    parts += ['## 2. Rule-shaped vs attribution-shaped', '',
              md_shape(sc_pooled, 'Pooled 5-dataset grid (75 paired instances, 5 clusters)')]
    for name, sc in sc_models.items():
        parts.append(md_shape(sc, f'{name} (15 paired instances, 1 dataset)'))
    parts += ['## 3. Retention as reliability, not level', '',
              'Retention = v6/ctx. The denominator is not commensurable across '
              'shapes, so the defensible statement is the spread and the worst '
              'case, not the mean.', '',
              '```', per_arm_ret.round(1).to_string(), '',
              by_shape_ret.round(1).to_string(), '```', '',
              '## 4. LaTeX table bodies', '',
              '`tab:erosion`:', '', '```latex', tex_erosion(five), '```', '',
              '`tab:models`:', '', '```latex', tex_models(per_model), '```', '']

    report = '\n'.join(parts)
    out = os.path.join(RES, 'stats_paired.md')
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(report)
    if not quiet:
        print(report)
    print(f'\n[written] {out}', file=sys.stderr)


if __name__ == '__main__':
    main()
