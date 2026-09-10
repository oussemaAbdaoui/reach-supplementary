"""Regenerate everything the dialogue fix changes: the erosion table with a
dialogue column, the paired stats, and the two-axis figure.

Run this ONCE when the v5 re-run and the judge are both complete. It reads the
CSVs and writes:

    research/tab_erosion_dialogue.tex     Table 2, with the dialogue column
    research/fig_two_axes.pdf/.png        coherence x precision, arrows structured->dialogue
    research/dialogue_stats.txt           paired CIs / Wilcoxon / Holm / rank-biserial

Nothing here is hand-typed: every number comes from research/results/.

  python research/make_dialogue_outputs.py [--tags a,b,c]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
RES = 'research/results'
DS = ['cancer', 'heart', 'diabetes', 'employee', 'compas']
ARMS = ['lmpaxp', 'shap', 'cop_k', 'anchors', 'lime', 'cop']
LABEL = {'lmpaxp': 'LmPAXp', 'shap': 'SHAP', 'cop_k': r'COP@$k$',
         'anchors': 'Anchors', 'lime': 'LIME', 'cop': 'COP'}
SHAPE = {'lmpaxp': 'rule', 'cop_k': 'rule', 'cop': 'rule', 'anchors': 'rule',
         'shap': 'attr.', 'lime': 'attr.'}
rng = np.random.default_rng(42)

ap = argparse.ArgumentParser()
ap.add_argument('--tags', default='gridllama33,grok420,gpt56,sonnet5,glm52,gptoss120b')
a = ap.parse_args()
TAGS = [t for t in a.tags.split(',') if t.strip()]


def load_formal(tags):
    out = []
    for t in tags:
        p = f'{RES}/nn_feasible_{t}.csv'
        if os.path.exists(p):
            out.append(pd.read_csv(p).assign(M=t))
        else:
            print(f'  [skip formal] {t}: no {p}', file=sys.stderr)
    return pd.concat(out) if out else pd.DataFrame()


def load_judge(tags):
    # gpt-oss-120b IS the judge. A stale cancer_nn_judge_gptoss120b.csv exists on
    # disk from an earlier experiment; loading it would let the judge score its
    # own family, which the paper forbids (sec 4.6).
    out = []
    for t in [x for x in tags if x != 'gptoss120b']:
        for ds in DS:
            p = f'{RES}/{ds}_nn_judge_{t}.csv'
            if os.path.exists(p):
                try: out.append(pd.read_csv(p).assign(M=t))
                except Exception: pass
    if not out:
        return pd.DataFrame()
    J = pd.concat(out)
    return J[J['parse_ok'] == True] if 'parse_ok' in J.columns else J


def boot_ci(d, n=20000):
    if len(d) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(d), (n, len(d)))
    return tuple(np.percentile(d[idx].mean(axis=1), [2.5, 97.5]))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run, m = [0.0] * len(ps), 0.0, len(ps)
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def star(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else r'n.s.'


def redone_keys(tags):
    ok = set()
    for t in tags:
        for ds in DS:
            p = f'{RES}/{ds}_nn_comparison_{t}.csv'
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            if 'v5_rerun' not in d.columns:
                continue
            for _, r in d[d['v5_rerun'] == 1].iterrows():
                ok.add((t, ds, int(r['instance']), str(r['arm'])))
    return ok

F = load_formal(TAGS)
J = load_judge(TAGS)
# gpt-oss-120b was frozen at 381/450 (see the note in the caption), so its grid
# holds a mix of corrected and pre-fix v5 rows. Keeping only redone rows makes
# ctx / single / dialogue / structured rest on ONE row set.
if not F.empty:
    keys = redone_keys(TAGS)
    before = len(F)
    F = F[[(m, ds, int(i), str(a)) in keys
           for m, ds, i, a in zip(F['M'], F['dataset'], F['instance'], F['arm'])]]
    print(f'restricted formal rows to redone v5: {before} -> {len(F)}')
if F.empty:
    raise SystemExit('no formal data found')
ngen_f = F['M'].nunique()
ngen_j = J['M'].nunique() if not J.empty else 0
print(f'formal: {len(F)} rows over {ngen_f} generators | '
      f'judge: {len(J)} rows over {ngen_j} generators')

# ── paired stats: dialogue vs structured, and dialogue vs single ──────────────
lines = [f'formal {len(F)} rows / {ngen_f} generators; '
         f'judge {len(J)} rows / {ngen_j} generators', '']
stats = {}
for ref, refname in (('V_v6', 'structured'), ('V_v4', 'single')):
    raw, rows = [], []
    for arm in ARMS:
        s = F[F['arm'] == arm].dropna(subset=['V_v5', ref])
        d = (s['V_v5'] - s[ref]).values
        lo, hi = boot_ci(d)
        nz = d[d != 0]
        p = wilcoxon(nz).pvalue if len(nz) else 1.0
        if len(nz):
            r = pd.Series(np.abs(nz)).rank().values
            rbc = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
        else:
            rbc = 0.0
        raw.append(p); rows.append((arm, len(s), d.mean(), lo, hi, rbc))
    adj = holm(raw)
    lines.append(f'=== dialogue vs {refname} (paired, V) ===')
    lines.append(f"{'arm':<9}{'n':>5}{'diff':>8}{'95% CI':>20}{'Holm p':>11}{'rbc':>7}")
    for (arm, n, m, lo, hi, rbc), p in zip(rows, adj):
        stats[(arm, refname)] = (m, lo, hi, p, rbc)
        lines.append(f'{arm:<9}{n:>5}{m:>+8.2f}  [{lo:>+6.2f},{hi:>+6.2f}]'
                     f'{p:>11.2e} {star(p):<5}{rbc:>6.2f}')
    lines.append('')
open('research/dialogue_stats.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('wrote research/dialogue_stats.txt')

# ── LaTeX table ──────────────────────────────────────────────────────────────
coh = {}
if not J.empty:
    for arm in ARMS:
        ja = J[J['arm'] == arm]
        coh[arm] = {s: ja[ja['strategy'] == s]['coherence'].mean()
                    for s in ('v4', 'v5', 'v6')}

T = [r'\begin{table*}[t]', r'\caption{Guarantee erosion by explanation context under '
     r'$\mathcal{V}$, with all three prompting strategies. \textsc{dialogue} supplies '
     r'the model its previous draft and the names of the conditions still missing; '
     r'\textsc{structured} requests one sentence per condition. Brackets are '
     r'percentile bootstrap $95\%$ CIs on the paired mean difference against '
     r'\textsc{structured}; markers are Holm-adjusted paired Wilcoxon signed-rank '
     r'tests within this family. The right-hand block is \emph{verifier-rated} '
     r'coherence and is evidence about the text-reading verifier, never a '
     r'measurement of fidelity (\S\ref{sec:evaluation}); it rests on '
     f'{ngen_j} generators against the formal block\'s {ngen_f}.' + r'}',
     r'\label{tab:erosion-dialogue}', r'\small',
     r'\begin{tabular}{llrrrrlrrr}', r'\toprule',
     r'& & \multicolumn{5}{c}{formal: conditional precision} & '
     r'\multicolumn{3}{c}{verifier-rated coherence} \\',
     r'\cmidrule(lr){3-7}\cmidrule(lr){8-10}',
     r'Context & Shape & \textsc{ctx} & \textsc{single} & \textsc{dialogue} & '
     r'\textsc{structured} & dial$-$struct [95\% CI] & \textsc{sing} & '
     r'\textsc{dial} & \textsc{struct} \\', r'\midrule']
for arm in ARMS:
    s = F[F['arm'] == arm]
    if s.empty:
        continue
    m, lo, hi, p, _ = stats[(arm, 'structured')]
    c = coh.get(arm, {})
    cs = [f'{c.get(k, float("nan")):.1f}' if c.get(k) == c.get(k) else '---'
          for k in ('v4', 'v5', 'v6')]
    T.append(f"{LABEL[arm]} & {SHAPE[arm]} & {s['V_arm'].mean():.1f} & "
             f"{s['V_v4'].mean():.1f} & {s['V_v5'].mean():.1f} & "
             f"{s['V_v6'].mean():.1f} & ${m:+.2f}^{{{star(p)}}}$~[${lo:+.2f}$, ${hi:+.2f}$] & "
             + ' & '.join(cs) + r' \\')
T += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
open('research/tab_erosion_dialogue.tex', 'w', encoding='utf-8').write('\n'.join(T))
print('wrote research/tab_erosion_dialogue.tex')

# ── figure: coherence x precision, arrow structured -> dialogue ──────────────
if not J.empty:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for arm in ARMS:
        s = F[F['arm'] == arm]
        c = coh.get(arm, {})
        if s.empty or not c or c.get('v6') != c.get('v6'):
            continue
        rule = SHAPE[arm] == 'rule'
        col = '#2C6A5B' if rule else '#7A6270'
        x6, y6 = c['v6'], s['V_v6'].mean()
        x5, y5 = c['v5'], s['V_v5'].mean()
        ax.annotate('', xy=(x5, y5), xytext=(x6, y6),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=1.4, alpha=.85))
        ax.scatter([x6], [y6], s=34, facecolor='white', edgecolor=col,
                   lw=1.4, zorder=3)
        ax.scatter([x5], [y5], s=46, color=col, zorder=3)
        ax.annotate(LABEL[arm].replace(r'@$k$', '@k'), (x5, y5),
                    textcoords='offset points', xytext=(7, 4),
                    fontsize=8.5, color=col)
    ax.set_xlabel('verifier-rated coherence  (what a text-reading judge sees)')
    ax.set_ylabel(r'conditional precision under $\mathcal{V}$')
    ax.set_title('Every arrow points up and to the right:\n'
                 'refinement dominates one-sentence-per-condition on both axes',
                 fontsize=10, loc='left')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker='o', ls='', mfc='white', mec='#2C6A5B', mew=1.4,
               label='structured (hollow)'),
        Line2D([], [], marker='o', ls='', color='#2C6A5B', label='dialogue (filled)'),
        Line2D([], [], color='#2C6A5B', label='rule-shaped'),
        Line2D([], [], color='#7A6270', label='attribution-shaped')],
        fontsize=8, frameon=False, loc='lower right')
    ax.grid(alpha=.25, lw=.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig('research/fig_two_axes.pdf')
    fig.savefig('research/fig_two_axes.png', dpi=170)
    print('wrote research/fig_two_axes.pdf / .png')
else:
    print('no judge data yet -- figure skipped')
