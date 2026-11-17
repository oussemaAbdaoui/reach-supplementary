"""Emit the body of tab:models, six generators, with the intact-transmission column.

The erosion / CI / significance / retention cells come from stats_paired.py, which
owns the paired statistics; this script only joins the intact-transmission rate onto
them, so the table cannot disagree with the statistics report.
"""
import os
import glob
import re
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
DS = ['cancer', 'compas', 'diabetes', 'employee', 'heart']
MODELS = [('gridllama33', 'L'), ('gptoss120b', 'O'), ('gpt56', 'T'),
          ('grok420', 'X'), ('sonnet5', 'C'), ('glm52', 'G')]
ARMS = ['shap', 'lmpaxp', 'cop_k', 'lime', 'cop', 'anchors']


def intact():
    """Share of instances whose context was restated with no omission and no
    contradiction, per (arm, model). This is the transmission event the section's
    claim is about; precision loss is exactly zero whenever it holds."""
    out = {}
    for tag, key in MODELS:
        v = pd.read_csv(os.path.join(R, f'nn_feasible_{tag}.csv'))
        v = v[v.V_arm.notna()][['dataset', 'instance', 'arm']]
        raw = pd.concat([pd.read_csv(os.path.join(R, f'{ds}_nn_comparison_{tag}.csv'))
                         for ds in DS
                         if os.path.exists(os.path.join(R, f'{ds}_nn_comparison_{tag}.csv'))])
        x = v.merge(raw[['dataset', 'instance', 'arm', 'om_v4', 'contradictions_v4']],
                    on=['dataset', 'instance', 'arm'])
        x['dropped'] = x.om_v4 + x.contradictions_v4
        for arm, grp in x.groupby('arm'):
            out[(arm, key)] = 100 * (grp.dropped == 0).mean()
    return out


def main():
    if not glob.glob(os.path.join('research', 'results', '*_nn_comparison_*.csv')):
        sys.exit('research/results/ is empty: no results ship with this bundle. '
                 'Generate them first (README, "Reproducing the study").')
    subprocess.run([sys.executable, os.path.join(HERE, 'stats_paired.py'), '--quiet'],
                   check=True, cwd=os.path.dirname(HERE))
    md = open(os.path.join(R, 'stats_paired.md'), encoding='utf-8').read()
    block = md.split('`tab:models`:')[1].split('```latex')[1].split('```')[0]

    ic = intact()
    lines, arm, order = [], None, iter(())
    for ln in block.strip().splitlines():
        if ln.startswith(r'\addlinespace'):
            lines.append(ln)
            continue
        head = ln.split('&')[0].strip()
        if head:                                  # first row of a context block
            arm = {'SHAP': 'shap', 'LmPAXp': 'lmpaxp', r'COP@$k$': 'cop_k',
                   'LIME': 'lime', 'COP': 'cop', 'Anchors': 'anchors'}[head]
            order = iter([k for _, k in MODELS])
        key = next(order)
        cells = ln.rstrip().rstrip(chr(92)).split('&')
        # insert Intact before Retained (index 6 of 8) and keep Recovery last
        cells = cells[:6] + [' %.1f' % ic[(arm, key)] + chr(92) + '% '] + cells[6:]
        lines.append('&'.join(cells).rstrip() + ' ' + chr(92)*2)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
