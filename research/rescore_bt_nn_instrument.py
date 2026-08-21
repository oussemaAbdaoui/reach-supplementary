"""
rescore_bt_nn_instrument.py — re-score the BT tier with the NN tier's metric.

WHY THIS EXISTS
---------------
The BT tier was scored with W_f (logical sufficiency against the rectified
tree's DNF); the NN tier with conditional precision estimated by MC-sampling
(bb_metrics.precision_hat). Because model AND metric both changed, the paper
could only compare the tiers qualitatively. This script removes the metric
half of that confound: it applies the NN tier's instrument, unchanged, to the
BT tier's stored explanations.

WHY THAT IS LEGITIMATE HERE
---------------------------
The BT tier's rectified classifier (`clf_rectified` in the cache) predicts
directly on the binarized space {0,1}^d -- exactly the setting bb_metrics
assumes (see its module docstring: predict_fn maps (n,d) {0,1} -> labels).
No distributional adaptation is needed and none is applied: the sampler,
n=1000, delta=0.05 and T=0.95 are identical to the NN tier's. The one
remaining difference across tiers is the model itself, which is the point.

WHAT IS AND IS NOT RECOMPUTED
-----------------------------
* Extracted explanations (v4, v6) -- recomputed for EVERY arm. These are the
  stored `bina_v4` / `bina_v6` literal lists, i.e. what the reader receives.
* Arm baseline -- recomputed for `pyxai_sr` ONLY, from the stored `sr_conds`.
  This is the arm whose erosion is the claim under test.
* shap / lime baselines -- NOT recomputed. Their arm terms are not stored, and
  reconstructing them means re-running KernelExplainer / LIME against the
  original-space wrapper with an identically rebuilt split. Approximating them
  would silently invent numbers, so those cells are left empty and the CSV
  says why. Filling them is a follow-up that must re-run `arm_term()`.

USAGE
-----
    python research/rescore_bt_nn_instrument.py                # cancer + diabetes
    python research/rescore_bt_nn_instrument.py cancer

Run from the project root (the directory holding `_bt_cache_*.pkl`).
"""

from __future__ import annotations

import ast
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
sys.path.insert(0, ROOT)
# The cache pickles reference pyxai classes; the vendored source is enough to
# resolve them for unpickling (the Linux-only c_explainer .so is not needed).
sys.path.insert(0, os.path.join(ROOT, 'supplementary_material', 'sources', 'pyxai'))

import joblib  # noqa: E402
from bb_metrics import precision_hat, precision_training  # noqa: E402

RESULTS = os.path.join(ROOT, 'research', 'results')
N_SAMPLES = 1000      # identical to the NN tier [Koriche §4.1]
DELTA = 0.05          # identical to the NN tier
T_WEAK = 0.95         # WeakPAXp threshold, identical to the NN tier
SEED = 42
THRESH_TOL = 1e-6     # float tolerance when matching a condition to a literal


# ── cache / config loading ────────────────────────────────────────────────────

def load_tier(dataset: str):
    """Return (predict_fn, X_bin_test, X_bin_train, lit_table, feature_names)."""
    cache = joblib.load(os.path.join(ROOT, f'_bt_cache_{dataset}.pkl'))
    clf = cache['clf_rectified']
    X_bin_test = np.asarray(cache['binarized_test_X'], dtype=np.int8)
    X_bin_train = np.asarray(cache['binarized_train_X'], dtype=np.int8)
    nb_binary = int(cache['nb_binary'])
    bt_map = cache['bt_map']

    if clf.n_features_in_ != nb_binary:
        raise RuntimeError(
            f'{dataset}: rectified classifier expects {clf.n_features_in_} '
            f'features but nb_binary={nb_binary}; the binary-space assumption '
            f'this script relies on does not hold.')

    cfg = yaml.safe_load(open(os.path.join(ROOT, 'datasets_config.yaml')))[dataset]
    df = pd.read_csv(cfg['chemin_csv'])
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    feature_names = list(df.drop(cfg['colonne_cible'], axis=1).columns)

    # bt_map is 1-indexed with a leading 0 placeholder: bt_map[L] describes
    # literal L, and its feature index is itself 1-based.
    # Binary-space column of literal L is L-1 (bina_cols == X_1..X_nb_binary).
    lit_table = {}
    for L in range(1, min(nb_binary, len(bt_map) - 1) + 1):
        info = bt_map[L]
        if not info or info == 0:
            continue
        feat_idx, _op, thresh = info
        lit_table[L] = (feature_names[feat_idx - 1], float(thresh))

    def predict_fn(Z):
        return clf.predict(np.asarray(Z, dtype=np.int8))

    return predict_fn, X_bin_test, X_bin_train, lit_table, feature_names


# ── parsing helpers ───────────────────────────────────────────────────────────

_COND = re.compile(r'^\s*(?P<f>.+?)\s*(?P<op><=|>=|<|>)\s*(?P<t>[-+0-9.eE]+)\s*$')


def parse_literals(cell) -> list[int] | None:
    """`bina_v4` / `bina_v6` cells: '[-1, -12, -71]' -> [-1, -12, -71]."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    try:
        val = ast.literal_eval(str(cell))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(val, (list, tuple)):
        return None
    out = []
    for v in val:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            return None
    return out


def sr_conds_to_literals(cell, lit_table) -> tuple[list[int] | None, int]:
    """
    `sr_conds` cells: "['perimeter_worst <= 106.199997', ...]" -> signed
    literals. Returns (literals, n_unmatched); a condition that matches no
    literal is dropped and counted rather than guessed at.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None, 0
    try:
        conds = ast.literal_eval(str(cell))
    except (ValueError, SyntaxError):
        return None, 0
    if not isinstance(conds, (list, tuple)):
        return None, 0

    lits, unmatched = [], 0
    for cond in conds:
        m = _COND.match(str(cond))
        if not m:
            unmatched += 1
            continue
        fname, op, thresh = m.group('f'), m.group('op'), float(m.group('t'))
        hit = next((L for L, (f, t) in lit_table.items()
                    if f == fname and abs(t - thresh) <= THRESH_TOL), None)
        if hit is None:
            unmatched += 1
            continue
        # bt_map stores the '>' side; '<=' / '<' is the negative literal.
        lits.append(hit if op in ('>', '>=') else -hit)
    return lits, unmatched


def to_indices(lits, d) -> list[int]:
    """Signed literals -> 0-based binary-feature indices, deduplicated."""
    idx = sorted({abs(int(L)) - 1 for L in lits})
    return [j for j in idx if 0 <= j < d]


# ── scoring ───────────────────────────────────────────────────────────────────

def score_set(predict_fn, x_bin, lits, d, X_bin_train=None):
    """
    Conditional precision over the feature set implied by `lits`, under BOTH
    distributions -- and the difference between them is not cosmetic here.

    `uniform`: D = Uniform({0,1}^d), the NN tier's choice, inherited from
    [Koriche §4.1]. At the BT tier the binarization is threshold-based, so a
    single original feature carries many indicator bits (up to 11 in cancer)
    that are mutually constrained: "area > 768" cannot hold while "area > 500"
    fails. A uniform draw over {0,1}^d violates such orderings with near
    certainty, so this estimate is dominated by points no instance can reach,
    where the classifier's behaviour is arbitrary. It is reported for
    continuity with the NN tier, NOT as the headline.

    `empirical`: D = the binarized training distribution, every point of which
    is feasible by construction [the arbitrary-D case Koriche Eq. (1) allows].
    A sufficient reason should score ~100 here; that it does is the check that
    the mapping is right.

    No Hoeffding bound is claimed for the empirical variant: its sample count
    is whatever the training set supplies, not a quantity we choose.
    """
    if lits is None:
        return dict(prec_u=None, prec_e=None, n_nb=None, size=None, wp_u=None)
    S = to_indices(lits, d)
    est = precision_hat(predict_fn, x_bin, S, n=N_SAMPLES, delta=DELTA, seed=SEED)
    prec_e, n_nb = (None, 0)
    if X_bin_train is not None:
        eps_e, n_nb = precision_training(predict_fn, x_bin, S, X_bin_train)
        prec_e = None if eps_e is None else round(100.0 * (1.0 - eps_e), 2)
    return dict(prec_u=round(100.0 * est.precision, 2), prec_e=prec_e,
                n_nb=n_nb, size=len(S), wp_u=bool(est.precision >= T_WEAK))


def rescore_dataset(dataset: str) -> pd.DataFrame | None:
    csv = os.path.join(RESULTS, f'{dataset}_bt_comparison.csv')
    if not os.path.exists(csv):
        print(f'  [skip] {csv} not found')
        return None

    df = pd.read_csv(csv)
    if 'arm' not in df.columns:
        print(f'  [skip] {dataset}: malformed CSV (no `arm` column)')
        return None

    predict_fn, X_bin_test, X_bin_train, lit_table, _ = load_tier(dataset)
    d = X_bin_test.shape[1]
    rows = []

    for _, r in df.iterrows():
        pos = int(r['pos'])
        if not (0 <= pos < len(X_bin_test)):
            continue
        x_bin = X_bin_test[pos]
        arm = str(r['arm'])

        # Arm baseline: SR only (see module docstring).
        if arm == 'pyxai_sr':
            sr_lits, unmatched = sr_conds_to_literals(r.get('sr_conds'), lit_table)
            base = score_set(predict_fn, x_bin, sr_lits, d, X_bin_train)
        else:
            unmatched = 0
            base = score_set(predict_fn, x_bin, None, d, X_bin_train)

        v4 = score_set(predict_fn, x_bin, parse_literals(r.get('bina_v4')), d, X_bin_train)
        v6 = score_set(predict_fn, x_bin, parse_literals(r.get('bina_v6')), d, X_bin_train)

        def ero(b, v):
            return None if b is None or v is None else round(b - v, 2)

        def ret(b, v):
            return None if not b or v is None else round(100.0 * v / b, 2)

        rows.append(dict(
            dataset=dataset, pos=pos, arm=arm,
            # uniform D -- the NN tier's choice (see score_set docstring)
            precU_arm=base['prec_u'], precU_v4=v4['prec_u'], precU_v6=v6['prec_u'],
            erosionU_v4=ero(base['prec_u'], v4['prec_u']),
            retainedU_v6=ret(base['prec_u'], v6['prec_u']),
            # empirical (feasible) D -- the headline for this tier
            precE_arm=base['prec_e'], precE_v4=v4['prec_e'], precE_v6=v6['prec_e'],
            erosionE_v4=ero(base['prec_e'], v4['prec_e']),
            retainedE_v6=ret(base['prec_e'], v6['prec_e']),
            n_neighbours_arm=base['n_nb'], n_neighbours_v4=v4['n_nb'],
            len_arm=base['size'], len_v4=v4['size'], len_v6=v6['size'],
            weak_paxpU_v4=v4['wp_u'], weak_paxpU_v6=v6['wp_u'],
            sr_conds_unmatched=unmatched,
            baseline_note=('' if arm == 'pyxai_sr'
                           else 'arm term not stored; needs arm_term() re-run'),
            # W_f metric, carried over for the side-by-side
            wf_arm=r.get('arm_baseline_score'), wf_v4=r.get('score_v4'),
            wf_v6=r.get('score_v6'),
        ))

    return pd.DataFrame(rows)


def main(argv):
    datasets = argv[1:] or ['cancer', 'diabetes']
    frames = []
    for ds in datasets:
        print(f'[{ds}] re-scoring with the NN instrument '
              f'(n={N_SAMPLES}, delta={DELTA}, T={T_WEAK}) ...')
        out = rescore_dataset(ds)
        if out is not None and len(out):
            frames.append(out)
            print(f'  {len(out)} rows')

    if not frames:
        print('nothing to write'); return 1

    allr = pd.concat(frames, ignore_index=True)
    dest = os.path.join(RESULTS, 'bt_rescored_nn_instrument.csv')
    allr.to_csv(dest, index=False)
    print(f'\nwrote {dest}')

    print('\n=== BT tier under the NN instrument (conditional precision) ===')
    print('\n-- D = uniform over {0,1}^d (the NN tier choice; see score_set '
          'docstring: at this tier most such points are INFEASIBLE) --')
    print(allr.groupby('arm')[['precU_arm', 'precU_v4', 'precU_v6',
                               'erosionU_v4', 'retainedU_v6']].mean().round(2).to_string())
    print('\n-- D = empirical binarized training distribution (feasible points only) --')
    print(allr.groupby('arm')[['precE_arm', 'precE_v4', 'precE_v6',
                               'erosionE_v4', 'retainedE_v6']].mean().round(2).to_string())

    print('\n=== the same rows under W_f (as originally reported) ===')
    print(allr.groupby('arm')[['wf_arm', 'wf_v4', 'wf_v6']].mean().round(2).to_string())

    miss = allr[allr.sr_conds_unmatched > 0]
    if len(miss):
        print(f'\nNOTE: {len(miss)} SR rows had unmatched conditions '
              f'(total {int(miss.sr_conds_unmatched.sum())}); those literals were '
              f'dropped, so their baselines are lower bounds.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
