"""
run_calibration_nn.py — part of the dense-network (NN) tier.

CALIBRATION GATE, no LLM. The complete black-box machinery of the NN tier is
exercised on a DECISION TREE — the one model where we still own the ground
truth — before a single LLM call is spent on the MLP:

  dataset (yaml, oversample, split seed-42: identical to run_comparison_bb.py)
    -> K-bins binarization, 3 bins/attribute, onehot   [Koriche §4.1]
    -> DecisionTreeClassifier (GridSearchCV, same grid as the DT tier)
       trained on {0,1}^d and then treated as a BLACK BOX (predict only)
    -> arms: tree_path | cop | cop_sat | lmpaxp | anchors | shap | lime | random
    -> scoring: precision_hat (MC, fix-S-sample-complement) + WeakPAXp@T
       [bb_metrics, J1] — the tier's ONLY live instrument
    -> ground truth (calibration crutch, tree-only, clearly quarantined):
       EXACT conditional precision computed analytically on the tree.

GO/NO-GO (§9 of the plan):
  1. INSTRUMENT   |eps_hat - eps_exact| <= Hoeffding half-width on >=90% rows
  2. SUFFICIENCY  tree_path arm: eps_exact == 0 on all instances (and eps_hat too)
  3. HIERARCHY    mean precision: formal arms (cop, lmpaxp) > shap/lime > random
  4. [K Table 2]  replication (informative): eps(cop) <= eps(sat), <= eps(anchors)

Usage:  python run_calibration_nn.py [dataset]      (default cancer)
        XAI_N_INSTANCES=5 for a smoke run (default 100, the [K §4.1] volume).
Output: research/results/nn_calibration_<dataset>.csv + printed verdict.
"""

from __future__ import annotations

import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

from bb_metrics import (precision_hat, precision_training,
                        hoeffding_half_width, PrecisionEstimate)
from cop_explainer import fit_cop_rule
from lmpaxp_explainer import fit_lmpaxp

DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else 'cancer'
N_INSTANCES  = int(os.getenv('XAI_N_INSTANCES', '100'))
RANDOM_SEED  = 42
T            = 0.95          # decision of 2026-07-02 (plan §7)
N_EVAL       = 1000          # [K §4.1]
DELTA        = 0.05

# Completion distribution. 'U' = Uniform({0,1}^d), the scheme inherited from
# [Koriche §4.1] / [Izza §4.2]. 'V' = uniform over ONE-HOT-VALID points, the
# structural correction of research/check_nn_feasible_sampling.py: under U
# essentially every completion violates the K-bins one-hot encoding and so
# encodes no possible instance. Both the MC estimator AND the exact tree
# oracle must be taken under the SAME distribution, or the calibration
# compares two different quantities.
SAMPLER      = os.getenv('XAI_SAMPLER', 'U').upper()
if SAMPLER not in ('U', 'V'):
    raise SystemExit(f'XAI_SAMPLER must be U or V, got {SAMPLER!r}')
BINS_PER_ATTR = 3            # KBinsDiscretizer(n_bins=3, onehot-dense)
CKPT_EVERY   = int(os.getenv('XAI_CKPT', '10'))   # flush partial CSV every N


def _out_path() -> str:
    suffix = '' if SAMPLER == 'U' else f'_{SAMPLER}'
    return f'research/results/nn_calibration_{DATASET_NAME}{suffix}.csv'


# ── Dataset prep — byte-for-byte the run_comparison_bb.py recipe ───────────────
def load_dataset(name: str):
    with open('datasets_config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)[name]
    df = pd.read_csv(cfg['chemin_csv'])
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    target = cfg['colonne_cible']
    counts = df[target].value_counts()
    maj, minn = counts.idxmax(), counts.idxmin()
    df_min = resample(df[df[target] == minn], replace=True,
                      n_samples=len(df[df[target] == maj]),
                      random_state=RANDOM_SEED)
    df = pd.concat([df[df[target] == maj], df_min])
    y = df[target]
    X = df.drop(target, axis=1)
    return train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)


# ── K-bins binarization [Koriche §4.1: 3 bins/attribute, onehot] ───────────────
def binarize(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Returns (Xtr_bin, Xte_bin, bin_feature_names, kbins)."""
    kb = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='quantile',
                          subsample=None)
    Xtr = kb.fit_transform(X_train.values)
    Xte = kb.transform(X_test.values)
    names = []
    for col, edges in zip(X_train.columns, kb.bin_edges_):
        for b in range(len(edges) - 1):
            names.append(f'{col}::[{edges[b]:.4g},{edges[b+1]:.4g})')
    return Xtr.astype(np.int8), Xte.astype(np.int8), names, kb


# ── EXACT conditional precision on a tree (calibration crutch, tree-only) ──────
def exact_precision_tree(sk_tree, x_bin: np.ndarray, S, pred: int) -> float:
    """
    Pr_z~U({0,1}^d)( f(z) = pred | z_S = x_S ), computed analytically by
    traversing the tree: unfixed features branch with mass 1/2 each; features
    fixed by S (or already decided on the path) follow their value. Linear in
    the number of nodes. WHITE-BOX — exists only to calibrate the MC estimator
    on the tree tier; the MLP tier has no equivalent (that is the point).
    """
    t = sk_tree.tree_
    Sset = set(int(j) for j in S)
    x_bin = np.asarray(x_bin).ravel()

    def rec(node: int, assigned: dict) -> float:
        f = int(t.feature[node])
        if f < 0:                                            # leaf
            cls = int(np.argmax(t.value[node][0]))
            return 1.0 if cls == pred else 0.0
        if f in Sset:
            val = int(x_bin[f])
        elif f in assigned:
            val = assigned[f]
        else:                                                # free bit: split mass
            a = dict(assigned)
            a[f] = 0
            left = rec(int(t.children_left[node]), a)
            a[f] = 1
            right = rec(int(t.children_right[node]), a)
            return 0.5 * left + 0.5 * right
        child = (t.children_left[node] if val <= t.threshold[node]
                 else t.children_right[node])
        return rec(int(child), assigned)

    return rec(0, {})


# ── Structurally valid completions (V): one active bin per attribute ──────────

def onehot_groups(d: int) -> list:
    """Bin indices grouped by source attribute: [[0,1,2],[3,4,5],...]."""
    if d % BINS_PER_ATTR:
        raise ValueError(f'd={d} is not a multiple of {BINS_PER_ATTR}')
    return [list(range(g, g + BINS_PER_ATTR))
            for g in range(0, d, BINS_PER_ATTR)]


def allowed_actives(x_bin: np.ndarray, S, groups: list) -> list:
    """
    Per group, the set of bins still eligible to be THE active one once S is
    fixed to x_S. A bin fixed to 1 by S determines its group outright; bins
    fixed to 0 are struck out. An empty set means x_S is not one-hot-valid.
    """
    x_bin = np.asarray(x_bin).ravel()
    Sset = set(int(j) for j in S)
    out = []
    for bins in groups:
        ones = [b for b in bins if b in Sset and x_bin[b] == 1]
        if ones:
            out.append({ones[0]})
        else:
            out.append({b for b in bins
                        if not (b in Sset and x_bin[b] == 0)})
    return out


def sample_conditional_valid(x_bin: np.ndarray, S, n: int,
                             rng: np.random.Generator,
                             groups: list) -> np.ndarray:
    """
    n samples z ~ Uniform({one-hot-valid points} | z_S = x_S).

    Every attribute gets exactly one active bin, drawn uniformly from those S
    still permits, so each row is both effective (S holds by construction) and
    realisable (it encodes an actual instance).
    """
    allowed = allowed_actives(x_bin, S, groups)
    d = int(np.asarray(x_bin).ravel().shape[0])
    Z = np.zeros((n, d), dtype=np.int8)
    for bins, A in zip(groups, allowed):
        if not A:
            raise ValueError('x_S violates the one-hot encoding: no valid '
                             f'completion exists for group {bins}')
        pick = rng.choice(sorted(A), size=n)
        Z[np.arange(n), pick] = 1
    return Z


def exact_precision_tree_valid(sk_tree, x_bin: np.ndarray, S, pred: int,
                               groups: list) -> float:
    """
    Pr_z~V( f(z) = pred | z_S = x_S ), exactly, by traversing the tree while
    carrying each group's set of still-eligible active bins. Testing bit b of
    group g splits that set: b is active with probability 1/|A_g| (and then
    A_g collapses to {b}), inactive otherwise (and b leaves A_g). Groups are
    independent, so the masses multiply along the path.

    WHITE-BOX — the calibration crutch, tree-only. Same role as
    exact_precision_tree, under the corrected distribution.
    """
    t = sk_tree.tree_
    x_bin = np.asarray(x_bin).ravel()
    group_of = {}
    for gi, bins in enumerate(groups):
        for b in bins:
            group_of[b] = gi
    init = [frozenset(A) for A in allowed_actives(x_bin, S, groups)]
    if any(not A for A in init):
        raise ValueError('x_S violates the one-hot encoding')

    def child_for(node: int, val: int) -> int:
        return int(t.children_left[node] if val <= t.threshold[node]
                   else t.children_right[node])

    def rec(node: int, allowed: tuple) -> float:
        f = int(t.feature[node])
        if f < 0:                                            # leaf
            return 1.0 if int(np.argmax(t.value[node][0])) == pred else 0.0
        gi = group_of[f]
        A = allowed[gi]
        p1 = (1.0 / len(A)) if f in A else 0.0
        total = 0.0
        if p1 > 0.0:
            nxt = list(allowed)
            nxt[gi] = frozenset({f})
            total += p1 * rec(child_for(node, 1), tuple(nxt))
        if p1 < 1.0:
            nxt = list(allowed)
            nxt[gi] = A - {f}
            total += (1.0 - p1) * rec(child_for(node, 0), tuple(nxt))
        return total

    return rec(0, tuple(init))


def _selftest_valid_sampler() -> None:
    """V-sampler and V-oracle must agree with each other on a known tree."""
    groups = onehot_groups(6)
    x = np.array([1, 0, 0, 0, 1, 0], dtype=np.int8)
    # every row one-hot per group, and S respected
    rng = np.random.default_rng(0)
    Z = sample_conditional_valid(x, [0, 1, 2], 500, rng, groups)
    assert (Z[:, 0:3].sum(axis=1) == 1).all(), 'group 0 not one-hot'
    assert (Z[:, 3:6].sum(axis=1) == 1).all(), 'group 1 not one-hot'
    assert (Z[:, 0] == 1).all(), 'S not respected'
    from sklearn.tree import DecisionTreeClassifier as _DT
    # tree that predicts 1 iff bit 4 is active
    Xt = np.array([[1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0],
                   [0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0]], dtype=float)
    yt = np.array([0, 1, 0, 1])
    tr = _DT(random_state=0).fit(Xt, yt)
    exact = exact_precision_tree_valid(tr, x, [0, 1, 2], 1, groups)
    mc = float(np.mean(tr.predict(
        sample_conditional_valid(x, [0, 1, 2], 20000,
                                 np.random.default_rng(1), groups)
        .astype(float)) == 1))
    assert abs(exact - mc) < 0.02, f'V oracle {exact} vs MC {mc}'
    print(f'[selftest] V sampler/oracle agree (exact={exact:.3f}, '
          f'mc={mc:.3f})', flush=True)


def _selftest_exact_precision():
    """Brute-force check of exact_precision_tree on a small enumerable space."""
    rng = np.random.default_rng(0)
    d = 8
    Xs = rng.integers(0, 2, size=(300, d)).astype(float)
    ys = ((Xs[:, 1] + Xs[:, 4] > 1) | (Xs[:, 6] == 1)).astype(int)
    small = DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xs, ys)
    x = np.ones(d, dtype=np.int8)
    pred = int(small.predict(x.reshape(1, -1).astype(float))[0])
    full = np.array(np.meshgrid(*[[0, 1]] * d)).T.reshape(-1, d).astype(float)
    for S in ([], [1, 4], [6], [0, 2, 3]):
        mask = np.all(full[:, S] == x[S], axis=1) if S else np.ones(len(full), bool)
        brute = float(np.mean(small.predict(full[mask]) == pred))
        exact = exact_precision_tree(small, x, S, pred)
        assert abs(brute - exact) < 1e-12, (S, brute, exact)
    print('  [PASS] exact_precision_tree == brute force on 2^8 space')


# ── tree_path arm: the instrument-check sufficient set ─────────────────────────
def tree_path_S(sk_tree, x_bin: np.ndarray) -> list:
    """Features tested on x's root->leaf path. Fixing them forces the same
    leaf, so its exact conditional precision is 1.0 by construction."""
    t = sk_tree.tree_
    node, S = 0, []
    while int(t.feature[node]) >= 0:
        f = int(t.feature[node])
        S.append(f)
        node = int(t.children_left[node] if x_bin[f] <= t.threshold[node]
                   else t.children_right[node])
    return sorted(set(S))


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    print(f'=== J4a calibration | dataset={DATASET_NAME} | N={N_INSTANCES} '
          f'| T={T} | n_eval={N_EVAL} | sampler={SAMPLER} ===', flush=True)
    _selftest_exact_precision()
    if SAMPLER == 'V':
        _selftest_valid_sampler()

    X_train, X_test, y_train, y_test = load_dataset(DATASET_NAME)
    Xtr_bin, Xte_bin, bin_names, _ = binarize(X_train, X_test)
    d = Xtr_bin.shape[1]
    GROUPS = onehot_groups(d)
    print(f'binarized: d={d} binary features '
          f'({X_train.shape[1]} raw x 3 bins onehot)', flush=True)

    # Black box c: a DT trained on {0,1}^d (same grid as the DT tier).
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_SEED),
        {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 7, 10],
         'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5],
         'max_features': ['sqrt', 'log2', None],
         'class_weight': [None, 'balanced']},
        scoring='accuracy', cv=5, n_jobs=-1)
    grid.fit(Xtr_bin.astype(float), y_train)
    dt = grid.best_estimator_
    acc = dt.score(Xte_bin.astype(float), y_test)
    print(f'black-box DT on binarized features | test acc {acc:.4f} '
          f'| depth {dt.get_depth()} leaves {dt.get_n_leaves()}', flush=True)

    # From here on: query access ONLY (except the quarantined exact oracle).
    predict = lambda Z: dt.predict(np.atleast_2d(np.asarray(Z)).astype(float))

    # SHAP / LIME / Anchors — all model-agnostic, all on {0,1}^d.
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    from anchor import anchor_tabular
    bg = shap.sample(Xtr_bin.astype(float), 100, random_state=RANDOM_SEED)
    shap_expl = shap.KernelExplainer(
        lambda Z: dt.predict_proba(np.atleast_2d(Z).astype(float)), bg)
    lime_expl = LimeTabularExplainer(
        Xtr_bin.astype(float), feature_names=bin_names,
        class_names=[str(c) for c in dt.classes_],
        categorical_features=list(range(d)), discretize_continuous=False,
        random_state=RANDOM_SEED)
    anchor_expl = anchor_tabular.AnchorTabularExplainer(
        class_names=[str(c) for c in dt.classes_], feature_names=bin_names,
        train_data=Xtr_bin.astype(float),
        categorical_names={j: ['0', '1'] for j in range(d)})

    def arm_shap(x_bin, pred, k):
        sv = shap_expl.shap_values(x_bin.astype(float), silent=True,
                                   l1_reg=f'num_features({d})')
        sv = np.asarray(sv)
        vec = (sv[pred] if sv.ndim == 2 and sv.shape[0] == len(dt.classes_)
               else (sv[..., pred] if sv.ndim > 1 else sv)).ravel()
        return sorted(np.argsort(-np.abs(vec))[:k].tolist())

    def arm_lime(x_bin, k):
        exp = lime_expl.explain_instance(x_bin.astype(float),
                                         lambda Z: dt.predict_proba(np.atleast_2d(Z).astype(float)),
                                         num_features=d, num_samples=1000)
        w = sorted(exp.as_map()[1], key=lambda t: -abs(t[1]))
        return sorted([int(f) for f, _ in w[:k]])

    def arm_anchors(x_bin):
        exp = anchor_expl.explain_instance(x_bin.astype(float), predict,
                                           threshold=T)
        return sorted(set(int(f) for f in exp.features()))

    random.seed(RANDOM_SEED)
    positions = random.sample(range(len(Xte_bin)), min(N_INSTANCES, len(Xte_bin)))
    rng_rand = np.random.default_rng(RANDOM_SEED)

    # Resume: `positions` is deterministic given (seed, N_INSTANCES), so any
    # instance already in the output CSV was scored under this exact recipe
    # and can be carried over. Lets a long run be driven in chunks.
    rows, done = [], set()
    if os.path.exists(_out_path()):
        prev = pd.read_csv(_out_path())
        keep = prev[prev['instance'].isin(positions)]
        rows = keep.to_dict('records')
        done = set(int(i) for i in keep['instance'].unique())
        print(f'resume: {len(done)} instances already scored '
              f'({len(rows)} rows) -> {len(positions) - len(done)} to go',
              flush=True)

    for it, pos in enumerate(positions):
        if pos in done:
            continue
        x = Xte_bin[pos]
        pred = int(predict(x)[0])

        cop = fit_cop_rule(predict, x, k=7, m=1000, timeout_s=60,
                           seed=RANDOM_SEED + pos)
        sat = fit_cop_rule(predict, x, k=7, m=1000, timeout_s=60,
                           variant='sat', seed=RANDOM_SEED + pos)
        lmp = fit_lmpaxp(predict, x, T=T, seed=RANDOM_SEED + pos)
        # Budget for the attribution arms = size of the CERTIFIED formal arm
        # (mirrors k_ref = len(SR) of the BT tier). NOT len(cop.S): the COP is
        # loss-optimal and can legitimately return very short/empty rules
        # (the [K Table 2] hard-dataset regime), which would collapse the
        # comparison.
        k_ref = max(len(lmp.S), 1)

        arms = {
            'tree_path': tree_path_S(dt, x),
            'cop':       cop.S,
            'cop_sat':   sat.S,
            'lmpaxp':    lmp.S,
            'anchors':   arm_anchors(x),
            'shap':      arm_shap(x, pred, k_ref),
            'lime':      arm_lime(x, k_ref),
            'random':    sorted(rng_rand.choice(d, size=k_ref,
                                                replace=False).tolist()),
        }

        for arm, S in arms.items():
            t0 = time.time()
            if SAMPLER == 'V':
                Zv = sample_conditional_valid(
                    x, S, N_EVAL, np.random.default_rng(RANDOM_SEED + 31 * pos),
                    GROUPS)
                eh = float(np.mean(
                    np.asarray(predict(Zv)).ravel() != pred))
                est = PrecisionEstimate(
                    eps_hat=eh, precision=1.0 - eh,
                    half_width=hoeffding_half_width(N_EVAL, DELTA),
                    n=N_EVAL, delta=DELTA, pred_class=pred)
                eps_exact = 1.0 - exact_precision_tree_valid(
                    dt, x, S, pred, GROUPS)
            else:
                est = precision_hat(predict, x, S, n=N_EVAL, delta=DELTA,
                                    seed=RANDOM_SEED + 31 * pos)
                eps_exact = 1.0 - exact_precision_tree(dt, x, S, pred)
            eps_tr, n_nb = precision_training(predict, x, S, Xtr_bin)
            rows.append({
                'instance': pos, 'arm': arm, 'pred': pred, 'len_S': len(S),
                'eps_hat': round(est.eps_hat, 4),
                'precision': round(est.precision, 4),
                'half_width': round(est.half_width, 4),
                'eps_exact': round(eps_exact, 6),
                'mc_agrees': abs(est.eps_hat - eps_exact) <= est.half_width,
                'weak_paxp': est.precision >= T,
                'weak_paxp_exact': (1.0 - eps_exact) >= T,
                'eps_training': None if eps_tr is None else round(eps_tr, 4),
                'n_train_neighbors': n_nb,
                'cop_loss': round(cop.empirical_loss, 4) if arm == 'cop'
                            else (round(sat.empirical_loss, 4) if arm == 'cop_sat'
                                  else None),
                'cop_status': cop.status if arm == 'cop'
                              else (sat.status if arm == 'cop_sat' else None),
                'base_rate_same': round(cop.base_rate_same, 3) if arm == 'cop'
                                  else None,   # risk #2 diagnostic (plan §8)
                'time_s': round(time.time() - t0, 2),
            })
        print(f'[{it+1:03d}/{len(positions)}] inst {pos} | ' + '  '.join(
            f'{a}:{1 - next(r["eps_hat"] for r in rows[-8:] if r["arm"] == a):.2f}'
            f'/{len(arms[a])}' for a in arms), flush=True)
        # Checkpoint: the run is long enough that losing it to an interrupt
        # costs an hour. Partial output is clearly marked by its row count.
        if (it + 1) % CKPT_EVERY == 0:
            os.makedirs('research/results', exist_ok=True)
            pd.DataFrame(rows).to_csv(_out_path(), index=False)

    df = pd.DataFrame(rows)
    os.makedirs('research/results', exist_ok=True)
    out = _out_path()
    df.to_csv(out, index=False)
    print(f'\nsaved -> {out}')

    # ── Verdict ────────────────────────────────────────────────────────────────
    print('\n=== per-arm means ===')
    g = df.groupby('arm').agg(
        len_S=('len_S', 'mean'), precision=('precision', 'mean'),
        eps_exact=('eps_exact', 'mean'), weak_paxp=('weak_paxp', 'mean'),
        weak_paxp_exact=('weak_paxp_exact', 'mean'),
        mc_agrees=('mc_agrees', 'mean'))
    print(g.round(3).to_string())

    p = g['precision']
    # Gate = instrument + the certified formal arm reproduces the known
    # hierarchy. The COP's position is NOT gated: it is loss-optimal with an
    # in-expectation guarantee — where it lands relative to the attribution
    # arms is a RESULT of the tier (the guarantee asymmetry, report §4), not
    # a calibration criterion.
    checks = {
        'C1 instrument: MC within half-width on >=90% rows':
            df['mc_agrees'].mean() >= 0.90,
        'C2 sufficiency: tree_path eps_exact == 0 everywhere':
            (df[df.arm == 'tree_path']['eps_exact'] == 0).all(),
        'C3 hierarchy: lmpaxp > max(shap,lime) > random':
            p['lmpaxp'] > max(p['shap'], p['lime'])
            and max(p['shap'], p['lime']) > p['random'],
    }
    info = {
        'I1 [K Table 2] replication: eps(cop) <= eps(sat)':
            (1 - p['cop']) <= (1 - p['cop_sat']) + 1e-9,
        'I2 [K Table 2] replication: eps(cop) <= eps(anchors)':
            (1 - p['cop']) <= (1 - p['anchors']) + 1e-9,
        'I3 cop above the attribution arms':
            p['cop'] > max(p['shap'], p['lime']),
    }
    print('\n=== GO/NO-GO (gate) ===')
    for name, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    print('=== informative (results, not gate) ===')
    for name, ok in info.items():
        print(f'  [{"yes" if ok else "no "}] {name}')
    go = all(checks.values())
    print(f'\n=== VERDICT: {"GO — instrument calibrated, proceed to J4b (MLP)" if go else "NO-GO — do not spend LLM budget"} ===')
    return 0 if go else 1


if __name__ == '__main__':
    sys.exit(main())
