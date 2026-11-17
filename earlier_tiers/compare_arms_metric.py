"""
Metric-layer comparison of 4 explanation arms, all on the single rectified DT.
NO LLM — pure formal layer.

Arms (all scored against ONE shared reference DNF = the rectified PyXAI tree's
opposite-class DNF, in PyXAI binary-var id space 1..nb):
  pyxai_sr   : PyXAI native sufficient_reason() on dt_model_pyxai      (I_P=100%)
  mytree_sr  : my_tree.find_sufficient_reason on clf_rectified mimic   (I_P~95%)
  shap       : SHAP (KernelExplainer) of the rectified tree -> top-k binvars -> term
  lime       : LIME of the rectified tree -> top-k binvars -> term

Usage:  python3 compare_arms_metric.py [dataset]   (default cancer)
Requires the distillation cache (run XAI_CACHE_ONLY=1 python3 run_comparison_bt.py <ds>).
"""
import sys, io, os, types, warnings
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from unittest.mock import MagicMock as _MM
_pb = _MM(); _pb.PBEnc = _MM()
sys.modules['pypblib'] = _pb; sys.modules['pypblib.pblib'] = _pb

_SUPP = os.environ.get('PYXAI_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyxai'))
_DIST = os.path.join(_SUPP, 'pyxai', 'examples', 'Distillation')
sys.path.insert(0, _SUPP); sys.path.insert(0, _DIST)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# c_explainer present -> real rectifier; mock only the heavy optional deps. If the compiled
# rectifier .so is absent (e.g. non-Linux), also mock c_explainer so the import succeeds — we
# only LOAD the cached rectified tree here, we don't rectify.
_HAS_CXX = os.path.isdir(_SUPP) and any(
    f.startswith('c_explainer') and f.endswith('.so') for f in os.listdir(_SUPP))
from unittest.mock import MagicMock
_mock_mods = ['distributed', 'docplex', 'docplex.mp', 'docplex.mp.model',
              'docplex.mp.dvar', 'docplex.mp.linear']
if not _HAS_CXX:
    _mock_mods = ['c_explainer'] + _mock_mods
for _m in _mock_mods:
    sys.modules.setdefault(_m, MagicMock())

import numpy as np, pandas as pd, joblib, yaml
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import my_tree as mt
from pyxai import Learning, Explainer, Tools
from pyxai.sources.core.structure.decisionTree import TypeLeaf
Tools.set_verbose(0)

DATASET = sys.argv[1] if len(sys.argv) > 1 else 'cancer'
CACHE   = f'_bt_cache_{DATASET}.pkl'
TOPK    = 7    # top-k features for SHAP/LIME terms

# ── load cache ────────────────────────────────────────────────────────────────
c = joblib.load(CACHE)
clf_rectified  = c['clf_rectified']
dt_model_pyxai = c['dt_model_pyxai']
binarized_test_X = c['binarized_test_X']
binarized_test_y = c['binarized_test_y']   # = BT's own predictions on test
nb = c['nb_binary']; bina_cols = c['bina_cols']; bt_map = c['bt_map']
print(f'[{DATASET}] nb={nb}  test={len(binarized_test_X)}  rectified nodes={dt_model_pyxai.n_nodes()}',
      flush=True)

# ── my_tree over the mimic (for mytree_sr) + card_bin <-> pyxai_id bijection ────
my_tree = mt.decision_tree()
my_tree.from_DecisionTreeClassifier(clf_rectified)
# my_tree.bina key = (pyxai_binvar_id, thresh~0.5), value = [card_bin, count]
cardbin_to_pyxai = {card_bin: pyxai_id
                    for (pyxai_id, _thr), (card_bin, _n) in my_tree.bina.items()}

def mytree_lits_to_pyxai(lits):
    out = []
    for L in lits:
        pid = cardbin_to_pyxai.get(abs(L))
        if pid is not None:
            out.append(pid if L > 0 else -pid)
    return out

# ── CRITICAL: dt_model_pyxai's explainer has its OWN binvar id space (1..#used-feats),
# NOT the BT binvar space (1..nb). map_features_to_id_binaries maps
#   (id_feature, op, thresh) -> [tree_binvar_id, ...]; id_feature == BT binvar id
# (the DT was trained on columns X_1..X_nb). We convert pyxai_sr AND the DNF from
# tree-id space into BT-binvar space so ALL arms share one literal space. ───────────
_dtid_to_btbinvar = {v[0]: idf for (idf, _op, _th), v in
                     dt_model_pyxai.map_features_to_id_binaries.items()}
def _dtid_lits_to_bt(lits):
    out = []
    for L in lits:
        bv = _dtid_to_btbinvar.get(abs(L))
        if bv is not None:
            out.append(bv if L > 0 else -bv)
    return out

# ── shared reference DNF: opposite-class paths of dt_model_pyxai, in BT-binvar space ──
def build_class_dnf(tree, target):
    """All root->leaf path cubes predicting `target`, literals converted to BT-binvar ids."""
    dnf = []
    for node in tree.compute_nodes_with_leaves(tree.root):
        if node.left.is_leaf() and node.left.is_prediction(target):
            dnf.append(_dtid_lits_to_bt(tree.create_cube(node, TypeLeaf.LEFT)))
        if node.right.is_leaf() and node.right.is_prediction(target):
            dnf.append(_dtid_lits_to_bt(tree.create_cube(node, TypeLeaf.RIGHT)))
    return dnf

DNF = {0: build_class_dnf(dt_model_pyxai, 0),
       1: build_class_dnf(dt_model_pyxai, 1)}
print(f'reference DNF clauses: class0={len(DNF[0])}  class1={len(DNF[1])}', flush=True)

# ── W_f: the ORIGINAL metric from the paper (theory-free) ──────────────────────
# Weighted model count over the 2^n binary-assignment space (binvars independent;
# the theory/ordering is NOT folded in). Verbatim from the paper for comparability.
def W_f(dnf, t, n):
    def check(C, td):
        for c in td:
            if -c in C: return 0
        return len(set(C + td))
    w = 0
    for cl in dnf:
        v = check(cl, t)
        if v: w += 2 ** (n - v)
    return w / 2 ** (n - len(t))

def formal_score(phi, term, n):
    if not term: return 0.0
    return (1 - W_f(phi, term, n)) * 100

# ── CP-2: graded fidelity on real training data ────────────────────────────────
# term (PyXAI ids) -> feature conditions (via bt_map) -> training instances that
# satisfy ALL conditions -> fraction the rectified tree predicts as a DIFFERENT
# class.  cp2_fidelity = (1 - eps) * 100.  Discriminates the non-sufficient arms
# (SHAP/LIME/mimic) that W_f collapses toward random.  All arms checked against the
# SAME model (dt_model_pyxai) — consistent with the shared-reference design.
def _lit_condition(lit):
    info = bt_map[abs(lit)]               # (orig_feat_idx_1based, GT, threshold)
    fidx, thr = info[0], info[2]
    return fidx - 1, ('>' if lit > 0 else '<='), thr   # 0-based feature col

def _binarize_orig(row):
    b = np.empty(nb, dtype=np.int8)
    for k in range(1, nb + 1):
        info = bt_map[k]
        b[k - 1] = 1 if row[info[0] - 1] > info[2] else 0
    return b

def cp2_fidelity(term, pred_class):
    """% of training neighbours (matching the term's conditions) that the rectified
    tree still assigns to pred_class.  100 = the conditions perfectly pin the class."""
    if not term: return None, 0
    mask = np.ones(len(X_train_orig), dtype=bool)
    Xtr = X_train_orig.values
    for lit in term:
        col, op, thr = _lit_condition(lit)
        mask &= (Xtr[:, col] > thr) if op == '>' else (Xtr[:, col] <= thr)
    n = int(mask.sum())
    if n == 0: return None, 0
    preds = [dt_model_pyxai.predict_instance(_binarize_orig(r)) for r in Xtr[mask]]
    n_diff = sum(int(p != pred_class) for p in preds)
    return round((1 - n_diff / n) * 100, 2), n

# ── arm: pyxai_sr (native sufficient_reason on the rectified tree) ─────────────
ert = Explainer.initialize(dt_model_pyxai)
def arm_pyxai_sr(pos):
    ert.set_instance(binarized_test_X[pos])
    return _dtid_lits_to_bt(list(ert.sufficient_reason()))   # -> BT-binvar space

# ── arm: mytree_sr (SR on the mimic, mapped to PyXAI ids) ──────────────────────
def arm_mytree_sr(pos):
    inst = binarized_test_X[pos]
    tgt  = my_tree.take_decision(instance=inst)
    lits = my_tree.find_sufficient_reason(instance=inst, target=tgt)
    return mytree_lits_to_pyxai(lits)

# ── per-instance prediction (for choosing the opposite class) ──────────────────
def predicted_class(pos):
    return int(dt_model_pyxai.predict_instance(binarized_test_X[pos]))

# ── Dataset split (aligned with run_comparison_bt.py) for the rectified-tree arms ──
# Reconstruct the split EXACTLY as run_comparison_bt.py so X_test_orig.iloc[i] aligns
# with binarized_test_X[i] (the cached rectified-tree binarized test instances).
RANDOM_SEED = 42
_cfg = yaml.safe_load(open('datasets_config.yaml', encoding='utf-8'))[DATASET]
_TARGET = _cfg['colonne_cible']
_df = pd.read_csv(_cfg['chemin_csv'])
_df.columns = _df.columns.str.lower().str.replace(' ', '_')
_cnt = _df[_TARGET].value_counts(); _maj, _min = _cnt.idxmax(), _cnt.idxmin()
_dmin = resample(_df[_df[_TARGET] == _min], replace=True,
                 n_samples=len(_df[_df[_TARGET] == _maj]), random_state=RANDOM_SEED)
_df = pd.concat([_df[_df[_TARGET] == _maj], _dmin])
_y = _df[_TARGET]; _X = _df.drop(_TARGET, axis=1)
X_train_orig, X_test_orig, _, _ = train_test_split(_X, _y, test_size=0.3, random_state=RANDOM_SEED)
X_train_orig = X_train_orig.reset_index(drop=True)
X_test_orig  = X_test_orig.reset_index(drop=True)
feat_names = list(_X.columns)

import shap, lime, lime.lime_tabular

# SHAP/LIME explain the RECTIFIED TREE (dt_model_pyxai) — the model that is actually
# scored — NOT the boosted tree. This removes the model-mismatch confound: pyxai_sr, the
# reference DNF and the SHAP/LIME arms all describe the same tree. dt_model_pyxai isn't an
# sklearn model → SHAP uses the model-agnostic KernelExplainer over a binarize→predict
# wrapper (LIME is model-agnostic already).
def _rect_predict_proba(Xorig):
    rows = np.atleast_2d(np.asarray(Xorig, dtype=float))
    out  = np.zeros((len(rows), 2))
    for i, row in enumerate(rows):
        out[i, int(dt_model_pyxai.predict_instance(_binarize_orig(row)))] = 1.0
    return out

def _shap_pred_vec(shap_out, idx, pc):
    """Predicted-class Shapley vector, robust to shap's list / 2-D / 3-D return shapes."""
    if isinstance(shap_out, list):            # [class0_arr, class1_arr]
        return np.asarray(shap_out[pc])[idx]
    a = np.asarray(shap_out)
    if a.ndim == 3:                           # (n, nfeat, nclass)
        return a[idx, :, pc]
    return a[idx]                             # (n, nfeat)

_shap_expl = shap.KernelExplainer(_rect_predict_proba, shap.kmeans(X_train_orig.values, 15))
_shap_all  = _shap_expl.shap_values(X_test_orig.values, silent=True,
                                    l1_reg=f'num_features({len(feat_names)})')
_lime_expl = lime.lime_tabular.LimeTabularExplainer(
    X_train_orig.values, feature_names=feat_names,
    class_names=['0', '1'], discretize_continuous=True, random_state=RANDOM_SEED)

def _feat_attr_to_term(attr_vec, pos, k):
    """Top-k binvars ranked by parent-feature |attribution|, signed by instance value."""
    inst_bin = binarized_test_X[pos]
    scored = []
    for b in range(1, nb + 1):
        info = bt_map[b] if b < len(bt_map) else 0
        if not info or info == 0: continue
        fidx = info[0]                       # 1-based original feature index
        scored.append((abs(float(attr_vec[fidx - 1])), b))
    scored.sort(key=lambda x: (-x[0], x[1]))  # importance desc, binvar id asc (deterministic)
    term = []
    for _, b in scored[:k]:
        term.append(b if inst_bin[b - 1] == 1 else -b)
    return term

def arm_shap(pos, k):
    sv = _shap_pred_vec(_shap_all, pos, predicted_class(pos))   # rectified-tree Shapley, pred class
    return _feat_attr_to_term(sv, pos, k)

def arm_lime(pos, k):
    exp = _lime_expl.explain_instance(
        X_test_orig.iloc[pos].values, _rect_predict_proba,
        num_features=len(feat_names), num_samples=1000)
    attr = np.zeros(len(feat_names))
    for fidx, w in exp.as_map()[1]:          # label 1 weights, fidx = column index
        attr[fidx] = w
    return _feat_attr_to_term(attr, pos, k)

# ── model-level I_P (relative accuracy vs BT) ──────────────────────────────────
ip_pyxai = np.mean([int(dt_model_pyxai.predict_instance(x) == t)
                    for x, t in zip(binarized_test_X, binarized_test_y)]) * 100
ip_mimic = np.mean([int(clf_rectified.predict([x])[0] == t)
                    for x, t in zip(binarized_test_X, binarized_test_y)]) * 100
print(f'I_P  pyxai_sr-tree={ip_pyxai:.2f}%   mytree_sr-mimic={ip_mimic:.2f}%', flush=True)

# ── run all 4 arms over all test instances ─────────────────────────────────────
rows = []
agree_term, agree_score = 0, 0
N = len(binarized_test_X)
for pos in range(N):
    pred = predicted_class(pos)
    phi  = DNF[1 - pred]                 # opposite-class DNF (shared reference)
    t_px = arm_pyxai_sr(pos)
    t_mt = arm_mytree_sr(pos)
    k    = max(len(t_px), 1)             # SHAP/LIME budget = pyxai_sr length
    t_sh = arm_shap(pos, k)
    t_li = arm_lime(pos, k)
    s_px = formal_score(phi, t_px, nb)
    s_mt = formal_score(phi, t_mt, nb)
    s_sh = formal_score(phi, t_sh, nb)
    s_li = formal_score(phi, t_li, nb)
    cp_px, _ = cp2_fidelity(t_px, pred)
    cp_mt, _ = cp2_fidelity(t_mt, pred)
    cp_sh, _ = cp2_fidelity(t_sh, pred)
    cp_li, _ = cp2_fidelity(t_li, pred)
    if set(t_px) == set(t_mt): agree_term += 1
    if abs(s_px - s_mt) < 1e-6: agree_score += 1
    rows.append({'pos': pos, 'pred': pred,
                 'pyxai_sr_len': len(t_px), 'mytree_sr_len': len(t_mt),
                 'shap_len': len(t_sh), 'lime_len': len(t_li),
                 'pyxai_sr_score': round(s_px, 2), 'mytree_sr_score': round(s_mt, 2),
                 'shap_score': round(s_sh, 2), 'lime_score': round(s_li, 2),
                 'pyxai_sr_cp2': cp_px, 'mytree_sr_cp2': cp_mt,
                 'shap_cp2': cp_sh, 'lime_cp2': cp_li})
    if (pos + 1) % 25 == 0: print(f'  ...{pos+1}/{N}', flush=True)

# ── random baseline (corrected space): k instance-consistent literals ──────────
import random as _rnd; _rnd.seed(0)
rb_fs, rb_cp = [], []
for pos in range(N):
    pred = predicted_class(pos); phi = DNF[1 - pred]; inst = binarized_test_X[pos]
    ert.set_instance(inst); k = max(len(_dtid_lits_to_bt(list(ert.sufficient_reason()))), 1)
    for _ in range(5):
        bs = _rnd.sample(range(1, nb + 1), k)
        term = [b if inst[b - 1] == 1 else -b for b in bs]
        rb_fs.append(formal_score(phi, term, nb))
        cp, _n = cp2_fidelity(term, pred)
        if cp is not None: rb_cp.append(cp)
print(f'random baseline (k=SR len): formal_score={np.mean(rb_fs):.2f}   CP-2 fidelity={np.mean(rb_cp):.2f}')

df = pd.DataFrame(rows)
def _summ(name):
    sc = df[f'{name}_score']; ln = df[f'{name}_len']; cp = df[f'{name}_cp2'].dropna()
    print(f'{name:11s}: mean len={ln.mean():5.2f}  formal_score={sc.mean():6.2f}'
          f'  (==100: {(sc>=100).sum():3d}/{N})   CP-2 fidelity={cp.mean():6.2f}')
print('\n=== 4-arm comparison (over %d test instances, k=pyxai_sr len) ===' % N)
print('  formal_score = W_f sufficiency (vs rectified-tree DNF);  CP-2 = graded fidelity on train data')
for a in ['pyxai_sr', 'mytree_sr', 'shap', 'lime']: _summ(a)
print(f'pyxai_sr vs mytree_sr agreement: identical term {agree_term}/{N}   identical score {agree_score}/{N}')

df.to_csv(f'_arms_metric_{DATASET}.csv', index=False)
print(f'\nwrote _arms_metric_{DATASET}.csv', flush=True)
