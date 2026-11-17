import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from unittest.mock import MagicMock
mock_pb = MagicMock(); mock_pb.PBEnc = MagicMock()
sys.modules['pypblib'] = mock_pb; sys.modules['pypblib.pblib'] = mock_pb

import os, ast, time, warnings, random, re
sys.path.append('build'); warnings.filterwarnings('ignore')
# Rectified trees can have ~10^6 nodes and be very deep; pickling/predict recurse per node.
# Raise the OS stack (best-effort) so deep recursion won't segfault, then lift Python's
# recursion ceiling. The run script should also `ulimit -s unlimited` before launching.
try:
    import resource as _resource
    _resource.setrlimit(_resource.RLIMIT_STACK,
                        (_resource.RLIM_INFINITY, _resource.RLIM_INFINITY))
except Exception:
    pass
sys.setrecursionlimit(2_000_000)

# ── Supplementary material paths ──────────────────────────────────────────────
# Path to a PyXAI source checkout (with its examples/Distillation folder);
# set PYXAI_SRC. See earlier_tiers/README.md.
_SUPP  = os.environ.get('PYXAI_SRC', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyxai'))
_DIST  = os.path.join(_SUPP, 'pyxai', 'examples', 'Distillation')
sys.path.insert(0, _SUPP)
sys.path.insert(0, _DIST)

import numpy as np, pandas as pd, yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import shap, my_tree as mt

# The compiled C++ rectifier (c_explainer) lives in the bundled pyxai dir. Build once
# with:  cd $PYXAI_SRC && python3 setup.py build_ext --inplace
# Using the REAL c_explainer (not a mock) enables theory-aware rectification (cxx=True,
# theory_cnf=...) exactly as in the reference distillation code [Audemard et al.] — grafted rules are simplified against the
# BT theory, removing logically-impossible branches. We only mock the other heavy deps.
# Detect ANY compiled c_explainer*.so (any Python/ABI tag) so a locally-built rectifier
# is used regardless of interpreter version (e.g. cpython-312 on Ubuntu 24.04).
_HAS_CXX = os.path.isdir(_SUPP) and any(
    f.startswith('c_explainer') and f.endswith('.so') for f in os.listdir(_SUPP))
from unittest.mock import MagicMock as _MagicMock
_mock_mods = ['distributed', 'docplex', 'docplex.mp', 'docplex.mp.model',
              'docplex.mp.dvar', 'docplex.mp.linear']
if not _HAS_CXX:
    _mock_mods = ['c_explainer'] + _mock_mods      # fall back to Python rectifier (no theory)
for _mod in _mock_mods:
    if _mod not in sys.modules:
        sys.modules[_mod] = _MagicMock()

from pyxai import Learning, Explainer, Tools
from pysat.solvers import Glucose3
from decisionNode import DecisionNode

load_dotenv('.env')
DATASET_NAME  = sys.argv[1] if len(sys.argv) > 1 else 'cancer'
N_INSTANCES   = int(os.getenv('XAI_N_INSTANCES', '15'))   # override for cheap smoke tests
RANDOM_SEED   = 42
SKIP_SWEEPS   = bool(os.getenv('XAI_SKIP_SWEEPS'))        # skip cp1/cp3/oracle diagnostics (speed)
VERBOSE       = os.getenv('XAI_VERBOSE', '1') == '1'      # live per-arm explanation→rules→score blocks
SAVE_TRACES   = os.getenv('XAI_SAVE_TRACES', '1') == '1'  # per-instance JSON traces + tee'd console log

# ── Persistent console log (tee stdout → research/results/logs/<ds>_bt_<ts>.log) ──
import datetime as _dt, json as _json
if SAVE_TRACES:
    os.makedirs('research/results/logs', exist_ok=True)
    _LOG_PATH = (f'research/results/logs/{DATASET_NAME}_bt_'
                 f'{_dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    class _Tee:
        def __init__(self, *s): self.streams = s
        def write(self, t):
            for st in self.streams:
                st.write(t); st.flush()
        def flush(self):
            for st in self.streams: st.flush()
    _logfile = open(_LOG_PATH, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.stdout, _logfile)
    print(f'Full console log → {_LOG_PATH}', flush=True)

# ── LLM provider (default Together AI; set XAI_PROVIDER=groq to switch back) ────
PROVIDER = os.getenv('XAI_PROVIDER', 'together').lower()
import re as _re
from openai import OpenAI
if PROVIDER == 'together':
    LLM_BASE_URL = 'https://api.together.xyz/v1'
    GROQ_MODEL   = sys.argv[2] if len(sys.argv) > 2 else 'meta-llama/Llama-3.3-70B-Instruct-Turbo'
    _api_keys    = [k for k in [os.getenv('TOGETHER_API_KEY'),
                                os.getenv('TOGETHER_API_KEY_2'),
                                os.getenv('TOGETHER_API_KEY_3')] if k]
    SLEEP_BETWEEN = float(os.getenv('XAI_SLEEP', '0'))     # paid tier -> no delay by default
else:
    LLM_BASE_URL = 'https://api.groq.com/openai/v1'
    GROQ_MODEL   = sys.argv[2] if len(sys.argv) > 2 else 'llama-3.3-70b-versatile'
    _api_keys    = [k for k in [os.getenv('GROQ_API_KEY'),
                                os.getenv('GROQ_API_KEY_2'),
                                os.getenv('GROQ_API_KEY_3')] if k]
    SLEEP_BETWEEN = float(os.getenv('XAI_SLEEP', '5'))
_key_idx = 0

def _get_client():
    return OpenAI(api_key=_api_keys[_key_idx % len(_api_keys)], base_url=LLM_BASE_URL)

print(f'Loaded {len(_api_keys)} API key(s)', flush=True)

def parse_rules_from_llm(raw):
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    text = re.sub(r'```(?:python)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL).strip()
    try:
        result = ast.literal_eval(text)
        if isinstance(result, list):
            return [str(r).lower().strip() for r in result]
    except:
        pass
    m = re.search(r'\[([^\[\]]*)\]', text, re.DOTALL)
    if m:
        try:
            result = ast.literal_eval('[' + m.group(1) + ']')
            if isinstance(result, list):
                return [str(r).lower().strip() for r in result]
        except:
            pass
    return []

def groq_llm(prompt, max_tokens=1024, max_retries=15):
    global _key_idx
    for attempt in range(max_retries):
        try:
            client = _get_client()
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.2, max_tokens=max_tokens, timeout=90.0)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate_limit' in err.lower():
                _key_idx += 1
                if attempt == 0 or _key_idx % len(_api_keys) == 0:
                    print(f'  [rate-limit] switching key', flush=True)
                m_ms = _re.search(r'(\d+)m([\d.]+)s', err)
                m_s  = (_re.search(r'try again in ([\d.]+)s', err) or
                        _re.search(r'Please try again in ([\d.]+)s', err, _re.IGNORECASE))
                if m_ms:   wait = int(m_ms.group(1))*60 + float(m_ms.group(2)) + 10
                elif m_s:  wait = float(m_s.group(1)) + 5
                else:      wait = 60*(attempt+1)
                if (attempt+1) < len(_api_keys): wait = min(wait, 5)
                print(f'  [rate-limit] waiting {wait:.0f}s', flush=True)
                time.sleep(wait)
            elif 'timed out' in err.lower() or 'timeout' in err.lower():
                wait = 15*(attempt+1)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f'Groq: max retries exceeded')

# ── Dataset loading ────────────────────────────────────────────────────────────
with open('datasets_config.yaml', 'r', encoding='utf-8') as f:
    dataset_config = yaml.safe_load(f).get(DATASET_NAME)
TARGET   = dataset_config['colonne_cible']
CSV_PATH = dataset_config['chemin_csv']

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.lower().str.replace(' ', '_')

counts = df[TARGET].value_counts()
maj, minn = counts.idxmax(), counts.idxmin()
df_min_os = resample(df[df[TARGET] == minn], replace=True,
                     n_samples=len(df[df[TARGET] == maj]), random_state=RANDOM_SEED)
df = pd.concat([df[df[TARGET] == maj], df_min_os])

y = df[TARGET]; X = df.drop(TARGET, axis=1)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED)
# Reset to avoid duplicate indices from oversampling
X_train_orig = X_train_orig.reset_index(drop=True)
X_test_orig  = X_test_orig.reset_index(drop=True)
y_train_orig = y_train_orig.reset_index(drop=True)
y_test_orig  = y_test_orig.reset_index(drop=True)
original_feature_names = list(X.columns)

# ── BT Distillation Pipeline (from the reference distillation code) ────────────
import joblib
_CACHE_FILE = f'_bt_cache_{DATASET_NAME}.pkl'

def _build_bina_inv(my_tree_obj, bt_map, original_feature_names):
    _bina_inv = {}; _feat_thresholds = {}
    for (feat_idx_bina, _thresh), (lit_id, _) in my_tree_obj.bina.items():
        k = feat_idx_bina
        if k < len(bt_map):
            info = bt_map[k]
            if info and info != 0:
                orig_feat_idx, _op, orig_thresh = info
                fname = original_feature_names[orig_feat_idx - 1]
                _bina_inv[ lit_id] = (fname, '>',  orig_thresh)
                _bina_inv[-lit_id] = (fname, '<=', orig_thresh)
                _feat_thresholds.setdefault(fname, [])
                if orig_thresh not in _feat_thresholds[fname]:
                    _feat_thresholds[fname].append(orig_thresh)
    for fname in _feat_thresholds:
        _feat_thresholds[fname] = sorted(_feat_thresholds[fname])
    return _bina_inv, _feat_thresholds

def python_ts_reason(explainer, n_iterations=10):
    """Pure-Python greedy TS reason — fallback when c_explainer not compiled."""
    binary_rep = list(explainer.binary_representation)
    best = binary_rep.copy()
    for _ in range(n_iterations):
        order = binary_rep.copy(); random.shuffle(order)
        candidate = binary_rep.copy()
        for lit in order:
            if lit not in candidate: continue
            temp = [l for l in candidate if l != lit]
            if explainer.is_implicant(tuple(temp)): candidate = temp
        if len(candidate) < len(best): best = candidate
    return tuple(sorted(best, key=abs))

if not os.path.exists(_CACHE_FILE):
    Tools.set_verbose(0)
    glucose = Glucose3()

    train_df_csv = pd.concat([X_train_orig, y_train_orig], axis=1)
    train_df_csv.to_csv('_bt_train.csv', index=False)

    print('Tuning BT hyperparameters...', flush=True)
    best_params_bt = DecisionNode.tuning('_bt_train.csv')

    print('Training BT...', flush=True)
    bt_learner = Learning.Xgboost('_bt_train.csv', learner_type=Learning.CLASSIFICATION)
    bt_model   = bt_learner.evaluate(method=Learning.HOLD_OUT, output=Learning.BT,
                                     seed=0, **best_params_bt)

    first_inst, _ = bt_learner.get_instances(bt_model, n=1)
    types_path = CSV_PATH.replace('.csv', '.types')
    bt_explainer = Explainer.initialize(
        bt_model, first_inst,
        features_type=types_path if os.path.exists(types_path) else None)

    _bt_raw = bt_learner.get_raw_models()[0]   # raw XGBClassifier (for SHAP/LIME-on-BT arms)

    for clause in bt_model.get_theory(bt_explainer.binary_representation):
        glucose.add_clause(clause)

    nb_binary = len(bt_explainer.binary_representation)
    bina_cols  = [f'X_{k}' for k in range(1, nb_binary + 1)]
    print(f'BT trained. Binary variables: {nb_binary}', flush=True)

    # Binarize training data
    binarized_train_X, binarized_train_y = [], []
    for i, instance in enumerate(bt_learner.data):
        bt_explainer.set_instance(instance)
        binarized_train_X.append([0 if l < 0 else 1 for l in bt_explainer.binary_representation])
        binarized_train_y.append(bt_learner.labels[i])

    # Binarize test data
    binarized_test_X, binarized_test_y = [], []
    for i in range(len(X_test_orig)):
        bt_explainer.set_instance(X_test_orig.iloc[i].values)
        binarized_test_X.append([0 if l < 0 else 1 for l in bt_explainer.binary_representation])
        binarized_test_y.append(bt_explainer.target_prediction)

    # Train initial DT on binarized data
    training_data_bina = pd.DataFrame(binarized_train_X, columns=bina_cols)
    training_data_bina['y'] = binarized_train_y
    dt_learner     = Learning.Scikitlearn(training_data_bina, learner_type=Learning.CLASSIFICATION)
    dt_model_pyxai = dt_learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT, seed=0)
    if isinstance(dt_model_pyxai, list): dt_model_pyxai = dt_model_pyxai[0]

    # Build theory + disagreements + rectification
    ert = Explainer.initialize(dt_model_pyxai)
    ths = bt_model.get_theory(bt_explainer.binary_representation)
    if ths:
        theorie        = DecisionNode.trasforme_list_tuple_to_binaire(ths, dt_model_pyxai)
        theorie_clause = ert.condi(conditions=theorie)
        theorie_clause = DecisionNode.list_to_tuple_pairs(theorie_clause)
    else:
        theorie_clause = []

    reasons_0, reasons_1 = [], []; n_disagree = 0
    for i, bina_inst in enumerate(binarized_test_X):
        bt_explainer.set_instance(X_test_orig.iloc[i].values)
        bt_pred = bt_explainer.target_prediction
        dt_pred = dt_model_pyxai.predict_instance(bina_inst)
        if dt_pred != bt_pred:
            n_disagree += 1
            ts_reason = python_ts_reason(bt_explainer, n_iterations=50)  # match the reference distillation code
            rule      = DecisionNode.trasforme_tuple_to_binaire(ts_reason, dt_model_pyxai)
            (reasons_1 if bt_pred == 1 else reasons_0).append(rule)

    print(f'Disagreements: {n_disagree}  |  rules_1: {len(reasons_1)}  rules_0: {len(reasons_0)}',
          flush=True)

    # Rectify. With the compiled C++ rectifier (cxx=True) we pass the raw binarised
    # reason as `conditions` and `theory_cnf=theorie_clause` — identical to the reference distillation code
    # main.py: grafted rules are simplified against the BT theory (drops impossible
    # branches). Without the .so we fall back to the theory-free Python path (the
    # ert.condi wrap is required there since that path doesn't re-parse conditions).
    if _HAS_CXX:
        print('Rectifying with compiled C++ rectifier + theory (reference path).', flush=True)
        for rule in reasons_1:
            dt_model_pyxai = ert.rectify(conditions=rule, label=1, cxx=True,
                                         tests=False, theory_cnf=theorie_clause)
        for rule in reasons_0:
            dt_model_pyxai = ert.rectify(conditions=rule, label=0, cxx=True,
                                         tests=False, theory_cnf=theorie_clause)
    else:
        print('WARNING: c_explainer not compiled — theory-FREE Python rectification.', flush=True)
        for rule in reasons_1:
            dt_model_pyxai = ert.rectify(conditions=ert.condi(conditions=rule), label=1, cxx=False, tests=False)
        for rule in reasons_0:
            dt_model_pyxai = ert.rectify(conditions=ert.condi(conditions=rule), label=0, cxx=False, tests=False)
    print(f'Rectified DT nodes: {dt_model_pyxai.n_nodes()}', flush=True)

    # Bridge: fit sklearn DT to rectified PyXAI DT
    rectified_labels = [dt_model_pyxai.predict_instance(inst) for inst in binarized_train_X]
    clf_rectified = DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1)
    clf_rectified.fit(binarized_train_X, rectified_labels)
    match = sum(clf_rectified.predict([inst])[0] == dt_model_pyxai.predict_instance(inst)
                for inst in binarized_train_X)
    print(f'sklearn/PyXAI agreement on train: {match}/{len(binarized_train_X)}', flush=True)

    bt_map = list(bt_model.map_id_binaries_to_features)

    # Save cache.  dt_model_pyxai (the rectified PyXAI tree) is joblib-picklable and is
    # persisted so the 4-arm comparison can score against the native rectified tree
    # (pyxai_sr arm + the shared reference DNF) without replaying rectification.
    joblib.dump({
        'clf_rectified':     clf_rectified,
        'dt_model_pyxai':    dt_model_pyxai,
        'binarized_train_X': binarized_train_X,
        'binarized_train_y': binarized_train_y,
        'binarized_test_X':  binarized_test_X,
        'binarized_test_y':  binarized_test_y,
        'nb_binary':         nb_binary,
        'bina_cols':         bina_cols,
        'bt_map':            bt_map,
        'best_params_bt':    best_params_bt,
    }, _CACHE_FILE)
    print(f'Distillation cache saved to {_CACHE_FILE}', flush=True)

else:
    print(f'Loading distillation cache from {_CACHE_FILE} — skipping distillation.', flush=True)
    _cache          = joblib.load(_CACHE_FILE)
    clf_rectified   = _cache['clf_rectified']
    dt_model_pyxai  = _cache.get('dt_model_pyxai')   # rectified PyXAI tree (may be absent in old caches)
    binarized_train_X = _cache['binarized_train_X']
    binarized_train_y = _cache['binarized_train_y']
    binarized_test_X  = _cache['binarized_test_X']
    binarized_test_y  = _cache['binarized_test_y']
    nb_binary         = _cache['nb_binary']
    bina_cols         = _cache['bina_cols']
    bt_map            = _cache['bt_map']
    best_params_bt    = _cache['best_params_bt']

    # Reload BT explainer (needed for _BinarizedDT predictor)
    Tools.set_verbose(0)
    train_df_csv = pd.concat([X_train_orig, y_train_orig], axis=1)
    train_df_csv.to_csv('_bt_train.csv', index=False)
    bt_learner_cache = Learning.Xgboost('_bt_train.csv', learner_type=Learning.CLASSIFICATION)
    bt_model_cache   = bt_learner_cache.evaluate(method=Learning.HOLD_OUT, output=Learning.BT,
                                                  seed=0, **best_params_bt)
    first_inst, _    = bt_learner_cache.get_instances(bt_model_cache, n=1)
    bt_explainer     = Explainer.initialize(bt_model_cache, first_inst)
    _bt_raw          = bt_learner_cache.get_raw_models()[0]   # raw XGBClassifier

# ── Shared: explain the REAL rectified PyXAI tree (not the sklearn mimic) ──────
# We feed the LLM PyXAI's native sufficient_reason() and score its recovered term against
# the rectified tree's own opposite-class DNF. dt_model_pyxai's explainer has its OWN binvar
# id space (1..#used-feats), NOT the BT binvar ids (1..nb); we convert SR + DNF into BT-binvar
# space so they align with bt_map / the binarization / SHAP. (See research/six_arm... + the
# pyxai-dt-idspace-gotcha memory.)
assert dt_model_pyxai is not None, 'cache lacks dt_model_pyxai — regenerate with XAI_CACHE_ONLY=1'
from pyxai.sources.core.structure.decisionTree import TypeLeaf
my_tree = mt.decision_tree()
my_tree.from_DecisionTreeClassifier(clf_rectified)   # kept for SHAP feature selection only

_dtid_to_btbinvar = {v[0]: idf for (idf, _op, _th), v in
                     dt_model_pyxai.map_features_to_id_binaries.items()}
def _dtid_lits_to_bt(lits):
    out = []
    for L in lits:
        bv = _dtid_to_btbinvar.get(abs(L))
        if bv is not None:
            out.append(bv if L > 0 else -bv)
    return out

def _build_class_dnf(tree, target):
    dnf = []
    for node in tree.compute_nodes_with_leaves(tree.root):
        if node.left.is_leaf() and node.left.is_prediction(target):
            dnf.append(_dtid_lits_to_bt(tree.create_cube(node, TypeLeaf.LEFT)))
        if node.right.is_leaf() and node.right.is_prediction(target):
            dnf.append(_dtid_lits_to_bt(tree.create_cube(node, TypeLeaf.RIGHT)))
    return dnf
DNF_BT = {0: _build_class_dnf(dt_model_pyxai, 0), 1: _build_class_dnf(dt_model_pyxai, 1)}

ert = Explainer.initialize(dt_model_pyxai)

# _bina_inv keyed by BT binvar id (1..nb): positive lit = (feat > thresh), negative = (<=).
_bina_inv, _feat_thresholds = {}, {}
for k in range(1, nb_binary + 1):
    info = bt_map[k] if k < len(bt_map) else 0
    if not info or info == 0: continue
    fname = original_feature_names[info[0] - 1]; thr = info[2]
    _bina_inv[k]  = (fname, '>',  thr)
    _bina_inv[-k] = (fname, '<=', thr)
    _feat_thresholds.setdefault(fname, [])
    if thr not in _feat_thresholds[fname]: _feat_thresholds[fname].append(thr)
for fname in _feat_thresholds: _feat_thresholds[fname] = sorted(_feat_thresholds[fname])

# ── I_P: relative accuracy of the explained surrogate (rectified PyXAI tree) vs the BT ──
# Now measured against dt_model_pyxai (the tree we actually explain), so I_P = 100% by
# construction — removing the mimic's ~5% generalization gap as a fidelity confound.
_ip_preds = [dt_model_pyxai.predict_instance(x) for x in binarized_test_X]
I_P = float(np.mean([int(p == t) for p, t in zip(_ip_preds, binarized_test_y)])) * 100
print(f'I_P (rectified PyXAI tree vs BT on test set): {I_P:.2f}%  '
      f'({sum(int(p==t) for p,t in zip(_ip_preds, binarized_test_y))}/{len(binarized_test_y)})',
      flush=True)

# ── Cache-only mode: regenerate the distillation cache (incl. dt_model_pyxai) and
# stop before any LLM calls. Used to prepare the 4-arm metric-layer comparison. ──
if os.getenv('XAI_CACHE_ONLY'):
    print('XAI_CACHE_ONLY set — distillation cache ready, exiting before LLM loop.', flush=True)
    sys.exit(0)

# ── SHAP on binarized DT, aggregated to original features ─────────────────────
_shap_explainer_bina = shap.TreeExplainer(clf_rectified)
_shap_values_bina    = _shap_explainer_bina.shap_values(
    pd.DataFrame(binarized_test_X, columns=bina_cols))

def _get_shap_orig(pos):
    """Return {orig_feature_name: abs_shap_value} aggregated from binarized SHAP."""
    if isinstance(_shap_values_bina, list):
        sv = _shap_values_bina[binarized_test_y[pos]][pos]
    elif _shap_values_bina.ndim == 3:
        sv = _shap_values_bina[pos, :, binarized_test_y[pos]]
    else:
        sv = _shap_values_bina[pos]

    agg = {}
    for k in range(1, nb_binary + 1):
        if k < len(bt_map):
            info = bt_map[k]
            if info and info != 0:
                orig_feat_idx = info[0]
                fname = original_feature_names[orig_feat_idx - 1]
                agg[fname] = agg.get(fname, 0.0) + abs(float(sv[k - 1]))
    return agg

# ── dt wrapper: accept original feature instances, binarize, predict with the
# rectified PyXAI tree (the model we explain) — keeps CP-2 / prediction consistent
# with the SR + reference DNF. ────────────────────────────────────────────────
class _BinarizedDT:
    def predict(self, X):
        results = []
        for row in np.atleast_2d(X):
            bt_explainer.set_instance(row)
            bina = [0 if l < 0 else 1 for l in bt_explainer.binary_representation]
            results.append(int(dt_model_pyxai.predict_instance(bina)))
        return np.array(results)

    def predict_proba(self, X):
        # one-hot proba from the DT's hard prediction (only argmax is used downstream)
        preds = self.predict(X)
        out = np.zeros((len(preds), 2))
        for i, p in enumerate(preds): out[i, int(p)] = 1.0
        return out

dt = _BinarizedDT()

# X_test / X_train exposed to evaluation functions (original feature space)
X_test  = X_test_orig.reset_index(drop=True)
X_train = X_train_orig.reset_index(drop=True)

print(f'bina entries: {len(my_tree.bina)}  |  X_test: {len(X_test)} instances', flush=True)

# ── 4-arm explanation builders (all terms in BT-binvar space 1..nb) ────────────
# Each arm provides the explanation shown to the LLM for one prompt; the LLM run is
# repeated per arm (never combined). pyxai_sr/mytree_sr = sufficient reasons; shap/lime =
# top-k binvars (k = pyxai_sr length) by the RECTIFIED TREE's attribution, signed by value.
import lime, lime.lime_tabular
_cardbin_to_btbinvar = {cb: bv for (bv, _t), (cb, _n) in my_tree.bina.items()}
# SHAP/LIME explain the RECTIFIED TREE (dt_model_pyxai via `dt`), NOT the boosted tree, so
# pyxai_sr, the reference DNF and the SHAP/LIME arms all describe the same model. The tree
# isn't sklearn → model-agnostic KernelExplainer over the binarize→predict wrapper, computed
# lazily per instance (only the sampled ones).
def _shap_pred_vec(shap_out, idx, pc):
    if isinstance(shap_out, list):
        return np.asarray(shap_out[pc])[idx]
    a = np.asarray(shap_out)
    if a.ndim == 3:
        return a[idx, :, pc]
    return a[idx]
_shap_rect_expl  = shap.KernelExplainer(dt.predict_proba, shap.kmeans(X_train.values, 15))
_shap_rect_cache = {}
def _rect_shap_vals(pos):
    if pos not in _shap_rect_cache:
        _shap_rect_cache[pos] = _shap_rect_expl.shap_values(
            X_test.values[pos:pos+1], silent=True,
            l1_reg=f'num_features({len(original_feature_names)})')
    return _shap_rect_cache[pos]
_lime_expl = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, feature_names=original_feature_names,
    class_names=['0', '1'], discretize_continuous=True, random_state=RANDOM_SEED)

def _feat_attr_to_term(attr_vec, pos, k):
    inst_bin = binarized_test_X[pos]; scored = []
    for b in range(1, nb_binary + 1):
        info = bt_map[b] if b < len(bt_map) else 0
        if not info or info == 0: continue
        scored.append((abs(float(attr_vec[info[0] - 1])), b))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [b if inst_bin[b - 1] == 1 else -b for _, b in scored[:k]]

def arm_term(arm, pos, k_ref):
    if arm == 'pyxai_sr':
        ert.set_instance(binarized_test_X[pos])
        return _dtid_lits_to_bt(list(ert.sufficient_reason()))
    if arm == 'mytree_sr':
        inst = binarized_test_X[pos]; tgt = my_tree.take_decision(instance=inst)
        lits = my_tree.find_sufficient_reason(instance=inst, target=tgt)
        return [(_cardbin_to_btbinvar[abs(l)] if l > 0 else -_cardbin_to_btbinvar[abs(l)])
                for l in lits if abs(l) in _cardbin_to_btbinvar]
    if arm == 'shap':
        pc = int(dt_model_pyxai.predict_instance(binarized_test_X[pos]))
        sv = _shap_pred_vec(_rect_shap_vals(pos), 0, pc)
        return _feat_attr_to_term(sv, pos, k_ref)
    if arm == 'lime':
        exp = _lime_expl.explain_instance(
            X_test.iloc[pos].values, dt.predict_proba,
            num_features=len(original_feature_names), num_samples=1000)
        attr = np.zeros(len(original_feature_names))
        for fidx, w in exp.as_map()[1]: attr[fidx] = w
        return _feat_attr_to_term(attr, pos, k_ref)
    raise ValueError(arm)

# Experiment arms: the symbolic SR (our method) vs the two post-hoc XAI baselines.
# mytree_sr (the sklearn mimic) is NOT an experimental arm — it was only the old
# adapter; the pyxai-vs-mimic comparison lives in research/ as methodological
# justification for using native PyXAI. (arm_term still supports 'mytree_sr' if needed.)
ARMS = ['pyxai_sr', 'shap', 'lime']

# ── Everything below is identical to run_comparison.py ────────────────────────

def sr_to_conditions(sr_lits):
    raw = [_bina_inv[l] for l in sr_lits if l in _bina_inv]
    tightest = {}
    for fname, op, thresh in raw:
        key = (fname, op)
        if key not in tightest:
            tightest[key] = thresh
        else:
            if op == '>'  and thresh > tightest[key]: tightest[key] = thresh
            if op == '<=' and thresh < tightest[key]: tightest[key] = thresh
    return [f'{fname} {op} {round(thresh, 6)}' for (fname, op), thresh in tightest.items()]

def bina_term_transform(term, features):
    bina_tree = my_tree.bina
    def parse(cond):
        if '<=' in cond:
            v, n = cond.split('<='); return (features.get_loc(v.strip()) + 1, 0, float(n.strip()))
        elif '<' in cond:
            v, n = cond.split('<');  return (features.get_loc(v.strip()) + 1, 0, float(n.strip()))
        elif '>=' in cond:
            v, n = cond.split('>='); return (features.get_loc(v.strip()) + 1, 1, float(n.strip()))
        elif '>' in cond:
            v, n = cond.split('>');  return (features.get_loc(v.strip()) + 1, 1, float(n.strip()))
        elif '=' in cond:
            v, n = cond.split('=');  return (features.get_loc(v.strip()) + 1, 1, float(n.strip()))
        raise ValueError(cond)
    parsed = [parse(c) for c in term]
    result = []
    for x, y_val, z in parsed:
        for key in [k for k in bina_tree if k[0] == x]:
            k, l = key; t, c = bina_tree[key]
            if y_val == 1 and z == l:   result.append(t);  break
            elif y_val == 0 and z == l: result.append(-t); break
    return result

def bina_term_transform_v2(term, features):
    bina_tree = my_tree.bina
    EPSILON   = 0.15

    def parse(cond):
        cond = cond.strip()
        if '<=' in cond:
            v, n = cond.split('<=', 1); return (features.get_loc(v.strip()) + 1, 0, float(n.strip()))
        elif '>=' in cond:
            v, n = cond.split('>=', 1); return (features.get_loc(v.strip()) + 1, 1, float(n.strip()))
        elif '==' in cond:
            v, n = cond.split('==', 1); val = float(n.strip())
            return (features.get_loc(v.strip()) + 1, 0 if val < 0.5 else 1, val)
        elif '<' in cond:
            v, n = cond.split('<', 1);  return (features.get_loc(v.strip()) + 1, 0, float(n.strip()))
        elif '>' in cond:
            v, n = cond.split('>', 1);  return (features.get_loc(v.strip()) + 1, 1, float(n.strip()))
        elif '=' in cond:
            v, n = cond.split('=', 1);  val = float(n.strip())
            return (features.get_loc(v.strip()) + 1, 0 if val < 0.5 else 1, val)
        raise ValueError(cond)

    result, seen = [], set()
    for cond in term:
        try: feat_idx, direction, ext_thresh = parse(cond)
        except: continue
        keys = [k for k in bina_tree if k[0] == feat_idx]
        if not keys: continue
        lit = None
        for key in keys:
            _, tree_thresh = key; t = bina_tree[key][0]
            if abs(tree_thresh - 0.5) < 1e-9:
                lit = t if direction == 1 else -t; break
            if abs(ext_thresh - tree_thresh) <= EPSILON:
                lit = t if direction == 1 else -t; break
        if lit is None:
            best = min(keys, key=lambda k: abs(k[1] - ext_thresh))
            dist = abs(best[1] - ext_thresh)
            if dist <= 1.0:
                t = bina_tree[best][0]; lit = t if direction == 1 else -t
            else:
                continue
        if -lit in seen: continue
        if lit not in seen: result.append(lit); seen.add(lit)
    return result

# ── W_f: the ORIGINAL metric from the paper (theory-free) ──────────────────────
# Weighted model count over the 2^n binary-assignment space (binvars treated as
# independent; the theory/ordering is NOT folded in). This is the metric as defined
# in the original work; kept verbatim for fidelity/comparability with the paper.
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

def formal_score(phi, bina_term, n):
    if not bina_term: return 0.0
    return (1 - W_f(phi, bina_term, n)) * 100

def cp1_optimal_sweep(phi, n_features, k_max, exact_threshold=60000):
    from itertools import combinations as _combs
    from math import comb as _comb
    all_lits = list(range(1, n_features + 1)) + list(range(-1, -(n_features + 1), -1))
    results  = []
    for k in range(1, k_max + 1):
        n_cands   = 2 * n_features
        n_subsets = _comb(n_cands, k)
        if n_subsets <= exact_threshold:
            best_score = 0.0
            for subset in _combs(all_lits, k):
                if any(-l in subset for l in subset): continue
                s = formal_score(phi, list(subset), n_features)
                if s > best_score: best_score = s
            results.append((k, round(best_score, 1), 'exact'))
        else:
            current = []
            for _ in range(k):
                best_gain, best_lit = -1.0, None
                for l in all_lits:
                    if l in current or -l in current: continue
                    s = formal_score(phi, current + [l], n_features)
                    if s > best_gain: best_gain = s; best_lit = l
                if best_lit is not None: current.append(best_lit)
            results.append((k, round(formal_score(phi, current, n_features), 1), 'greedy'))
        if results[-1][1] >= 100.0:
            for kk in range(k + 1, k_max + 1): results.append((kk, 100.0, 'triv'))
            break
    return results

def cp3_size_sweep(bina_term, phi, n_features):
    from itertools import combinations as _combs
    if not bina_term: return [], None, None, None
    n = len(bina_term); sweep = []
    for k in range(1, n + 1):
        best_k_score = 0.0
        for subset in _combs(bina_term, k):
            s = formal_score(phi, list(subset), n_features)
            if s > best_k_score: best_k_score = s
        sweep.append((k, round(best_k_score, 1)))
        if best_k_score >= 100.0:
            for kk in range(k + 1, n + 1): sweep.append((kk, 100.0))
            break
    min_k_80  = next((k for k, s in sweep if s >= 80.0),  None)
    min_k_90  = next((k for k, s in sweep if s >= 90.0),  None)
    min_k_100 = next((k for k, s in sweep if s >= 100.0), None)
    return sweep, min_k_80, min_k_90, min_k_100

def compute_precision_error(s_literals, x_instance, x_pred_class):
    if not s_literals: return None, 0
    conditions = []
    for lit in s_literals:
        if lit in _bina_inv:
            fname, op, thresh = _bina_inv[lit]
            conditions.append((fname, op, thresh))
    if not conditions: return None, 0
    mask = np.ones(len(X_train), dtype=bool)
    for fname, op, thresh in conditions:
        if fname not in X_train.columns: continue
        col = X_train[fname].values
        if op == '>': mask &= col > thresh
        else:         mask &= col <= thresh
    n_neighbors = int(mask.sum())
    if n_neighbors == 0: return None, 0
    neighbors      = X_train.values[mask]
    neighbor_preds = dt.predict(neighbors)
    n_different    = int((neighbor_preds != x_pred_class).sum())
    return round(n_different / n_neighbors, 4), n_neighbors

def compute_precision_error_oracle(s_literals, x_pred_class, m=1000, seed=42):
    import math
    if not s_literals: return None, 0, None
    conditions = []
    for lit in s_literals:
        if lit in _bina_inv:
            fname, op, thresh = _bina_inv[lit]
            conditions.append((fname, op, thresh))
    if not conditions: return None, 0, None
    rng = np.random.RandomState(seed)
    feat_mins = X_train.min(); feat_maxs = X_train.max()
    random_df = pd.DataFrame({
        col: rng.uniform(feat_mins[col], feat_maxs[col], m)
        for col in X_train.columns})
    mask = np.ones(m, dtype=bool)
    for fname, op, thresh in conditions:
        if fname not in random_df.columns: continue
        col = random_df[fname].values
        if op == '>': mask &= col > thresh
        else:         mask &= col <= thresh
    n_neighbors = int(mask.sum())
    if n_neighbors == 0: return None, 0, None
    neighbors      = random_df.values[mask]
    neighbor_preds = dt.predict(neighbors)
    n_different    = int((neighbor_preds != x_pred_class).sum())
    eps_oracle     = round(n_different / n_neighbors, 4)
    delta          = 0.05
    pac_bound      = round(math.sqrt(math.log(2.0/delta) / (2.0*n_neighbors)), 4)
    return eps_oracle, n_neighbors, pac_bound

def lit_to_feature(lit):
    if  lit in _bina_inv: return _bina_inv[ lit][0]
    if -lit in _bina_inv: return _bina_inv[-lit][0]
    return None

def missed_features(sr_lits, bina_term):
    term_feats = {lit_to_feature(l) for l in bina_term} - {None}
    sr_feats   = {lit_to_feature(l) for l in sr_lits}   - {None}
    return sorted(sr_feats - term_feats)

def decompose_errors(sr_lits, bina_term):
    sr_set   = set(sr_lits)
    term_set = set(bina_term)
    commissions = [l for l in bina_term if -l in sr_set]
    redundant   = [l for l in bina_term if l not in sr_set and -l not in sr_set]
    omissions   = [l for l in sr_lits  if l not in term_set and -l not in term_set]
    filtered    = [l for l in bina_term if -l not in sr_set]
    return {
        'n_omissions':      len(omissions),
        'n_commissions':    len(commissions),
        'n_redundant':      len(redundant),
        'omission_feats':   sorted({lit_to_feature(l) for l in omissions}   - {None}),
        'commission_feats': sorted({lit_to_feature(l) for l in commissions} - {None}),
        'filtered_term':    filtered,
    }

MAX_ROUNDS = 3

PROMPT = (
    "System: You are a machine learning expert explaining a model decision to a non-expert.\n"
    "Model objective: {obj}\n"
    "Instance: {inst}\n"
    "Predicted class: {cls}\n"
    "{shap}\n\n"
    "Instructions: Be concise. Focus on the most influential features and their exact values. "
    "No special formatting.\n"
    "Question: Explain the reasons behind this prediction."
)

PROMPT_V3 = (
    "System: You are a machine learning expert explaining a model decision to a non-expert.\n"
    "Model objective: {obj}\n"
    "Instance: {inst}\n"
    "Predicted class: {cls}\n"
    "{shap}\n\n"
    "Instructions: Be concise. When stating conditions, use ONLY the threshold values listed "
    "in brackets above for each feature - do not invent other values. "
    "No special formatting.\n"
    "Question: Explain the reasons behind this prediction."
)

PROMPT_V4 = (
    "System: You are a machine learning expert explaining a model decision to a non-expert.\n"
    "Model objective: {obj}\n"
    "Instance: {inst}\n"
    "Predicted class: {cls}\n"
    "{sr}\n\n"
    "Instructions: Explain in plain language why this prediction was made, referring ONLY to "
    "the information in the context above. Do NOT invent numeric thresholds that are not given "
    "in the context. State each feature's condition at most once and never in two opposite "
    "directions. Be concise. No special formatting.\n"
    "Question: Explain the reasons behind this prediction."
)

PROMPT_V6 = (
    "System: You are a machine learning expert explaining a model decision.\n"
    "Model objective: {obj}\n"
    "Instance: {inst}\n"
    "Predicted class: {cls}\n"
    "{sr}\n\n"
    "Instructions: Write EXACTLY one numbered sentence per item listed above. Do NOT merge or "
    "skip any item. State each feature at most once and never in two opposite directions. "
    "Use ONLY the numeric values that appear in the context above; do NOT invent thresholds.\n"
    "Answer:"
)

EXTRACT = (
    "You are an expert extracting logical conditions from ML explanations.\n"
    "Extract all conditions as: <variable> <operator> <numeric_value>\n"
    "Allowed variables: {feats}\n"
    "Rules: variable=1 -> variable>0.5 | variable=0 -> variable<0.5 | no combined lines\n"
    "Example explanation: {ex_expl}\n"
    "Example output: {ex_rules}\n"
    "Constraints: output ONLY the Python list, no extra text, no code blocks.\n"
    "Explanation: {expl}\n"
    "Output:"
)

REFINE = (
    "You are a machine learning expert refining an explanation based on feedback.\n"
    "Model objective: {obj}\n"
    "Instance: {inst}\n"
    "Predicted class: {cls}\n"
    "{sr}\n\n"
    "Features from the context above not yet reflected in your explanation: {feedback}\n"
    "IMPORTANT: Keep the conditions you already stated correctly and add a condition for each "
    "listed feature ONLY if the context above supports it. State each feature at most once and "
    "NEVER in two opposite directions. Do NOT invent numeric thresholds that are not in the "
    "context. Be concise and specific. No special formatting."
)

# feature index for bina_term_transform (original feature names, 1-indexed)
_orig_features_idx = pd.Index(original_feature_names)

# ── Leak-free per-arm context (matches run_comparison_xai.py) ──────────────────
# pyxai_sr -> SR conditions WITH thresholds (legitimate: a formal SR is exact numbers).
# shap     -> importance + direction, NO thresholds (no tree-threshold leak).
# lime     -> LIME's own discretized descriptors + weight (its data bins, not tree splits).
# Instance values are shown via inst_repr to all arms (legitimate observed data).
def build_shap_ctx_bt(pos, k):
    pc = int(dt_model_pyxai.predict_instance(binarized_test_X[pos]))
    sv = _shap_pred_vec(_rect_shap_vals(pos), 0, pc)
    pairs = sorted(((abs(float(sv[j])), j) for j in range(len(original_feature_names))),
                   key=lambda x: -x[0])[:max(k, 1)]
    total = sum(m for m, _ in pairs) or 1.0
    lines, own = ['SHAP feature importance (contribution to this prediction):'], []
    for m, j in pairs:
        f = original_feature_names[j]; own.append(f)
        lines.append(f'  {f}: {m/total*100:.1f}% ({"positive" if float(sv[j]) > 0 else "negative"})')
    return '\n'.join(lines), sorted(set(own))

def build_lime_ctx_bt(pos, k):
    try:
        exp = _lime_expl.explain_instance(X_test.iloc[pos].values, dt.predict_proba,
                                          num_features=max(k, 1), num_samples=1000)
        lines, own = ['LIME local explanation (linear approximation around this instance):'], []
        for feat_desc, weight in exp.as_list():
            lines.append(f'  {feat_desc}: weight={weight:+.4f} '
                         f'({"supports prediction" if weight > 0 else "opposes prediction"})')
            for fn in original_feature_names:
                if fn in feat_desc: own.append(fn); break
        return '\n'.join(lines), sorted(set(own))
    except Exception as e:
        return f'LIME: unavailable ({e})', []

def arm_context(arm, pos, sr_lits, k_ref):
    """Return (context_text_with_header, own_feature_names) — leak-free for shap/lime."""
    if arm in ('pyxai_sr', 'mytree_sr'):
        conds = sr_to_conditions(sr_lits)
        ctx = ('Formal sufficient reason - the minimal set of conditions that guarantees '
               'this prediction:\n' + '\n'.join(f'  {c}' for c in conds))
        return ctx, sorted({lit_to_feature(l) for l in sr_lits} - {None})
    if arm == 'shap': return build_shap_ctx_bt(pos, k_ref)
    if arm == 'lime': return build_lime_ctx_bt(pos, k_ref)
    raise ValueError(arm)

# feature_name -> sorted [(threshold, +lit_id)]; per-instance feature values (set by
# run_strategies). Used to map a direction-only 0.5 placeholder to the split NEAREST the
# instance's own value of that feature (correct literal), instead of dropping it.
_feat_splits = {}
for _lid, (_fn, _op, _thr) in _bina_inv.items():
    if _lid > 0: _feat_splits.setdefault(_fn, []).append((_thr, _lid))
for _f in _feat_splits: _feat_splits[_f].sort()
_CUR_INST_VALS = {}

def bina_term_transform_bt(term):
    """
    Map extracted rule strings -> BT-binvar literals via _bina_inv.

    A 0.5 value is a DIRECTION-ONLY placeholder (the extractor had no real threshold —
    typical for SHAP/LIME). It is mapped by direction to the split NEAREST THE INSTANCE'S
    OWN VALUE of that feature (scale-independent, never dropped). Real thresholds use the
    EPSILON distance guard as before.
    """
    EPSILON = 1.0
    result, seen = [], set()

    for cond in term:
        cond = cond.strip()
        try:
            if '<=' in cond:
                fname, thr = cond.split('<=', 1); direction = 0
            elif '>=' in cond:
                fname, thr = cond.split('>=', 1); direction = 1
            elif '==' in cond:
                fname, thr = cond.split('==', 1)
                val = float(thr.strip()); direction = 0 if val < 0.5 else 1
            elif '<' in cond:
                fname, thr = cond.split('<',  1); direction = 0
            elif '>' in cond:
                fname, thr = cond.split('>',  1); direction = 1
            elif '=' in cond:
                fname, thr = cond.split('=',  1)
                val = float(thr.strip()); direction = 0 if val < 0.5 else 1
            else:
                continue
            fname = fname.strip(); ext_thresh = float(thr.strip())
        except:
            continue

        is_ph = abs(ext_thresh - 0.5) < 1e-9
        if is_ph:
            splits = _feat_splits.get(fname)
            if not splits: continue
            v = _CUR_INST_VALS.get(fname)
            thr_pick, lid = (min(splits, key=lambda s: abs(s[0] - v)) if v is not None
                             else splits[len(splits)//2])
            best_lit = lid if direction == 1 else -lid
        else:
            best_lit, best_dist = None, float('inf')
            for lit_id, (bname, bop, bthresh) in _bina_inv.items():
                if bname != fname or lit_id <= 0: continue
                dist = abs(bthresh - ext_thresh)
                if dist < best_dist:
                    best_dist = dist
                    best_lit = lit_id if direction == 1 else -lit_id
            if best_lit is None or best_dist > EPSILON: continue

        if -best_lit in seen: continue
        if best_lit not in seen:
            result.append(best_lit)
            seen.add(best_lit)

    return result

# ── Per-(instance,arm) pipeline ───────────────────────────────────────────────
# run_strategies runs the 3 LLM strategies (v4/v5/v6) for ONE arm's explanation
# (sr_lits, in BT-binvar space) on one instance, and scores them against the
# rectified-tree DNF. run_instance (below) builds the 4 arm terms and calls this
# once per arm, so the output is a 4-arm × 3-strategy grid (B ⊇ A: each arm's own
# formal_score on sr_lits is the Comparison-A baseline).
def run_strategies(pos, arm_name, sr_lits, ctx_text, own_feats):
    global _CUR_INST_VALS
    # Original feature instance (for LLM context)
    instance      = X_test.iloc[pos]
    instance_dict = dict(instance)
    # instance feature values -> placeholder mapper maps 0.5 to nearest split to the value
    _CUR_INST_VALS = dict(instance)

    # Binarized instance (for formal reasoning)
    instance_bina = binarized_test_X[pos]

    # Prediction via BT-faithful rectified DT
    proba         = dt.predict_proba(instance.values.reshape(1, -1))[0]
    classe_predite = int(proba.argmax())

    _MAX_INST_FEATS = 20

    # Formal reasoning on the binarized instance, against the REAL rectified PyXAI tree.
    # SR + DNF are converted from the tree's id space into BT-binvar space (1..nb).
    target  = int(dt_model_pyxai.predict_instance(instance_bina))
    phi     = DNF_BT[1 - target]            # opposite-class DNF, BT-binvar space
    n_feats = nb_binary

    # This arm's explanation term (passed in) and its own formal_score = Comparison-A baseline
    arm_baseline_score = formal_score(phi, sr_lits, n_feats)
    _k_max_cp1   = max(len(sr_lits), 1)
    if SKIP_SWEEPS:
        cp1_sweep, cp1_k100, cp1_score_at_sr = [], None, None
    else:
        cp1_sweep    = cp1_optimal_sweep(phi, n_feats, _k_max_cp1)
        cp1_k100     = next((k for k, s, _ in cp1_sweep if s >= 100.0), None)
        cp1_score_at_sr = next((s for k, s, _ in cp1_sweep if k == _k_max_cp1), None)

    # Per-arm (no cross-context bleed): show instance VALUES + extract allow-list for THIS
    # arm's own features only. own_feats = arm's surfaced features (SR / SHAP / LIME).
    _arm_feats  = sorted(set(own_feats) | ({lit_to_feature(l) for l in sr_lits} - {None}))
    inst_repr   = {f: (round(instance_dict[f], 4) if isinstance(instance_dict[f], float)
                       else instance_dict[f])
                   for f in _arm_feats if f in instance_dict}
    feats_short = _arm_feats

    ex = dataset_config['tests_result']['exemple']

    # Record of this arm's term as readable conditions (for CSV/trace only — NOT shown to the
    # LLM; the LLM sees the leak-free ctx_text instead).
    sr_conds = sr_to_conditions(sr_lits)

    # v4 — context is the arm's leak-free block (thresholds only for pyxai_sr)
    sr_ctx = ctx_text

    prompt_v4 = PROMPT_V4.format(
        obj=dataset_config['objectif_classification'],
        inst=inst_repr, cls=classe_predite, sr=sr_ctx)

    time.sleep(SLEEP_BETWEEN)
    try:
        explanation_v4 = groq_llm(prompt_v4)
    except Exception as e:
        return {'pos': pos, 'arm': arm_name, 'error': str(e)}

    ext_prompt_v4 = EXTRACT.format(
        feats=feats_short,
        ex_expl=str(ex['explication']).strip()[:300],
        ex_rules=str(ex['regles']).strip(),
        expl=explanation_v4)

    time.sleep(SLEEP_BETWEEN)
    try:
        rules_raw_v4 = groq_llm(ext_prompt_v4, max_tokens=2048)
    except Exception as e:
        return {'pos': pos, 'arm': arm_name, 'error': str(e)}

    rules_v4 = parse_rules_from_llm(rules_raw_v4)
    bina_v4  = bina_term_transform_bt(rules_v4)
    score_v4 = formal_score(phi, bina_v4, n_feats)

    # v5 iterative refinement
    v5_scores  = [score_v4]
    v5_rounds  = 0
    cur_expl   = explanation_v4
    cur_bina   = bina_v4
    cur_score  = score_v4

    for rnd in range(1, MAX_ROUNDS + 1):
        if cur_score >= 100.0: break
        # Dialogue signal = this arm's OWN features not yet captured (names only, no
        # thresholds, no W_f score) -> no ground-truth leak, contexts stay parallel.
        missed = missed_features(sr_lits, cur_bina)
        feedback = (str(missed) if missed
                    else "(all context features already captured; refine for clarity)")

        refine_prompt = REFINE.format(
            obj=dataset_config['objectif_classification'],
            inst=inst_repr, cls=classe_predite,
            sr=sr_ctx, feedback=feedback)

        time.sleep(SLEEP_BETWEEN)
        try:
            cur_expl = groq_llm(refine_prompt)
        except:
            break

        ext_refine = EXTRACT.format(
            feats=feats_short,
            ex_expl=str(ex['explication']).strip()[:300],
            ex_rules=str(ex['regles']).strip(),
            expl=cur_expl)

        time.sleep(SLEEP_BETWEEN)
        try:
            raw = groq_llm(ext_refine, max_tokens=2048)
        except:
            break

        rules_r   = parse_rules_from_llm(raw)
        cur_bina  = bina_term_transform_bt(rules_r)
        cur_score = formal_score(phi, cur_bina, n_feats)
        v5_scores.append(cur_score)
        v5_rounds = rnd

    score_v5_final = v5_scores[-1]
    score_v5_best  = max(v5_scores)

    err = decompose_errors(sr_lits, cur_bina)
    score_v5_filtered = formal_score(phi, err['filtered_term'], n_feats)

    # v6 structured
    prompt_v6 = PROMPT_V6.format(
        obj=dataset_config['objectif_classification'],
        inst=inst_repr, cls=classe_predite, sr=sr_ctx)

    time.sleep(SLEEP_BETWEEN)
    try:
        explanation_v6 = groq_llm(prompt_v6)
    except:
        explanation_v6 = None

    if explanation_v6:
        ext_prompt_v6 = EXTRACT.format(
            feats=feats_short,
            ex_expl=str(ex['explication']).strip()[:300],
            ex_rules=str(ex['regles']).strip(),
            expl=explanation_v6)
        time.sleep(SLEEP_BETWEEN)
        try:
            rules_raw_v6 = groq_llm(ext_prompt_v6, max_tokens=2048)
        except:
            rules_raw_v6 = '[]'
        rules_v6 = parse_rules_from_llm(rules_raw_v6)
    else:
        rules_v6 = []

    bina_v6  = bina_term_transform_bt(rules_v6)
    score_v6 = formal_score(phi, bina_v6, n_feats)

    # CP-2 training-based
    eps_sr, n_sr = compute_precision_error(sr_lits,  instance, classe_predite)
    eps_v4, n_v4 = compute_precision_error(bina_v4,  instance, classe_predite)
    eps_v5, n_v5 = compute_precision_error(cur_bina, instance, classe_predite)
    eps_v6, n_v6 = compute_precision_error(bina_v6,  instance, classe_predite)

    # CP-2 oracle (1000 random samples — skippable)
    if SKIP_SWEEPS:
        eps_oracle_sr = n_oracle_sr = pac_sr = None
        eps_oracle_v6 = n_oracle_v6 = pac_v6 = None
    else:
        _oracle_seed = int(pos) * 17 + 3
        eps_oracle_sr, n_oracle_sr, pac_sr = compute_precision_error_oracle(
            sr_lits, classe_predite, m=1000, seed=_oracle_seed)
        eps_oracle_v6, n_oracle_v6, pac_v6 = compute_precision_error_oracle(
            bina_v6, classe_predite, m=1000, seed=_oracle_seed)

    def _div(formal, eps):
        if eps is None: return None
        return round(abs(formal - (1 - eps) * 100), 2)

    div_v4 = _div(score_v4, eps_v4)
    div_v5 = _div(score_v5_final, eps_v5)
    div_v6 = _div(score_v6, eps_v6)
    orthogonality_ok = all(d is None or d <= 15 for d in [div_v4, div_v5, div_v6])

    # CP-3
    if SKIP_SWEEPS:
        cp3_sweep_v5, cp3_k80_v5, cp3_k90_v5, cp3_k100_v5 = [], None, None, None
        cp3_sweep_v6, cp3_k80_v6, cp3_k90_v6, cp3_k100_v6 = [], None, None, None
        cp3_score_at_sr_v5 = cp3_score_at_sr_v6 = None
    else:
        cp3_sweep_v5, cp3_k80_v5, cp3_k90_v5, cp3_k100_v5 = cp3_size_sweep(cur_bina, phi, n_feats)
        cp3_sweep_v6, cp3_k80_v6, cp3_k90_v6, cp3_k100_v6 = cp3_size_sweep(bina_v6,  phi, n_feats)
        sr_size = len(sr_lits)
        cp3_score_at_sr_v5 = next((s for k, s in cp3_sweep_v5 if k == sr_size), None)
        cp3_score_at_sr_v6 = next((s for k, s in cp3_sweep_v6 if k == sr_size), None)

    return {
        'pos': pos, 'arm': arm_name, 'arm_baseline_score': round(arm_baseline_score, 2),
        'target': int(target), 'class_pred': classe_predite,
        'n_rules_v4': len(rules_v4), 'n_rules_v6': len(rules_v6),
        'n_sr_conds': len(sr_conds),
        'rules_v4': rules_v4, 'rules_v6': rules_v6, 'sr_conds': sr_conds,
        'explanation_v4': explanation_v4, 'explanation_v5': cur_expl,
        'explanation_v6': explanation_v6,
        'bina_v4': bina_v4, 'bina_v5': cur_bina, 'bina_v6': bina_v6,
        'lits_v4': len(bina_v4), 'lits_v6': len(bina_v6),
        'score_v4': round(score_v4, 2),
        'score_v5': round(score_v5_final, 2), 'score_v5_best': round(score_v5_best, 2),
        'score_v5_filtered': round(score_v5_filtered, 2),
        'score_v6': round(score_v6, 2),
        'v5_scores': [round(s, 2) for s in v5_scores], 'v5_rounds': v5_rounds,
        'delta_v5': round(score_v5_final - score_v4, 2),
        'delta_v6': round(score_v6 - score_v5_final, 2),
        'n_omissions': err['n_omissions'], 'n_commissions': err['n_commissions'],
        'n_redundant': err['n_redundant'],
        'omission_feats': err['omission_feats'], 'commission_feats': err['commission_feats'],
        'eps_sr': eps_sr,   'n_sr_neighbors': n_sr,
        'eps_v4': eps_v4,   'n_v4_neighbors': n_v4,
        'eps_v5': eps_v5,   'n_v5_neighbors': n_v5,
        'eps_v6': eps_v6,   'n_v6_neighbors': n_v6,
        'div_v4': div_v4, 'div_v5': div_v5, 'div_v6': div_v6,
        'orthogonality_ok': orthogonality_ok,
        'eps_oracle_sr': eps_oracle_sr, 'n_oracle_sr': n_oracle_sr, 'pac_bound_sr': pac_sr,
        'eps_oracle_v6': eps_oracle_v6, 'n_oracle_v6': n_oracle_v6, 'pac_bound_v6': pac_v6,
        'cp1_sweep': cp1_sweep, 'cp1_k100': cp1_k100, 'cp1_score_at_sr': cp1_score_at_sr,
        'cp3_sweep_v5': cp3_sweep_v5, 'cp3_k80_v5': cp3_k80_v5,
        'cp3_k90_v5': cp3_k90_v5,    'cp3_k100_v5': cp3_k100_v5,
        'cp3_score_at_sr_v5': cp3_score_at_sr_v5,
        'cp3_sweep_v6': cp3_sweep_v6, 'cp3_k80_v6': cp3_k80_v6,
        'cp3_k90_v6': cp3_k90_v6,    'cp3_k100_v6': cp3_k100_v6,
        'cp3_score_at_sr_v6': cp3_score_at_sr_v6,
    }

def run_instance(pos):
    """Build the 4 arm explanations for this instance and run the 3 LLM strategies on
    each (separate prompts). k for the SHAP/LIME arms = the pyxai_sr length. Returns a
    list of rows (one per arm)."""
    k_ref = max(len(arm_term('pyxai_sr', pos, 1)), 1)
    rows = []
    for arm in ARMS:
        sr_lits = arm_term(arm, pos, k_ref)
        ctx_text, own_feats = arm_context(arm, pos, sr_lits, k_ref)   # leak-free per arm
        rows.append(run_strategies(pos, arm, sr_lits, ctx_text, own_feats))
    return rows

# ── No-LLM formal-layer check: verify the PyXAI-native wiring before spending budget.
# For a few instances: (1) the SR is a true sufficient reason vs the new DNF (==100);
# (2) the extractor round-trips sr_to_conditions(SR) -> bina_term_transform_bt back to a
# term that still scores high (validates _bina_inv + the BT-binvar literal space). ──
# ── White-box self-test (no LLM): exercises the leak-free context builders + the
# placeholder mapper directly, with assertions. XAI_WHITEBOX=1 python run_comparison_bt.py <ds>
if os.getenv('XAI_WHITEBOX'):
    print('=== XAI_WHITEBOX (no LLM): leak-free context + mapper self-test ===', flush=True)
    _fails = []
    def _ok(name, cond):
        print(f'  [{"PASS" if cond else "FAIL"}] {name}', flush=True)
        if not cond: _fails.append(name)
    _pos = 0
    _kref = max(len(arm_term('pyxai_sr', _pos, 1)), 1)
    _sr   = arm_term('pyxai_sr', _pos, _kref)
    _sh   = arm_term('shap', _pos, _kref)
    _li   = arm_term('lime', _pos, _kref)
    _ctx_sr, _ = arm_context('pyxai_sr', _pos, _sr, _kref)
    _ctx_sh, _ = arm_context('shap',     _pos, _sh, _kref)
    _ctx_li, _ = arm_context('lime',     _pos, _li, _kref)
    import re as _r2
    # all numeric tree thresholds for the SR features (to check SHAP/LIME don't leak them)
    _tree_thrs = {round(t, 3) for (f, sp) in _feat_splits.items() for (t, _l) in sp}
    def _has_tree_thr(txt):
        nums = {round(float(x), 3) for x in _r2.findall(r'-?\d+\.\d+', txt)}
        return bool(nums & _tree_thrs)
    _ok('pyxai_sr context contains tree thresholds (expected)', _has_tree_thr(_ctx_sr))
    _ok('shap context contains NO tree thresholds (leak-free)', not _has_tree_thr(_ctx_sh))
    _ok('shap context is importance%% + direction', '%' in _ctx_sh and 'SHAP' in _ctx_sh)
    _ok('lime context is LIME descriptors (weight=)', 'weight=' in _ctx_li)
    # placeholder mapper: a 0.5 direction-only placeholder maps to the split NEAREST the
    # instance value, not 0.5, not dropped.
    _feat_multi = next((f for f, sp in _feat_splits.items() if len(sp) >= 2), None)
    if _feat_multi:
        _splits = [t for (t, _l) in _feat_splits[_feat_multi]]
        _vmid = (_splits[0] + _splits[-1]) / 2.0
        globals()['_CUR_INST_VALS'] = {_feat_multi: _vmid}
        _lit = bina_term_transform_bt([f'{_feat_multi} > 0.5'])
        _expect_thr = min(_splits, key=lambda s: abs(s - _vmid))
        _got = _bina_inv.get(_lit[0], (None, None, None))[2] if _lit else None
        _ok(f'placeholder maps to instance-nearest split ({_feat_multi})',
            bool(_lit) and abs(_got - _expect_thr) < 1e-9)
        globals()['_CUR_INST_VALS'] = {}
    # real-threshold path: a condition near a real split maps within EPSILON
    if _sr:
        _f0 = _bina_inv[_sr[0]][0]; _t0 = _bina_inv[_sr[0]][2]
        _lit2 = bina_term_transform_bt([f'{_f0} > {_t0}'])
        _ok('real threshold maps (EPSILON path)', bool(_lit2))
    print(f'=== WHITEBOX: {"ALL PASS" if not _fails else "FAILURES: "+str(_fails)} ===', flush=True)
    sys.exit(0 if not _fails else 1)

if os.getenv('XAI_FORMAL_CHECK'):
    random.seed(RANDOM_SEED)
    chk = random.sample(range(len(X_test)), min(5, len(X_test)))
    print(f'=== XAI_FORMAL_CHECK (no LLM): per-instance, arms={ARMS} ===', flush=True)
    for pos in chk:
        inst_bina = binarized_test_X[pos]
        tgt  = int(dt_model_pyxai.predict_instance(inst_bina))
        phi  = DNF_BT[1 - tgt]
        k_ref = max(len(arm_term('pyxai_sr', pos, 1)), 1)
        print(f' pos {pos:3d} (pred={tgt}, k_ref={k_ref}):', flush=True)
        for arm in ARMS:
            t = arm_term(arm, pos, k_ref)
            s = formal_score(phi, t, nb_binary)
            conds = sr_to_conditions(t)
            s_rt = formal_score(phi, bina_term_transform_bt(conds), nb_binary)
            eps, n_nb = compute_precision_error(t, X_test.iloc[pos], tgt)
            print(f'    {arm:10s} |term|={len(t):2d}  baseline_score={s:6.2f}  '
                  f'roundtrip={s_rt:6.2f}  CP2_eps={eps if eps is None else round(eps,3)} (n={n_nb})',
                  flush=True)
    print('Expected: pyxai_sr baseline=100 & roundtrip=100 & eps=0; others lower.', flush=True)
    sys.exit(0)

# ── Main loop ─────────────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
indices = random.sample(range(len(X_test)), min(N_INSTANCES, len(X_test)))
results = []
if SAVE_TRACES:
    _TRACE_DIR = f'research/results/bt_traces/{DATASET_NAME}'
    os.makedirs(_TRACE_DIR, exist_ok=True)

def _vblock(row):
    """Verbose per-arm block: SR/context → LLM explanation → rules → literals → score."""
    arm = row.get('arm', '?')
    print(f'   ┌─ arm={arm}  baseline={row["arm_baseline_score"]:.1f}%', flush=True)
    print(f'   │ SR/context conds : {row.get("sr_conds")}', flush=True)
    for tag, ek, rk, sk in [('v4', 'explanation_v4', 'rules_v4', 'score_v4'),
                            ('v5', 'explanation_v5', None,       'score_v5'),
                            ('v6', 'explanation_v6', 'rules_v6', 'score_v6')]:
        expl = str(row.get(ek) or '').strip().replace('\n', ' ')
        if len(expl) > 240: expl = expl[:240] + '…'
        print(f'   │ [{tag}] {sk.split("_")[1]}={row.get(sk)}%  expl: {expl}', flush=True)
        if rk: print(f'   │      rules: {row.get(rk)}', flush=True)
    print(f'   └─ v5 trajectory={row.get("v5_scores")}  rounds={row.get("v5_rounds")}', flush=True)

for i, pos in enumerate(indices):
    print(f'[{i+1:02d}/{N_INSTANCES}] instance {pos}', flush=True)
    inst_rows = run_instance(pos)
    for row in inst_rows:                    # one row per arm
        results.append(row)
        arm = row.get('arm', '?')
        if 'error' in row:
            print(f'    {arm:10s} ERROR: {row["error"]}')
        else:
            print(f'    {arm:10s} base={row["arm_baseline_score"]:.1f}%  '
                  f'v4={row["score_v4"]:.1f}%  '
                  f'v5={row["score_v5"]:.1f}% (best={row["score_v5_best"]:.1f}%)  '
                  f'v6={row["score_v6"]:.1f}%  rnds={row["v5_rounds"]}')
            if VERBOSE:
                _vblock(row)
    # per-instance JSON trace (all arms, full text chain)
    if SAVE_TRACES:
        ok_rows = [r for r in inst_rows if 'error' not in r]
        trace = {'instance_id': pos, 'dataset': DATASET_NAME, 'llm': GROQ_MODEL,
                 'arms': {r['arm']: r for r in ok_rows}}
        with open(f'{_TRACE_DIR}/inst_{pos}.json', 'w', encoding='utf-8') as _tf:
            _json.dump(trace, _tf, indent=2, default=str)

# ── Summary ───────────────────────────────────────────────────────────────────
ok     = [r for r in results if 'error' not in r]
df_res = pd.DataFrame(ok)

print()
print(f'Dataset: {DATASET_NAME}  |  LLM: {GROQ_MODEL}  |  Model: BT→DT (distillation+rectification)')
print(f'I_P (rectified DT fidelity to BT): {I_P:.2f}%   '
      f'[reliability of the explained surrogate vs the boosted tree]')
print(f'Instances evaluated: {pd.DataFrame(ok)["pos"].nunique() if ok else 0}/{N_INSTANCES}'
      f'   (arm-runs: {len(ok)} across {len(ARMS)} arms)')
print()
if len(ok) == 0:
    n_err = sum(1 for r in results if 'error' in r)
    print(f'No instances succeeded ({n_err} errored — e.g. LLM connectivity). Skipping summary table.')
    errs = {r.get('error', '?') for r in results if 'error' in r}
    print('Errors seen:', '; '.join(list(errs)[:3]))
    sys.exit(1)
# Per-arm × per-strategy LLM fidelity (mean formal score). 'baseline' = the arm's own
# term scored vs the rectified-tree DNF (Comparison A); v4/v5/v6 = LLM reproduction.
print(f'{"arm":<12} {"baseline":>9} {"v4 (SR)":>9} {"v5 final":>9} {"v5 best":>9} {"v6 struct":>10}   n')
print('-' * 78)
for arm in ARMS:
    d = df_res[df_res['arm'] == arm]
    if len(d) == 0: continue
    print(f'{arm:<12} {d.arm_baseline_score.mean():>9.2f} {d.score_v4.mean():>9.2f} '
          f'{d.score_v5.mean():>9.2f} {d.score_v5_best.mean():>9.2f} {d.score_v6.mean():>10.2f}  {len(d):>3d}')
print('-' * 78)
print('baseline = explanation method fidelity (Comparison A); v4/v5/v6 = LLM fidelity per strategy')

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('research/results', exist_ok=True)
csv_path = f'research/results/{DATASET_NAME}_bt_comparison.csv'
df_res['dataset'] = DATASET_NAME; df_res['llm'] = GROQ_MODEL; df_res['model'] = 'BT_distilled'
df_res['I_P_rectified_vs_BT'] = round(I_P, 2)
df_res['rules_v4_str']   = [str(r['rules_v4'])          for r in ok]
df_res['rules_v6_str']   = [str(r['rules_v6'])          for r in ok]
df_res['sr_str']         = [str(r['sr_conds'])          for r in ok]
df_res['v5_trajectory']  = [str(r['v5_scores'])         for r in ok]
df_res['omission_feats'] = [str(r['omission_feats'])    for r in ok]
df_res['commission_feats']= [str(r['commission_feats']) for r in ok]
df_res['cp1_sweep_str']  = [str(r['cp1_sweep'])         for r in ok]
df_res['cp3_sweep_v5_str']= [str(r['cp3_sweep_v5'])     for r in ok]
df_res['cp3_sweep_v6_str']= [str(r['cp3_sweep_v6'])     for r in ok]
df_res.to_csv(csv_path, index=False)
print(f'\nResults saved to {csv_path}')
