import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from unittest.mock import MagicMock
mock_pb = MagicMock(); mock_pb.PBEnc = MagicMock()
sys.modules['pypblib'] = mock_pb; sys.modules['pypblib.pblib'] = mock_pb

import os, ast, re, time, math, warnings, random, signal
sys.path.append('build'); warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import shap
from lime.lime_tabular import LimeTabularExplainer
from anchor import anchor_tabular
import my_tree as mt

load_dotenv('.env')
DATASET_NAME  = sys.argv[1] if len(sys.argv) > 1 else 'cancer'
GROQ_MODEL    = sys.argv[2] if len(sys.argv) > 2 else 'llama-3.3-70b-versatile'
N_INSTANCES   = 15
RANDOM_SEED   = 42
GROQ_BASE_URL = 'https://api.groq.com/openai/v1'
SLEEP_BETWEEN = 5
MAX_ROUNDS    = 3

# ── Key rotation ──────────────────────────────────────────────────────────────
from openai import OpenAI
_api_keys = [k for k in [
    os.getenv('GROQ_API_KEY'),
    os.getenv('GROQ_API_KEY_2'),
    os.getenv('GROQ_API_KEY_3'),
] if k]
_key_idx = 0

def _get_client():
    return OpenAI(api_key=_api_keys[_key_idx % len(_api_keys)], base_url=GROQ_BASE_URL)

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
                next_key_num = (_key_idx % len(_api_keys)) + 1
                if attempt == 0 or _key_idx % len(_api_keys) == 0:
                    print(f'  [rate-limit] switching to key {next_key_num}/{len(_api_keys)}', flush=True)
                wait = None
                m_ms = re.search(r'(\d+)m([\d.]+)s', err)
                m_s  = (re.search(r'try again in ([\d.]+)s', err) or
                        re.search(r'Please try again in ([\d.]+)s', err, re.IGNORECASE) or
                        re.search(r'after ([\d.]+) second', err))
                if m_ms:  wait = int(m_ms.group(1)) * 60 + float(m_ms.group(2)) + 10
                elif m_s: wait = float(m_s.group(1)) + 5
                else:     wait = 60 * (attempt + 1)
                if (attempt + 1) < len(_api_keys): wait = min(wait, 5)
                print(f'  [rate-limit] waiting {wait:.0f}s', flush=True)
                time.sleep(wait)
            elif 'timed out' in err.lower() or 'timeout' in err.lower():
                wait = 15 * (attempt + 1)
                print(f'  [timeout] waiting {wait:.0f}s', flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f'Groq: max retries exceeded')

# ── Dataset ───────────────────────────────────────────────────────────────────
with open('datasets_config.yaml', 'r', encoding='utf-8') as f:
    dataset_config = yaml.safe_load(f).get(DATASET_NAME)
TARGET = dataset_config['colonne_cible']
df = pd.read_csv(dataset_config['chemin_csv'])
df.columns = df.columns.str.lower().str.replace(' ', '_')

counts = df[TARGET].value_counts()
maj, minn = counts.idxmax(), counts.idxmin()
df_min_os = resample(df[df[TARGET] == minn], replace=True,
                     n_samples=len(df[df[TARGET] == maj]), random_state=RANDOM_SEED)
df = pd.concat([df[df[TARGET] == maj], df_min_os])

y = df[TARGET]; X = df.drop(TARGET, axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED)

param_grid = {
    'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2', None], 'class_weight': [None, 'balanced'],
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_SEED),
                    param_grid, scoring='accuracy', cv=5, n_jobs=1, verbose=0)
grid.fit(X_train, y_train)
dt = grid.best_estimator_
print(f'Tree trained. Test acc: {dt.score(X_test, y_test):.4f}')

my_tree = mt.decision_tree()
my_tree.from_DecisionTreeClassifier(dt)
features = df.columns.drop(TARGET)

# ── SHAP explainer ────────────────────────────────────────────────────────────
shap_explainer = shap.TreeExplainer(dt)
shap_values    = shap_explainer.shap_values(X_test)
print(f'bina entries: {len(my_tree.bina)}  |  X_test: {len(X_test)} instances')

# ── LIME explainer ────────────────────────────────────────────────────────────
lime_explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=features.tolist(),
    class_names=[str(c) for c in dt.classes_],
    discretize_continuous=True,
    random_state=RANDOM_SEED
)

# ── Anchors explainer ─────────────────────────────────────────────────────────
anchor_explainer = anchor_tabular.AnchorTabularExplainer(
    class_names=[str(c) for c in dt.classes_],
    feature_names=features.tolist(),
    train_data=X_train.values
)

# ── Threshold lookup ──────────────────────────────────────────────────────────
_feat_thresholds = {}
for (feat_idx, thresh) in my_tree.bina.keys():
    fname = features[feat_idx - 1]
    _feat_thresholds.setdefault(fname, [])
    if thresh not in _feat_thresholds[fname]:
        _feat_thresholds[fname].append(thresh)
for fname in _feat_thresholds:
    _feat_thresholds[fname].sort()

_bina_inv = {}
for (fi, th), (lid, _) in my_tree.bina.items():
    fn = features[fi - 1]
    _bina_inv[ lid] = (fn, '>',  th)
    _bina_inv[-lid] = (fn, '<=', th)

# ── Instance selection ────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
positions = random.sample(range(len(X_test)), min(N_INSTANCES, len(X_test)))

# ── Evaluation helpers ────────────────────────────────────────────────────────
def formal_score(phi, term, n):
    if not term or not phi: return 0.0
    w = 0
    for cl in phi:
        if not any(-c in cl for c in term):
            w += 2 ** (n - len(set(list(cl) + list(term))))
    denom = 2 ** (n - len(term))
    return round((1 - w / denom) * 100, 2) if denom > 0 else 0.0

def bina_term_transform_v2(term, feats, bina, strict=True):
    """Map natural-language conditions to binary literals.
    strict=True  (SHAP/LIME): require distance <= 1.0 to nearest threshold
    strict=False (Anchors):   always use nearest threshold — Anchors approximates
                               the same boundary as the tree, direction is what matters
    """
    EPSILON = 0.15
    def parse(cond):
        cond = cond.strip()
        if '<=' in cond: v, n = cond.split('<=', 1); return (feats.get_loc(v.strip())+1, 0, float(n.strip()))
        elif '>=' in cond: v, n = cond.split('>=', 1); return (feats.get_loc(v.strip())+1, 1, float(n.strip()))
        elif '==' in cond:
            v, n = cond.split('==', 1); val = float(n.strip())
            return (feats.get_loc(v.strip())+1, 0 if val < 0.5 else 1, val)
        elif '<' in cond: v, n = cond.split('<', 1); return (feats.get_loc(v.strip())+1, 0, float(n.strip()))
        elif '>' in cond: v, n = cond.split('>', 1); return (feats.get_loc(v.strip())+1, 1, float(n.strip()))
        elif '=' in cond:
            v, n = cond.split('=', 1); val = float(n.strip())
            return (feats.get_loc(v.strip())+1, 0 if val < 0.5 else 1, val)
        raise ValueError(cond)
    result, seen = [], set()
    for cond in term:
        try: feat_idx, direction, ext_thresh = parse(cond)
        except: continue
        keys = [k for k in bina if k[0] == feat_idx]
        if not keys: continue
        lit = None
        for key in keys:
            _, tree_thresh = key; t = bina[key][0]
            if abs(tree_thresh - 0.5) < 1e-9:
                lit = t if direction == 1 else -t; break
            if abs(ext_thresh - tree_thresh) <= EPSILON:
                lit = t if direction == 1 else -t; break
        if lit is None:
            best = min(keys, key=lambda k: abs(k[1] - ext_thresh))
            if not strict or abs(best[1] - ext_thresh) <= 1.0:
                t = bina[best][0]; lit = t if direction == 1 else -t
            else: continue
        if -lit in seen: continue
        if lit not in seen: result.append(lit); seen.add(lit)
    return result

def cp1_optimal_sweep(phi, n, k_max, exact_threshold=60000):
    from itertools import combinations
    all_lits = list(range(1, n+1)) + list(range(-n, 0))
    results = []
    best_score = 0.0
    for k in range(1, k_max + 1):
        from math import comb
        n_comb = comb(len(all_lits), k)
        if n_comb <= exact_threshold:
            best_k, best_sub = 0.0, []
            for subset in combinations(all_lits, k):
                if any(-l in subset for l in subset): continue
                s = formal_score(phi, list(subset), n)
                if s > best_k: best_k = s; best_sub = list(subset)
        else:
            current = []
            for _ in range(k):
                best_lit, best_s = None, -1
                for l in all_lits:
                    if l in current or -l in current: continue
                    s = formal_score(phi, current + [l], n)
                    if s > best_s: best_s = s; best_lit = l
                if best_lit: current.append(best_lit)
            best_k = formal_score(phi, current, n)
        best_score = max(best_score, best_k)
        results.append((k, best_score, 'exact' if n_comb <= exact_threshold else 'greedy'))
        if best_score >= 100.0: break
    return results

def lit_to_feature(lit):
    if  lit in _bina_inv: return _bina_inv[ lit][0]
    if -lit in _bina_inv: return _bina_inv[-lit][0]
    return None

def decompose_errors(sr_lits, bina_term):
    sr_set = set(sr_lits); term_set = set(bina_term)
    commissions = [l for l in bina_term if -l in sr_set]
    redundant   = [l for l in bina_term if l not in sr_set and -l not in sr_set]
    omissions   = [l for l in sr_lits  if l not in term_set and -l not in term_set]
    filtered    = [l for l in bina_term if -l not in sr_set]
    return {
        'n_omissions': len(omissions), 'n_commissions': len(commissions),
        'n_redundant':  len(redundant), 'filtered_term': filtered,
        'omission_feats':   sorted({lit_to_feature(l) for l in omissions}   - {None}),
        'commission_feats': sorted({lit_to_feature(l) for l in commissions} - {None}),
    }

# ── XAI context builders ──────────────────────────────────────────────────────
def build_shap_ctx(sv, total, top7, thresholds=False):
    lines = ['SHAP feature importance (contribution to this prediction):']
    for fname, pct in top7:
        direction = 'positive' if sv[list(X_test.columns).index(fname)] > 0 else 'negative'
        if thresholds and fname in _feat_thresholds:
            thresh_str = ', '.join(f'{t:.4f}' for t in _feat_thresholds[fname])
            lines.append(f'  {fname}: {pct:.1f}% ({direction}) [tree thresholds: {thresh_str}]')
        else:
            lines.append(f'  {fname}: {pct:.1f}% ({direction})')
    return '\n'.join(lines)

def build_lime_ctx(instance, pred_class):
    try:
        exp = lime_explainer.explain_instance(
            instance.values, dt.predict_proba,
            num_features=min(10, len(features)), top_labels=1)
        lines = ['LIME local explanation (linear approximation around this instance):']
        for feat_desc, weight in exp.as_list():
            direction = 'supports prediction' if weight > 0 else 'opposes prediction'
            lines.append(f'  {feat_desc}: weight={weight:+.4f} ({direction})')
        return '\n'.join(lines)
    except Exception as e:
        return f'LIME: unavailable ({e})'

def build_anchors_ctx(instance, timeout_sec=30):
    """Returns (anchors_str, anchors_conditions_list)."""
    try:
        def _handler(sig, frame): raise TimeoutError('Anchors timeout')
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_sec)
        exp = anchor_explainer.explain_instance(
            instance.values, dt.predict, threshold=0.90)
        signal.alarm(0)
        if not exp.names():
            return 'Anchors: no rule found (prediction too uncertain).', []
        conds = list(exp.names())
        prec  = exp.precision()
        cov   = exp.coverage()
        lines = [f'Anchors rule (precision={prec:.1%}, coverage={cov:.1%}):']
        lines += [f'  IF {c}' for c in conds]
        lines += [f'  THEN prediction = {dt.predict(instance.values.reshape(1,-1))[0]}']
        return '\n'.join(lines), conds
    except Exception as e:
        signal.alarm(0)
        return f'Anchors: unavailable ({e})', []

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPT_BB = (
    "System: You are a machine learning expert explaining a model decision to a non-expert.\n"
    "Model objective: {obj}\n"
    "Instance features: {inst}\n"
    "Predicted class: {cls}\n"
    "{xai_ctx}\n\n"
    "Instructions: Explain why this prediction was made. Be concise and specific. "
    "Reference the feature values and conditions that matter. No special formatting.\n"
    "Question: Explain the reasons behind this prediction."
)

PROMPT_BB_STRUCTURED = (
    "System: You are a machine learning expert explaining a model decision to a non-expert.\n"
    "Model objective: {obj}\n"
    "Instance features: {inst}\n"
    "Predicted class: {cls}\n"
    "{xai_ctx}\n\n"
    "Instructions: Write your explanation as a numbered list where EACH item states "
    "exactly ONE condition (feature name, direction, and value). "
    "Do NOT merge multiple conditions into one sentence. "
    "Cover every condition from the XAI context above.\n"
    "Question: List the conditions explaining this prediction."
)

PROMPT_BB_REFINE = (
    "System: You are a machine learning expert refining an explanation.\n"
    "Model objective: {obj}\n"
    "Instance features: {inst}\n"
    "Predicted class: {cls}\n"
    "{xai_ctx}\n\n"
    "Your previous explanation scored {score:.1f}% formal fidelity (100% = perfect).\n"
    "Important features not yet captured in your explanation: {missing}\n"
    "IMPORTANT: Keep all correct conditions from your previous explanation. "
    "Add specific conditions for the missing features listed above.\n"
    "Question: Provide a refined explanation."
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

# ── Per-instance pipeline ─────────────────────────────────────────────────────
def run_instance(pos):
    instance       = X_test.iloc[pos]
    instance_dict  = dict(instance)
    proba          = dt.predict_proba(instance.values.reshape(1, -1))[0]
    classe_predite = int(proba.argmax())

    target  = my_tree.take_decision(instance=instance)
    phi     = my_tree.to_DNF(target=1 - target, threshold_clauses=False)[0].clauses
    n_feats = len(my_tree.bina)

    # SR — used ONLY for evaluation, never shown to LLM
    sr_lits  = my_tree.find_sufficient_reason(instance=instance, target=target)
    sr_size  = len(sr_lits)

    # CP-1 sweep (formal upper bound — no LLM)
    cp1_sweep    = cp1_optimal_sweep(phi, n_feats, max(sr_size, 1))
    cp1_k100     = next((k for k, s, _ in cp1_sweep if s >= 100.0), None)
    cp1_score_at_sr = next((s for k, s, _ in cp1_sweep if k == sr_size), None)

    # SHAP values for this instance
    if isinstance(shap_values, list):
        sv = shap_values[classe_predite][pos]
    elif shap_values.ndim == 3:
        sv = shap_values[pos, :, classe_predite]
    else:
        sv = shap_values[pos]
    total = np.abs(sv).sum()
    shap_pct = {f: float(abs(sv[j]) / total * 100)
                for j, f in enumerate(X_test.columns)} if total > 0 else {}
    top7 = sorted(shap_pct.items(), key=lambda x: -x[1])[:7]

    # Instance repr (top SHAP features)
    _MAX_INST_FEATS = 20
    top_feats = [f for f, _ in top7]
    inst_short = {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in instance_dict.items()
                  if k in top_feats or len(instance_dict) <= _MAX_INST_FEATS}
    if len(instance_dict) > _MAX_INST_FEATS and len(inst_short) < len(instance_dict):
        inst_short['...'] = f'({len(instance_dict) - len(inst_short)} more features)'
    inst_repr = inst_short

    # Feature list for EXTRACT — SHAP top features (no SR)
    feats_short = sorted({f for f, _ in top7})

    ex = dataset_config['tests_result']['exemple']

    # ── Build XAI contexts ────────────────────────────────────────────────────
    shap_ctx      = build_shap_ctx(sv, total, top7, thresholds=False)
    lime_ctx      = build_lime_ctx(instance, classe_predite)
    anchors_str, anchors_conds = build_anchors_ctx(instance)
    combined_ctx  = shap_ctx + '\n\n' + anchors_str

    # ── v2: SHAP basic ────────────────────────────────────────────────────────
    prompt_v2 = PROMPT_BB.format(
        obj=dataset_config['objectif_classification'],
        inst=inst_repr, cls=classe_predite, xai_ctx=shap_ctx)
    time.sleep(SLEEP_BETWEEN)
    try:   expl_v2 = groq_llm(prompt_v2)
    except Exception as e: return {'pos': pos, 'error': str(e)}

    ext_prompt_v2 = EXTRACT.format(feats=feats_short,
        ex_expl=str(ex['explication']).strip()[:300],
        ex_rules=str(ex['regles']).strip(), expl=expl_v2)
    time.sleep(SLEEP_BETWEEN)
    try:   raw_v2 = groq_llm(ext_prompt_v2, max_tokens=2048)
    except Exception as e: return {'pos': pos, 'error': str(e)}
    rules_v2  = parse_rules_from_llm(raw_v2)
    bina_v2   = bina_term_transform_v2(rules_v2, features, my_tree.bina)
    score_v2  = formal_score(phi, bina_v2, n_feats)

    # ── v4a: LIME context ─────────────────────────────────────────────────────
    prompt_v4a = PROMPT_BB.format(
        obj=dataset_config['objectif_classification'],
        inst=inst_repr, cls=classe_predite, xai_ctx=lime_ctx)
    time.sleep(SLEEP_BETWEEN)
    try:   expl_v4a = groq_llm(prompt_v4a)
    except Exception as e: expl_v4a = expl_v2

    ext_prompt_v4a = EXTRACT.format(feats=feats_short,
        ex_expl=str(ex['explication']).strip()[:300],
        ex_rules=str(ex['regles']).strip(), expl=expl_v4a)
    time.sleep(SLEEP_BETWEEN)
    try:   raw_v4a = groq_llm(ext_prompt_v4a, max_tokens=2048)
    except: raw_v4a = '[]'
    rules_v4a = parse_rules_from_llm(raw_v4a)
    bina_v4a  = bina_term_transform_v2(rules_v4a, features, my_tree.bina)
    score_v4a = formal_score(phi, bina_v4a, n_feats)

    # ── v4b: Anchors context ──────────────────────────────────────────────────
    prompt_v4b = PROMPT_BB.format(
        obj=dataset_config['objectif_classification'],
        inst=inst_repr, cls=classe_predite, xai_ctx=anchors_str)
    time.sleep(SLEEP_BETWEEN)
    try:   expl_v4b = groq_llm(prompt_v4b)
    except Exception as e: expl_v4b = expl_v2

    ext_prompt_v4b = EXTRACT.format(feats=feats_short,
        ex_expl=str(ex['explication']).strip()[:300],
        ex_rules=str(ex['regles']).strip(), expl=expl_v4b)
    time.sleep(SLEEP_BETWEEN)
    try:   raw_v4b = groq_llm(ext_prompt_v4b, max_tokens=2048)
    except: raw_v4b = '[]'
    rules_v4b = parse_rules_from_llm(raw_v4b)
    # strict=False: Anchors uses its own discretization, threshold values may
    # not exactly match tree splits — use nearest threshold regardless of distance
    bina_v4b  = bina_term_transform_v2(rules_v4b, features, my_tree.bina, strict=False)
    score_v4b = formal_score(phi, bina_v4b, n_feats)

    # Also evaluate raw Anchors conditions directly (before LLM)
    bina_anchors_raw = bina_term_transform_v2(
        [c.lower() for c in anchors_conds], features, my_tree.bina, strict=False)
    score_anchors_raw = formal_score(phi, bina_anchors_raw, n_feats)

    # ── v4c: Combined SHAP + Anchors ──────────────────────────────────────────
    prompt_v4c = PROMPT_BB.format(
        obj=dataset_config['objectif_classification'],
        inst=inst_repr, cls=classe_predite, xai_ctx=combined_ctx)
    time.sleep(SLEEP_BETWEEN)
    try:   expl_v4c = groq_llm(prompt_v4c)
    except Exception as e: expl_v4c = expl_v2

    ext_prompt_v4c = EXTRACT.format(feats=feats_short,
        ex_expl=str(ex['explication']).strip()[:300],
        ex_rules=str(ex['regles']).strip(), expl=expl_v4c)
    time.sleep(SLEEP_BETWEEN)
    try:   raw_v4c = groq_llm(ext_prompt_v4c, max_tokens=2048)
    except: raw_v4c = '[]'
    rules_v4c = parse_rules_from_llm(raw_v4c)
    # strict=False: combined context includes Anchors conditions
    bina_v4c  = bina_term_transform_v2(rules_v4c, features, my_tree.bina, strict=False)
    score_v4c = formal_score(phi, bina_v4c, n_feats)

    # ── v5: Structured format (best BB context = Anchors or Combined) ─────────
    best_ctx = combined_ctx  # use combined as it has most information
    prompt_v5 = PROMPT_BB_STRUCTURED.format(
        obj=dataset_config['objectif_classification'],
        inst=inst_repr, cls=classe_predite, xai_ctx=best_ctx)
    time.sleep(SLEEP_BETWEEN)
    try:   expl_v5 = groq_llm(prompt_v5)
    except Exception as e: expl_v5 = expl_v4c

    ext_prompt_v5 = EXTRACT.format(feats=feats_short,
        ex_expl=str(ex['explication']).strip()[:300],
        ex_rules=str(ex['regles']).strip(), expl=expl_v5)
    time.sleep(SLEEP_BETWEEN)
    try:   raw_v5 = groq_llm(ext_prompt_v5, max_tokens=2048)
    except: raw_v5 = '[]'
    rules_v5 = parse_rules_from_llm(raw_v5)
    bina_v5  = bina_term_transform_v2(rules_v5, features, my_tree.bina)
    score_v5 = formal_score(phi, bina_v5, n_feats)

    # ── v6: Refinement with SHAP feedback (no SR) ─────────────────────────────
    # Feedback: which high-importance SHAP features weren't captured
    v6_scores  = [score_v4c]
    cur_expl   = expl_v4c
    cur_bina   = bina_v4c
    cur_score  = score_v4c
    v6_rounds  = 0

    for rnd in range(1, MAX_ROUNDS + 1):
        if cur_score >= 100.0: break
        # Find high-importance features not mentioned
        captured_feats = {lit_to_feature(l) for l in cur_bina} - {None}
        missing_shap = [f for f, pct in top7
                        if pct > 8.0 and f not in captured_feats]
        if not missing_shap:
            missing_shap = [f for f, pct in top7[:3] if f not in captured_feats]
        if not missing_shap: break

        refine_prompt = PROMPT_BB_REFINE.format(
            obj=dataset_config['objectif_classification'],
            inst=inst_repr, cls=classe_predite,
            xai_ctx=combined_ctx, score=cur_score,
            missing=missing_shap)
        time.sleep(SLEEP_BETWEEN)
        try:   cur_expl = groq_llm(refine_prompt)
        except: break

        ext_refine = EXTRACT.format(feats=feats_short,
            ex_expl=str(ex['explication']).strip()[:300],
            ex_rules=str(ex['regles']).strip(), expl=cur_expl)
        time.sleep(SLEEP_BETWEEN)
        try:   raw_r = groq_llm(ext_refine, max_tokens=2048)
        except: break
        rules_r   = parse_rules_from_llm(raw_r)
        cur_bina  = bina_term_transform_v2(rules_r, features, my_tree.bina)
        cur_score = formal_score(phi, cur_bina, n_feats)
        v6_scores.append(cur_score)
        v6_rounds = rnd

    score_v6_final = v6_scores[-1]
    score_v6_best  = max(v6_scores)

    # ── Error decomposition (against SR ground truth) ─────────────────────────
    err = decompose_errors(sr_lits, cur_bina)

    # ── CP-1 summary string ───────────────────────────────────────────────────
    cp1_str = f'opt_k100={cp1_k100}'

    # ── Print progress ────────────────────────────────────────────────────────
    anchors_ok = 'ok' if anchors_conds else 'no-anchor'
    print(
        f'  [{pos:>3}] '
        f'v2={score_v2:.1f}%  '
        f'v4a(lime)={score_v4a:.1f}%  '
        f'v4b(anchor)={score_v4b:.1f}%  '
        f'v4c(comb)={score_v4c:.1f}%  '
        f'v5(struct)={score_v5:.1f}%  '
        f'v6(refine)={score_v6_final:.1f}%(best={score_v6_best:.1f}%)  '
        f'SR={sr_size}  anchor={anchors_ok}  '
        f'err=[om={err["n_omissions"]} cm={err["n_commissions"]}]  '
        f'cp1=[{cp1_str}]',
        flush=True
    )

    return {
        'dataset': DATASET_NAME, 'llm': GROQ_MODEL, 'pos': pos,
        'target': target, 'class_pred': classe_predite,
        'n_sr_conds': sr_size, 'anchor_ok': int(bool(anchors_conds)),
        'score_anchors_raw': score_anchors_raw,
        'n_rules_v2': len(rules_v2),
        'n_rules_v4a': len(rules_v4a), 'n_rules_v4b': len(rules_v4b),
        'n_rules_v4c': len(rules_v4c), 'n_rules_v5': len(rules_v5),
        'score_v2': score_v2,
        'score_v4a': score_v4a, 'score_v4b': score_v4b,
        'score_v4c': score_v4c, 'score_v5': score_v5,
        'score_v6': score_v6_final, 'score_v6_best': score_v6_best,
        'v6_rounds': v6_rounds,
        'n_omissions': err['n_omissions'], 'n_commissions': err['n_commissions'],
        'n_redundant': err['n_redundant'],
        'cp1_k100': cp1_k100, 'cp1_score_at_sr': cp1_score_at_sr,
        'cp1_sweep_str': str([(k, s) for k, s, _ in cp1_sweep]),
        'rules_v2_str': str(rules_v2), 'rules_v4b_str': str(rules_v4b),
        'rules_v4c_str': str(rules_v4c), 'rules_v6_str': str(cur_bina),
        'anchors_conds_str': str(anchors_conds), 'sr_str': str([
            f'{_bina_inv[l][0]} {_bina_inv[l][1]} {_bina_inv[l][2]}'
            for l in sr_lits if l in _bina_inv
        ]),
        'omission_feats': str(err['omission_feats']),
        'commission_feats': str(err['commission_feats']),
        'v6_trajectory': str(v6_scores),
    }

# ── Main loop ─────────────────────────────────────────────────────────────────
print(f'\nRunning {len(positions)} instances on {DATASET_NAME} with {GROQ_MODEL}')
print(f'Propositions: BB-1(SHAP) | BB-2(LIME) | BB-3(Anchors) | '
      f'BB-4(SHAP+Anchors) | BB-5(structured) | BB-6(refinement)')
print()

results = []
for i, pos in enumerate(positions):
    print(f'[{i+1:02d}/{len(positions)}] instance {pos} ...', flush=True)
    r = run_instance(pos)
    if 'error' in r:
        print(f'  ERROR: {r["error"]}', flush=True)
    else:
        results.append(r)

# ── Save CSV ──────────────────────────────────────────────────────────────────
os.makedirs('research/results', exist_ok=True)
out_path = f'research/results/{DATASET_NAME}_bb_comparison.csv'
df_out = pd.DataFrame(results)
df_out.to_csv(out_path, index=False)
print(f'\nResults saved to {out_path}')

# ── Summary ───────────────────────────────────────────────────────────────────
if results:
    n = len(results)
    print(f'\n{"="*65}')
    print(f'  SUMMARY — {DATASET_NAME.upper()} ({GROQ_MODEL}, n={n})')
    print(f'{"="*65}')
    for col, label in [
        ('score_v2',      'Prop BB-1  SHAP basic         '),
        ('score_v4a',     'Prop BB-2  LIME               '),
        ('score_v4b',     'Prop BB-3  Anchors            '),
        ('score_v4c',     'Prop BB-4  SHAP + Anchors     '),
        ('score_v5',      'Prop BB-5  Structured format  '),
        ('score_v6',      'Prop BB-6  Refinement (final) '),
        ('score_v6_best', 'Prop BB-6  Refinement (best)  '),
    ]:
        vals = [r[col] for r in results if col in r]
        if vals:
            mean  = sum(vals) / len(vals)
            p100  = sum(1 for v in vals if v == 100)
            p0    = sum(1 for v in vals if v == 0)
            print(f'  {label}  mean={mean:6.1f}%  100%={p100}/{n}  0%={p0}/{n}')
    print()
    anchor_ok = sum(1 for r in results if r.get('anchor_ok', 0))
    print(f'  Anchors found      : {anchor_ok}/{n}')
    print(f'  Zero commissions   : {sum(1 for r in results if r["n_commissions"]==0)}/{n}')
    cp1_vals = [r['cp1_k100'] for r in results if r.get('cp1_k100')]
    if cp1_vals:
        print(f'  CP-1 k100 mean     : {sum(cp1_vals)/len(cp1_vals):.2f}')
    print(f'{"="*65}')
