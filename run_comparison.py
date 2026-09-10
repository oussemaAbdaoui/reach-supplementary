import sys, io
# Force UTF-8 stdout so Unicode in print() works on Windows cp1252 terminals
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from unittest.mock import MagicMock
mock_pb = MagicMock(); mock_pb.PBEnc = MagicMock()
sys.modules['pypblib'] = mock_pb; sys.modules['pypblib.pblib'] = mock_pb

import os, ast, time, warnings, random, re
sys.path.append('build'); warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import shap, my_tree as mt

load_dotenv('.env')
DATASET_NAME  = sys.argv[1] if len(sys.argv) > 1 else 'vote_house_84'
N_INSTANCES   = 15
RANDOM_SEED   = 42
GROQ_MODEL    = sys.argv[2] if len(sys.argv) > 2 else 'llama-3.3-70b-versatile'
GROQ_BASE_URL = 'https://api.groq.com/openai/v1'
SLEEP_BETWEEN = 5

# ── Key rotation: cycle through all available keys on rate-limit ───────────────
import re as _re
from openai import OpenAI

_api_keys = [k for k in [
    os.getenv('GROQ_API_KEY'),
    os.getenv('GROQ_API_KEY_2'),
    os.getenv('GROQ_API_KEY_3'),
] if k]
_key_idx  = 0

def _get_client():
    return OpenAI(api_key=_api_keys[_key_idx % len(_api_keys)], base_url=GROQ_BASE_URL)

print(f'Loaded {len(_api_keys)} API key(s)', flush=True)

def parse_rules_from_llm(raw):
    """Robustly extract a Python list from LLM output.
    Handles: thinking models (<think>...</think>), markdown code blocks,
    extra prose around the list."""
    # Strip thinking blocks (qwen3, deepseek-r1, etc.)
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    # Strip markdown code fences
    text = re.sub(r'```(?:python)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL).strip()
    # Try direct parse first
    try:
        result = ast.literal_eval(text)
        if isinstance(result, list):
            return [str(r).lower().strip() for r in result]
    except:
        pass
    # Regex fallback: find the first [...] block
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
    """Call Groq with automatic retry + key rotation on rate-limit."""
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
                # Rotate to next key immediately
                _key_idx += 1
                next_key_num = (_key_idx % len(_api_keys)) + 1
                if attempt == 0 or _key_idx % len(_api_keys) == 0:
                    print(f'  [rate-limit] switching to key {next_key_num}/{len(_api_keys)}  err: {err[50:200]}', flush=True)
                # Parse wait time: "13m20.928s", "45.3s", "after X seconds"
                wait = None
                m_ms = _re.search(r'(\d+)m([\d.]+)s', err)
                m_s  = (_re.search(r'try again in ([\d.]+)s', err) or
                        _re.search(r'Please try again in ([\d.]+)s', err, _re.IGNORECASE) or
                        _re.search(r'after ([\d.]+) second', err) or
                        _re.search(r'in ([\d.]+) second', err))
                if m_ms:
                    wait = int(m_ms.group(1)) * 60 + float(m_ms.group(2)) + 10
                elif m_s:
                    wait = float(m_s.group(1)) + 5
                else:
                    wait = 60 * (attempt + 1)
                # If we still have unused keys, only do a short pause before retry
                keys_exhausted = (attempt + 1) >= len(_api_keys)
                if not keys_exhausted:
                    wait = min(wait, 5)
                print(f'  [rate-limit] waiting {wait:.0f}s', flush=True)
                time.sleep(wait)
            elif 'timed out' in err.lower() or 'timeout' in err.lower():
                wait = 15 * (attempt + 1)
                print(f'  [timeout] waiting {wait:.0f}s', flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f'Groq: max retries ({max_retries}) exceeded')

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
shap_explainer = shap.TreeExplainer(dt)
shap_values    = shap_explainer.shap_values(X_test)
print(f'bina entries: {len(my_tree.bina)}  |  X_test: {len(X_test)} instances')

# ── Threshold lookup: feature_name → sorted list of tree split thresholds ─────
_feat_thresholds = {}
for (feat_idx, thresh) in my_tree.bina.keys():
    fname = features[feat_idx - 1]   # feat_idx is 1-indexed
    _feat_thresholds.setdefault(fname, [])
    if thresh not in _feat_thresholds[fname]:
        _feat_thresholds[fname].append(thresh)
for fname in _feat_thresholds:
    _feat_thresholds[fname] = sorted(_feat_thresholds[fname])

# ── Inverse bina lookup: lit_id → (feature_name, operator, threshold) ─────────
_bina_inv = {}
for (feat_idx, thresh), (lit_id, _) in my_tree.bina.items():
    fname = features[feat_idx - 1]
    _bina_inv[ lit_id] = (fname, '>',  thresh)
    _bina_inv[-lit_id] = (fname, '<=', thresh)

def sr_to_conditions(sr_lits):
    """Convert SR literal list to simplified human-readable conditions.
    Keeps only the tightest bound per (feature, direction) pair."""
    raw = [_bina_inv[l] for l in sr_lits if l in _bina_inv]
    # group by (feature, op) and keep tightest threshold
    tightest = {}
    for fname, op, thresh in raw:
        key = (fname, op)
        if key not in tightest:
            tightest[key] = thresh
        else:
            # for '>'  keep highest threshold (tightest lower bound)
            # for '<=' keep lowest  threshold (tightest upper bound)
            if op == '>'  and thresh > tightest[key]: tightest[key] = thresh
            if op == '<=' and thresh < tightest[key]: tightest[key] = thresh
    return [f'{fname} {op} {round(thresh, 6)}' for (fname, op), thresh in tightest.items()]

# ── Transform functions ───────────────────────────────────────────────────────
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
    for x, y, z in parsed:
        for key in [k for k in bina_tree if k[0] == x]:
            k, l = key; t, c = bina_tree[key]
            if y == 1 and z == l:   result.append(t);  break
            elif y == 0 and z == l: result.append(-t); break
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
    """
    Prop CP-1: For each k=1,...,k_max, find the globally optimal k-literal
    conjunction from ALL binary variables that maximizes formal_score(phi, t, n).

    This gives the theoretical upper bound on explanation fidelity at each size k,
    independent of what any LLM extracts. Compare with CP-3 (subset of LLM term).

    Uses exact search for C(2n, k) <= exact_threshold, greedy otherwise.
    Greedy: at each step add the literal giving the largest marginal score gain.

    Returns: list of (k, best_score, method)  where method is 'exact' or 'greedy'
    """
    from itertools import combinations as _combs
    from math import comb as _comb

    all_lits = list(range(1, n_features + 1)) + list(range(-1, -(n_features + 1), -1))
    results  = []

    for k in range(1, k_max + 1):
        n_cands  = 2 * n_features
        n_subsets = _comb(n_cands, k)

        if n_subsets <= exact_threshold:
            best_score = 0.0
            for subset in _combs(all_lits, k):
                # Skip subsets with contradictory pairs (l and -l both present)
                if any(-l in subset for l in subset):
                    continue
                s = formal_score(phi, list(subset), n_features)
                if s > best_score:
                    best_score = s
            results.append((k, round(best_score, 1), 'exact'))
        else:
            # Greedy: build term one literal at a time
            current = []
            for _ in range(k):
                best_gain = -1.0
                best_lit  = None
                for l in all_lits:
                    if l in current or -l in current:
                        continue
                    s = formal_score(phi, current + [l], n_features)
                    if s > best_gain:
                        best_gain = s
                        best_lit  = l
                if best_lit is not None:
                    current.append(best_lit)
            results.append((k, round(formal_score(phi, current, n_features), 1), 'greedy'))

        if results[-1][1] >= 100.0:
            for kk in range(k + 1, k_max + 1):
                results.append((kk, 100.0, 'triv'))
            break

    return results


def cp3_size_sweep(bina_term, phi, n_features):
    """
    Prop CP-3: Size-bounded rule minimization.
    For each k=1,...,|bina_term|, find the k-literal subset with the best formal score.

    Mathematical insight: adding a literal l to term t:
      - If l contradicts clauses in phi  -> removes them from W_f sum -> score IMPROVES
      - If l appears in phi clauses      -> those terms double in W_f sum -> score WORSENS
      - If l is neutral (not in phi)     -> W_f unchanged
    So the optimal subset is not always the full term; smaller subsets can score better.

    Returns:
      sweep    : list of (k, best_score_pct)
      min_k_80 : smallest k with score >= 80%, or None
      min_k_90 : smallest k with score >= 90%, or None
      min_k_100: smallest k with score >= 100%, or None
    """
    from itertools import combinations as _combs
    if not bina_term:
        return [], None, None, None

    n   = len(bina_term)
    sweep = []

    for k in range(1, n + 1):
        best_k_score = 0.0
        for subset in _combs(bina_term, k):
            s = formal_score(phi, list(subset), n_features)
            if s > best_k_score:
                best_k_score = s
        sweep.append((k, round(best_k_score, 1)))
        if best_k_score >= 100.0:
            # Fill remaining k slots with 100
            for kk in range(k + 1, n + 1):
                sweep.append((kk, 100.0))
            break

    min_k_80  = next((k for k, s in sweep if s >= 80.0),  None)
    min_k_90  = next((k for k, s in sweep if s >= 90.0),  None)
    min_k_100 = next((k for k, s in sweep if s >= 100.0), None)
    return sweep, min_k_80, min_k_90, min_k_100


def compute_precision_error(s_literals, x_instance, x_pred_class):
    """
    Prop CP-2: estimate eps_hat_{f,x}(S) from training data.

    eps_hat_{f,x}(S) = |{z in X_train : z satisfies all conditions in S, f(z) != f(x)}|
                  ─────────────────────────────────────────────────────────────────
                          |{z in X_train : z satisfies all conditions in S}|

    This is the empirical precision error - the fraction of training instances
    that share x's conditions on S but receive a different model prediction.
    For S = SR this should be 0 by definition; for extracted terms it measures
    how faithfully the rule captures the model's decision boundary.

    Returns (precision_error, n_neighbors):
      precision_error : float in [0,1], or None if no training instances satisfy S
      n_neighbors     : int - number of training instances in the S-defined region
    """
    if not s_literals:
        return None, 0

    # Map each literal to a readable condition via _bina_inv
    conditions = []
    for lit in s_literals:
        if lit in _bina_inv:
            fname, op, thresh = _bina_inv[lit]
            conditions.append((fname, op, thresh))
    if not conditions:
        return None, 0

    # Build a boolean mask over X_train rows
    mask = np.ones(len(X_train), dtype=bool)
    for fname, op, thresh in conditions:
        if fname not in X_train.columns:
            continue
        col = X_train[fname].values
        if op == '>':
            mask &= col > thresh
        else:          # '<='
            mask &= col <= thresh

    neighbors = X_train.values[mask]
    n_neighbors = int(mask.sum())
    if n_neighbors == 0:
        return None, 0

    neighbor_preds = dt.predict(neighbors)
    n_different    = int((neighbor_preds != x_pred_class).sum())
    return round(n_different / n_neighbors, 4), n_neighbors


def compute_precision_error_oracle(s_literals, x_pred_class, m=1000, seed=42):
    """
    Oracle sampling version of CP-2, following Koriche et al. (2023).

    Instead of restricting to training instances, generates m random instances
    uniformly from the feature space (D = uniform over [min, max] per feature,
    estimated from training data) and queries the decision tree on each.

    eps_hat_oracle(S) = |{z ~ D : z satisfies S, f(z) != f(x)}|
                        ─────────────────────────────────────────
                        |{z ~ D : z satisfies S}|

    This covers the full feature space, not just the training distribution.
    Critically important for high-dimensional datasets (e.g., diabetes n=78)
    where training data covers an infinitesimal fraction of {0,1}^n.

    Also returns a PAC-style confidence bound via Hoeffding's inequality:
      With probability >= 0.95, the true precision error is within
      pac_bound = sqrt(ln(2/delta) / (2 * n_neighbors)) of eps_hat_oracle.
    (Hoeffding form of Theorem 1 in Koriche et al.)

    Returns (eps_oracle, n_neighbors, pac_bound):
      eps_oracle  : float in [0,1], or None if no sampled instance satisfies S
      n_neighbors : int - number of sampled instances in the S-defined region
      pac_bound   : float - 95% confidence half-width (None if n_neighbors == 0)
    """
    import math
    if not s_literals:
        return None, 0, None

    conditions = []
    for lit in s_literals:
        if lit in _bina_inv:
            fname, op, thresh = _bina_inv[lit]
            conditions.append((fname, op, thresh))
    if not conditions:
        return None, 0, None

    rng = np.random.RandomState(seed)
    feat_mins = X_train.min()
    feat_maxs = X_train.max()

    # Sample m random instances uniformly over training feature ranges
    random_df = pd.DataFrame({
        col: rng.uniform(feat_mins[col], feat_maxs[col], m)
        for col in X_train.columns
    })

    # Filter to instances satisfying all conditions in S
    mask = np.ones(m, dtype=bool)
    for fname, op, thresh in conditions:
        if fname not in random_df.columns:
            continue
        col = random_df[fname].values
        if op == '>':
            mask &= col > thresh
        else:   # '<='
            mask &= col <= thresh

    n_neighbors = int(mask.sum())
    if n_neighbors == 0:
        return None, 0, None

    neighbors      = random_df.values[mask]
    neighbor_preds = dt.predict(neighbors)
    n_different    = int((neighbor_preds != x_pred_class).sum())
    eps_oracle     = round(n_different / n_neighbors, 4)

    # Hoeffding 95% confidence bound
    delta     = 0.05
    pac_bound = round(math.sqrt(math.log(2.0 / delta) / (2.0 * n_neighbors)), 4)

    return eps_oracle, n_neighbors, pac_bound

def lit_to_feature(lit):
    """Return feature name for a literal (positive or negative)."""
    if  lit in _bina_inv: return _bina_inv[ lit][0]
    if -lit in _bina_inv: return _bina_inv[-lit][0]
    return None

def missed_features(sr_lits, bina_term):
    """Feature names in SR not represented in the extracted bina_term."""
    term_feats = {lit_to_feature(l) for l in bina_term} - {None}
    sr_feats   = {lit_to_feature(l) for l in sr_lits}   - {None}
    return sorted(sr_feats - term_feats)

def decompose_errors(sr_lits, bina_term):
    """
    Prop 5: decompose extraction errors into omission, commission, redundant.

    - Omission  : SR literal whose feature is absent from bina_term entirely
                  (LLM never mentioned that feature's condition)
    - Commission: bina_term literal whose negation is in SR
                  (LLM stated the wrong direction - e.g. glucose<=127 when SR has glucose>127)
    - Redundant : bina_term literal neither in SR nor contradicting SR
                  (LLM over-specified with an extra, non-SR condition)

    filtered_term removes commission errors; scoring it tests whether commissions
    cause the residual gap or omissions do.
    """
    sr_set   = set(sr_lits)
    term_set = set(bina_term)

    commissions = [l for l in bina_term if -l in sr_set]
    redundant   = [l for l in bina_term if l not in sr_set and -l not in sr_set]
    omissions   = [l for l in sr_lits  if l not in term_set and -l not in term_set]
    filtered    = [l for l in bina_term if -l not in sr_set]   # remove commissions only

    return {
        'n_omissions':     len(omissions),
        'n_commissions':   len(commissions),
        'n_redundant':     len(redundant),
        'omission_feats':  sorted({lit_to_feature(l) for l in omissions} - {None}),
        'commission_feats':sorted({lit_to_feature(l) for l in commissions} - {None}),
        'filtered_term':   filtered,
    }

MAX_ROUNDS = 3   # refinement rounds after the initial explanation

# ── Prompts ───────────────────────────────────────────────────────────────────
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
    "Formal sufficient reason - the minimal set of conditions that guarantees this prediction:\n"
    "{sr}\n\n"
    "Instructions: Explain in plain language why this prediction was made, "
    "referring to the conditions above. Be concise. No special formatting.\n"
    "Question: Explain the reasons behind this prediction."
)

PROMPT_V6 = (
    "System: You are a machine learning expert explaining a model decision.\n"
    "Model objective: {obj}\n"
    "Instance: {inst}\n"
    "Predicted class: {cls}\n"
    "The following conditions formally guarantee this prediction:\n"
    "{sr}\n\n"
    "Instructions: Write EXACTLY one numbered sentence per condition listed above. "
    "Do NOT merge or skip any condition. "
    "Include the specific numeric value from each condition in your sentence.\n"
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
    "You are a machine learning expert refining an explanation based on formal feedback.\n"
    "Model objective: {obj}\n"
    "Instance: {inst}\n"
    "Predicted class: {cls}\n"
    "Formal sufficient reason:\n{sr}\n\n"
    "Your previous explanation scored {score:.1f}% formal fidelity (100% = perfect).\n"
    "{feedback}\n"
    "IMPORTANT: Keep all conditions you stated correctly in your previous explanation. "
    "Only add or clarify conditions for the missing features listed above. "
    "Be concise and specific. No special formatting."
)

# ── Per-instance pipeline ─────────────────────────────────────────────────────
def run_instance(pos):
    instance      = X_test.iloc[pos]
    instance_dict = dict(instance)
    proba         = dt.predict_proba(instance.values.reshape(1, -1))[0]
    classe_predite = int(proba.argmax())

    # Limit instance repr to top-20 features by SHAP to keep prompts short for
    # large datasets (e.g. diabetes with 78 features). SR conditions already
    # contain the exact values that matter; the full dict is unnecessary.
    _MAX_INST_FEATS = 20

    target  = my_tree.take_decision(instance=instance)
    phi     = my_tree.to_DNF(target=1 - target, threshold_clauses=False)[0].clauses
    n_feats = len(my_tree.bina)

    # ── Prop CP-1: globally optimal explanation at each size k ────────────────
    # k_max = SR size (computed below) — we sweep k=1...|SR|.
    # Done here (before LLM calls) since it needs only phi and n_feats.
    _sr_lits_cp1 = my_tree.find_sufficient_reason(instance=instance, target=target)
    _k_max_cp1   = max(len(_sr_lits_cp1), 1)
    cp1_sweep    = cp1_optimal_sweep(phi, n_feats, _k_max_cp1)
    cp1_k100     = next((k for k, s, _ in cp1_sweep if s >= 100.0), None)
    cp1_score_at_sr = next((s for k, s, _ in cp1_sweep if k == _k_max_cp1), None)

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

    # Truncate instance dict for prompt (keep top SHAP features to limit token usage)
    top_feats = [f for f, _ in top7]
    inst_short = {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in instance_dict.items()
                  if k in top_feats or len(instance_dict) <= _MAX_INST_FEATS}
    if len(instance_dict) > _MAX_INST_FEATS and len(inst_short) < len(instance_dict):
        inst_short['...'] = f'({len(instance_dict) - len(inst_short)} more features)'
    inst_repr = inst_short

    # Feature list for EXTRACT prompt - limit to SR + SHAP features to reduce tokens
    # (SR features are the only ones that matter for W_f scoring)
    _sr_lits_pre  = my_tree.find_sufficient_reason(instance=instance, target=target)
    _sr_feats_pre = {lit_to_feature(l) for l in _sr_lits_pre} - {None}
    feats_short   = sorted(_sr_feats_pre | {f for f, _ in top7})

    # v1/v2 SHAP context - no thresholds
    shap_ctx = 'SHAP contributions (%):\n' + '\n'.join(
        f'  {k}: {v:.1f}%' for k, v in top7)

    # v3 SHAP context - with tree thresholds injected
    shap_ctx_v3 = 'SHAP contributions (%) and decision tree split thresholds:\n' + '\n'.join(
        f'  {k}: {v:.1f}%  [splits: {_feat_thresholds.get(k, ["N/A"])}]'
        for k, v in top7)

    # ── v1 / v2 pipeline (shared explanation) ────────────────────────────────
    prompt = PROMPT.format(
        obj=dataset_config['objectif_classification'],
        inst=inst_repr, cls=classe_predite, shap=shap_ctx)

    time.sleep(SLEEP_BETWEEN)
    try:
        explanation = groq_llm(prompt)
    except Exception as e:
        return {'pos': pos, 'error': str(e)}

    ex = dataset_config['tests_result']['exemple']
    ext_prompt = EXTRACT.format(
        feats=feats_short,
        ex_expl=str(ex['explication']).strip()[:300],
        ex_rules=str(ex['regles']).strip(),
        expl=explanation)

    time.sleep(SLEEP_BETWEEN)
    try:
        rules_raw = groq_llm(ext_prompt, max_tokens=2048)
    except Exception as e:
        return {'pos': pos, 'error': str(e)}

    rules = parse_rules_from_llm(rules_raw)

    try:
        bina_v1  = bina_term_transform(rules, features)
        score_v1 = formal_score(phi, bina_v1, n_feats)
    except:
        bina_v1 = []; score_v1 = 0.0

    bina_v2  = bina_term_transform_v2(rules, features)
    score_v2 = formal_score(phi, bina_v2, n_feats)

    # ── v3 pipeline (threshold-explicit prompt) ───────────────────────────────
    prompt_v3 = PROMPT_V3.format(
        obj=dataset_config['objectif_classification'],
        inst=inst_repr, cls=classe_predite, shap=shap_ctx_v3)

    time.sleep(SLEEP_BETWEEN)
    try:
        explanation_v3 = groq_llm(prompt_v3)
    except Exception as e:
        return {'pos': pos, 'error': str(e)}

    ext_prompt_v3 = EXTRACT.format(
        feats=feats_short,
        ex_expl=str(ex['explication']).strip()[:300],
        ex_rules=str(ex['regles']).strip(),
        expl=explanation_v3)

    time.sleep(SLEEP_BETWEEN)
    try:
        rules_raw_v3 = groq_llm(ext_prompt_v3, max_tokens=2048)
    except Exception as e:
        return {'pos': pos, 'error': str(e)}

    rules_v3 = parse_rules_from_llm(rules_raw_v3)

    bina_v3  = bina_term_transform_v2(rules_v3, features)   # use v2 mapper
    score_v3 = formal_score(phi, bina_v3, n_feats)

    # ── v4 pipeline (SR as primary context) ──────────────────────────────────
    sr_lits  = _sr_lits_pre   # already computed above for feats_short
    sr_conds = sr_to_conditions(sr_lits)
    sr_ctx = '\n'.join(f'  {c}' for c in sr_conds)

    prompt_v4 = PROMPT_V4.format(
        obj=dataset_config['objectif_classification'],
        inst=inst_repr, cls=classe_predite, sr=sr_ctx)

    time.sleep(SLEEP_BETWEEN)
    try:
        explanation_v4 = groq_llm(prompt_v4)
    except Exception as e:
        return {'pos': pos, 'error': str(e)}

    ext_prompt_v4 = EXTRACT.format(
        feats=feats_short,
        ex_expl=str(ex['explication']).strip()[:300],
        ex_rules=str(ex['regles']).strip(),
        expl=explanation_v4)

    time.sleep(SLEEP_BETWEEN)
    try:
        rules_raw_v4 = groq_llm(ext_prompt_v4, max_tokens=2048)
    except Exception as e:
        return {'pos': pos, 'error': str(e)}

    rules_v4 = parse_rules_from_llm(rules_raw_v4)

    bina_v4  = bina_term_transform_v2(rules_v4, features)
    score_v4 = formal_score(phi, bina_v4, n_feats)

    # ── v5 pipeline (iterative dialogic refinement, max MAX_ROUNDS) ───────────
    v5_scores  = [score_v4]          # round 0 = same as v4
    v5_rounds  = 0                   # rounds actually used
    cur_expl   = explanation_v4
    cur_bina   = bina_v4
    cur_score  = score_v4

    for rnd in range(1, MAX_ROUNDS + 1):
        if cur_score >= 100.0:
            break                    # already perfect - stop early
        missed = missed_features(sr_lits, cur_bina)
        if not missed:
            feedback = "All key conditions were captured. Refine the explanation for clarity."
        else:
            feedback = (f"The following features had conditions that were not recovered: "
                        f"{missed}. Ensure your explanation mentions specific conditions on each.")

        refine_prompt = REFINE.format(
            obj=dataset_config['objectif_classification'],
            inst=inst_repr, cls=classe_predite,
            sr=sr_ctx, score=cur_score, feedback=feedback)

        time.sleep(SLEEP_BETWEEN)
        try:
            cur_expl = groq_llm(refine_prompt)
        except Exception as e:
            break   # quota or timeout - keep last score

        ext_refine = EXTRACT.format(
            feats=feats_short,
            ex_expl=str(ex['explication']).strip()[:300],
            ex_rules=str(ex['regles']).strip(),
            expl=cur_expl)

        time.sleep(SLEEP_BETWEEN)
        try:
            raw = groq_llm(ext_refine, max_tokens=2048)
        except Exception as e:
            break

        rules_r = parse_rules_from_llm(raw)

        cur_bina  = bina_term_transform_v2(rules_r, features)
        cur_score = formal_score(phi, cur_bina, n_feats)
        v5_scores.append(cur_score)
        v5_rounds = rnd

    score_v5_final = v5_scores[-1]
    score_v5_best  = max(v5_scores)

    # ── Prop 5: commission-filtered scoring (diagnostic, no LLM calls) ────────
    err = decompose_errors(sr_lits, cur_bina)
    score_v5_filtered = formal_score(phi, err['filtered_term'], n_feats)

    # ── v6 pipeline (Prop 6: structured output format) ────────────────────────
    # Force LLM to write one numbered sentence per SR condition → prevents
    # compression of multiple conditions into a single prose sentence.
    prompt_v6 = PROMPT_V6.format(
        obj=dataset_config['objectif_classification'],
        inst=inst_repr, cls=classe_predite, sr=sr_ctx)

    time.sleep(SLEEP_BETWEEN)
    try:
        explanation_v6 = groq_llm(prompt_v6)
    except Exception as e:
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
        except Exception:
            rules_raw_v6 = '[]'

        rules_v6 = parse_rules_from_llm(rules_raw_v6)
    else:
        rules_v6 = []

    bina_v6  = bina_term_transform_v2(rules_v6, features)
    score_v6 = formal_score(phi, bina_v6, n_feats)

    # ── Prop CP-2: sample-based precision error eps_hat_{f,x}(S) ──────────────────
    # Computes eps_hat for the SR (ground truth - should be ~0) and each extracted
    # term. Cross-validates W_f: if formal_score ≈ (1 - eps_hat)×100, the DNF was
    # orthogonal and W_f is reliable. Divergence flags a non-orthogonality issue.
    eps_sr, n_sr   = compute_precision_error(sr_lits,  instance, classe_predite)
    eps_v2, n_v2   = compute_precision_error(bina_v2,  instance, classe_predite)
    eps_v4, n_v4   = compute_precision_error(bina_v4,  instance, classe_predite)
    eps_v5, n_v5   = compute_precision_error(cur_bina, instance, classe_predite)
    eps_v6, n_v6   = compute_precision_error(bina_v6,  instance, classe_predite)

    # CP-2 oracle: Koriche et al. (2023) oracle sampling over full feature space.
    # m=1000 random instances sampled uniformly from training feature ranges.
    # Covers the full space, not just training distribution.
    _oracle_seed = int(pos) * 17 + 3   # deterministic per instance
    eps_oracle_sr,  n_oracle_sr,  pac_sr  = compute_precision_error_oracle(
        sr_lits,  classe_predite, m=1000, seed=_oracle_seed)
    eps_oracle_v6,  n_oracle_v6,  pac_v6  = compute_precision_error_oracle(
        bina_v6,  classe_predite, m=1000, seed=_oracle_seed)

    # Divergence flag: |formal_score - (1-eps_hat)×100| > 15 pts on any metric
    def _div(formal, eps):
        if eps is None: return None
        return round(abs(formal - (1 - eps) * 100), 2)

    div_v2 = _div(score_v2, eps_v2)
    div_v4 = _div(score_v4, eps_v4)
    div_v5 = _div(score_v5_final, eps_v5)
    div_v6 = _div(score_v6, eps_v6)
    # Oracle divergence: compare W_f score vs oracle precision error
    div_oracle_sr = _div(100.0, eps_oracle_sr)   # SR should be ~0 error
    div_oracle_v6 = _div(score_v6, eps_oracle_v6)
    orthogonality_ok = all(
        d is None or d <= 15
        for d in [div_v2, div_v4, div_v5, div_v6]
    )

    # ── Prop CP-3: size-bounded rule minimization ──────────────────────────────
    # Sweep k=1...|bina_v5|: what is the best formal score achievable with only k
    # literals from the extracted term? Reports min k needed to hit 80/90/100%.
    # Also: does the k=|SR| subset match SR quality?
    cp3_sweep_v5, cp3_k80_v5, cp3_k90_v5, cp3_k100_v5 = cp3_size_sweep(
        cur_bina, phi, n_feats)
    cp3_sweep_v6, cp3_k80_v6, cp3_k90_v6, cp3_k100_v6 = cp3_size_sweep(
        bina_v6, phi, n_feats)

    # Score at k = |SR| (how good is the SR-sized best subset of extracted term?)
    sr_size = len(sr_lits)
    cp3_score_at_sr_v5 = next(
        (s for k, s in cp3_sweep_v5 if k == sr_size), None)
    cp3_score_at_sr_v6 = next(
        (s for k, s in cp3_sweep_v6 if k == sr_size), None)

    return {
        'pos':               pos,
        'target':            int(target),
        'class_pred':        classe_predite,
        'n_rules':           len(rules),
        'n_rules_v4':        len(rules_v4),
        'n_rules_v6':        len(rules_v6),
        'n_sr_conds':        len(sr_conds),
        'rules':             rules,
        'rules_v4':          rules_v4,
        'rules_v6':          rules_v6,
        'sr_conds':          sr_conds,
        'lits_v1':           len(bina_v1),
        'lits_v2':           len(bina_v2),
        'lits_v4':           len(bina_v4),
        'lits_v6':           len(bina_v6),
        'score_v1':          round(score_v1,          2),
        'score_v2':          round(score_v2,          2),
        'score_v4':          round(score_v4,          2),
        'score_v5':          round(score_v5_final,    2),
        'score_v5_best':     round(score_v5_best,     2),
        'score_v5_filtered': round(score_v5_filtered, 2),
        'score_v6':          round(score_v6,          2),
        'v5_scores':         [round(s, 2) for s in v5_scores],
        'v5_rounds':         v5_rounds,
        'delta_v2':          round(score_v2          - score_v1,       2),
        'delta_v4':          round(score_v4          - score_v2,       2),
        'delta_v5':          round(score_v5_final    - score_v4,       2),
        'delta_v6':          round(score_v6          - score_v5_final, 2),
        # Prop 5 error decomposition
        'n_omissions':       err['n_omissions'],
        'n_commissions':     err['n_commissions'],
        'n_redundant':       err['n_redundant'],
        'omission_feats':    err['omission_feats'],
        'commission_feats':  err['commission_feats'],
        # Prop CP-2: training-based precision error
        'eps_sr':            eps_sr,   'n_sr_neighbors':  n_sr,
        'eps_v2':            eps_v2,   'n_v2_neighbors':  n_v2,
        'eps_v4':            eps_v4,   'n_v4_neighbors':  n_v4,
        'eps_v5':            eps_v5,   'n_v5_neighbors':  n_v5,
        'eps_v6':            eps_v6,   'n_v6_neighbors':  n_v6,
        'div_v2':            div_v2,   'div_v4':          div_v4,
        'div_v5':            div_v5,   'div_v6':          div_v6,
        'orthogonality_ok':  orthogonality_ok,
        # CP-2 oracle: Koriche et al. (2023) – uniform sampling over feature space
        'eps_oracle_sr':     eps_oracle_sr,  'n_oracle_sr':   n_oracle_sr,
        'eps_oracle_v6':     eps_oracle_v6,  'n_oracle_v6':   n_oracle_v6,
        'pac_bound_sr':      pac_sr,
        'pac_bound_v6':      pac_v6,
        'div_oracle_sr':     div_oracle_sr,
        'div_oracle_v6':     div_oracle_v6,
        # Prop CP-1: globally optimal explanation sweep
        'cp1_sweep':          cp1_sweep,
        'cp1_k100':           cp1_k100,
        'cp1_score_at_sr':    cp1_score_at_sr,
        # Prop CP-3: size-bounded rule minimization
        'cp3_sweep_v5':       cp3_sweep_v5,
        'cp3_k80_v5':         cp3_k80_v5,
        'cp3_k90_v5':         cp3_k90_v5,
        'cp3_k100_v5':        cp3_k100_v5,
        'cp3_score_at_sr_v5': cp3_score_at_sr_v5,
        'cp3_sweep_v6':       cp3_sweep_v6,
        'cp3_k80_v6':         cp3_k80_v6,
        'cp3_k90_v6':         cp3_k90_v6,
        'cp3_k100_v6':        cp3_k100_v6,
        'cp3_score_at_sr_v6': cp3_score_at_sr_v6,
    }

# ── Main loop ─────────────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
indices = random.sample(range(len(X_test)), N_INSTANCES)
results = []
for i, pos in enumerate(indices):
    print(f'[{i+1:02d}/{N_INSTANCES}] instance {pos} ...', end=' ', flush=True)
    row = run_instance(pos)
    results.append(row)
    if 'error' in row:
        print(f'ERROR: {row["error"]}')
    else:
        eps_sr_str = f'{row["eps_sr"]:.3f}' if row["eps_sr"] is not None else 'N/A'
        k80  = row["cp3_k80_v5"]  if row["cp3_k80_v5"]  is not None else '-'
        k100 = row["cp3_k100_v5"] if row["cp3_k100_v5"] is not None else '-'
        cp1k = row["cp1_k100"]    if row["cp1_k100"]    is not None else '-'
        print(f'rules={row["n_rules"]}  '
              f'v2={row["score_v2"]:.1f}%  '
              f'v4={row["score_v4"]:.1f}%  '
              f'v5={row["score_v5"]:.1f}% (best={row["score_v5_best"]:.1f}%)  '
              f'v6={row["score_v6"]:.1f}%  '
              f'eps_sr={eps_sr_str}  '
              f'ok={row["orthogonality_ok"]}  '
              f'err=[om={row["n_omissions"]} cm={row["n_commissions"]}]  '
              f'rnds={row["v5_rounds"]}  '
              f'cp1=[opt_k100={cp1k}]  '
              f'cp3=[k80={k80} k100={k100}/SR={row["n_sr_conds"]}]')

# ── Summary ───────────────────────────────────────────────────────────────────
ok     = [r for r in results if 'error' not in r]
df_res = pd.DataFrame(ok)

print()
print(f'Dataset: {DATASET_NAME}  |  LLM: {GROQ_MODEL}')
print(f'Instances evaluated: {len(ok)}/{N_INSTANCES}')
print()
print(f'{"Metric":<34} {"v2":>8} {"v4 (SR)":>9} {"v5 final":>10} {"v5 best":>9} {"v6 struct":>10}')
print('-' * 85)
print(f'{"Mean formal score (%)":<34} {df_res.score_v2.mean():>8.2f} {df_res.score_v4.mean():>9.2f} {df_res.score_v5.mean():>10.2f} {df_res.score_v5_best.mean():>9.2f} {df_res.score_v6.mean():>10.2f}')
print(f'{"Median formal score (%)":<34} {df_res.score_v2.median():>8.2f} {df_res.score_v4.median():>9.2f} {df_res.score_v5.median():>10.2f} {df_res.score_v5_best.median():>9.2f} {df_res.score_v6.median():>10.2f}')
print(f'{"Score = 0%":<34} {(df_res.score_v2==0).sum():>8} {(df_res.score_v4==0).sum():>9} {(df_res.score_v5==0).sum():>10} {(df_res.score_v5_best==0).sum():>9} {(df_res.score_v6==0).sum():>10}')
print(f'{"Score > 90%":<34} {(df_res.score_v2>90).sum():>8} {(df_res.score_v4>90).sum():>9} {(df_res.score_v5>90).sum():>10} {(df_res.score_v5_best>90).sum():>9} {(df_res.score_v6>90).sum():>10}')
print(f'{"Score = 100%":<34} {(df_res.score_v2==100).sum():>8} {(df_res.score_v4==100).sum():>9} {(df_res.score_v5==100).sum():>10} {(df_res.score_v5_best==100).sum():>9} {(df_res.score_v6==100).sum():>10}')
print('-' * 85)
print(f'{"Mean delta v4-v2 (SR gain)":<34} {df_res.delta_v4.mean():>+38.2f}')
print(f'{"Mean delta v5-v4 (dialogue gain)":<34} {df_res.delta_v5.mean():>+38.2f}')
print(f'{"Mean delta v6-v5 (struct gain)":<34} {df_res.delta_v6.mean():>+38.2f}')
print(f'{"Instances v6 > v5":<34} {(df_res.delta_v6 > 0).sum():>38}')
print(f'{"Instances v6 = v5":<34} {(df_res.delta_v6 == 0).sum():>38}')
print(f'{"Instances v6 < v5":<34} {(df_res.delta_v6 < 0).sum():>38}')

# ── Prop 5 error decomposition summary ────────────────────────────────────────
print()
print('Prop 5 - Error decomposition (on v5-final bina_term):')
print(f'  Mean omissions   per instance: {df_res.n_omissions.mean():.2f}')
print(f'  Mean commissions per instance: {df_res.n_commissions.mean():.2f}')
print(f'  Mean redundant   per instance: {df_res.n_redundant.mean():.2f}')
print(f'  Instances with 0 commissions: {(df_res.n_commissions==0).sum()}/{len(ok)}')
print(f'  Instances with 0 omissions:   {(df_res.n_omissions==0).sum()}/{len(ok)}')

# ── Prop CP-2 summary ─────────────────────────────────────────────────────────
eps_sr_vals  = [r['eps_sr']  for r in ok if r['eps_sr']  is not None]
eps_v4_vals  = [r['eps_v4']  for r in ok if r['eps_v4']  is not None]
eps_v5_vals  = [r['eps_v5']  for r in ok if r['eps_v5']  is not None]
eps_v6_vals  = [r['eps_v6']  for r in ok if r['eps_v6']  is not None]
ok_count     = sum(1 for r in ok if r['orthogonality_ok'])
print()
print('Prop CP-2 - Precision error eps_hat_{f,x}(S) cross-validation:')
print(f'  [Training-based]')
print(f'  SR   eps_hat mean (should be ~0): {sum(eps_sr_vals)/len(eps_sr_vals):.4f}'
      if eps_sr_vals else '  SR   eps_hat: N/A (no training neighbors)')
print(f'  v4   eps_hat mean: {sum(eps_v4_vals)/len(eps_v4_vals):.4f}'  if eps_v4_vals else '  v4   eps_hat: N/A')
print(f'  v5   eps_hat mean: {sum(eps_v5_vals)/len(eps_v5_vals):.4f}'  if eps_v5_vals else '  v5   eps_hat: N/A')
print(f'  v6   eps_hat mean: {sum(eps_v6_vals)/len(eps_v6_vals):.4f}'  if eps_v6_vals else '  v6   eps_hat: N/A')
print(f'  W_f / eps_hat divergence <=15 pts: {ok_count}/{len(ok)} instances (orthogonality OK)')
# Oracle summary (Koriche et al. 2023)
eps_oracle_sr_vals = [r['eps_oracle_sr'] for r in ok if r['eps_oracle_sr'] is not None]
eps_oracle_v6_vals = [r['eps_oracle_v6'] for r in ok if r['eps_oracle_v6'] is not None]
pac_sr_vals        = [r['pac_bound_sr']  for r in ok if r['pac_bound_sr']  is not None]
pac_v6_vals        = [r['pac_bound_v6']  for r in ok if r['pac_bound_v6']  is not None]
print(f'  [Oracle sampling — Koriche et al. (2023), m=1000, D=Uniform(train ranges)]')
print(f'  SR   eps_oracle mean (should be ~0): {sum(eps_oracle_sr_vals)/len(eps_oracle_sr_vals):.4f}  '
      f'(PAC bound ±{sum(pac_sr_vals)/len(pac_sr_vals):.4f})'
      if eps_oracle_sr_vals else '  SR   eps_oracle: N/A')
print(f'  v6   eps_oracle mean: {sum(eps_oracle_v6_vals)/len(eps_oracle_v6_vals):.4f}  '
      f'(PAC bound ±{sum(pac_v6_vals)/len(pac_v6_vals):.4f})'
      if eps_oracle_v6_vals else '  v6   eps_oracle: N/A')

# Correlation: formal_score vs (1 - eps) * 100
import warnings
def _corr(scores, eps_list):
    pairs = [(s, (1-e)*100) for s, e in zip(scores, eps_list) if e is not None]
    if len(pairs) < 2: return float('nan')
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = (sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys)) ** 0.5
    return round(num/den, 4) if den > 0 else float('nan')

print(f'  Pearson r(formal_score_v4, 1-eps_v4): {_corr(df_res.score_v4.tolist(), [r["eps_v4"] for r in ok]):.4f}')
print(f'  Pearson r(formal_score_v6, 1-eps_v6): {_corr(df_res.score_v6.tolist(), [r["eps_v6"] for r in ok]):.4f}')

print()
print(f'{"#":<4} {"SR":>4} {"v2":>7} {"v4":>7} {"v5":>7} {"v6":>7} {"dv6":>6}  '
      f'{"eps_sr":>6} {"eps_v4":>6} {"eps_v6":>6} {"ok":>3}  v5 trajectory')
print('-' * 108)
for r in ok:
    traj   = ' -> '.join(f'{s:.0f}' for s in r['v5_scores'])
    e_sr   = f'{r["eps_sr"]:.3f}'  if r["eps_sr"]  is not None else ' N/A'
    e_v4   = f'{r["eps_v4"]:.3f}'  if r["eps_v4"]  is not None else ' N/A'
    e_v6   = f'{r["eps_v6"]:.3f}'  if r["eps_v6"]  is not None else ' N/A'
    ok_str = 'Y' if r['orthogonality_ok'] else 'N'
    print(f'{r["pos"]:<4} {r["n_sr_conds"]:>4} '
          f'{r["score_v2"]:>7.1f} {r["score_v4"]:>7.1f} '
          f'{r["score_v5"]:>7.1f} {r["score_v6"]:>7.1f} {r["delta_v6"]:>+6.1f}  '
          f'{e_sr:>6} {e_v4:>6} {e_v6:>6} {ok_str:>3}  {traj}')

# ── Prop CP-1 summary ─────────────────────────────────────────────────────────
print()
print('Prop CP-1 - Globally optimal explanation (upper bound at each size k):')
cp1_k100_vals = [r['cp1_k100'] for r in ok if r['cp1_k100'] is not None]
cp1_sat_sr    = [r['cp1_score_at_sr'] for r in ok if r['cp1_score_at_sr'] is not None]
print(f'  Instances where optimal k=1 reaches 100%: {sum(1 for r in ok if r.get("cp1_k100")==1)}/{len(ok)}')
print(f'  Instances where optimal k<=SR reaches 100%: {len(cp1_k100_vals)}/{len(ok)}')
print(f'  Mean optimal k100: {sum(cp1_k100_vals)/len(cp1_k100_vals):.2f}' if cp1_k100_vals else '  No instance reaches 100%')
print(f'  Mean CP-1 score at k=|SR|: {sum(cp1_sat_sr)/len(cp1_sat_sr):.1f}%' if cp1_sat_sr else '  N/A')
print()
print(f'  CP-1 vs CP-3 comparison (gap = optimal - LLM best):')
for r in ok:
    cp3_best = max((s for k, s in (r['cp3_sweep_v5'] or [])), default=0)
    cp1_best = max((s for k, s, _ in (r['cp1_sweep'] or [])), default=0)
    gap = round(cp1_best - cp3_best, 1)
    print(f'    inst {r["pos"]}: CP-1 best={cp1_best:.1f}%  CP-3 best={cp3_best:.1f}%  gap={gap:+.1f}%  CP-1 sweep={[(k,s) for k,s,_ in (r["cp1_sweep"] or [])]}')

# ── Prop CP-3 summary ─────────────────────────────────────────────────────────
print()
print('Prop CP-3 - Size-bounded rule minimization (v5 extracted term):')
sr_sizes    = [r['n_sr_conds']    for r in ok]
k80_vals_v5 = [r['cp3_k80_v5']   for r in ok if r['cp3_k80_v5']  is not None]
k90_vals_v5 = [r['cp3_k90_v5']   for r in ok if r['cp3_k90_v5']  is not None]
k100_vals_v5= [r['cp3_k100_v5']  for r in ok if r['cp3_k100_v5'] is not None]
sat_sr_v5   = [r['cp3_score_at_sr_v5'] for r in ok if r['cp3_score_at_sr_v5'] is not None]
print(f'  Mean SR size:              {sum(sr_sizes)/len(sr_sizes):.2f} conditions')
print(f'  Mean k to reach >= 80%:   {sum(k80_vals_v5)/len(k80_vals_v5):.2f}  ({len(k80_vals_v5)}/{len(ok)} instances reach 80%)' if k80_vals_v5 else '  No instance reaches 80%')
print(f'  Mean k to reach >= 90%:   {sum(k90_vals_v5)/len(k90_vals_v5):.2f}  ({len(k90_vals_v5)}/{len(ok)} instances reach 90%)' if k90_vals_v5 else '  No instance reaches 90%')
print(f'  Mean k to reach = 100%:   {sum(k100_vals_v5)/len(k100_vals_v5):.2f}  ({len(k100_vals_v5)}/{len(ok)} instances reach 100%)' if k100_vals_v5 else '  No instance reaches 100%')
print(f'  Mean score at k=|SR|:      {sum(sat_sr_v5)/len(sat_sr_v5):.1f}%' if sat_sr_v5 else '  N/A')
print()
print(f'{"#":<4} {"SR":>3} {"lits_v5":>7} {"k80":>4} {"k90":>4} {"k100":>5} {"score@SR":>9}  sweep (k: score%)')
print('-' * 80)
for r in ok:
    sweep_str = '  '.join(f'{k}:{s:.0f}' for k, s in (r['cp3_sweep_v5'] or []))
    sat_str   = f'{r["cp3_score_at_sr_v5"]:.1f}%' if r['cp3_score_at_sr_v5'] is not None else 'N/A'
    k80_s  = str(r['cp3_k80_v5'])  if r['cp3_k80_v5']  is not None else '-'
    k90_s  = str(r['cp3_k90_v5'])  if r['cp3_k90_v5']  is not None else '-'
    k100_s = str(r['cp3_k100_v5']) if r['cp3_k100_v5'] is not None else '-'
    print(f'{r["pos"]:<4} {r["n_sr_conds"]:>3} {r["lits_v2"]:>7} '
          f'{k80_s:>4} {k90_s:>4} {k100_s:>5} {sat_str:>9}  [{sweep_str}]')

# ── Save results ──────────────────────────────────────────────────────────────
os.makedirs('research/results', exist_ok=True)
csv_path = f'research/results/{DATASET_NAME}_prop6_comparison.csv'
df_res['dataset']         = DATASET_NAME
df_res['llm']             = GROQ_MODEL
df_res['rules_str']       = [str(r['rules'])           for r in ok]
df_res['rules_v4_str']    = [str(r['rules_v4'])        for r in ok]
df_res['rules_v6_str']    = [str(r['rules_v6'])        for r in ok]
df_res['sr_str']          = [str(r['sr_conds'])        for r in ok]
df_res['v5_trajectory']   = [str(r['v5_scores'])       for r in ok]
df_res['omission_feats']  = [str(r['omission_feats'])  for r in ok]
df_res['commission_feats']  = [str(r['commission_feats'])  for r in ok]
df_res['cp1_sweep_str']      = [str(r['cp1_sweep'])          for r in ok]
df_res['cp3_sweep_v5_str']  = [str(r['cp3_sweep_v5'])      for r in ok]
df_res['cp3_sweep_v6_str']  = [str(r['cp3_sweep_v6'])      for r in ok]
cols = ['dataset', 'llm', 'pos', 'target', 'class_pred',
        'n_rules', 'n_rules_v4', 'n_rules_v6', 'n_sr_conds',
        'lits_v2', 'lits_v4', 'lits_v6',
        'score_v1', 'score_v2', 'score_v4', 'score_v5', 'score_v5_best',
        'score_v5_filtered', 'score_v6',
        'delta_v2', 'delta_v4', 'delta_v5', 'delta_v6',
        'v5_rounds', 'n_omissions', 'n_commissions', 'n_redundant',
        # Prop CP-2 (training-based)
        'eps_sr', 'n_sr_neighbors',
        'eps_v2', 'n_v2_neighbors',
        'eps_v4', 'n_v4_neighbors',
        'eps_v5', 'n_v5_neighbors',
        'eps_v6', 'n_v6_neighbors',
        'div_v2', 'div_v4', 'div_v5', 'div_v6', 'orthogonality_ok',
        # CP-2 oracle (Koriche et al. 2023)
        'eps_oracle_sr', 'n_oracle_sr', 'pac_bound_sr',
        'eps_oracle_v6', 'n_oracle_v6', 'pac_bound_v6',
        'div_oracle_sr', 'div_oracle_v6',
        # Prop CP-1
        'cp1_k100', 'cp1_score_at_sr', 'cp1_sweep_str',
        # Prop CP-3
        'cp3_k80_v5', 'cp3_k90_v5', 'cp3_k100_v5', 'cp3_score_at_sr_v5',
        'cp3_k80_v6', 'cp3_k90_v6', 'cp3_k100_v6', 'cp3_score_at_sr_v6',
        'cp3_sweep_v5_str', 'cp3_sweep_v6_str',
        'rules_str', 'rules_v4_str', 'rules_v6_str', 'sr_str',
        'v5_trajectory', 'omission_feats', 'commission_feats']
df_res[cols].to_csv(csv_path, index=False)
print(f'\nResults saved to {csv_path}')
