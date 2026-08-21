"""
run_comparison_nn.py — J4b of the NN tier (research/PLAN_NN_TIER.md §6).

End-to-end LLM-fidelity experiment on a dense MLP treated as a BLACK BOX.
Mirrors the BT tier's arm x strategy grid (run_comparison_bt.py) with the
NN-tier machinery validated in J1-J3 and calibrated in J4a:

  model  : sklearn MLPClassifier, default params, trained on the K-bins
           binarized {0,1}^d features                     [Koriche §4.1/§2]
  arms   : cop | lmpaxp | anchors | shap | lime   (5 arms, decision 2026-07-02)
  metric : precision_hat (MC, fix-S-sample-complement) [bb_metrics] — W_f is
           dead at this tier; score_vX = precision(extracted term) x 100,
           baseline = precision(arm's own term) x 100; WeakPAXp@T pass flags.
  LLM    : v4 direct / v5 dialogue (<=3 rounds, leak-free feedback = feature
           NAMES only) / v6 structured — frozen from the previous tiers.

Leak-free contexts (per-arm):
  cop     -> the learned rule's conditions with bin intervals (the rule IS
             exact conditions — legitimate, like the SR was)
  lmpaxp  -> conditions + "verified precision >= T"
  anchors -> the anchor rule as produced (its own conditions, its precision)
  shap    -> importance % + direction only, NO intervals
  lime    -> weight + direction only, NO intervals

Usage:
  python run_comparison_nn.py [dataset] [model]      (default cancer, Llama-3.3-70B)
  XAI_NO_LLM=1        -> baselines only (the 100-instance no-LLM table)
  XAI_N_INSTANCES=..  -> default 15 (LLM) / 100 (no-LLM)
  XAI_PROVIDER=groq   -> switch provider (default together)
"""

from __future__ import annotations

import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import ast
import datetime as _dt
import json
import os
import random
import re
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
from sklearn.neural_network import MLPClassifier

from bb_metrics import precision_hat
from cop_explainer import fit_cop_rule
from lmpaxp_explainer import fit_lmpaxp
from run_calibration_nn import load_dataset, binarize

load_dotenv('.env')

DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else 'cancer'
NO_LLM       = bool(os.getenv('XAI_NO_LLM'))
N_INSTANCES  = int(os.getenv('XAI_N_INSTANCES', '100' if NO_LLM else '15'))
RANDOM_SEED  = 42
T            = 0.95
N_EVAL       = 1000
DELTA        = 0.05
# Ceiling for the reasoning-model retry in llm(). Raise if a model needs
# more headroom; it only bounds the retry, it is not the default budget.
MAX_TOKENS_CAP = int(os.getenv('XAI_MAX_TOKENS_CAP', '8192'))
MAX_ROUNDS   = 3
# Anchors sampling batch. Drives its preallocation (batch_size * 10000 rows,
# twice) and therefore its peak memory. See the note at the anchors arm.
ANCHOR_BATCH = int(os.getenv('XAI_ANCHOR_BATCH', '100'))
SAVE_TRACES  = os.getenv('XAI_SAVE_TRACES', '1') == '1'

# ── Persistent console log (tee) — same infra as the BT tier ───────────────────
if SAVE_TRACES:
    os.makedirs('research/results/logs', exist_ok=True)
    # The tag goes in the FILENAME, not just the CSV: two runs launched in the
    # same second (e.g. two models in parallel) otherwise open the same log with
    # 'w' and silently truncate each other's output. The PID keeps even two
    # same-tag runs apart.
    _LOG_TAG = os.getenv('XAI_OUT_TAG', '').strip()
    _LOG_PATH = (f'research/results/logs/{DATASET_NAME}_nn_'
                 f'{(_LOG_TAG + "_") if _LOG_TAG else ""}'
                 f'{_dt.datetime.now().strftime("%Y%m%d_%H%M%S")}_'
                 f'{os.getpid()}.log')
    class _Tee:
        def __init__(self, *s): self.streams = s
        def write(self, t):
            for st in self.streams: st.write(t); st.flush()
        def flush(self):
            for st in self.streams: st.flush()
    sys.stdout = _Tee(sys.stdout, open(_LOG_PATH, 'w', encoding='utf-8'))
    print(f'Full console log -> {_LOG_PATH}', flush=True)

# ── LLM provider (copied from run_comparison_bt.py — Together default) ─────────
PROVIDER = os.getenv('XAI_PROVIDER', 'together').lower()
from openai import OpenAI
if PROVIDER == 'together':
    LLM_BASE_URL = 'https://api.together.xyz/v1'
    LLM_MODEL    = sys.argv[2] if len(sys.argv) > 2 else 'meta-llama/Llama-3.3-70B-Instruct-Turbo'
    _api_keys    = [k for k in [os.getenv('TOGETHER_API_KEY'),
                                os.getenv('TOGETHER_API_KEY_2'),
                                os.getenv('TOGETHER_API_KEY_3')] if k]
    SLEEP_BETWEEN = float(os.getenv('XAI_SLEEP', '0'))
else:
    LLM_BASE_URL = 'https://api.groq.com/openai/v1'
    LLM_MODEL    = sys.argv[2] if len(sys.argv) > 2 else 'llama-3.3-70b-versatile'
    _api_keys    = [k for k in [os.getenv('GROQ_API_KEY'),
                                os.getenv('GROQ_API_KEY_2'),
                                os.getenv('GROQ_API_KEY_3')] if k]
    SLEEP_BETWEEN = float(os.getenv('XAI_SLEEP', '5'))
_key_idx = 0

def _get_client():
    return OpenAI(api_key=_api_keys[_key_idx % len(_api_keys)], base_url=LLM_BASE_URL)

def llm(prompt, max_tokens=1024, max_retries=15):
    """One completion.

    REASONING MODELS: a thinking model spends its token budget on hidden
    reasoning BEFORE emitting any content, so a budget sized for a plain
    chat model comes back with finish_reason='length' and content=None --
    silently, as an empty answer rather than an API error. Measured on
    thinkingmachines/Inkling-Small: ~1100 completion tokens are consumed
    before the first content token, so max_tokens=1024 yields None every
    time. We therefore retry with a larger budget when that pattern is seen,
    rather than letting a null answer propagate into the scores.
    """
    global _key_idx
    budget = max_tokens
    for attempt in range(max_retries):
        try:
            resp = _get_client().chat.completions.create(
                model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}],
                temperature=0.2, max_tokens=budget, timeout=180.0)
            choice = resp.choices[0]
            content = choice.message.content
            if content is None or not content.strip():
                if budget < MAX_TOKENS_CAP:
                    budget = min(budget * 4, MAX_TOKENS_CAP)
                    print(f'  [empty content, finish={choice.finish_reason}] '
                          f'raising max_tokens to {budget}', flush=True)
                    continue
                raise RuntimeError(
                    f'empty content at max_tokens={budget} '
                    f'(finish_reason={choice.finish_reason})')
            return content.strip()
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate_limit' in err.lower():
                _key_idx += 1
                m_s = re.search(r'try again in ([\d.]+)s', err, re.IGNORECASE)
                wait = float(m_s.group(1)) + 5 if m_s else 30 * (attempt + 1)
                if (attempt + 1) < max(len(_api_keys), 1): wait = min(wait, 5)
                print(f'  [rate-limit] waiting {wait:.0f}s', flush=True)
                time.sleep(wait)
            elif 'time' in err.lower() and 'out' in err.lower():
                time.sleep(15 * (attempt + 1))
            elif any(s in err.lower() for s in
                     ('connection', 'econnreset', 'broken pipe', 'ssl',
                      'bad gateway', 'service unavailable', 'overloaded',
                      'remote end closed', '502', '503', '504', '529')):
                # TRANSIENT NETWORK FAILURE. These used to fall through to
                # `raise`, which killed the whole arm on a single dropped
                # socket -- one blip cost 3 of 5 arms on an instance, and the
                # row was written as an error that instance-level resume would
                # never retry. Retry with backoff instead; a genuinely dead
                # endpoint still exits via max_retries.
                wait = min(10 * (attempt + 1), 60)
                print(f'  [network: {err[:60]}] retry in {wait}s', flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError('LLM: max retries exceeded')

def parse_rules_from_llm(raw):
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    text = re.sub(r'```(?:python)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL).strip()
    for cand in (text,):
        try:
            r = ast.literal_eval(cand)
            if isinstance(r, list): return [str(x).lower().strip() for x in r]
        except Exception: pass
    m = re.search(r'\[([^\[\]]*)\]', text, re.DOTALL)
    if m:
        try:
            r = ast.literal_eval('[' + m.group(1) + ']')
            if isinstance(r, list): return [str(x).lower().strip() for x in r]
        except Exception: pass
    return []

# ── Data + black-box MLP ────────────────────────────────────────────────────────
with open('datasets_config.yaml', 'r', encoding='utf-8') as f:
    import yaml
    dataset_config = yaml.safe_load(f)[DATASET_NAME]

X_train, X_test, y_train, y_test = load_dataset(DATASET_NAME)
Xtr_bin, Xte_bin, bin_names_long, kbins = binarize(X_train, X_test)
d = Xtr_bin.shape[1]

# Short, extraction-friendly bin ids: <col>_b<i>  (+ a human legend per bin)
bin_ids, bin_legend = [], {}
for col, edges in zip(X_train.columns, kbins.bin_edges_):
    for b in range(len(edges) - 1):
        bid = f'{col}_b{b}'
        bin_ids.append(bid)
        bin_legend[bid] = f'{col} in [{edges[b]:.4g}, {edges[b+1]:.4g})'
_id_to_idx = {bid: j for j, bid in enumerate(bin_ids)}

# One-hot groups: one consecutive index block per original column. Used to draw
# the COP example oracle from realisable points -- see valid_oracle_samples.
_ONEHOT_GROUPS, _j = [], 0
for _col, _edges in zip(X_train.columns, kbins.bin_edges_):
    _nb = len(_edges) - 1
    _ONEHOT_GROUPS.append(np.arange(_j, _j + _nb))
    _j += _nb


_COP_SWEEP_CACHE = {}


def _cop_best_k(pos, k_ref):
    """k that maximised measured precision for this instance in the stored sweep.

    Reads <dataset>_cop_sweep_V.csv, the cop@k curve fitted on one-hot-valid
    oracle samples. Falls back to k_ref when the instance or the file is
    absent, so a missing sweep degrades to the length-matched arm rather than
    failing the run. The sweep's own K_MAX bounds the search, so k* is the best
    length WITHIN THAT RANGE and not necessarily the global optimum.
    """
    if not _COP_SWEEP_CACHE:
        path = f'research/results/{DATASET_NAME}_cop_sweep_V.csv'
        if not os.path.exists(path):
            print(f'  [cop_best] {path} missing -> falling back to k_ref',
                  flush=True)
            _COP_SWEEP_CACHE['__missing__'] = True
        else:
            s = pd.read_csv(path)
            s = s[(s.method == 'cop@k') & s.precision.notna()]
            for inst, g in s.groupby('instance'):
                row = g.loc[g.precision.idxmax()]
                _COP_SWEEP_CACHE[int(inst)] = (int(row.k_requested),
                                               float(row.precision))
    if pos in _COP_SWEEP_CACHE:
        k_star, p_sel = _COP_SWEEP_CACHE[pos]
        print(f'  [cop_best] inst {pos}: k*={k_star} '
              f'(selection precision {p_sel:.1f}, re-certified below)',
              flush=True)
        return k_star
    return k_ref


def valid_oracle_samples(m, rng):
    """m draws with exactly one active bin per attribute.

    [K §2/§4.1] specifies the example oracle as D = Uniform({0,1}^d), and
    fit_cop_rule implements that verbatim. Here d is a K-bins ONE-HOT encoding,
    so a uniform draw respects one-hot on all attributes with probability
    (3/8)^m -- effectively never, and COP would minimise its loss on points
    that encode no possible instance while being evaluated on realisable ones
    (\\S sampler). Measured over eight cancer instances, that mismatch costs
    8.5 precision points on average and produces the entire below-chance tail
    (worst case 18.9 under uniform, 58.7 under this sampler). We therefore pass
    the oracle samples in rather than letting the default uniform draw stand;
    the solver, k, m, seed and timeout are untouched.
    """
    Z = np.zeros((m, d), dtype=np.int8)
    for g in _ONEHOT_GROUPS:
        Z[np.arange(m), g[rng.integers(0, len(g), size=m)]] = 1
    return Z

mlp = MLPClassifier(random_state=RANDOM_SEED).fit(Xtr_bin.astype(float), y_train)
acc = mlp.score(Xte_bin.astype(float), y_test)
print(f'black-box MLP (sklearn defaults, [K §4.1]) on d={d} binarized features '
      f'| test acc {acc:.4f}', flush=True)

predict = lambda Z: mlp.predict(np.atleast_2d(np.asarray(Z)).astype(float))
predict_proba = lambda Z: mlp.predict_proba(np.atleast_2d(np.asarray(Z)).astype(float))

# ── Model-agnostic explainers (all on {0,1}^d) ─────────────────────────────────
import shap
from anchor import anchor_tabular
from lime.lime_tabular import LimeTabularExplainer

_bg = shap.sample(Xtr_bin.astype(float), 100, random_state=RANDOM_SEED)
_shap_expl = shap.KernelExplainer(predict_proba, _bg)
_lime_expl = LimeTabularExplainer(
    Xtr_bin.astype(float), feature_names=bin_ids,
    class_names=[str(c) for c in mlp.classes_],
    categorical_features=list(range(d)), discretize_continuous=False,
    random_state=RANDOM_SEED)
_anchor_expl = anchor_tabular.AnchorTabularExplainer(
    class_names=[str(c) for c in mlp.classes_], feature_names=bin_ids,
    train_data=Xtr_bin.astype(float),
    categorical_names={j: ['0', '1'] for j in range(d)})

def _shap_vec(x_bin, pred):
    sv = np.asarray(_shap_expl.shap_values(x_bin.astype(float), silent=True,
                                           l1_reg=f'num_features({d})'))
    if sv.ndim == 2 and sv.shape[0] == len(mlp.classes_): return sv[pred].ravel()
    if sv.ndim > 1: return sv[..., pred].ravel()
    return sv.ravel()

# ── Arm terms (S) + leak-free contexts ─────────────────────────────────────────
def conds_of(S, x_bin):
    """Readable conditions of a term: bin id + legend + x's membership."""
    return [f'{bin_ids[j]} = {int(x_bin[j])}  ({bin_legend[bin_ids[j]]}'
            f'{" — holds" if x_bin[j] else " — does NOT hold"})' for j in S]

def build_arm(arm, x_bin, pred, k_ref, seed):
    """Returns (S, context_text, own_feature_ids, meta)."""
    if arm in ('cop', 'cop_k', 'cop_best'):
        # cop   : floating-K Table-1 optimum (loss-optimal size)  [K Table 1]
        # cop_k : loss-minimal rule pinned AT size k_ref — equal budget with
        #         the certified arm, per the [K Fig. 2] size protocol
        _rng = np.random.default_rng(seed)
        _Z = valid_oracle_samples(5000, _rng)
        if arm == 'cop_best':
            # PRECISION-OPTIMAL LENGTH. cop@k is not monotone in k -- a larger
            # k can express any smaller rule, so a dip is the 60 s solver
            # returning best-so-far, not a property of the rule class. Free-k
            # picks by empirical LOSS and cop_k pins to the certified length,
            # so both can sit off the peak. Here k* is the k that maximised
            # measured precision in the stored sweep.
            #
            # The selection estimate is NOT reused as the score. Choosing the
            # max of ~9 estimates each carrying a Hoeffding half-width of 4.3
            # points is a winner's curse worth several points, so the chosen
            # rule is re-fitted and re-certified against a fresh draw here,
            # exactly as fit_lmpaxp re-certifies its final S.
            k_star = _cop_best_k(seed - RANDOM_SEED, k_ref)
        else:
            k_star = k_ref if arm == 'cop_k' else 7
        r = fit_cop_rule(predict, x_bin, k=k_star,
                         m=5000, timeout_s=60, seed=seed,
                         samples=_Z, labels=np.asarray(predict(_Z)).ravel(),
                         exact_k=(arm in ('cop_k', 'cop_best')))
        ctx = ('Rule learned by constraint optimization (agrees with the model '
               'on this instance; PAC guarantee in expectation):\n' +
               ('\n'.join(f'  {c}' for c in conds_of(r.S, x_bin))
                if r.S else '  (empty rule: the loss-optimal monomial has no condition)'))
        return r.S, ctx, [bin_ids[j] for j in r.S], {
            'cop_loss': r.empirical_loss, 'cop_status': r.status,
            'cop_variant': r.variant, 'base_rate_same': r.base_rate_same}
    if arm == 'lmpaxp':
        r = fit_lmpaxp(predict, x_bin, T=T, seed=seed)
        ctx = (f'Probabilistic sufficient explanation (verified precision '
               f'>= {T:.0%} for this instance):\n' +
               '\n'.join(f'  {c}' for c in conds_of(r.S, x_bin)))
        return r.S, ctx, [bin_ids[j] for j in r.S], {
            'lmp_precision_cert': r.precision_cert}
    if arm == 'anchors':
        # MEMORY: anchor_base sets prealloc_size = batch_size * 10000 and then
        # vstacks TWO arrays of that many rows (int64 `data` + float64
        # `raw_data`). At the library default batch_size=100 and d=90 that is
        # ~687 MiB each, ~1.4 GB per instance, which fails on a machine with
        # little free RAM. batch_size only sets how many samples are drawn per
        # KL-LUCB round; the precision threshold T and confidence delta are
        # unchanged, so a smaller batch costs iterations, not guarantees.
        # Default kept at 100 so stored runs stay reproducible; lower it with
        # XAI_ANCHOR_BATCH when memory is tight.
        exp = _anchor_expl.explain_instance(x_bin.astype(float), predict,
                                            threshold=T,
                                            batch_size=ANCHOR_BATCH)
        S = sorted(set(int(f) for f in exp.features()))
        ctx = ('Anchor rule (IF-THEN, estimated precision '
               f'{exp.precision():.2f}):\n  IF ' +
               ' AND '.join(exp.names()) + f' THEN class {pred}')
        return S, ctx, [bin_ids[j] for j in S], {'anchor_prec': exp.precision()}
    if arm == 'shap':
        vec = _shap_vec(x_bin, pred)
        top = np.argsort(-np.abs(vec))[:k_ref].tolist()
        tot = np.abs(vec[top]).sum() or 1.0
        ctx = 'SHAP feature importance (contribution to this prediction):\n' + \
            '\n'.join(f'  {bin_ids[j]}: {abs(vec[j])/tot*100:.1f}% '
                      f'({"supports" if vec[j] > 0 else "opposes"} the prediction)'
                      for j in top)
        return sorted(top), ctx, [bin_ids[j] for j in top], {}
    if arm == 'lime':
        exp = _lime_expl.explain_instance(x_bin.astype(float), predict_proba,
                                          num_features=d, num_samples=1000)
        w = sorted(exp.as_map()[1], key=lambda t: -abs(t[1]))[:k_ref]
        S = sorted(int(f) for f, _ in w)
        ctx = 'LIME local explanation (linear approximation weights):\n' + \
            '\n'.join(f'  {bin_ids[int(f)]}: weight={wt:+.4f} '
                      f'({"supports" if wt > 0 else "opposes"} the prediction)'
                      for f, wt in w)
        return S, ctx, [bin_ids[int(f)] for f, _ in w], {}
    raise ValueError(arm)

# ── Extracted rules -> S (bin-id mapper; polarity checked against x) ───────────
_rule_re = re.compile(r'([a-z0-9_]+_b\d+)\s*(>=|<=|==|=|>|<)\s*([\d.]+)')

def rules_to_S(rules, x_bin):
    """Map extracted 'col_b2 > 0.5' strings to bin indices. A condition whose
    stated polarity contradicts x is DROPPED and counted (the explanation set
    formalism fixes features to x's values [K §2] — a contradicting literal is
    a commission, not a fixable feature)."""
    S, contradictions, unknown = [], 0, 0
    for r in rules:
        m = _rule_re.search(str(r).lower().replace(' ', ''))
        if not m:
            unknown += 1
            continue
        bid, op, val = m.group(1), m.group(2), float(m.group(3))
        j = _id_to_idx.get(bid)
        if j is None:
            unknown += 1
            continue
        want = 1 if (op in ('>', '>=') or (op in ('=', '==') and val >= 0.5)) else 0
        if int(x_bin[j]) != want:
            contradictions += 1
            continue
        if j not in S:
            S.append(j)
    return sorted(S), contradictions, unknown

# ── Prompts (leak-free wording, frozen v4/v5/v6 pattern) ───────────────────────
PROMPT_V4 = (
    "System: You are a machine learning expert explaining a model decision to a non-expert.\n"
    "Model objective: {obj}\n"
    "Instance (binary indicator features; name = value): {inst}\n"
    "Predicted class: {cls}\n"
    "{ctx}\n\n"
    "Instructions: Explain in plain language why this prediction was made, referring ONLY "
    "to the information in the context above. Do NOT invent conditions that are not given. "
    "State each indicator at most once and never in two opposite directions. Be concise. "
    "No special formatting.\n"
    "Write for a reader with no technical background: name each measurement in ordinary "
    "words, and give its numeric range when the context supplies one (for example, \"the "
    "worst texture reading is between 12.0 and 23.3\"). Where the context gives only an "
    "importance or direction, describe how the measurement pushed the prediction rather "
    "than inventing a range, and never write a vague placeholder such as \"a certain "
    "range\". NEVER write the internal codes that end in _b0, _b1, _b2 -- they are "
    "meaningless to the reader. When a condition does not hold, say so explicitly.\n"
    "Question: Explain the reasons behind this prediction."
)

PROMPT_V6 = (
    "System: You are a machine learning expert explaining a model decision.\n"
    "Model objective: {obj}\n"
    "Instance (binary indicator features; name = value): {inst}\n"
    "Predicted class: {cls}\n"
    "{ctx}\n\n"
    "Instructions: Write EXACTLY one numbered sentence per item listed above. Do NOT merge "
    "or skip any item. State each indicator at most once and never in two opposite "
    "directions.\n"
    "Write each sentence for a reader with no technical background. Name the measurement in "
    "ordinary words -- for example \"the worst texture reading\" -- and then:\n"
    "  * if the item above gives a numeric range, state that range and say plainly whether "
    "it is true for this case, e.g. \"The worst texture reading is between 12.0 and 23.3, "
    "which is true here.\";\n"
    "  * if the item gives only an importance or a direction and no range, say how that "
    "measurement pushed the prediction instead. When it supports the prediction write e.g. "
    "\"The worst concavity reading pushed the prediction towards this class.\"; when it "
    "opposes the prediction write e.g. \"The standard error of the area pushed against this "
    "class, but was outweighed.\" An opposing item is still a reason and must be stated -- "
    "never call it irrelevant and never skip it.\n"
    "Do NOT invent a range that was not given, and never write a vague placeholder such as "
    "\"a certain range\" or \"a specific range\". NEVER write the internal codes that end "
    "in _b0, _b1, _b2 -- they are meaningless to the reader. Every sentence must be about "
    "one listed item: do not spend a sentence restating the prediction or summarising. One "
    "item per sentence, and every item must appear.\n"
    "Answer:"
)

REFINE = (
    "System: You are a machine learning expert refining an explanation based on feedback.\n"
    "Model objective: {obj}\n"
    "Instance (binary indicator features; name = value): {inst}\n"
    "Predicted class: {cls}\n"
    "{ctx}\n\n"
    "Indicators from the context above not yet reflected in your explanation: {missing}\n"
    "IMPORTANT: Keep the conditions you already stated correctly and add one for each listed "
    "indicator ONLY as supported by the context above. Never state an indicator in two "
    "opposite directions. Be concise. No special formatting.\n"
    "Keep writing for a reader with no technical background: measurements named in ordinary "
    "words with their numeric ranges, and NEVER the internal codes ending in _b0, _b1, _b2."
)

EXTRACT = (
    "You are an expert extracting logical conditions from ML explanations.\n"
    "Extract every condition as: <indicator> <operator> <value>\n"
    "Allowed indicators, and the measurement range each one stands for:\n{legend}\n"
    "The explanation is written for a lay reader, so it names measurements in words and "
    "ranges in numbers instead of using the indicator codes. Match each condition it states "
    "to the indicator whose measurement and range it describes.\n"
    "Rules: the explanation says the range holds / is true for this case -> "
    "'indicator > 0.5' | it says the range does not hold / is not true -> "
    "'indicator < 0.5' | one condition per list item | no variables outside the list above\n"
    "Example output: ['glucose_b2 > 0.5', 'age_b0 < 0.5']\n"
    "Constraints: output ONLY the Python list, no extra text, no code blocks.\n"
    "Explanation: {expl}\n"
    "Output:"
)

# ── Per-arm strategies ─────────────────────────────────────────────────────────
def score_of(S, x_bin, seed):
    est = precision_hat(predict, x_bin, S, n=N_EVAL, delta=DELTA, seed=seed)
    return est

def run_strategies(pos, arm, x_bin, pred, S_arm, ctx, own_ids, meta):
    inst_repr = {bin_ids[j]: int(x_bin[j]) for j in S_arm} or \
                {bin_ids[j]: int(x_bin[j]) for j in range(min(d, 5))}
    obj = dataset_config['objectif_classification']
    base = score_of(S_arm, x_bin, RANDOM_SEED + 31 * pos)

    # The explanation no longer contains indicator codes (they are unreadable to
    # the lay reader the pipeline is written for), so the extractor is given the
    # code -> measurement-range legend and matches on the range instead.
    legend = '\n'.join(f'  {bid} : {bin_legend[bid]}' for bid in own_ids)

    def explain_extract(prompt):
        time.sleep(SLEEP_BETWEEN)
        expl = llm(prompt)
        time.sleep(SLEEP_BETWEEN)
        raw = llm(EXTRACT.format(legend=legend, expl=expl), max_tokens=2048)
        rules = parse_rules_from_llm(raw)
        S, contr, unk = rules_to_S(rules, x_bin)
        est = score_of(S, x_bin, RANDOM_SEED + 31 * pos)
        return expl, rules, S, contr, unk, est

    # v4 direct
    e4, r4, S4, c4, u4, est4 = explain_extract(
        PROMPT_V4.format(obj=obj, inst=inst_repr, cls=pred, ctx=ctx))

    # v5 dialogue (start from v4; feedback = missing indicator NAMES only)
    cur_expl, cur_S, cur_est = e4, S4, est4
    v5_scores, rounds = [round(est4.precision * 100, 2)], 0
    for rnd in range(1, MAX_ROUNDS + 1):
        # LEAK-FREE FEEDBACK -- do not "improve" this into ranges or polarity.
        # The refine loop may name only the missing indicators. An earlier
        # revision sent bin_legend intervals plus a holds/does-NOT-hold flag,
        # which is safe for the rule arms (their context already states both)
        # but hands the ATTRIBUTION arms exactly the two things their context
        # withholds by design: which interval a feature sits in, and which side
        # of it this instance falls on. That would manufacture the recovery the
        # paper attributes to context shape. Names only, as in the original.
        missing = [bin_ids[j] for j in S_arm if j not in cur_S]
        if not missing and cur_est.precision >= T:
            break
        fb = missing if missing else \
            '(all context indicators captured; refine for clarity)'
        try:
            e, r, S, c, u, est = explain_extract(
                REFINE.format(obj=obj, inst=inst_repr, cls=pred, ctx=ctx, missing=fb))
        except Exception:
            break
        cur_expl, cur_S, cur_est, rounds = e, S, est, rnd
        v5_scores.append(round(est.precision * 100, 2))

    # v6 structured
    e6, r6, S6, c6, u6, est6 = explain_extract(
        PROMPT_V6.format(obj=obj, inst=inst_repr, cls=pred, ctx=ctx))

    def om_cm(S):
        return (len([j for j in S_arm if j not in S]),
                len([j for j in S if j not in S_arm]))
    om4, cm4 = om_cm(S4); om5, cm5 = om_cm(cur_S); om6, cm6 = om_cm(S6)

    return {
        'dataset': DATASET_NAME, 'llm': LLM_MODEL, 'instance': pos, 'arm': arm,
        'pred': pred, 'len_S_arm': len(S_arm),
        'baseline_precision': round(base.precision * 100, 2),
        'baseline_weak_paxp': base.precision >= T,
        'score_v4': round(est4.precision * 100, 2),
        'score_v5': round(cur_est.precision * 100, 2),
        'score_v5_best': max(v5_scores),
        'score_v6': round(est6.precision * 100, 2),
        'weak_paxp_v4': est4.precision >= T,
        'weak_paxp_v5': cur_est.precision >= T,
        'weak_paxp_v6': est6.precision >= T,
        'len_v4': len(S4), 'len_v5': len(cur_S), 'len_v6': len(S6),
        'om_v4': om4, 'cm_v4': cm4, 'om_v5': om5, 'cm_v5': cm5,
        'om_v6': om6, 'cm_v6': cm6,
        'contradictions_v4': c4, 'contradictions_v6': c6,
        'v5_rounds': rounds, 'v5_scores': str(v5_scores),
        'half_width': round(base.half_width, 4),
        'ctx': ctx, 'explanation_v4': e4, 'explanation_v5': cur_expl,
        'explanation_v6': e6, 'rules_v4': str(r4), 'rules_v6': str(r6),
        'S_arm': str(S_arm), 'S_v4': str(S4), 'S_v5': str(cur_S), 'S_v6': str(S6),
        **meta,
    }

# ── Main loop ──────────────────────────────────────────────────────────────────
# XAI_ARMS=cop_k (comma-separated) runs a subset; outputs get an arm suffix so
# partial re-runs can never overwrite the full-grid CSV/traces.
_ARMS_ENV = os.getenv('XAI_ARMS', '')
ARMS = ([a.strip() for a in _ARMS_ENV.split(',') if a.strip()]
        if _ARMS_ENV else ['cop', 'lmpaxp', 'anchors', 'shap', 'lime'])
# XAI_OUT_TAG=<tag> appends to the CSV name and the trace directory. Use it for
# any run that is NOT the canonical full-grid run for this dataset -- a second
# model, a smoke test, a re-run -- so it cannot silently overwrite the stored
# baseline. The default (no tag) keeps the historical filenames.
_OUT_TAG = os.getenv('XAI_OUT_TAG', '').strip()
_OUT_SUFFIX = (('_' + '_'.join(ARMS)) if _ARMS_ENV else '') \
              + (('_' + _OUT_TAG) if _OUT_TAG else '')
random.seed(RANDOM_SEED)
positions = random.sample(range(len(Xte_bin)), min(N_INSTANCES, len(Xte_bin)))
if SAVE_TRACES and not NO_LLM:
    _TRACE_DIR = f'research/results/nn_traces/{DATASET_NAME}{_OUT_SUFFIX}'
    os.makedirs(_TRACE_DIR, exist_ok=True)

_OUT_CSV = (f'research/results/{DATASET_NAME}_nn_comparison'
            f'{("_nollm" if NO_LLM else "")}{_OUT_SUFFIX}.csv')
# Checkpoint + resume. A full 15-instance grid is ~650 LLM calls and tens of
# minutes; without this an interruption loses the whole run and the money spent
# on it. Instances already present in the CSV are carried over and skipped.
CKPT_EVERY = int(os.getenv('XAI_CKPT', '1'))
results, _done = [], set()
if os.path.exists(_OUT_CSV):
    _prev = pd.read_csv(_OUT_CSV)
    _keep = _prev[_prev['instance'].isin(positions)]
    results = _keep.to_dict('records')
    _done = set(int(i) for i in _keep['instance'].unique())
    print(f'resume: {len(_done)} instances already done '
          f'({len(results)} rows) -> {len(positions) - len(_done)} to go',
          flush=True)

for it, pos in enumerate(positions):
    if pos in _done:
        continue
    x = Xte_bin[pos]
    pred = int(predict(x)[0])
    # certified arm first: its size sets the attribution budget (plan §8bis.2)
    lmp = fit_lmpaxp(predict, x, T=T, seed=RANDOM_SEED + pos)
    k_ref = max(len(lmp.S), 1)
    print(f'[{it+1:03d}/{len(positions)}] instance {pos} (pred={pred}, '
          f'k_ref={k_ref})', flush=True)

    inst_rows = []
    for arm in ARMS:
        if arm == 'lmpaxp':
            S_arm = lmp.S
            ctx = (f'Probabilistic sufficient explanation (verified precision '
                   f'>= {T:.0%} for this instance):\n' +
                   '\n'.join(f'  {c}' for c in conds_of(lmp.S, x)))
            own, meta = [bin_ids[j] for j in lmp.S], \
                        {'lmp_precision_cert': lmp.precision_cert}
        else:
            S_arm, ctx, own, meta = build_arm(arm, x, pred, k_ref,
                                              RANDOM_SEED + pos)
        if NO_LLM:
            est = score_of(S_arm, x, RANDOM_SEED + 31 * pos)
            row = {'dataset': DATASET_NAME, 'instance': pos, 'arm': arm,
                   'pred': pred, 'len_S_arm': len(S_arm),
                   'baseline_precision': round(est.precision * 100, 2),
                   'baseline_weak_paxp': est.precision >= T,
                   'half_width': round(est.half_width, 4),
                   'S_arm': str(S_arm), **meta}
        else:
            try:
                row = run_strategies(pos, arm, x, pred, S_arm, ctx, own, meta)
            except Exception as e:
                row = {'dataset': DATASET_NAME, 'instance': pos, 'arm': arm,
                       'error': str(e)[:300]}
        inst_rows.append(row)
        results.append(row)
        if 'error' in row:
            print(f'  [{arm}] ERROR {row["error"]}', flush=True)
        elif NO_LLM:
            print(f'  [{arm}] baseline={row["baseline_precision"]:.1f}% '
                  f'len={row["len_S_arm"]}', flush=True)
        else:
            print(f'  [{arm}] base={row["baseline_precision"]:.1f}% '
                  f'v4={row["score_v4"]:.1f}% v5={row["score_v5"]:.1f}% '
                  f'v6={row["score_v6"]:.1f}% (len {row["len_S_arm"]}, '
                  f'om_v5={row["om_v5"]})', flush=True)
    if SAVE_TRACES and not NO_LLM:
        with open(f'{_TRACE_DIR}/inst_{pos}.json', 'w', encoding='utf-8') as tf:
            json.dump({'instance_id': pos, 'dataset': DATASET_NAME,
                       'llm': LLM_MODEL,
                       'arms': {r['arm']: r for r in inst_rows}},
                      tf, indent=2, default=str)
    if (it + 1) % CKPT_EVERY == 0:
        os.makedirs('research/results', exist_ok=True)
        pd.DataFrame(results).to_csv(_OUT_CSV, index=False)

df = pd.DataFrame(results)
os.makedirs('research/results', exist_ok=True)
out = _OUT_CSV
df.to_csv(out, index=False)
print(f'\nsaved -> {out}')

ok = df[~df.get('error', pd.Series(index=df.index, dtype=object)).notna()] \
    if 'error' in df.columns else df
print('\n=== per-arm means ===')
cols = ['len_S_arm', 'baseline_precision'] + \
       ([] if NO_LLM else ['score_v4', 'score_v5', 'score_v5_best', 'score_v6'])
print(ok.groupby('arm')[cols].mean().round(2).to_string())
if not NO_LLM:
    print('\nWeakPAXp pass rates (baseline / v5):')
    print(ok.groupby('arm')[['baseline_weak_paxp', 'weak_paxp_v5']]
          .mean().round(2).to_string())
