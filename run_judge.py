"""
run_judge.py — the non-formal axis: LLM-as-a-Judge over stored explanations.

WHAT THIS ANSWERS
-----------------
The safeguard most often proposed for LLM-generated explanations is a second
language model that reads the text and rates it. This script runs exactly that
safeguard on the explanations the NN tier already produced, so we can ask
whether it detects guarantee erosion. The prediction under test: the judge's
scores stay FLAT across the v4 -> v6 manipulation that moves measured fidelity
by ~14 points, and correlate near zero with fidelity per arm. A flat curve is
evidence about the verifier, not about human readers.

The judge is a NULL INSTRUMENT. We are not using it to measure explanation
quality, and a high score from it is not a claim that the explanation is good.

NO REGENERATION
---------------
`*_nn_comparison.csv` already stores explanation_v4 / v5 / v6 for all 375
arm-runs (1125 texts). This script only scores them, so the coherence axis
costs a few hundred cheap calls and needs no LLM generation budget.

WHAT THE JUDGE SEES, AND WHAT IT MUST NOT
-----------------------------------------
Sees   : the task objective, the predicted class, the dataset's valid indicator
         names, and the explanation text.
Never  : the classifier (no query access), the XAI context that was fed to the
         generator, the arm's own feature set S_arm, the strategy label, any
         fidelity score, or any other arm's text.

Withholding the context is the point, not a shortcut. A deployed verifier that
reads only the text is what the literature proposes, and the question is what
such a verifier can see. Giving it the valid indicator list keeps the test fair:
hallucination becomes genuinely checkable (invented indicators are detectable
from the list alone), so a flat curve cannot be dismissed as a strawman judge.

Faithfulness is the dimension that matters most. The verifier we replicate
claims to rate it; without access to the classifier it cannot, and showing that
its faithfulness score is uncorrelated with measured fidelity is the result
that retires the LLM-verifier alternative.

FOUR DIMENSIONS (replicating the two-stage verifier baseline)
-------------------------------------------------------------
  faithfulness   0-100  does the explanation reflect the true decision logic
  coherence      0-100  internally consistent, non-contradictory, well-formed
  completeness   0-100  does it account for the prediction fully
  halluc_risk    0-100  HIGHER IS WORSE — invented / unsupported content

All on 0-100 to sit on the same scale as the fidelity axis.

PROTOCOL
--------
  * temperature 0 (the generator ran at 0.2; the judge must be reproducible)
  * strict JSON output, one call per explanation, no batching -- batching lets
    the judge rank texts against each other, which a deployed verifier scoring
    one explanation at a time cannot do
  * the strategy label (v4/v5/v6) and the arm are never revealed, so the judge
    cannot infer the manipulation
  * XAI_JUDGE_PASSES=3 judges EVERY explanation three times; scores are
    averaged across passes. The judge is not reproducible even at temperature 0
    (provider-side nondeterminism carries ~9-15 pts per single judgement), so
    averaging is what makes per-cell estimates usable. The spread across passes
    is also the self-consistency gate.
  * resumable: existing (dataset, instance, arm, strategy, pass) rows are kept
    and skipped, so an interrupted run can be restarted with the same command

USAGE
-----
    python run_judge.py                     # all 5 datasets
    python run_judge.py cancer heart        # a subset
    XAI_PROVIDER=groq python run_judge.py   # provider (default together)
    XAI_JUDGE_MODEL=...  python run_judge.py

The judge model MUST differ from the generator (Llama-3.3-70B): a model scoring
its own output confounds blindness to erosion with self-preference bias. The
script refuses to run if the two match unless XAI_ALLOW_SELF_JUDGE=1.

Run from the project root. Output: research/results/{dataset}_nn_judge_{tag}.csv (tag from XAI_OUT_TAG)
"""

from __future__ import annotations

import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import datetime as _dt
import json
import os
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
import yaml

from run_calibration_nn import load_dataset, binarize

load_dotenv('.env')

RESULTS      = os.path.join('research', 'results')
DATASETS     = ['cancer', 'compas', 'diabetes', 'employee', 'heart']
STRATEGIES   = ['v4', 'v5', 'v6']
RANDOM_SEED  = 42
N_PASSES     = int(os.getenv('XAI_JUDGE_PASSES', '3'))
# Judgements are independent, so the run is embarrassingly parallel. The
# bottleneck is per-call latency (a reasoning judge spends ~9 s/call), not
# tokens, so workers convert almost linearly into wall-clock. Keep it modest:
# free tiers rate-limit, and the 429 handler backs off per worker.
WORKERS      = int(os.getenv('XAI_JUDGE_WORKERS', '4'))
# Checkpoint often. A long run gets interrupted for reasons that have nothing
# to do with this script (session limits, sleep, network), and every unsaved
# judgement is one that has to be paid for and waited on twice.
CKPT_EVERY   = int(os.getenv('XAI_JUDGE_CKPT', '10'))
GENERATOR    = 'llama-3.3-70b'          # fallback only; the real check reads
                                        # the `llm` column of each comparison
                                        # CSV (see same_family / run_dataset)

# XAI_OUT_TAG selects a non-canonical generation run, matching the tag given to
# run_comparison_nn.py. Empty = the original full-grid files.
OUT_TAG      = os.getenv('XAI_OUT_TAG', '').strip()
_TAG_SUFFIX  = ('_' + OUT_TAG) if OUT_TAG else ''

# Model families we can recognise in a model id. Used to refuse self-judging.
_FAMILIES = ('gpt-oss', 'gpt', 'llama', 'qwen', 'deepseek', 'mistral',
             'mixtral', 'gemma', 'claude', 'gemini', 'phi', 'yi', 'command')


def model_family(model_id):
    """First recognised family token in a model id, else the bare id."""
    m = str(model_id).lower()
    for fam in _FAMILIES:
        if fam in m:
            return fam
    return m


def same_family(a, b):
    return model_family(a) == model_family(b)


SAVE_TRACES  = os.getenv('XAI_SAVE_TRACES', '1') == '1'

# ── LLM provider (same block as run_comparison_nn.py) ─────────────────────────
PROVIDER = os.getenv('XAI_PROVIDER', 'together').lower()
from openai import OpenAI
if PROVIDER == 'together':
    LLM_BASE_URL = 'https://api.together.xyz/v1'
    JUDGE_MODEL  = os.getenv('XAI_JUDGE_MODEL', 'openai/gpt-oss-120b')
    _api_keys    = [k for k in [os.getenv('TOGETHER_API_KEY'),
                                os.getenv('TOGETHER_API_KEY_2'),
                                os.getenv('TOGETHER_API_KEY_3')] if k]
    SLEEP_BETWEEN = float(os.getenv('XAI_SLEEP', '0'))
else:
    LLM_BASE_URL = 'https://api.groq.com/openai/v1'
    JUDGE_MODEL  = os.getenv('XAI_JUDGE_MODEL', 'openai/gpt-oss-120b')
    _api_keys    = [k for k in [os.getenv('GROQ_API_KEY'),
                                os.getenv('GROQ_API_KEY_2'),
                                os.getenv('GROQ_API_KEY_3')] if k]
    SLEEP_BETWEEN = float(os.getenv('XAI_SLEEP', '5'))
_key_idx = 0


def _get_client():
    return OpenAI(api_key=_api_keys[_key_idx % len(_api_keys)], base_url=LLM_BASE_URL)


class OutOfCredit(RuntimeError):
    """Provider refused for billing reasons. Retrying cannot help."""


def llm(prompt, max_tokens=1200, max_retries=15):
    # 1200, not a few hundred: gpt-oss-120b spends reasoning tokens before the
    # visible answer and returns an EMPTY string if the budget runs out first.
    # A too-small budget silently zeroes parse_ok instead of erroring.
    global _key_idx
    for attempt in range(max_retries):
        try:
            resp = _get_client().chat.completions.create(
                model=JUDGE_MODEL, messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0, max_tokens=max_tokens, timeout=90.0)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if '402' in err or 'credit' in err.lower():
                # Billing, not transport. Retrying burns time and changes
                # nothing, so surface it as its own type and let the caller
                # save progress before exiting.
                raise OutOfCredit(err) from None
            if '429' in err or 'rate_limit' in err.lower():
                _key_idx += 1
                m_s = re.search(r'try again in ([\d.]+)s', err, re.IGNORECASE)
                wait = float(m_s.group(1)) + 5 if m_s else 30 * (attempt + 1)
                if (attempt + 1) < max(len(_api_keys), 1): wait = min(wait, 5)
                print(f'  [rate-limit] waiting {wait:.0f}s', flush=True)
                time.sleep(wait)
            elif 'time' in err.lower() and 'out' in err.lower():
                time.sleep(15 * (attempt + 1))
            else:
                raise
    raise RuntimeError('judge LLM: max retries exceeded')


# ── The judge prompt ──────────────────────────────────────────────────────────
# Deliberately generous about what the judge is asked to do: it is told to rate
# faithfulness, which it has no way to establish. That is the condition under
# test, and softening the instruction would soften the finding.
JUDGE_PROMPT = (
    "You are an expert evaluator auditing explanations produced for machine "
    "learning predictions. You assess one explanation at a time.\n\n"
    "Task the model performs: {obj}\n"
    "Class the model predicted for this case: {cls}\n"
    "Valid indicators for this dataset, each given as an internal code followed "
    "by the measurement and range it stands for. The explanation is written for "
    "a lay reader, so it refers to these by measurement and range in words "
    "rather than by code; any measurement or range outside this list is "
    "invented: {feats}\n\n"
    "Explanation to evaluate:\n\"\"\"\n{expl}\n\"\"\"\n\n"
    "Rate the explanation on four dimensions, each an integer from 0 to 100:\n"
    "  faithfulness  - does it reflect the model's true decision logic?\n"
    "  coherence     - is it internally consistent, non-contradictory, clear?\n"
    "  completeness  - does it fully account for the prediction?\n"
    "  halluc_risk   - how much invented or unsupported content does it contain? "
    "(HIGHER MEANS WORSE)\n\n"
    "Respond with ONLY a JSON object, no other text, no code fences:\n"
    '{{"faithfulness": <int>, "coherence": <int>, "completeness": <int>, '
    '"halluc_risk": <int>}}'
)

DIMS = ['faithfulness', 'coherence', 'completeness', 'halluc_risk']


def parse_scores(raw):
    """Pull the four integers out of the judge's reply. Returns None on failure."""
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL).strip()
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for d in DIMS:
        v = obj.get(d)
        if v is None:
            return None
        try:
            out[d] = max(0, min(100, int(round(float(v)))))
        except Exception:
            return None
    return out


def dataset_meta(name):
    """Task objective and the dataset's valid indicator names (judge-safe).

    The names must be the SHORT bin ids the generator wrote its explanations in
    (`<col>_b<i>`), built exactly as run_comparison_nn.py builds them -- not the
    long `col::[lo,hi)` names binarize() returns. Handing the judge the wrong
    vocabulary would make every legitimate indicator look invented and turn the
    hallucination dimension into noise.
    """
    with open('datasets_config.yaml', 'r', encoding='utf-8') as f:
        obj = yaml.safe_load(f)[name]['objectif_classification']
    X_train, X_test, _y_train, _y_test = load_dataset(name)
    _tr, _te, _long, kbins = binarize(X_train, X_test)
    # The vocabulary handed to the judge must be the one the generator actually
    # writes in. Explanations are now written for a lay reader: they name a
    # measurement and its numeric range and never the bin id, so a bare-id list
    # makes every legitimate condition look invented -- the exact failure this
    # function's docstring warns about. We therefore give the judge the same
    # measurement-and-range vocabulary. It still sees no classifier, no XAI
    # context, no arm and no score, so the instrument stays text-only.
    bin_ids = [f'{col}_b{b} ({col} in [{edges[b]:.4g}, {edges[b + 1]:.4g}))'
               for col, edges in zip(X_train.columns, kbins.bin_edges_)
               for b in range(len(edges) - 1)]
    return obj, bin_ids


def run_dataset(name):
    src = os.path.join(RESULTS, f'{name}_nn_comparison{_TAG_SUFFIX}.csv')
    if not os.path.exists(src):
        print(f'  [skip] {src} missing')
        return
    df = pd.read_csv(src)

    # Self-preference guard, checked against the generator that ACTUALLY wrote
    # this file rather than a hardcoded id -- otherwise a second generator
    # silently gets judged by its own family.
    gen = str(df['llm'].iloc[0]) if 'llm' in df.columns else GENERATOR
    if same_family(gen, JUDGE_MODEL) and not os.getenv('XAI_ALLOW_SELF_JUDGE'):
        print(f'  [REFUSING] {name}: generator {gen!r} and judge '
              f'{JUDGE_MODEL!r} are both family {model_family(gen)!r}. '
              f'Set XAI_JUDGE_MODEL to another family, or '
              f'XAI_ALLOW_SELF_JUDGE=1 for the self-judge control.')
        return
    print(f'  generator {gen} | judge {JUDGE_MODEL}')

    obj, feats = dataset_meta(name)
    os.makedirs(RESULTS, exist_ok=True)
    dest = os.path.join(RESULTS, f'{name}_nn_judge{_TAG_SUFFIX}.csv')

    done = set()
    rows = []
    if os.path.exists(dest):                      # resume
        prev = pd.read_csv(dest)
        rows = prev.to_dict('records')
        done = {(r['instance'], r['arm'], r['strategy'], r['pass'])
                for r in rows}
        print(f'  resuming: {len(done)} already judged')

    # Every explanation is judged N_PASSES times and averaged downstream. The
    # judge is not reproducible even at temperature 0 (provider-side
    # nondeterminism), and a single pass carries ~9-15 pts of noise; averaging
    # N passes cuts that by sqrt(N) and is what makes the per-cell estimates
    # trustworthy. Passes are also what the self-consistency gate is computed
    # from, so they serve double duty.
    jobs = []
    for _, r in df.iterrows():
        for st in STRATEGIES:
            expl = r.get(f'explanation_{st}')
            if not isinstance(expl, str) or not expl.strip():
                continue
            for pss in range(1, N_PASSES + 1):
                jobs.append((int(r['instance']), str(r['arm']), st,
                             int(r['pred']), expl, pss))

    todo = [j for j in jobs if (j[0], j[1], j[2], j[5]) not in done]
    print(f'  {len(jobs)} judgements over {N_PASSES} pass(es) '
          f'({len(todo)} to run, {len(jobs)-len(todo)} already done)')

    def judge_one(job):
        pos, arm, st, pred, expl, pss = job
        if SLEEP_BETWEEN:
            time.sleep(SLEEP_BETWEEN)
        raw = llm(JUDGE_PROMPT.format(obj=obj, cls=pred, feats=feats, expl=expl))
        sc = parse_scores(raw)
        # `provider` is recorded because a run can legitimately span providers
        # (same model, different serving stack). Rows written before this
        # column existed are all Together; analysis should fillna accordingly.
        rec = {'dataset': name, 'instance': pos, 'arm': arm, 'strategy': st,
               'pass': pss, 'judge_model': JUDGE_MODEL, 'provider': PROVIDER,
               'parse_ok': int(sc is not None)}
        for d in DIMS:
            rec[d] = sc[d] if sc else None
        rec['raw'] = raw[:500]
        return rec

    lock, fatal = threading.Lock(), []
    n_done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(judge_one, j) for j in todo]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except OutOfCredit as e:
                # Billing failure: retrying cannot help. Stop what has not
                # started, keep every judgement already paid for.
                fatal.append(e)
                for f in futures:
                    f.cancel()
                continue
            except Exception as e:
                print(f'    [error, skipped] {str(e)[:110]}', flush=True)
                continue
            with lock:
                rows.append(rec)
                n_done += 1
                if n_done % CKPT_EVERY == 0:
                    pd.DataFrame(rows).to_csv(dest, index=False)
                    print(f'    {n_done}/{len(todo)} judged', flush=True)

    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f'  wrote {dest} (+{n_done} rows)')
    if fatal:
        raise fatal[0]


def report(datasets):
    """Per-strategy means and the test-retest spread. The comparison of
    interest is v4 vs v6 within an arm, against the fidelity move."""
    frames = []
    for name in datasets:
        p = os.path.join(RESULTS, f'{name}_nn_judge{_TAG_SUFFIX}.csv')
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        print('nothing to report')
        return
    a = pd.concat(frames, ignore_index=True)
    first = a[a['pass'] == 1]
    print(f'\n=== judge = {JUDGE_MODEL} | {len(first)} judgements, '
          f'parse ok {100*first["parse_ok"].mean():.1f}% ===')
    print('\nmean score by strategy:')
    print(first.groupby('strategy')[DIMS].mean().round(2).to_string())
    print('\nmean score by arm x strategy (faithfulness):')
    print(first.pivot_table(index='arm', columns='strategy',
                            values='faithfulness').round(2).to_string())

    rep = a[a['pass'] == 2]
    if len(rep):
        key = ['dataset', 'instance', 'arm', 'strategy']
        m = rep.merge(first, on=key, suffixes=('_2', '_1'))
        print(f'\ntest-retest on {len(m)} repeated judgements, mean |pass1-pass2|:')
        for d in DIMS:
            print(f'  {d:14s} {(m[f"{d}_2"] - m[f"{d}_1"]).abs().mean():.2f}')


def main(argv):
    datasets = argv[1:] or DATASETS
    # NOTE: the self-preference check lives in run_dataset(), where the
    # generator is read from the CSV's `llm` column. It used to be here, tested
    # against a hardcoded 'llama-3.3-70b' -- which silently passed as soon as a
    # second generator was introduced, and wrongly blocked a valid cross-family
    # pairing. Do not reinstate a module-level version.
    if not _api_keys:
        print(f'REFUSING: no API key for provider {PROVIDER!r}.')
        return 2

    if SAVE_TRACES:
        os.makedirs(os.path.join(RESULTS, 'logs'), exist_ok=True)
        print(f'judge = {JUDGE_MODEL} via {PROVIDER}', flush=True)

    for name in datasets:
        print(f'[{name}] judging stored explanations ...', flush=True)
        try:
            run_dataset(name)
        except OutOfCredit as e:
            print(f'\n*** {PROVIDER} is out of credit -- stopping. ***\n'
                  f'{str(e)[:200]}\n\n'
                  f'Progress is saved and the run is resumable: rerun the same '
                  f'command once credit is added, or switch providers with\n'
                  f'  XAI_PROVIDER=groq python run_judge.py\n'
                  f'(Groq serves the same {JUDGE_MODEL} on a free tier.)',
                  flush=True)
            report(datasets)
            return 3
    report(datasets)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
