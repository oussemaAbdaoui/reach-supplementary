"""Re-run ONLY the dialogue (v5) path with the fixed REFINE prompt.

Why this exists: REFINE originally carried no {prev}, and llm() is a single-turn
call, so each "refine" round regenerated from scratch instead of revising. The
prompt is fixed in run_comparison_nn.py; this script replays the loop over the
stored grids so v4 and v6 stay byte-identical and only the dialogue column moves.

  - starts from the stored explanation_v4 / S_v4, so the draft is the published one
  - replays the stored `ctx`, so no explainer and no solver runs (no SHAP, LIME,
    Anchors or CP-SAT) -- that is why this is cheap in RAM and why Anchors is not
    a bottleneck
  - re-scores with the pipeline's own estimator and seeding, so the new v5 is
    commensurable with the untouched v4/v6
  - writes back ONLY the v5 columns, checkpointed per instance

usage:  python research/rerun_dialogue.py <dataset> <tag> <model-id> <provider>
  e.g.  python research/rerun_dialogue.py cancer grok420 grok-4.20-0309-non-reasoning xai
env:    XAI_RERUN_DRY=1   wire-check, no API calls
"""
from __future__ import annotations
import ast, os, shutil, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

if len(sys.argv) < 5:
    raise SystemExit(__doc__)
DATASET, TAG, MODEL, PROVIDER = sys.argv[1:5]
DRY = os.getenv('XAI_RERUN_DRY', '') == '1'

# The pinned parser is the instrument. If it is not pinned, a cross-generator
# comparison becomes a comparison of generator-parser pairs (paper sec 4.5).
if os.getenv('XAI_EXTRACTOR', 'meta-llama/Llama-3.3-70B-Instruct-Turbo') != \
        'meta-llama/Llama-3.3-70B-Instruct-Turbo':
    raise SystemExit('XAI_EXTRACTOR must stay at its default for this re-run')

import pandas as pd

# Duplicate guard. Two processes on one grid CSV both hold their own DataFrame
# and both write it back, so the later save silently discards the earlier one's
# work. Observed 2026-09-05 on cancer/grok420, and recorded in the 08-30 handoff
# as having happened before. An exclusive lock makes it impossible.
_LOCK = f'research/results/logs/.lock_{DATASET}_{TAG}'
os.makedirs('research/results/logs', exist_ok=True)
try:
    _fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(_fd, str(os.getpid()).encode()); os.close(_fd)
except FileExistsError:
    raise SystemExit(f'{DATASET}/{TAG} already running (lock {_LOCK}); '
                     f'delete the lock if that is stale')
import atexit
atexit.register(lambda: os.path.exists(_LOCK) and os.remove(_LOCK))

MARKER = '# ── Main loop'
src = open('run_comparison_nn.py', encoding='utf-8').read()
if '{prev}' not in src[src.index('REFINE = ('):src.index('EXTRACT = (')]:
    raise SystemExit('run_comparison_nn.py REFINE has no {prev}: apply the fix first')

ns = {'__name__': 'pipeline_partial', '__file__': os.path.abspath('run_comparison_nn.py')}
_argv = sys.argv[:]
sys.argv = ['run_comparison_nn.py', DATASET, MODEL]
os.environ['XAI_PROVIDER'] = PROVIDER
exec(compile(src[:src.index(MARKER)], 'run_comparison_nn.py', 'exec'), ns)
sys.argv = _argv

# Spread the Together load across keys from the first call. run_comparison_nn.py
# advances _key_idx only inside the 429 handler, and the pinned extractor indexes
# the SAME counter, so every concurrent process starts on key 1 and the spare
# keys stay idle until something already failed. Offsetting the starting index
# per (dataset, tag) balances generation AND extraction immediately.
_nkeys = max(len(ns.get('_extractor_keys') or []), len(ns.get('_api_keys') or []), 1)
if _nkeys > 1:
    ns['_key_idx'] = (hash((DATASET, TAG)) % _nkeys + _nkeys) % _nkeys
    print(f'  [keys] {_nkeys} available, starting at index {ns["_key_idx"]}', flush=True)

llm, EXTRACT, REFINE = ns['llm'], ns['EXTRACT'], ns['REFINE']
parse_rules, rules_to_S = ns['parse_rules_from_llm'], ns['rules_to_S']
bin_ids, bin_legend, Xte_bin = ns['bin_ids'], ns['bin_legend'], ns['Xte_bin']
predict, T = ns['predict'], ns['T']
precision_hat = ns['precision_hat']
N_EVAL, DELTA, SEED = ns['N_EVAL'], ns['DELTA'], ns['RANDOM_SEED']
MAX_ROUNDS = ns.get('MAX_ROUNDS', 3)
obj = ns['dataset_config']['objectif_classification']
CreditExhausted = ns['CreditExhausted']

CSV = f'research/results/{DATASET}_nn_comparison_{TAG}.csv'
if not os.path.exists(CSV):
    raise SystemExit(f'no such grid: {CSV}')
if not os.path.exists(CSV + '.preRefineFix.bak'):
    shutil.copy2(CSV, CSV + '.preRefineFix.bak')

df = pd.read_csv(CSV)
if 'v5_rerun' not in df.columns:
    df['v5_rerun'] = 0
todo = df.index[df['v5_rerun'] != 1].tolist()
print(f'[{DATASET}/{TAG}] {len(df)} rows, {len(todo)} to redo, model={MODEL} '
      f'provider={PROVIDER}{" DRY" if DRY else ""}', flush=True)


def as_set(s):
    try:
        v = ast.literal_eval(s) if isinstance(s, str) else s
        return list(v) if v else []
    except Exception:
        return []


def score_of(S, x_bin, seed):
    return precision_hat(predict, x_bin, S, n=N_EVAL, delta=DELTA, seed=seed)


done = 0
for i in todo:
    r = df.loc[i]
    pos, arm = int(r['instance']), str(r['arm'])
    ctx, e4 = r.get('ctx'), r.get('explanation_v4')
    if not isinstance(ctx, str) or not isinstance(e4, str):
        df.at[i, 'v5_rerun'] = 1
        continue
    S_arm, S4 = as_set(r['S_arm']), as_set(r['S_v4'])
    x_bin = Xte_bin[pos]
    pred = int(predict(x_bin)[0])
    legend = '\n'.join(f'  {bin_ids[j]} : {bin_legend[bin_ids[j]]}' for j in S_arm)
    inst_repr = {bin_ids[j]: int(x_bin[j]) for j in S_arm}
    seed = SEED + 31 * pos

    if DRY:
        df.at[i, 'v5_rerun'] = 1
        done += 1
        continue

    est4 = score_of(S4, x_bin, seed)
    cur_expl, cur_S, cur_est = e4, list(S4), est4
    v5_scores, rounds = [round(est4.precision * 100, 2)], 0
    for rnd in range(1, MAX_ROUNDS + 1):
        missing = [bin_ids[j] for j in S_arm if j not in cur_S]
        # Matches run_comparison_nn.py: exit once the context has arrived. The
        # old `and cur_est.precision >= T` could never be met by an arm whose
        # own context precision is below T, so 45% of rounds ran after om=0 and
        # each re-roll risked dropping conditions already transmitted.
        if not missing:
            break
        fb = missing if missing else '(all context indicators captured; refine for clarity)'
        try:
            expl = llm(REFINE.format(obj=obj, inst=inst_repr, cls=pred, ctx=ctx,
                                     missing=fb, prev=cur_expl))
            raw = llm(EXTRACT.format(legend=legend, expl=expl), max_tokens=2048)
        except CreditExhausted:
            df.to_csv(CSV, index=False)
            raise
        except Exception as e:
            print(f'  !! {type(e).__name__}: {str(e)[:100]}', flush=True)
            break
        S, contr, unk = rules_to_S(parse_rules(raw), x_bin)
        est = score_of(S, x_bin, seed)
        cur_expl, cur_S, cur_est, rounds = expl, S, est, rnd
        v5_scores.append(round(est.precision * 100, 2))

    df.at[i, 'explanation_v5'] = cur_expl
    df.at[i, 'S_v5'] = str(sorted(cur_S))
    df.at[i, 'score_v5'] = round(cur_est.precision * 100, 2)
    df.at[i, 'score_v5_best'] = max(v5_scores)
    df.at[i, 'weak_paxp_v5'] = bool(cur_est.precision >= T)
    df.at[i, 'len_v5'] = len(cur_S)
    df.at[i, 'om_v5'] = len([j for j in S_arm if j not in cur_S])
    df.at[i, 'cm_v5'] = len([j for j in cur_S if j not in S_arm])
    df.at[i, 'v5_rounds'] = rounds
    df.at[i, 'v5_scores'] = str(v5_scores)
    df.at[i, 'v5_rerun'] = 1
    done += 1
    print(f'  inst {pos:>4} {arm:<8} v5={df.at[i, "score_v5"]:6.2f} '
          f'(was {r["score_v5"]}) om={df.at[i, "om_v5"]} rounds={rounds}', flush=True)
    if done % 10 == 0:
        df.to_csv(CSV, index=False)

df.to_csv(CSV, index=False)
print(f'\n[{DATASET}/{TAG}] wrote {done} rows -> {CSV}')
if not DRY and 'spend_report' in ns:
    print('  spend:', ns['spend_report']())
