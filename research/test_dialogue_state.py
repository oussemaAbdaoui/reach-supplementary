"""Minimal probe: does the refine loop behave differently if the model can see
its own previous explanation?

Variant A = REFINE exactly as shipped (no prior draft in the prompt).
Variant B = REFINE + the prior draft, i.e. an actual dialogue.

Both start from the SAME stored explanation_v4, so the only thing that differs
is the loop. Contexts are replayed from the stored `ctx` column, so no explainer
and no solver is re-run and the contexts are identical to the paper's.

Read-only with respect to run_comparison_nn.py: its source is exec'd up to the
main-loop marker to obtain predict / bin_ids / bin_legend / rules_to_S / llm,
and nothing below that marker runs.

  XAI_PROBE_DRY=1     no API calls; just prove the wiring and print the plan
  XAI_PROBE_N=8       instances per dataset (default 8)
  XAI_PROBE_DS=cancer comma-separated datasets
"""
from __future__ import annotations
import ast, csv, difflib, glob, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

DRY      = os.getenv('XAI_PROBE_DRY', '') == '1'
N_INST   = int(os.getenv('XAI_PROBE_N', '8'))
DATASETS = [d.strip() for d in os.getenv('XAI_PROBE_DS', 'cancer').split(',') if d.strip()]
MODEL    = os.getenv('XAI_PROBE_MODEL', 'grok-4.20-0309-non-reasoning')
PROVIDER = os.getenv('XAI_PROBE_PROVIDER', 'xai')
TAG      = os.getenv('XAI_PROBE_TAG', 'grok420')
ARMS     = ['cop', 'lmpaxp', 'anchors', 'shap', 'lime', 'cop_k']

MARKER = '# ── Main loop'


def load_pipeline(dataset):
    """exec run_comparison_nn.py up to the main loop; return its namespace."""
    src = open('run_comparison_nn.py', encoding='utf-8').read()
    cut = src.index(MARKER)
    ns = {'__name__': 'pipeline_partial', '__file__': os.path.abspath('run_comparison_nn.py')}
    argv, environ = sys.argv[:], dict(os.environ)
    sys.argv = ['run_comparison_nn.py', dataset, MODEL]
    os.environ['XAI_PROVIDER'] = PROVIDER
    os.environ['XAI_NO_LLM'] = '0'
    try:
        exec(compile(src[:cut], 'run_comparison_nn.py', 'exec'), ns)
    finally:
        sys.argv = argv
        os.environ.clear(); os.environ.update(environ)
        os.environ['XAI_PROVIDER'] = PROVIDER
    return ns


def stored_rows(dataset):
    rows, seen = [], set()
    for f in glob.glob(f'research/results/{dataset}_nn_comparison*{TAG}.csv'):
        for r in csv.DictReader(open(f, encoding='utf-8', errors='replace')):
            k = (r['dataset'], r['instance'], r['arm'])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows


def as_set(s):
    try:
        v = ast.literal_eval(s)
        return set(v) if v else set()
    except Exception:
        return None


REFINE_B = (
    "System: You are a machine learning expert refining an explanation based on feedback.\n"
    "Model objective: {obj}\n"
    "Instance (binary indicator features; name = value): {inst}\n"
    "Predicted class: {cls}\n"
    "{ctx}\n\n"
    "Your previous explanation:\n\"\"\"\n{prev}\n\"\"\"\n\n"
    "Indicators from the context above not yet reflected in your explanation: {missing}\n"
    "IMPORTANT: Keep the conditions you already stated correctly and add one for each listed "
    "indicator ONLY as supported by the context above. Never state an indicator in two "
    "opposite directions. Be concise. No special formatting.\n"
    "Keep writing for a reader with no technical background: measurements named in ordinary "
    "words with their numeric ranges, and NEVER the internal codes ending in _b0, _b1, _b2."
)

out_rows = []
for ds in DATASETS:
    stored = stored_rows(ds)
    if not stored:
        print(f'[{ds}] no stored rows for tag {TAG} -- skipped')
        continue
    ns = load_pipeline(ds)
    llm, EXTRACT, REFINE_A = ns['llm'], ns['EXTRACT'], ns['REFINE']
    parse_rules, rules_to_S = ns['parse_rules_from_llm'], ns['rules_to_S']
    bin_ids, bin_legend = ns['bin_ids'], ns['bin_legend']
    Xte_bin, predict, T = ns['Xte_bin'], ns['predict'], ns['T']
    MAX_ROUNDS = ns.get('MAX_ROUNDS', 3)
    obj = ns['dataset_config']['objectif_classification']

    positions = sorted({int(r['instance']) for r in stored})[:N_INST]
    print(f'[{ds}] {len(positions)} instances x {len(ARMS)} arms, '
          f'{"DRY" if DRY else MODEL}')

    for pos in positions:
        x_bin = Xte_bin[pos]
        pred = int(predict(x_bin)[0])
        for arm in ARMS:
            rec = next((r for r in stored
                        if int(r['instance']) == pos and r['arm'] == arm), None)
            if not rec or not rec.get('ctx') or not rec.get('explanation_v4'):
                continue
            S_arm, S4 = as_set(rec['S_arm']), as_set(rec['S_v4'])
            if S_arm is None or S4 is None:
                continue
            ctx, e4 = rec['ctx'], rec['explanation_v4']
            legend = '\n'.join(f'  {bin_ids[j]} : {bin_legend[bin_ids[j]]}' for j in S_arm)
            inst_repr = {bin_ids[j]: int(x_bin[j]) for j in S_arm}

            if DRY:
                out_rows.append({'dataset': ds, 'instance': pos, 'arm': arm,
                                 'planned_calls': 2 * MAX_ROUNDS * 2})
                continue

            def run(variant):
                cur_expl, cur_S = e4, set(S4)
                rounds = 0
                for _ in range(MAX_ROUNDS):
                    missing = [bin_ids[j] for j in S_arm if j not in cur_S]
                    if not missing:
                        break
                    fb = missing
                    if variant == 'A':
                        p = REFINE_A.format(obj=obj, inst=inst_repr, cls=pred,
                                            ctx=ctx, missing=fb)
                    else:
                        p = REFINE_B.format(obj=obj, inst=inst_repr, cls=pred,
                                            ctx=ctx, missing=fb, prev=cur_expl)
                    try:
                        expl = llm(p)
                        raw = llm(EXTRACT.format(legend=legend, expl=expl), max_tokens=2048)
                    except Exception as e:
                        print(f'    !! {type(e).__name__}: {str(e)[:120]}')
                        break
                    S, _c, _u = rules_to_S(parse_rules(raw), x_bin)
                    cur_expl, cur_S, rounds = expl, set(S), rounds + 1
                return cur_expl, cur_S, rounds

            eA, SA, rA = run('A')
            eB, SB, rB = run('B')
            out_rows.append({
                'dataset': ds, 'instance': pos, 'arm': arm, 'n_ctx': len(S_arm),
                'lostA': len(S4 - SA), 'lostB': len(S4 - SB),
                'supA': int(S4 <= SA), 'supB': int(S4 <= SB),
                'missA': len(S_arm - SA), 'missB': len(S_arm - SB),
                'simA': round(difflib.SequenceMatcher(None, e4, eA).ratio(), 3),
                'simB': round(difflib.SequenceMatcher(None, e4, eB).ratio(), 3),
                'roundsA': rA, 'roundsB': rB,
            })
            print(f'  inst {pos:>4} {arm:<8} lost A={out_rows[-1]["lostA"]} B={out_rows[-1]["lostB"]}'
                  f'  still-missing A={out_rows[-1]["missA"]} B={out_rows[-1]["missB"]}'
                  f'  sim A={out_rows[-1]["simA"]} B={out_rows[-1]["simB"]}', flush=True)

os.makedirs('research/results', exist_ok=True)
dest = f'research/results/dialogue_state_probe_{TAG}.csv'
if out_rows:
    with open(dest, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f'\nsaved -> {dest}  ({len(out_rows)} rows)')
if DRY:
    tot = sum(r.get('planned_calls', 0) for r in out_rows)
    print(f'DRY: {len(out_rows)} arm-instances, <= {tot} LLM calls '
          f'({tot // 2} per variant)')
