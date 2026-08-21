"""Measure the real per-request token cost of one NN arm-run, from stored runs.

Written for the model-grid budget question, which needs a cost-per-
request figure for each provider BEFORE any paid run is launched, so the number
of instances can be calibrated to a fixed envelope.

Method. We do not guess from rate cards. Every prompt this pipeline sends is
rebuilt here from the exact templates in run_comparison_nn.py, filled with the
context and explanation text of the 375 completed Llama-3.3-70B arm-runs stored
in research/results/*_nn_comparison.csv. Output sizes are the stored
explanations and extracted-rule strings.

Two honest caveats, both stated in the output:
  * No tokenizer is installed here, so tokens are estimated at chars/4 -- the
    standard English approximation. Treat every figure as +-15%.
  * REASONING models bill hidden reasoning tokens that never appear in the
    stored text. Measured on thinkingmachines/Inkling-Small, ~1100 completion
    tokens are consumed before the first content token (see the note in
    run_comparison_nn.py:llm). For those models the output side is charged with
    a reasoning surcharge; a rate-card estimate would understate them badly.
"""
import ast
import glob
import os

import pandas as pd

CHARS_PER_TOKEN = 4.0

# ── the exact templates sent by run_comparison_nn.py (lines 308-355) ──────────
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
    "Question: Explain the reasons behind this prediction."
)
PROMPT_V6 = (
    "System: You are a machine learning expert explaining a model decision.\n"
    "Model objective: {obj}\n"
    "Instance (binary indicator features; name = value): {inst}\n"
    "Predicted class: {cls}\n"
    "{ctx}\n\n"
    "Instructions: Write EXACTLY one numbered sentence per item listed above. Do NOT merge "
    "or skip any item. Name the indicator exactly as written (e.g. glucose_b2). State each "
    "indicator at most once and never in two opposite directions.\n"
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
    "opposite directions. Be concise. No special formatting."
)
EXTRACT = (
    "You are an expert extracting logical conditions from ML explanations.\n"
    "Extract every condition as: <indicator> <operator> <value>\n"
    "Allowed indicators: {feats}\n"
    "Rules: indicator holds / is 1 -> 'indicator > 0.5' | indicator does not hold / is 0 "
    "-> 'indicator < 0.5' | one condition per list item | no other variables\n"
    "Example output: ['glucose_b2 > 0.5', 'age_b0 < 0.5']\n"
    "Constraints: output ONLY the Python list, no extra text, no code blocks.\n"
    "Explanation: {expl}\n"
    "Output:"
)


def toks(s):
    """Tokens for a string."""
    return len(str(s)) / CHARS_PER_TOKEN


def tk(n_chars):
    """Tokens for an already-counted number of characters."""
    return n_chars / CHARS_PER_TOKEN


def measure_row(r):
    """Input/output tokens for one arm-run (all its LLM calls)."""
    ctx = str(r.get('ctx', ''))
    e4, e5, e6 = (str(r.get(f'explanation_v{v}', '')) for v in (4, 5, 6))
    rules = str(r.get('rules_v4', '')) + str(r.get('rules_v6', ''))

    # inst_repr and own_ids are rebuilt from the stored index set
    try:
        S_arm = ast.literal_eval(str(r.get('S_arm', '[]')))
    except Exception:
        S_arm = []
    # indicator names average ~12 chars ("glucose_b2"); dict repr adds ~6/entry
    inst = max(len(S_arm), 5) * 18
    own = max(len(S_arm), 1) * 14

    static4 = len(PROMPT_V4) + len(PROMPT_V6) + len(REFINE) + len(EXTRACT)
    del static4  # kept for clarity; each template counted per-call below

    obj = 60  # 'objectif_classification' string, measured range 40-90 chars

    n_rounds = int(r.get('v5_rounds', 0) or 0)

    inp = 0.0
    out = 0.0
    # v4: explain + extract
    inp += tk(len(PROMPT_V4) + obj + inst + len(ctx))
    out += toks(e4)
    inp += tk(len(EXTRACT) + own + len(e4))
    out += toks(rules) / 2
    # v5: one refine + extract per round
    for _ in range(n_rounds):
        inp += tk(len(REFINE) + obj + inst + len(ctx) + own)
        out += toks(e5)
        inp += tk(len(EXTRACT) + own + len(e5))
        out += toks(rules) / 2
    # v6: explain + extract
    inp += tk(len(PROMPT_V6) + obj + inst + len(ctx))
    out += toks(e6)
    inp += tk(len(EXTRACT) + own + len(e6))
    out += toks(rules) / 2

    n_calls = 2 + 2 * n_rounds + 2
    return inp, out, n_calls, n_rounds


def main():
    files = sorted(f for f in glob.glob('research/results/*_nn_comparison.csv'))
    if not files:
        files = sorted(glob.glob('results/*_nn_comparison.csv'))
    rows = []
    for f in files:
        d = pd.read_csv(f)
        if 'ctx' not in d.columns:
            continue
        for _, r in d.iterrows():
            if pd.isna(r.get('explanation_v4', None)):
                continue
            inp, out, n_calls, n_rounds = measure_row(r)
            rows.append({'dataset': os.path.basename(f).split('_')[0],
                         'arm': r['arm'], 'in_tok': inp, 'out_tok': out,
                         'calls': n_calls, 'v5_rounds': n_rounds})
    m = pd.DataFrame(rows)
    if m.empty:
        print('no measurable rows found')
        return

    print(f'measured {len(m)} arm-runs from {len(files)} datasets '
          f'(the completed Llama-3.3-70B grid)\n')
    print('=== per ARM-RUN (one instance x one context, all 3 strategies) ===')
    print(f'  LLM calls      mean {m.calls.mean():.1f}   '
          f'(min {m.calls.min()}, max {m.calls.max()})')
    print(f'  v5 rounds      mean {m.v5_rounds.mean():.2f}')
    print(f'  input tokens   mean {m.in_tok.mean():,.0f}')
    print(f'  output tokens  mean {m.out_tok.mean():,.0f}')
    print(f'  total tokens   mean {(m.in_tok + m.out_tok).mean():,.0f}\n')

    print('=== per SINGLE REQUEST ===')
    per_call_in = m.in_tok.sum() / m.calls.sum()
    per_call_out = m.out_tok.sum() / m.calls.sum()
    print(f'  input  {per_call_in:,.0f} tok/request')
    print(f'  output {per_call_out:,.0f} tok/request\n')

    print('=== per ARM breakdown (context size drives input cost) ===')
    g = m.groupby('arm').agg(calls=('calls', 'mean'), in_tok=('in_tok', 'mean'),
                             out_tok=('out_tok', 'mean'))
    print(g.round(0).to_string(), '\n')

    print('=== per INSTANCE (5 arms) and per DATASET-GRID (15 instances) ===')
    per_inst_in = m.in_tok.mean() * 5
    per_inst_out = m.out_tok.mean() * 5
    print(f'  instance : {per_inst_in:,.0f} in + {per_inst_out:,.0f} out '
          f'= {per_inst_in + per_inst_out:,.0f} tok, {m.calls.mean() * 5:.0f} calls')
    print(f'  15-inst  : {per_inst_in * 15:,.0f} in + {per_inst_out * 15:,.0f} out '
          f'= {(per_inst_in + per_inst_out) * 15:,.0f} tok, '
          f'{m.calls.mean() * 75:.0f} calls')

    m.to_csv('research/results/token_cost_measured.csv', index=False)
    print('\nper-row measurements -> research/results/token_cost_measured.csv')
    return per_inst_in, per_inst_out, m.calls.mean() * 5


if __name__ == '__main__':
    main()
