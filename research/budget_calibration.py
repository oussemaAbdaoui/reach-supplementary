"""Calibrate the paid-model grid to a fixed euro envelope.

Consumes the measured per-request token counts from measure_token_cost.py and
the July-2026 verified price table in LLM_API_BUDGET.md, and answers the
supervisor's question: what does one request cost on each provider, and how
many instances fit in 30 EUR?

Everything here is per-token arithmetic on MEASURED traffic. The only estimates
are the chars/4 tokenizer approximation and the USD/EUR rate, both flagged.
"""

USD_PER_EUR = 1.08          # state the assumption; re-check before spending
BUDGET_EUR = 30.0

# ── measured on the completed 375 arm-run Llama grid (measure_token_cost.py) ──
IN_PER_INSTANCE = 12_260     # 5 arms, all strategies
OUT_PER_INSTANCE = 4_025
CALLS_PER_INSTANCE = 44
IN_PER_CALL = 280
OUT_PER_CALL = 92

# ── prices USD / 1M tokens, verified July 2026 (research/LLM_API_BUDGET.md) ──
MODELS = [
    # (label, provider, $in, $out, reasoning?)
    ('GPT-5.6 Terra',      'OpenAI',    2.50, 15.00, False),
    ('GPT-4o',             'OpenAI',    2.50, 10.00, False),
    ('Claude Sonnet 5',    'Anthropic', 3.00, 15.00, False),
    ('Claude Haiku 4.5',   'Anthropic', 1.00,  5.00, False),
    ('Claude Opus 4.8',    'Anthropic', 5.00, 25.00, False),
    ('Gemini 2.5 Pro',     'Google',    1.25, 10.00, False),
    ('Gemini 2.5 Flash',   'Google',    0.30,  2.50, False),
    ('Grok 4',             'xAI',       3.00, 15.00, False),
    ('DeepSeek v4-pro',    'DeepSeek',  0.435, 0.87, False),
    ('Llama-3.3-70B',      'Together',  0.88,  0.88, False),
]

# A thinking model bills hidden reasoning tokens that never reach the transcript.
# Measured on thinkingmachines/Inkling-Small: ~1100 completion tokens are spent
# before the first content token (run_comparison_nn.py:llm). That multiplies the
# OUTPUT side by roughly this factor, and a rate-card estimate misses it.
REASONING_OUT_MULT = (1100 + OUT_PER_CALL) / OUT_PER_CALL


def cost(n_in, n_out, p_in, p_out, reasoning=False):
    if reasoning:
        n_out *= REASONING_OUT_MULT
    return (n_in / 1e6) * p_in + (n_out / 1e6) * p_out


def eur(usd):
    return usd / USD_PER_EUR


def main():
    print(f'Measured traffic: {IN_PER_CALL} in / {OUT_PER_CALL} out tokens per '
          f'request; {CALLS_PER_INSTANCE} requests per instance (5 arms).')
    print(f'Budget {BUDGET_EUR:.0f} EUR at {USD_PER_EUR} USD/EUR = '
          f'${BUDGET_EUR * USD_PER_EUR:.2f}\n')

    print('| Model | Provider | $/request | EUR/instance | EUR/15-inst grid | '
          'instances per 30 EUR |')
    print('|---|---|---:|---:|---:|---:|')
    rows = []
    for label, prov, pin, pout, reas in MODELS:
        c_call = cost(IN_PER_CALL, OUT_PER_CALL, pin, pout, reas)
        c_inst = cost(IN_PER_INSTANCE, OUT_PER_INSTANCE, pin, pout, reas)
        c_grid = c_inst * 15
        n_max = int(BUDGET_EUR * USD_PER_EUR / c_inst)
        rows.append((label, prov, c_call, c_inst, c_grid, n_max))
        print(f'| {label} | {prov} | ${c_call:.5f} | {eur(c_inst):.3f} | '
              f'{eur(c_grid):.2f} | {n_max:,} |')

    print('\n--- Candidate grids, 4 paid models for diversity ---')
    for name, picks, n_inst, n_ds in [
        ('A: 4 frontier, 3 datasets x 15 instances',
         ['GPT-5.6 Terra', 'Claude Sonnet 5', 'Gemini 2.5 Pro', 'Grok 4'], 15, 3),
        ('B: 4 frontier, 5 datasets x 15 instances (full replication)',
         ['GPT-5.6 Terra', 'Claude Sonnet 5', 'Gemini 2.5 Pro', 'Grok 4'], 15, 5),
        ('C: 4 frontier incl. Opus, 3 datasets x 10 instances',
         ['GPT-5.6 Terra', 'Claude Sonnet 5', 'Claude Opus 4.8', 'Gemini 2.5 Pro'], 10, 3),
        ('D: 5 models (adds Haiku), 2 datasets x 10 instances',
         ['GPT-5.6 Terra', 'Claude Sonnet 5', 'Claude Haiku 4.5', 'Gemini 2.5 Pro',
          'Grok 4'], 10, 2),
    ]:
        total = 0.0
        for label, prov, pin, pout, reas in MODELS:
            if label in picks:
                total += cost(IN_PER_INSTANCE, OUT_PER_INSTANCE, pin, pout, reas) \
                         * n_inst * n_ds
        arm_runs = len(picks) * n_ds * n_inst * 5
        print(f'  {name}')
        print(f'      {arm_runs} arm-runs | ${total:.2f} = {eur(total):.2f} EUR '
              f'| with 1.5x retry margin {eur(total) * 1.5:.2f} EUR '
              f'| {"FITS" if eur(total) * 1.5 <= BUDGET_EUR else "OVER"}')

    print(f'\nReasoning-model warning: output billed x{REASONING_OUT_MULT:.1f}.')
    for label, prov, pin, pout, _ in MODELS:
        if label in ('GPT-5.6 Terra', 'Claude Sonnet 5'):
            plain = cost(IN_PER_INSTANCE, OUT_PER_INSTANCE, pin, pout, False)
            think = cost(IN_PER_INSTANCE, OUT_PER_INSTANCE, pin, pout, True)
            print(f'  {label}: {eur(plain):.3f} EUR/instance -> '
                  f'{eur(think):.3f} EUR/instance if reasoning is on '
                  f'({think / plain:.1f}x)')


if __name__ == '__main__':
    main()
