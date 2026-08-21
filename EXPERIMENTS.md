# Extended results

Nine runs behind the paper's later sections, with the numbers each produced and
the file each is reproducible from. None of them called a paid API: the
generation runs were already stored, and everything here is solver-side or
free-tier scoring.

| | Run | Finding |
|---|---|---|
| E1 | Reader-facing prompts, four silent-failure fixes | See `METHODS_NOTES.md` §2–3 |
| E2 | Second generator (Cogito-671B) | Both headline findings reproduce; erosion *magnitude* is model-dependent |
| E3 | Judge / coherence axis | The proposed safeguard is blind within a context and penalises the repair by 34 points |
| E4 | Cost model from measured traffic | 280 in / 92 out tokens per request; 44 calls per instance |
| E5 | COP sampler probe (8 instances) | The below-chance tail looks like a fitting artifact |
| E6 | COP length sweep under `V` (k = 1…18) | Best-length COP stays 13–15 points short; two sweeps disagree by up to 5 at the same k |
| E7 | `cop_best` arm | Length maximising precision, re-certified |
| E8 | Full V-fitted COP refit, both generators | Both COP arms refitted and merged: 75 → 90 rows per grid |
| E9 | V re-scoring of both per-model grids | Confirms E5 at full scale; the gap survives but moves from tail to median |

---

## E2 — Second generator
`research/results/cancer_nn_comparison_cogito671b.csv`

15 instances × 5 contexts × 3 strategies on the current pipeline. All figures
under the structurally valid sampler `V`; the run reproduces its stored `U`
values to 0.00 points before re-scoring, so the two distributions are compared
on identical explanations.

| context | ctx | v4 | erosion | v6 | retained | WeakPAXp ctx→v6 |
|---|---:|---:|---:|---:|---:|---|
| SHAP | 99.13 | 94.65 | −4.47 | 94.05 | 94.9% | 1.00 → 0.73 |
| LmPAXp | 95.57 | 90.45 | −5.12 | 95.57 | **100.0%** | 0.67 → 0.67 |
| LIME | 88.77 | 62.91 | −25.87 | 63.43 | 71.5% | 0.47 → 0.00 |

Read under the inherited uniform sampler the same run reports **12.2** points of
erosion for the certified context rather than 5.1 — again roughly twice its
true size.

---

## E3 — The coherence axis
`research/results/cancer_nn_judge_cogito671b.csv` · `research/analyze_judge.py`

675 judgements (225 explanations × 3 passes), 658 parsed (97.5%). Judge is a
120B open model at temperature 0, from a different family than the generator.

**Between contexts it looks usable.** LmPAXp rates highest on faithfulness
(50.5) and lowest on hallucination risk (19.9); Anchors lowest (27.3) and most
hallucinated (47.2) — an ordering that agrees with the measured one.

**Within a context it is blind.** Pooled correlation between verifier-rated
faithfulness and measured fidelity is +0.242, produced entirely by differences
*between* arms. Within each arm: −0.04 (LmPAXp), +0.05 (COP), −0.10 (Anchors),
−0.00 (SHAP), −0.00 (LIME).

**It points the wrong way on the repair.** Moving the certified context from
bounded dialogue to structured generation raises measured fidelity 95.6 → 96.3
while the verifier cuts faithfulness 62.8 → 28.5. Across contexts, verifier
faithfulness falls 42.9 → 40.9 → 29.8 and completeness 44.6 → 41.5 → 32.2 from
v4 to v6 — structured output reads as a flat enumeration while being logically
exact.

**Its grounded dimension is a vocabulary match.** Re-scoring the same
explanations against internal bin identifiers moved LmPAXp from 19.9 to 76.7 on
hallucination risk and 50.5 to 21.5 on faithfulness, with no change to the
explanations.

*Limits.* Repeat judgements at temperature 0 differ by 7.5 points on
faithfulness and 14.9 on hallucination risk even averaged over three passes;
only effects above roughly 8 points are readable. Protocol-compliant coverage is
84 of 1,125 texts.

---

## E4 — Cost model from measured traffic
`research/measure_token_cost.py`, `budget_calibration.py`

Measured on 375 executed arm-runs rather than taken from a price list.

| measured quantity | value |
|---|---:|
| LLM calls per arm-run | 8.8 (4–10) |
| calls per instance (5 arms) | 44 |
| input tokens / request | 280 |
| output tokens / request | 92 |
| tokens per 15-instance grid | 244,275 |

Input dominates (≈12,260 tokens/instance against 4,025 out) because the XAI
context is re-injected at each refinement round.

**A reasoning model bills hidden reasoning tokens** — ~1,100 measured before the
first content token — so output is billed ≈13×, and some arms do not complete at
all inside a token budget that is generous for a chat model. Run generators in
non-reasoning mode unless you intend otherwise.

---

## E5 — Is COP's weak tail its guarantee type, or its fit?
`research/results/cop_sampler_check_cancer.csv` · `research/cop_sampler_check.py`

`fit_cop_rule` draws its example oracle from `Uniform({0,1}^d)`, verbatim from
its source. But *d* is a K-bins **one-hot** encoding, so a uniform draw is
one-hot-valid on all *m* attributes with probability `(3/8)^m` — effectively
never. COP therefore minimises its loss on points that encode no possible
instance, and is then evaluated under `V`.

Each instance is fitted twice with the *same* solver, k, m, seed and timeout,
changing only the oracle's distribution; both rules are scored under `V`.

| 8 instances | mean | worst | below 50 |
|---|---:|---:|---:|
| fitted under `U` | 64.1 | 18.9 | 1 |
| fitted under `V` | **72.6** | **58.7** | **0** |

---

## E6 — COP length sweep under `V`
`research/results/cancer_cop_sweep_V.csv`, `cancer_cop_sweep_V18.csv`

Two sweeps: A capped k at 9, B extended to 18. **Quote B.**

| k | 1 | 3 | 5 | 7 | 9 | 12 | 15 | 18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean precision | 55.1 | 65.7 | 71.3 | 75.5 | 77.8 | 82.5 | 83.1 | 85.1 |

- `cop_free` at its optimum: **83.29** at mean length 17.87
- `lmpaxp` reference: **96.35** at mean length 12.80
- equal budget: **cop@13 = 81.43** vs **lmpaxp = 96.35**

Lengthening the learned rule buys real precision and never closes the gap. The
disagreement between the two sweeps at identical k is discussed in
`METHODS_NOTES.md` §4.

---

## E8 — Full V-fitted refit
`research/results/cancer_nn_comparison_cop{,_k}_*.csv` · `research/run_cop_pipeline.sh`

Both COP arms refitted under `V` for both generators and merged into the cancer
grids, taking them from 5 arms to 6.

| grid | arm | before | after |
|---|---|---:|---:|
| cogito671b | cop | 65.92 | 70.11 |
| llama33 | cop | 65.34 | 70.11 |

---

## E9 — V re-scoring of both per-model grids
`research/results/nn_feasible_sampling_*.csv` · `research/rescore_models_V.py`

The comparison CSVs store **U-scored** values; every number the paper quotes is
**V-scored**, and that lives in a separate family of files. The new COP arms
existed only U-scored, so they were not comparable to anything else. This run
produced the missing values for all 90 rows per generator, at a reproduction
gate of **0.00 points**.

**1 — the below-chance tail is an artifact, at full scale.**

| arm | fit | mean | worst | below 50 |
|---|---|---:|---:|---:|
| COP (free k) | `U` | 73.1 | 18.9 | 1 |
| COP (free k) | `V` | **76.7** | **60.0** | **0** |
| COP@k | `U` | 86.5 | 45.0 | 1 |
| COP@k | `V` | **88.3** | **72.8** | **0** |

**2 — the gap survives and moves to the median.** At matched length the
refitted rule is 7.3 points behind, behind on 11 of 15 instances, paired median
+6.1, WeakPAXp 0.13 against 0.67.

**3 — erosion at matched length is not regime-neutral.**

| generator | COP@k | LmPAXp | ratio |
|---|---:|---:|---:|
| Llama-3.3-70B | −3.8 | −1.7 | 2.2× |
| Cogito-671B | −13.7 | −5.1 | 2.7× |

**4 — one non-replication.** Anchors — rule-shaped, near chance — retains 98.9%
on one generator but 91.2% on the other, below the 98.3% that SHAP retains
there. The structural diagnostic still separates by shape on both: rule arms
leave 0.0–1.3 conditions unstated under structured generation, attribution arms
5.5–8.3.

**5 — the refit is not a fixed point.** See `METHODS_NOTES.md` §4.

**Still open.** The refit covers `cancer` only. The other four datasets remain
fitted under `U`, including two below-chance instances (*employee* 49.0,
*heart* 48.2). Solver-side only, no API cost.
