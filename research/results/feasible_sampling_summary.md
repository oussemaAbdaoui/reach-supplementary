# Feasible-sampling check — does the erosion result survive a valid distribution?

Run 2026-08-04. Scripts: `research/check_nn_feasible_sampling.py` (NN),
`research/rescore_bt_nn_instrument.py` (BT).
Raw output: `nn_feasible_sampling_check.csv`, `bt_rescored_nn_instrument.csv`.

**Reproduction gate: max |U_recomputed − stored| = 0.00 pts on all five
datasets.** The rebuilt MLP/split/binarization is bit-exact against the runs
that produced `*_nn_comparison.csv`, so the V and E columns are directly
comparable to the published U column.

## The problem

The tier estimates conditional precision under D = Uniform({0,1}^d), inherited
from [Koriche §4.1]. But d is a K-bins **one-hot** encoding: 3 mutually
exclusive bins per attribute. A uniform draw satisfies one-hot on one attribute
with probability 3/8, and on all m attributes with probability (3/8)^m — i.e.
never. The published numbers are therefore dominated by points that encode no
possible instance.

At the BT tier this is fatal: a sufficient reason, which must score 100 by
definition, scores **52** (chance) under uniform D and **92** under a feasible
D. That is what prompted this check.

## NN tier, all 5 datasets, 375 arm-runs, Llama-3.3-70B

| | U: uniform {0,1}^d | V: one-hot-valid uniform | E: empirical train |
|---|---|---|---|
| **LmPAXp erosion (v4)** | **23.71** | **14.28** | **10.23** |
| next-largest eroder | lime 10.99 | lime 8.70 | lime 6.84 |
| LmPAXp retained (v6) | 99.7% | 100.0% | 99.9% |
| rule arms retained | 99.6–100.0% | 99.9–100.0% | 100.0% |
| attribution arms retained | 89.4 / 92.0% | 92.9 / 96.8% | 94.8 / 94.3% |
| context span (arm) | 73.9–97.1 (23.1) | 80.1–96.9 (16.9) | 83.7–99.5 (15.8) |
| output span (v4) | 66.0–73.6 (**7.6**) | 76.3–89.7 (**13.4**) | 81.4–95.8 (**14.5**) |

### What survives

1. **Erosion grows with guarantee strength.** LmPAXp is the largest eroder
   under all three distributions (1.5×–2.2× the next arm). The *direction* of
   the headline finding is robust.
2. **Recovery is unaffected.** LmPAXp retains ~100% under v6 in every
   distribution — the strongest and most robust result in the study.
3. **Rule vs attribution split holds.** All three rule-shaped arms retain
   ≥99.6% everywhere; both attribution-shaped arms stay below 97% everywhere.
4. **The guarantee ordering is still not delivered to the reader.** Under V,
   LmPAXp enters ranked 1st (96.9) and its single-pass output ranks 3rd
   (82.6), behind SHAP (89.7). The certified context still buys the reader
   nothing over an attribution.

### What does not survive

1. **The magnitude.** Erosion of the certified context is 23.7 under U but
   10.2–14.3 under a feasible D. Roughly **57% of the headline number is
   infeasible-sampling inflation**. Every erosion figure in the paper must be
   restated.
2. **"Collapses into an 8-point band".** The output span is 7.6 under U but
   13.4–14.5 under a feasible D. The flattening claim must be weakened to the
   qualitative version (ordering not preserved), which does hold.

## BT tier, cancer + diabetes, 89 arm-runs

| metric | SR arm | v4 | v6 | erosion |
|---|---|---|---|---|
| conditional precision, **feasible D** | 92.31 | 92.00 | 92.31 | **0.0** |
| conditional precision, uniform D | 52.13 | 52.83 | 52.13 | −0.7 |
| W_f (as originally reported) | 100.0 | 67.9 | 70.7 | −32.1 |

**The BT erosion is a W_f phenomenon and does not appear under the
conditional-precision instrument.** The literals the LLM dropped were
logically necessary for the sufficiency proof but behaviourally almost inert
on feasible inputs. The two metrics disagree sharply about the same
explanations — which is itself a finding, but it is not the cross-metric
replication we hoped for.

Caveat: at BT only the `pyxai_sr` baseline is reconstructible from stored
artefacts; SHAP/LIME arm terms were not saved and would need an `arm_term()`
re-run.

## Length-matched COP@k, all 5 datasets (added 2026-08-08)

The `cop_k` sweep lives in `*_nn_comparison_cop_k.csv` and was outside the
first pass, so the COP-vs-LmPAXp head-to-head was stuck on the invalid uniform
sampler. `check_nn_feasible_sampling.py` now re-scores it too, into
`nn_feasible_sampling_cop_k.csv`. **Reproduction gate: 0.00 pts on all five
datasets.** 63 of 75 instances solved (heart: 3/15 — the solver returns no rule
at LmPAXp's length within the timeout). Of the 63, 10 proved OPTIMAL and 53
returned best-feasible at timeout, so COP@k is a *lower bound* on the regime.

Paired on those 63 instances, under V:

| arm | mean \|S\| | ctx | min ctx | WeakPAXp | v4 | erosion | v6 retained |
|---|---|---|---|---|---|---|---|
| cop (free k) | 2.56 | 78.74 | 18.9 | 0.302 | 75.08 | 3.66 | 99.8% |
| cop@k | 7.44 | 91.79 | 45.0 | 0.603 | 78.71 | 13.08 | 99.7% |
| lmpaxp | 7.44 | 97.31 | 83.7 | 0.794 | 82.31 | 15.00 | 99.9% |

Under U the same arm reads ctx 90.72 / v4 68.48 / erosion 22.24; under E,
94.87 / 84.90 / 9.97. Direction identical across all three.

### Two things this changes

1. **The regimes differ in the tail, not the median.** Paired gap: median 0
   (27/63 identical), LmPAXp ahead on 24, COP@k ahead on 12; mean 5.51 carried
   entirely by a tail where COP@k drops below chance (3 instances, min 45.0;
   18.9 at free length) and LmPAXp never goes below 83.7. This is Koriche's own
   in-expectation caveat, observed.
2. **"Erosion grows with guarantee strength" was confounded with length.**
   At identical length a far weaker guarantee erodes 13.08 vs LmPAXp's 15.00 —
   within 1.9 pts. Single-pass erosion tracks *how many conditions the context
   states*, not what it proves; the same rule at 2.56 conditions erodes 3.66.
   The paper's §5.2 causal reading was corrected accordingly; the *finding*
   (certified context loses most, ordering not delivered) is untouched, and
   shape still governs recovery.

## Consequences

- The distributional assumption is not a limitations-section detail. It
  inflates measured erosion ~2.3× at NN and destroys the metric entirely at BT.
  Correcting it is a contribution, not a caveat.
- Recommended headline distribution: **V (one-hot-valid uniform)** — it keeps
  the uniform, model-independent spirit of [Koriche §4.1] while restricting to
  points the encoding can represent. Report U alongside for continuity.
- The paper's erosion numbers, the abstract, and Table `tab:erosion` all need
  regenerating from the V column.
