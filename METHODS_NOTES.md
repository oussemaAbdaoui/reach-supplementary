# Methods notes

Protocol decisions that shaped the results, including two that would have
invalidated them if they had gone the other way. These are recorded here rather
than in the paper because they are about how the apparatus was built, not about
what it found.

---

## 1. The refine loop returns the draft and the missing names, nothing else

Bounded dialogue gives the generator up to three refine-and-extract rounds.
Each round returns the model **its own previous draft** together with **the
names of the indicators that draft did not state**. An earlier implementation
returned the names without the draft; because `llm()` is a single-turn call
with no history, the model then regenerated from scratch every round instead
of revising, and dropped conditions it had already transmitted. That is
corrected, and the results here are from the corrected loop.

Beyond the draft, the feedback carries **the names of missing indicators only** — never
thresholds, never ranges, never a holds / does-not-hold flag against the
instance, and never any fidelity score.

An interim revision of the pipeline returned ranges *plus* a holds/does-not-hold
flag. For the rule-shaped arms this is harmless: their context already states
both. For the attribution-shaped arms it is fatal to the experiment, because
those two facts are **exactly what an attribution context withholds by design**.
A ranked attribution says which features matter and in which direction they
push; it never says which side of a threshold this instance falls on. Feeding
that back through the loop would have handed the attribution arms the missing
polarity and **manufactured the very recovery the paper attributes to context
shape**.

This was caught before any reported run. It is worth stating plainly because it
is the failure mode a sceptical reader should look for in work of this kind: the
recovery result is only meaningful if the attribution arms never receive, by any
path, the information their context is defined as lacking.

The same principle governs the parser. It is supplied with the
identifier-to-range legend so it can match prose against binary features; that
legend goes to the **parser only**, never to the generator.

---

## 2. Reader-facing explanations, and why two prompt generations exist

Explanations were originally written in internal bin identifiers — *"the
indicator `concavity_mean_b0` is 1"*. They scored well and were meaningless to
the reader the pipeline exists to serve, which invalidated the framing rather
than the numbers.

Generation now names the measurement and its range in ordinary words. Measured
on `cancer` / instance 163:

| | before | after |
|---|---:|---:|
| raw identifiers per explanation | 15 | 0 |
| fidelity | 98.0 | 98.0 |
| certified arm, single pass | 70.0 | 98.0 |

**Consequence.** The five-dataset grid (375 arm-runs) predates this change and
is not text-comparable to anything produced after it. Files tagged
`_cogito671b` or `_llama33_newprompts` are on the current prompts; the plain
`{dataset}_nn_comparison.csv` files are on the earlier ones. No table in the
paper mixes them.

This change also broke the judge, which had been handed bare bin identifiers as
its "valid indicator" vocabulary while explanations had moved to prose — making
every legitimate condition look invented. Re-scoring the same explanations under
each vocabulary is now reported as a finding in its own right: the verifier's
one grounded dimension turns out to be a vocabulary match rather than a check on
content.

---

## 3. Four silent failures, each of which corrupted results without raising

1. **Log truncation.** The log filename carried no model tag, so two runs
   started in the same second opened the same file with `w` and truncated each
   other. Filenames now carry model and PID.
2. **Anchors memory.** Anchors preallocated `batch_size × 10000` rows twice
   (~1.4 GB at *d*=90) and died on an 8 GB machine. `XAI_ANCHOR_BATCH` makes it
   configurable; the default is unchanged so stored runs stay reproducible.
3. **Retry scope.** Only 429s and timeouts were retried, so a single dropped
   socket killed a whole arm and wrote an error row that instance-level resume
   never revisits.
4. **Judge vocabulary.** See §2 above.

---

## 4. Solver variation is larger than one comparison suggests

The COP explainer runs CP-SAT under a fixed timeout and returns its best
feasible rule, not necessarily an optimum. Two independent invocations of the
identical program at matched length returned the same rule on 14 of 15
instances, moving the mean by **0.2 points**.

That understates it. Comparing the two length sweeps in `results/cop/` at
*identical* k — same program, same k, different invocation — gives
disagreements of up to **5 points**:

| k | 3 | 5 | 7 | 9 |
|---|---:|---:|---:|---:|
| `cancer_cop_sweep_V.csv` | 60.8 | 69.6 | 71.5 | 75.5 |
| `cancer_cop_sweep_V18.csv` | 65.7 | 71.3 | 75.5 | 77.8 |

The trend is stable across both — monotone increase, plateau near 85, never
reaching the certified 96.4 — but no single per-*k* figure should be read as a
measurement until the sweep is repeated under a pinned seed. The paper reports
5 points as the working bound on solver variation for this reason.

---

## 5. Estimator details worth knowing

- Conditional precision fixes *S* to the instance's values **by construction**
  and samples only the complement. An earlier estimator sampled the full space
  and filtered post hoc, which starves the effective sample count for long
  condition sets and invalidates the Hoeffding bound. Nothing in `results/`
  uses that path.
- `n=1000`, `δ=0.05`, giving a half-width of ±0.043.
- Probabilistic-sufficiency rates are tested at the point estimate, so they are
  optimistic by at most the half-width.
- The dialogue strategy admits two summaries — final-round and best-over-rounds.
  The latter is a maximum over up to three draws and biased upward; the
  final-round figure is the one reported.
