# REACH — supplementary material

Code and extended results for *REACH: Democratized, Not Delivered: A
Query-Access Framework for Measuring and Repairing Guarantee Erosion in
LLM-Mediated Explanation*.

Everything the paper reports is reproducible from this tree. It also contains
the runs the paper only points at: the full judge sweep, the COP length sweeps,
the sampler probe, the cross-tier re-scoring, and the raw generated
explanations.

---

## What is here

```
appendix.pdf             the extended appendix, sections A–G
video_figure.mp4         the video figure, 4 min 36 s
*.py                     the framework — generation, parsing, both scoring axes
datasets/                the five datasets, exactly as used
datasets_config.yaml     per-dataset paths, targets, few-shot examples
research/
  *.py                   analysis scripts; each produces reported numbers
  results/
    *_nn_comparison*.csv        dense-network grids, per dataset and generator
    nn_feasible_sampling_*.csv  the U / V / E re-estimations
    *cop_sweep*.csv             COP length sweeps
    cop_sampler_check_*.csv     the fitting-distribution probe
    *judge*.csv                 coherence axis, every judgement
    *bt*.csv                    boosted-tree tier, both metrics
    nn_traces/ bt_traces/       raw LLM explanations, per instance and arm
    logs/                       stdout of every run, incl. reproduction gates
requirements.txt
METHODS_NOTES.md         engineering decisions and near-misses behind the protocol
EXPERIMENTS.md           the runs behind the paper's later sections, per file
```

**The extended write-up is `appendix.pdf`**, a document separate from the paper
and included here in full. Sections A–G cover the extended background, the full
notation, the framework as implemented, a step-by-step trace of one real
instance through every stage, the complete result tables, three deployment
scenarios, and the disclosure of LLM use in writing. This repository holds what
the appendix cannot: the code, the data, and every stored run.

`video_figure.mp4` introduces the problem and the result for a reader meeting
the work for the first time.

The layout is flat and mirrors the working repository on purpose: the scripts
resolve their imports and data paths relative to this root, so **run everything
from the tree root** and they work unmodified. A tidier `code/` hierarchy was
tried and broke every path.

---

## Where each claim in the paper comes from

Every figure in the paper is re-derived by `research/audit_numbers.py`, which
reads only the files below and checks each value against the number as printed.
It reports `238 pass, 0 fail`. Run it first; it needs no API key.

The paper reports six generators, written here as the tags the result files use:
`gridllama33` (Llama-3.3-70B), `gptoss120b`, `gpt56` (gpt-5.6-terra),
`grok420`, `sonnet5` (Claude Sonnet 5) and `glm52` (GLM-5.2). Five datasets:
`cancer`, `compas`, `diabetes`, `employee`, `heart`.

| Paper | Claim | File |
|---|---|---|
| Sec. 4.1, Tab. 2 | Instrument validated against exact ground truth | `nn_calibration_cancer.csv` (under U), `nn_calibration_cancer_V.csv` (under V) |
| Sec. 4.2 | Uniform sampler scores a *sufficient* reason at chance | the same five grids, `V_*` against `score_*` columns |
| Sec. 4.3, Tab. 3 | Erosion by context; certified loses 0.49 | `<ds>_nn_comparison_gridllama33.csv` for the five datasets, joined to `nn_feasible_gridllama33.csv` |
| Sec. 4.4 | Recovery by structured generation; shape decides | same files, `V_v4` / `V_v5` / `V_v6` columns |
| Sec. 4.5, Tab. 4 | Accessibility-fidelity frontier | same files, `len_S_arm` column |
| Sec. 4.6, Tab. 5 | Guarantee regimes, Llama-3.3-70B alone, rule refitted under V | `nn_feasible_gridllama33.csv` with the same five grids |
| Sec. 4.6 | Length sweep plateaus, never reaches the single-pass figure | `cancer_cop_sweep_V18.csv` |
| Sec. 4.7 | The same explanations under the earlier structural metric | `research/rescore_bt_nn_instrument.py`, which writes `bt_rescored_nn_instrument.csv`. This section is the one the audit does not cover; re-run the script to reproduce it |
| Sec. 4.8 | Judge axis; the judge penalizes the repair | `<ds>_nn_judge_<tag>.csv` for `gridllama33`, `gpt56`, `grok420`, `sonnet5`, joined to `nn_feasible_<tag>.csv` |
| Sec. 4.9, Tab. 6 | Across six generators | `research/build_tab_models.py`, over `nn_feasible_<tag>.csv` and `<ds>_nn_comparison_<tag>.csv` for all six tags; paired statistics from `research/stats_paired.py` |
| Sec. 6.2 | Solver variation across COP lengths | `cancer_cop_sweep_V18.csv`, the k=18 sweep; the shorter `cancer_cop_sweep_V.csv` is the earlier range |

`research/results/MANIFEST.md` classifies every result file in this tree, so it
is always clear which runs are behind the paper and which are kept only for
provenance.

## Reproducing

Nothing below calls a paid API. The generation runs are already stored in
`research/results/nn_traces/`; the scoring, fitting and re-estimation steps
re-derive every number in the paper from those artefacts.

```bash
pip install -r requirements.txt
# run from this directory

# the sampler correction, all five datasets, U / V / E side by side
python research/check_nn_feasible_sampling.py

# the same for the two per-model grids (prints a reproduction gate; must be 0.00)
python research/rescore_models_V.py

# is COP's weak tail its guarantee type, or its fitting distribution?
python research/cop_sampler_check.py

# COP length sweep, and the refit + merge pipeline
bash research/run_cop_pipeline.sh

# coherence axis
python research/analyze_judge.py

# cost model, measured from real traffic rather than a price list
python research/measure_token_cost.py && python research/budget_calibration.py
```

**Reproduction gate.** `check_nn_feasible_sampling.py` recomputes each stored
`U` value before reporting anything under `V`, and prints the maximum absolute
deviation. Every re-estimation in the paper passes at **0.00 points**, and so
does a clean run of this tree. If that gate does not print 0.00 on your
machine, the rebuilt model diverged and no other number in the run can be
compared to the paper.

Note that re-running a script overwrites its own output file in
`research/results/`. Copy the tree first if you want to diff against what
shipped.

**Regenerating explanations** needs API credentials and is driven by
`run_comparison_nn.py`; put keys in a `.env` at this root. `XAI_ARMS`,
`XAI_OUT_TAG` and `XAI_N_INSTANCES` scope a run — see `research/run_cop_pipeline.sh`
for a worked example. Reasoning models bill hidden reasoning tokens (~1,100
before the first content token, output billed ≈13×); run generators in
non-reasoning mode unless you mean not to.

---

## Two things to know before reading the results

**Three generations of result file exist and are not text-comparable.** The
paper reports the third alone: the six-generator grid tagged `_gridllama33`,
`_gptoss120b`, `_gpt56`, `_grok420`, `_sonnet5` and `_glm52`, run in September
on the reader-facing prompts of the paper's Sec. 3.5, reseeded so every
generator sees identical contexts. The untagged `{dataset}_nn_comparison.csv`
files are from July, when explanations were written in internal bin
identifiers. The files tagged `_cogito671b` and `_llama33_newprompts` are from
August, the first runs on reader-facing prompts; Cogito-671B stopped being
served and was superseded by the six-generator grid rather than extended.
The first two generations are kept for provenance and are cited nowhere in the
paper. No table mixes generations; see `METHODS_NOTES.md` Sec. 2.

**Superseded files are not shipped.** Where a result was recomputed, only the
current version is here. The one case where both versions matter — COP fitted
under `U` versus under `V` — is kept as two separate files, since the paper
reports both.

---

## Scope and limits

Five tabular datasets; the pipeline assumes tabular, binarizable features. The
fidelity estimator is uniform over structurally valid points, which is not the
distribution any particular person is drawn from. The coherence axis is a
language model's judgment used as a null instrument, not a proxy for what a
person finds clear — and its run is partial, at 84 of 1,125 texts under the
three-pass protocol. No human participants were involved in any part of this
work.

The datasets are redistributed here as used, for reproducibility; each remains
under its own original terms.
