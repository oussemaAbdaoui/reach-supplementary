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

| Paper | Claim | File |
|---|---|---|
| §4.1, Tab. 2 | Instrument validated against exact ground truth | `research/results/nn_calibration_cancer*.csv` |
| §4.2 | Uniform sampler scores a *sufficient* reason at chance | `research/rescore_bt_nn_instrument.py`, `research/results/bt_rescored_nn_instrument.csv` |
| §4.3, Tab. 3 | Erosion by context; certified loses 14.3 | `research/results/*_nn_comparison.csv` |
| §4.4 | Recovery to 100%; shape decides | same, `score_v6` columns |
| §4.5, Tab. 4 | Accessibility–fidelity frontier | same, `len_*` columns |
| §4.6, Tab. 5(a) | Regimes, rule fitted under `U` | `research/results/nn_feasible_sampling_cop_k.csv` |
| §4.6, Tab. 5(b) | Regimes, rule refitted under `V` | `research/results/nn_feasible_sampling_{cogito671b,llama33_newprompts}.csv` |
| §4.6 | Length sweep plateaus near 85, never reaches 96.4 | `research/results/cancer_cop_sweep_V18.csv` |
| §4.7 | Same explanations under the structural metric | `research/results/bt_rescored_nn_instrument.csv` |
| §4.8 | Judge blind within a context; −34 on the repair | `research/results/cancer_nn_judge_cogito671b.csv` |
| §4.9, Tab. 6 | Two generators | `research/results/nn_feasible_sampling_*.csv` |
| §6.2 | Solver variation, 0.2 to 5 points | `cancer_cop_sweep_V.csv` vs `cancer_cop_sweep_V18.csv` |

---

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

**Two prompt generations exist and are not text-comparable.** The five-dataset
grid was produced when explanations were written in internal bin identifiers.
Everything tagged `_cogito671b` or `_llama33_newprompts` uses the reader-facing
prompts described in the paper's §3.5. No table in the paper mixes them; see
`METHODS_NOTES.md` §2.

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
