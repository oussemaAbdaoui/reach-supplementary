# Result files: what each one is

`research/audit_numbers.py` re-derives every number printed in the paper and
checks it against the printed value. It reports **238 pass, 0 fail**. This file
records which runs it reads, and what the rest of the tree is.

Counts: 167 CSV files here, 62 of them read during that verification.

---

## Behind the reported numbers

These 62 files are the ones opened when the paper's numbers are verified. The
grid is five datasets (cancer, compas, diabetes, employee, heart) crossed with
six generators, written here as the tags the filenames use: `gridllama33`
(Llama-3.3-70B), `gptoss120b`, `gpt56` (gpt-5.6-terra), `grok420`, `sonnet5`
(Claude Sonnet 5), `glm52` (GLM-5.2). The judge axis covers four of the six:
gpt-oss-120b is itself the judge, and GLM-5.2 was not judged.

### Per-generator grids

One row per (dataset, instance, arm). Precision, length and parser diagnostics.

- `cancer_nn_comparison_glm52.csv`
- `cancer_nn_comparison_gpt56.csv`
- `cancer_nn_comparison_gptoss120b.csv`
- `cancer_nn_comparison_gridllama33.csv`
- `cancer_nn_comparison_grok420.csv`
- `cancer_nn_comparison_sonnet5.csv`
- `compas_nn_comparison_glm52.csv`
- `compas_nn_comparison_gpt56.csv`
- `compas_nn_comparison_gptoss120b.csv`
- `compas_nn_comparison_gridllama33.csv`
- `compas_nn_comparison_grok420.csv`
- `compas_nn_comparison_sonnet5.csv`
- `diabetes_nn_comparison_glm52.csv`
- `diabetes_nn_comparison_gpt56.csv`
- `diabetes_nn_comparison_gptoss120b.csv`
- `diabetes_nn_comparison_gridllama33.csv`
- `diabetes_nn_comparison_grok420.csv`
- `diabetes_nn_comparison_sonnet5.csv`
- `employee_nn_comparison_glm52.csv`
- `employee_nn_comparison_gpt56.csv`
- `employee_nn_comparison_gptoss120b.csv`
- `employee_nn_comparison_gridllama33.csv`
- `employee_nn_comparison_grok420.csv`
- `employee_nn_comparison_sonnet5.csv`
- `heart_nn_comparison_glm52.csv`
- `heart_nn_comparison_gpt56.csv`
- `heart_nn_comparison_gptoss120b.csv`
- `heart_nn_comparison_gridllama33.csv`
- `heart_nn_comparison_grok420.csv`
- `heart_nn_comparison_sonnet5.csv`

### Judge axis

Every judgement, four generators.

- `cancer_nn_judge.csv`
- `cancer_nn_judge_gpt56.csv`
- `cancer_nn_judge_gridllama33.csv`
- `cancer_nn_judge_grok420.csv`
- `cancer_nn_judge_sonnet5.csv`
- `compas_nn_judge_gpt56.csv`
- `compas_nn_judge_gridllama33.csv`
- `compas_nn_judge_grok420.csv`
- `compas_nn_judge_sonnet5.csv`
- `diabetes_nn_judge_gpt56.csv`
- `diabetes_nn_judge_gridllama33.csv`
- `diabetes_nn_judge_grok420.csv`
- `diabetes_nn_judge_sonnet5.csv`
- `employee_nn_judge_gpt56.csv`
- `employee_nn_judge_gridllama33.csv`
- `employee_nn_judge_grok420.csv`
- `employee_nn_judge_sonnet5.csv`
- `heart_nn_judge.csv`
- `heart_nn_judge_gpt56.csv`
- `heart_nn_judge_gridllama33.csv`
- `heart_nn_judge_grok420.csv`
- `heart_nn_judge_sonnet5.csv`

### Fidelity re-estimation under V

The structurally valid sampler, per generator.

- `nn_feasible_glm52.csv`
- `nn_feasible_gpt56.csv`
- `nn_feasible_gptoss120b.csv`
- `nn_feasible_gridllama33.csv`
- `nn_feasible_grok420.csv`
- `nn_feasible_sampling_check.csv`
- `nn_feasible_sonnet5.csv`

### Instrument calibration

The instrument against exact ground truth, under U and under V.

- `nn_calibration_cancer.csv`
- `nn_calibration_cancer_V.csv`

### Other files read during verification



- `cancer_cop_sweep_V18.csv`

---

## Kept for provenance, not reported

Nothing below is behind a number in the paper. It is here so the record is
complete and so a reader can see what was tried.

### Per-arm COP runs, merged into the grids above

The COP arms were re-run one arm at a time and folded back into each generator's grid with `research/merge_cop_arm.py`. These are those partial files; their rows are already present in the grids.

- `cancer_nn_comparison_cop_best_gridllama33.csv`
- `cancer_nn_comparison_cop_best_smokebest.csv`
- `cancer_nn_comparison_cop_cogito671b.csv`
- `cancer_nn_comparison_cop_k.csv`
- `cancer_nn_comparison_cop_k_cogito671b.csv`
- `cancer_nn_comparison_cop_k_glm52.csv`
- `cancer_nn_comparison_cop_k_gpt56.csv`
- `cancer_nn_comparison_cop_k_gptoss120b.csv`
- `cancer_nn_comparison_cop_k_gptoss20b.csv`
- `cancer_nn_comparison_cop_k_gridllama33.csv`
- `cancer_nn_comparison_cop_k_grok420.csv`
- `cancer_nn_comparison_cop_k_llama33_newprompts.csv`
- `cancer_nn_comparison_cop_k_sonnet5.csv`
- `cancer_nn_comparison_cop_llama33_newprompts.csv`
- `compas_nn_comparison_cop_best_gridllama33.csv`
- `compas_nn_comparison_cop_k.csv`
- `compas_nn_comparison_cop_k_glm52.csv`
- `compas_nn_comparison_cop_k_gpt56.csv`
- `compas_nn_comparison_cop_k_gptoss120b.csv`
- `compas_nn_comparison_cop_k_gridllama33.csv`
- `compas_nn_comparison_cop_k_grok420.csv`
- `compas_nn_comparison_cop_k_sonnet5.csv`
- `diabetes_nn_comparison_cop_best_gridllama33.csv`
- `diabetes_nn_comparison_cop_k.csv`
- `diabetes_nn_comparison_cop_k_glm52.csv`
- `diabetes_nn_comparison_cop_k_gpt56.csv`
- `diabetes_nn_comparison_cop_k_gptoss120b.csv`
- `diabetes_nn_comparison_cop_k_gridllama33.csv`
- `diabetes_nn_comparison_cop_k_grok420.csv`
- `diabetes_nn_comparison_cop_k_sonnet5.csv`
- `employee_nn_comparison_cop_best_gridllama33.csv`
- `employee_nn_comparison_cop_k.csv`
- `employee_nn_comparison_cop_k_glm52.csv`
- `employee_nn_comparison_cop_k_gpt56.csv`
- `employee_nn_comparison_cop_k_gptoss120b.csv`
- `employee_nn_comparison_cop_k_gridllama33.csv`
- `employee_nn_comparison_cop_k_grok420.csv`
- `employee_nn_comparison_cop_k_sonnet5.csv`
- `heart_nn_comparison_cop_best_gridllama33.csv`
- `heart_nn_comparison_cop_k.csv`
- `heart_nn_comparison_cop_k_glm52.csv`
- `heart_nn_comparison_cop_k_gpt56.csv`
- `heart_nn_comparison_cop_k_gptoss120b.csv`
- `heart_nn_comparison_cop_k_gptoss20b.csv`
- `heart_nn_comparison_cop_k_gridllama33.csv`
- `heart_nn_comparison_cop_k_grok420.csv`
- `heart_nn_comparison_cop_k_sonnet5.csv`

### July grid, earlier prompts

Produced when explanations were written in internal bin identifiers. Not text-comparable to anything after it. See `METHODS_NOTES.md` Sec. 2.

- `cancer_nn_comparison.csv`
- `compas_nn_comparison.csv`
- `diabetes_nn_comparison.csv`
- `employee_nn_comparison.csv`
- `heart_nn_comparison.csv`

### August runs, first reader-facing prompts

The first runs on the prompts the paper describes, superseded by the September six-generator grid. Cogito-671B stopped being served and was not carried forward; it appears nowhere in the paper.

- `cancer_nn_comparison_cogito671b.csv`
- `cancer_nn_comparison_cogito671b_pinned.csv`
- `cancer_nn_comparison_llama33_newprompts.csv`
- `cancer_nn_judge_cogito671b.csv`
- `nn_feasible_sampling_cogito671b.csv`
- `nn_feasible_sampling_cogito671b_pinned.csv`
- `nn_feasible_sampling_llama33_newprompts.csv`

### Prompt and parser iterations

Single-condition variations run while the prompts and the parser were being settled. gpt-oss-20b was run as a same-family capacity control and is not among the six generators the paper reports.

- `cancer_nn_comparison_gptoss20b.csv`
- `cancer_nn_comparison_inkling_small.csv`
- `cancer_nn_comparison_lime_shap_readable2.csv`
- `cancer_nn_comparison_lime_shap_readable3.csv`
- `cancer_nn_comparison_lmpaxp_readable.csv`
- `cancer_nn_comparison_pinnedparser.csv`
- `cancer_nn_comparison_proof2.csv`
- `cancer_nn_comparison_readable1.csv`
- `diabetes_nn_comparison_gptoss20b.csv`
- `heart_nn_comparison_gptoss20b.csv`

### COP length sweeps

Selection estimates over rule length. The paper reads the k=18 cancer sweep; the others are the shorter and per-dataset versions.

- `cancer_cop_sweep_V.csv`
- `cancer_cop_sweep_V18pv.csv`
- `compas_cop_sweep_V.csv`
- `diabetes_cop_sweep.csv`
- `diabetes_cop_sweep_V.csv`
- `diabetes_cop_sweep_det.csv`
- `employee_cop_sweep_V.csv`
- `heart_cop_sweep_V.csv`

### Other model tiers

Decision-tree, boosted-tree and black-box runs from the complexity progression. The paper's results section reports the dense-network tier.

- `cancer_bb_comparison.csv`
- `cancer_bt_chat_abtest.csv`
- `cancer_bt_chat_reeval.csv`
- `cancer_bt_comparison.csv`
- `cancer_prop1_comparison.csv`
- `cancer_prop2_comparison.csv`
- `cancer_prop3_comparison.csv`
- `cancer_prop4_comparison.csv`
- `cancer_prop5_comparison.csv`
- `cancer_prop6_comparison.csv`
- `compas_bt_comparison.csv`
- `diabetes_bt_comparison.csv`
- `diabetes_prop1_comparison.csv`
- `diabetes_prop2_comparison.csv`
- `diabetes_prop3_comparison.csv`
- `diabetes_prop4_comparison.csv`
- `diabetes_prop6_comparison.csv`
- `vote_house_84_prop1_comparison.csv`

### Probes and diagnostics

One-off checks: the fitting-distribution probe, the reasoning-token and cost probes, model availability, the pinned-parser re-parse, and the cross-tier re-scoring.

- `bt_rescored_nn_instrument.csv`
- `cop_sampler_check_cancer.csv`
- `dialogue_state_probe_grok420.csv`
- `model_availability.csv`
- `reasoning_probe.csv`
- `reparse_pinned_extractor.csv`
- `token_cost_measured.csv`

### Remaining files

Individually run controls and checks that do not group with the above.

- `cancer_nn_comparison_nollm.csv`
- `cancer_nn_judge_gptoss120b.csv`
- `nn_feasible_sampling_cop_k.csv`

