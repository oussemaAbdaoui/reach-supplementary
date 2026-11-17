# Earlier tiers

The paper's main results are on a dense network accessed by query only (the
scripts at the root). This folder holds the two earlier tiers of the same
pipeline. They are kept because one result of the paper comes from them: at the
boosted-tree tier, where ground truth is available, a provably sufficient
explanation scores at chance under the uniform completion sampler and close to
its true value under a feasible one (§4 and Appendix D.2).

| script | tier |
|---|---|
| `run_comparison.py` | transparent decision tree, sufficient reason given to the LLM (the pipeline the paper extends) |
| `run_comparison_bb.py` | decision tree explained by SHAP, LIME and Anchors |
| `run_comparison_bt.py` | boosted trees, distilled into a rectified decision tree with PyXAI |
| `compare_arms_metric.py` | metric-layer comparison of the arms on the rectified tree (no LLM) |
| `rescore_bt_nn_instrument.py` | re-scores the boosted-tree explanations with the dense-network instrument under the uniform and the empirical distribution (no LLM) |
| `my_tree.py`, `encodage_CNF.py` | decision-tree and CNF utilities used by the scripts above |

Run them from the repository root, for example
`python earlier_tiers/rescore_bt_nn_instrument.py cancer`.

The boosted-tree tier needs a PyXAI source checkout that includes its
`examples/Distillation` folder (and, for the theory-aware rectifier, its
compiled C++ extension). Point `PYXAI_SRC` at that checkout.
`run_comparison_bt.py` writes `_bt_cache_<dataset>.pkl` and
`research/results/<dataset>_bt_comparison.csv`, which
`rescore_bt_nn_instrument.py` then reads.

These scripts use the first-generation prompts (Groq for the tree tiers,
Together by default for the boosted-tree tier), so they are not comparable with the dense-network grid except through
the re-scoring above.
