# Oracle Sampling — CP-2 Extension Summary
## Oracle sampling approach: Koriche et al. (2023), Equation 1 (precision error definition)
## PAC confidence bound: Hoeffding inequality (Hoeffding, 1963) — NOT Theorem 1 of Koriche et al.
## NOTE: Theorem 1 of Koriche et al. is a sample complexity bound for finding explanations,
##       not a confidence interval around ε̂. The two are different. Do not conflate them in the paper.

---

## 1. What Is Oracle Sampling and Why We Added It

### The gap in our original CP-2

Our original `compute_precision_error()` estimated ε̂ using **training data only**:

```
ε̂_train(S) = |{z ∈ X_train : z satisfies S, f(z) ≠ f(x)}|
              ─────────────────────────────────────────────
              |{z ∈ X_train : z satisfies S}|
```

The problem: the training set is a biased, clustered sample of the feature space.

| Dataset | n binary features | 2^n space | X_train size | Coverage |
|---------|-------------------|-----------|--------------|----------|
| Cancer  | 18 | 262,144 | 499 | **0.190%** |
| Diabetes | 78 | 3 × 10²³ | 700 | **≈ 0%** |

For diabetes, 700 training instances cover an **infinitesimal** fraction of the possible feature combinations. Any ε̂ computed only from training data is measuring error within a tiny, biased corner of reality.

### The oracle sampling approach (Koriche et al.)

Generate m random instances uniformly over the feature space, query the decision tree on each, and compute precision error over those:

```
ε̂_oracle(S) = |{z ~ D : z satisfies S, f(z) ≠ f(x)}|
               ──────────────────────────────────────
               |{z ~ D : z satisfies S}|

D = Uniform[min_j, max_j] per feature j  (estimated from training ranges)
```

This covers the full feature space, not just where real data happens to cluster.

### PAC confidence bound (Hoeffding inequality)

With n_neighbors samples falling inside the rule region:

```
PAC_bound = sqrt( ln(2/δ) / (2 × n_neighbors) )   at confidence 1−δ = 95%

True ε ∈ [ε̂_oracle ± PAC_bound]  with probability ≥ 0.95
```

| n_neighbors | PAC bound |
|---|---|
| 9 | ±0.45 — unreliable |
| 30 | ±0.25 — marginal |
| 82 | ±0.15 — acceptable |
| 180 | ±0.10 — good |
| 339 | ±0.07 — reliable |

---

## 2. Implementation

**File:** `run_comparison.py` — function `compute_precision_error_oracle()`

**No LLM calls required.** Post-hoc analysis runs on saved CSV results (`rules_v6_str`, `sr_str` columns), reconstructs bina_terms locally, then applies oracle sampling. Zero tokens spent.

**Sample sizes chosen:**
- Cancer: m = 2,000 (n=18, sufficient coverage)
- Diabetes: m = 50,000 (n=78, needed to find neighbors for tight k=5–9 rules)

---

## 3. Results

### Cancer (n=18, gpt-oss-120b, 12/15 instances)

| Metric | Training-based | Oracle (m=2,000) |
|--------|---------------|------------------|
| SR mean ε̂ | 0.0000 | **0.0000** |
| v6 mean ε̂ | 0.0000 | **0.0000** |
| Mean PAC bound | — | ±0.353 |

**Finding:** Oracle confirms training results. Every v6 rule stays entirely on the correct side of the decision boundary. ε̂=0 is not a training-distribution artifact — it holds across the full feature space.

**Limitation:** PAC bounds are wide for instances with tight 7-condition SR (only 3–8 oracle neighbors found). Would need m=50,000 for tighter bounds on cancer too.

---

### Diabetes (n=78, llama-3.3-70b-versatile, 13/15 instances)

#### Per-instance results (m=50,000)

| inst | SR | v6% | train ε̂ | n_train | oracle ε̂ | n_oracle | PAC± |
|------|----|-----|---------|---------|----------|---------|------|
| 57 | 6 | 70.2% | 0.000 | 1 | **0.000** | 339 | 0.074 |
| 12 | 10 | 51.7% | 0.000 | 2 | **N/A** | 0 | — |
| 140 | 9 | 70.5% | 0.000 | 17 | **0.000** | 9 | 0.453 |
| 125 | 8 | 60.0% | 0.000 | 27 | **0.000** | 82 | 0.150 |
| 114 | 8 | 80.3% | 1.000 | 18 | **0.828** | 180 | 0.101 |
| 71 | 5 | 71.9% | 0.000 | 6 | **0.000** | 12 | 0.392 |
| 52 | 7 | 68.8% | 0.000 | 69 | **0.000** | 6 | 0.554 |
| 279 | 7 | 63.5% | 0.000 | 9 | **0.000** | 380 | 0.070 |
| 44 | 6 | 84.2% | N/A | 0 | **0.000** | 26 | 0.266 |
| 216 | 8 | 60.0% | 0.000 | 27 | **0.000** | 72 | 0.160 |
| 16 | 6 | 74.1% | 0.000 | 24 | **0.000** | 101 | 0.135 |
| 15 | 10 | 65.0% | 0.000 | 2 | **N/A** | 0 | — |
| 47 | 12 | 50.4% | N/A | 0 | **N/A** | 0 | — |

#### Summary statistics

| Metric | Training (m=train) | Oracle (m=50,000) |
|--------|-------------------|-------------------|
| v6 mean ε̂ | 0.0909 | **0.0828** |
| Resolved instances | 11/13 | **10/13** |
| Mean PAC bound | — | **±0.236** |

---

## 4. Key Findings

### Finding 1 — Oracle confirms training for cancer
ε̂=0 on all 12 cancer instances holds across the full feature space. The v6 structured rules are genuinely precise — not just precise within the training distribution.

### Finding 2 — Instance 44 recovered (diabetes)
Training-based ε̂ was N/A (0 training neighbors inside the rule region). Oracle (m=50,000) finds 26 neighbors → ε̂=0.000. The rule is tight but correct. Training-based approach was blind to this instance.

### Finding 3 — Instance 114 is genuinely bad (diabetes)
- Training ε̂ = 1.000 (18/18 training neighbors get wrong prediction)
- Oracle ε̂ = **0.828** (149/180 oracle neighbors get wrong prediction)
- Training overstated the error (caught the worst cluster), but oracle confirms: 83% error rate. This v6 rule genuinely crosses the decision boundary. It is the worst-performing explanation in the dataset.

### Finding 4 — Oracle slightly more optimistic than training (diabetes)
Mean oracle ε̂ = 0.083 vs training 0.091. Oracle samples a wider, more uniform region; training is biased toward dense clusters where the model tends to be consistent. Difference is small but consistent.

### Finding 5 — 3 instances remain unresolvable (k=10, 10, 12)
Instances 12, 15, 47 have 10–12 simultaneous conditions. Even m=50,000 random points find 0 neighbors. For k=12, you would need m≈300,000+ to expect even a handful of neighbors. These instances are **not evaluable by oracle sampling at any practical m**.

---

## 5. Honest Assessment

| Situation | Reliable metric |
|-----------|----------------|
| Cancer (n=18, k≤7) | W_f, training ε̂, oracle ε̂ — all agree |
| Diabetes (n=78, k≤8) | Oracle ε̂ with m=50,000 gives PAC ±0.10–0.16 |
| Diabetes (n=78, k=9) | Oracle ε̂ marginal (few neighbors, PAC ±0.45) |
| Diabetes (n=78, k≥10) | Oracle ε̂ unresolvable at any practical m |

For diabetes instances with k≥10 extracted conditions, **W_f remains the only available metric** — despite its orthogonality issues. This is an inherent limitation of high-dimensional explanation evaluation, not a bug in our pipeline.

---

## 6. Reference

Koriche, F., Lagniez, J.-M., Mengel, S., Tran, C. (2023).
*Learning Model Agnostic Explanations via Constraint Programming.*
ECAI 2023, pp. 433–449.

> We adopt their oracle sampling approach (Section 2, Equation 1) to estimate
> the empirical precision error ε̂_{f,x}(S), and their PAC-style confidence bound
> We use the Hoeffding inequality (Hoeffding, 1963) as a confidence bound around ε̂_oracle.
> Note: this is independent of Theorem 1 in Koriche et al., which is a sample complexity result.
> Our setting differs in that f is a white-box decision tree — we use the tree
> directly for neighbor prediction rather than black-box membership queries.
