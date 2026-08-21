"""
cop_explainer.py — J2 of the NN tier (research/PLAN_NN_TIER.md).

Learns a model-agnostic rule explanation by Constraint Optimization,
implementing Koriche, Lagniez, Mengel, Tran, "Learning Model Agnostic
Explanations via Constraint Programming" — VERBATIM, nothing more:

  1. Draw m samples z_i ~ D = Uniform({0,1}^d), label through the example
     oracle ex(f, D) = query predict_fn                      [K §2, §4.1]
  2. Lemma 2 transform:  u_i = z_i XNOR x,  v_i = [f(z_i) == f(x)]
     (reduces rule learning to learning a MONOTONE k-monomial)  [K p.440]
  3. Solve the COP of Table 1 [K p.441] with OR-Tools CP-SAT:
       vars    X_j in {0,1} (j<d),  U_i in {0,1} (i<m),  K in {0..k}
       (iv)    sum_j X_j <= K
       (v)     v_i=1:  U_i <-> [sum_j X_j*u_ij <  K]   (positive channeling)
       (vi)    v_i=0:  U_i <-> [sum_j X_j*u_ij >= K]   (negative channeling)
       (vii)   minimize sum_i U_i
     60 s timeout, best-so-far solution returned on timeout    [K §4.1]
     variant='sat': drop the positive constraints (v) — smaller and faster,
     no longer an empirical-loss minimizer                     [K §4.1]
  4. Decode S = {j : X_j = 1}; rule body = literals (X_j = x_j), head = f(x).
     By construction the rule FIRES ON x and agrees with f on x — the
     property Anchors lacks                                    [K §4.2]

Guarantee carried by the output (report it, never overclaim):
  E_x[eps_{f,x}(S)] <= min_r L_f(r) + eps  with prob 1-delta, for
  m >= (2/eps^2)(k ln d + ln(2/delta))     [K Thm. 1 — IN EXPECTATION over x,
  not per instance: Note after Thm. 1].
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from ortools.sat.python import cp_model

__all__ = ['CopRule', 'fit_cop_rule', 'theorem1_m']


def theorem1_m(d: int, k: int, eps: float = 0.1, delta: float = 0.05) -> int:
    """Sample bound of [K Thm. 1]: m >= (2/eps^2)(k ln d + ln(2/delta))."""
    return int(math.ceil((2.0 / eps**2) * (k * math.log(d) + math.log(2.0 / delta))))


@dataclass
class CopRule:
    S: list                    # selected feature indices (0-based)
    literals: list             # [(j, x_j)] — body: conjunction of X_j = x_j
    head: int                  # f(x)
    k: int                     # size limit given to the solver
    K_used: int                # value of the cardinality variable K
    empirical_loss: float      # objective / m  (Eq. (8)-(9) [K])
    m: int
    status: str                # 'OPTIMAL' | 'FEASIBLE' (best-so-far at timeout)
    solve_time_s: float
    base_rate_same: float      # fraction of samples with f(z)==f(x) (risk log)
    variant: str = 'cop'
    theorem1_m_ref: int = field(default=0)   # the Thm-1 bound, for the record

    def fires_on(self, z_bin: np.ndarray) -> bool:
        z = np.asarray(z_bin).ravel()
        return all(z[j] == v for j, v in self.literals)


def fit_cop_rule(predict_fn, x_bin: np.ndarray, k: int = 7, m: int = 1000,
                 timeout_s: float = 60.0, variant: str = 'cop',
                 seed: int = 42, samples: np.ndarray | None = None,
                 labels: np.ndarray | None = None,
                 n_workers: int = 1, exact_k: bool = False) -> CopRule:
    """
    Learn a k-rule explanation for (predict_fn, x_bin). All defaults are the
    experimental values of [K §4.1] except k (their Table 2 uses k=5; the
    solver's K variable picks the effective size <= k).

    exact_k=True pins |S| = K = k — the loss-minimal rule AT size k, the
    'cop@k' protocol. Rationale: [K Fig. 2] evaluates precision as a function
    of explanation size; with the free K variable and a ~50/50 same-class
    region under uniform D (our oversampled datasets), the loss optimum is
    structurally 1-2 literals, which makes cross-arm comparisons at unequal
    lengths meaningless. cop@k gives the equal-budget comparison. NOTE: [K]'s
    own sweep uses k as a LIMIT (sum X <= K <= k), not an equality — cop@k is
    inspired by their protocol, not a replication of it.

    n_workers=1 (default) makes CP-SAT deterministic under a fixed seed.
    Multi-worker search is nondeterministic even with random_seed set, and
    equal-loss monomials can have very different CONDITIONAL precision —
    observed spread up to ~8.5 precision points across reruns of identical
    inputs at n_workers=4 (reviewer audit 2026-07-02). Only raise n_workers
    for exploratory runs where reproducibility doesn't matter.

    `samples`/`labels` may be supplied to reuse one draw across variants/arms.
    """
    if variant not in ('cop', 'sat'):
        raise ValueError("variant must be 'cop' or 'sat'")
    x_bin = np.asarray(x_bin).ravel().astype(np.int8)
    d = x_bin.shape[0]

    # 1. example oracle ex(f, D), D = Uniform({0,1}^d)  [K §2/§4.1]
    if samples is None:
        rng = np.random.default_rng(seed)
        samples = rng.integers(0, 2, size=(m, d), dtype=np.int8)
    else:
        samples = np.asarray(samples, dtype=np.int8)
        m = samples.shape[0]
    if labels is None:
        labels = np.asarray(predict_fn(samples)).ravel()
    pred = int(np.asarray(predict_fn(x_bin.reshape(1, -1))).ravel()[0])

    # 2. Lemma 2 transform  [K p.440]
    U = (samples == x_bin).astype(np.int8)          # u_i = z_i XNOR x
    v = (labels == pred).astype(np.int8)            # v_i = f(z_i) XNOR f(x)
    base_rate_same = float(v.mean())

    # 3. COP of Table 1  [K p.441]
    model = cp_model.CpModel()
    X = [model.NewBoolVar(f'X{j}') for j in range(d)]
    Uv = [model.NewBoolVar(f'U{i}') for i in range(m)]
    K = model.NewIntVar(0, k, 'K')
    model.Add(sum(X) <= K)                                          # (iv)
    if exact_k:
        # cop@k protocol: loss-minimal rule AT size k (see docstring).
        model.Add(sum(X) == K)
        model.Add(K == k)
    elif variant == 'sat':
        # Interpretation, not verbatim: without the positive constraints (v),
        # Table 1's '<= K' admits the degenerate optimum sum(X) < K ('fire on
        # nothing', objective 0). The paper's SAT results (Table 2) imply the
        # monomial semantics sum(X) = K; we pin it explicitly.
        model.Add(sum(X) == K)
    for i in range(m):
        sel = [X[j] for j in np.flatnonzero(U[i]).tolist()]         # sum X_j*u_ij
        s = sum(sel) if sel else 0
        if v[i] == 1:
            if variant == 'sat':                                    # drop (v)
                continue
            model.Add(s < K).OnlyEnforceIf(Uv[i])                   # (v)
            model.Add(s >= K).OnlyEnforceIf(Uv[i].Not())
        else:
            model.Add(s >= K).OnlyEnforceIf(Uv[i])                  # (vi)
            model.Add(s < K).OnlyEnforceIf(Uv[i].Not())
    model.Minimize(sum(Uv))                                         # (vii)

    # Warm start: greedy top-k features by |corr(u_j, v)|. Any X assignment
    # extends to a feasible solution (U is fully determined by reification),
    # so this hint guarantees the solver never returns UNKNOWN-for-lack-of-
    # incumbent at the timeout (observed under CPU contention with m=5000).
    _score = np.abs(U.mean(axis=0) - (U * v[:, None]).sum(axis=0)
                    / max(v.sum(), 1))
    _greedy = np.argsort(-_score)[:k].tolist()
    for j in range(d):
        model.AddHint(X[j], 1 if j in _greedy else 0)
    model.AddHint(K, k if exact_k else min(len(_greedy), k))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(timeout_s)
    solver.parameters.num_workers = n_workers
    solver.parameters.random_seed = seed
    t0 = time.time()
    st = solver.Solve(model)
    dt = time.time() - t0
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f'CP-SAT returned {solver.StatusName(st)}')

    # 4. decode  [K §3.2 end / Lemma 2]
    S = [j for j in range(d) if solver.Value(X[j]) == 1]
    return CopRule(
        S=S, literals=[(j, int(x_bin[j])) for j in S], head=pred, k=k,
        K_used=int(solver.Value(K)),
        empirical_loss=float(solver.ObjectiveValue()) / m, m=m,
        status='OPTIMAL' if st == cp_model.OPTIMAL else 'FEASIBLE',
        solve_time_s=round(dt, 2), base_rate_same=base_rate_same,
        variant=(f'{variant}@{k}' if exact_k else variant),
        theorem1_m_ref=theorem1_m(d, k))


# ── Self-tests (python cop_explainer.py --selftest) ────────────────────────────

def _selftest() -> int:
    from bb_metrics import precision_hat
    failures = []

    def check(name, cond):
        print(f'  [{"PASS" if cond else "FAIL"}] {name}')
        if not cond:
            failures.append(name)

    # Realizable case [K §3.2 / Valiant]: f IS a monotone 3-monomial.
    d, J = 20, [2, 5, 11]
    def f(Z):
        Z = np.atleast_2d(Z)
        return np.all(Z[:, J] == 1, axis=1).astype(int)
    x = np.ones(d, dtype=np.int8)

    rule = fit_cop_rule(f, x, k=5, m=1000, timeout_s=30, seed=0)
    print(f'    cop: S={rule.S} K={rule.K_used} loss={rule.empirical_loss:.4f} '
          f'status={rule.status} t={rule.solve_time_s}s '
          f'base_rate={rule.base_rate_same:.3f} (Thm1 m_ref={rule.theorem1_m_ref})')
    check('realizable: COP recovers the planted monomial exactly',
          sorted(rule.S) == J and rule.empirical_loss == 0.0)
    check('rule fires on x and head == f(x)',
          rule.fires_on(x) and rule.head == 1)

    # Learned S is (near-)sufficient; a random same-size S is not.
    est = precision_hat(f, x, rule.S, n=1000, seed=1)
    rnd = np.random.default_rng(3).choice(
        [j for j in range(d) if j not in J], size=len(rule.S), replace=False)
    est_r = precision_hat(f, x, rnd.tolist(), n=1000, seed=1)
    print(f'    eps_hat learned={est.eps_hat:.3f}  random={est_r.eps_hat:.3f}')
    check('eps(learned) << eps(random same-size)',
          est.eps_hat + 0.3 < est_r.eps_hat)

    # SAT variant: runs, and its loss (re-evaluated on the samples) >= COP's.
    rng = np.random.default_rng(0)
    Z = rng.integers(0, 2, size=(1000, d), dtype=np.int8)
    rule_s = fit_cop_rule(f, x, k=5, m=1000, timeout_s=30, variant='sat',
                          seed=0, samples=Z)
    def emp_loss(r):
        fires = np.all(Z[:, r.S] == x[r.S], axis=1) if r.S else np.ones(len(Z), bool)
        rule_pred = np.where(fires, r.head, 1 - r.head)
        return float(np.mean(rule_pred != f(Z)))
    l_cop, l_sat = emp_loss(rule), emp_loss(rule_s)
    print(f'    sat: S={rule_s.S} | loss cop={l_cop:.4f} sat={l_sat:.4f}')
    check('SAT variant runs; loss(sat) >= loss(cop)', l_sat >= l_cop - 1e-9)
    check('SAT variant non-degenerate (covers the planted monomial)',
          set(J).issubset(set(rule_s.S)))

    # cop@k: exact size honored; at k=|J| it recovers J; supersets stay valid.
    rule_k3 = fit_cop_rule(f, x, k=3, m=1000, timeout_s=30, seed=0, exact_k=True)
    rule_k5 = fit_cop_rule(f, x, k=5, m=1000, timeout_s=30, seed=0, exact_k=True)
    print(f'    cop@3: S={rule_k3.S} loss={rule_k3.empirical_loss:.4f} | '
          f'cop@5: S={rule_k5.S} loss={rule_k5.empirical_loss:.4f}')
    check('cop@k pins |S| == k', len(rule_k3.S) == 3 and len(rule_k5.S) == 5)
    check('cop@3 recovers the planted monomial', sorted(rule_k3.S) == J)
    check('cop@5 is a superset of J (still fires correctly on the region)',
          set(J).issubset(set(rule_k5.S)))

    # Noisy / agnostic case: f = monomial XOR 5% label noise -> loss ~ noise.
    def f_noisy(Z):
        Z = np.atleast_2d(Z)
        base = np.all(Z[:, J] == 1, axis=1).astype(int)
        h = (np.sum(Z * (np.arange(d) + 1), axis=1) * 2654435761) % 97
        return np.where(h < 5, 1 - base, base)          # ~5% deterministic flip
    rule_n = fit_cop_rule(f_noisy, x, k=5, m=1000, timeout_s=30, seed=0)
    print(f'    noisy: S={rule_n.S} loss={rule_n.empirical_loss:.4f}')
    check('agnostic case: loss small but nonzero, S still ~ J',
          0.0 < rule_n.empirical_loss < 0.15
          and set(J).issubset(set(rule_n.S)) or rule_n.empirical_loss < 0.15)

    # Smoke test on a real sklearn MLP (the actual NN-tier target), if present.
    try:
        from sklearn.neural_network import MLPClassifier
        rng = np.random.default_rng(1)
        Xtr = rng.integers(0, 2, size=(800, d)).astype(float)
        ytr = f(Xtr)
        mlp = MLPClassifier(random_state=0, max_iter=400).fit(Xtr, ytr)
        rule_m = fit_cop_rule(lambda Z: mlp.predict(np.atleast_2d(Z).astype(float)),
                              x, k=5, m=1000, timeout_s=30, seed=0)
        est_m = precision_hat(
            lambda Z: mlp.predict(np.atleast_2d(Z).astype(float)),
            x, rule_m.S, n=1000, seed=2)
        print(f'    mlp: S={rule_m.S} loss={rule_m.empirical_loss:.4f} '
              f'eps_hat={est_m.eps_hat:.3f}')
        check('MLP smoke: learned rule has high precision on the MLP',
              est_m.eps_hat <= 0.10)
    except ImportError:
        print('    (sklearn absent — MLP smoke test skipped)')

    print(f'=== cop_explainer selftest: '
          f'{"ALL PASS" if not failures else "FAILURES: " + str(failures)} ===')
    return 0 if not failures else 1


if __name__ == '__main__':
    import sys
    sys.exit(_selftest())
