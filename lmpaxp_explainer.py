"""
lmpaxp_explainer.py — J3 of the NN tier (research/PLAN_NN_TIER.md).

Locally-minimal probabilistic abductive explanation (LmPAXp) computed 100%
black-box, implementing Izza, Meel, Marques-Silva (ECAI 2024):

  - WeakPAXp test, Eq. (4):   Pr_z( f(z) = f(x) | z_S = x_S ) >= T
  - checker = MC-sampling (§4.2): fix S, sample the complement, Hoeffding
    (Prop. 1) — query access only, NO encoding, NO ApproxMC. This is the
    variant that makes the method applicable to a dense MLP (report §4,
    correction of 2026-07-02).
  - search = deletion-based linear pass (Alg. 1): start from the full feature
    set (the AXp initialization requires a formal reasoner — unavailable on a
    black box), traverse features in heuristic order, drop feature j whenever
    the reduced set still passes the WeakPAXp test.
  - heuristic order (§4.3): features sorted by an MC importance score,
    least-contributing first — empirically yields smaller explanations.
  - delta budget: Alg. 1 line 5 calls the counter at confidence delta/m; we
    split delta the same way across the d deletion tests (strict), and spend
    a final fresh-seed re-certification at full delta on the returned S.

Output guarantee (per instance, unlike the COP's in-expectation bound):
  Pr[ Pr_x(S) >= T - eps_test ] >= 1 - delta   [Izza, Alg. 1 output spec],
  with eps_test the Hoeffding half-width of the per-test sample size.

Local minimality is w.r.t. the single deletion pass, exactly as in Alg. 1
(WeakPAXp is non-monotone [Izza §2.5], so no subset-minimality is claimed —
in practice locally-minimal ~ subset-minimal in 99.8% of DT cases [Izza §1]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bb_metrics import hoeffding_half_width, hoeffding_n, precision_hat

__all__ = ['LmPAXp', 'fit_lmpaxp']


@dataclass
class LmPAXp:
    S: list                   # remaining fixed features (0-based indices)
    literals: list            # [(j, x_j)] body, same convention as CopRule
    head: int                 # f(x)
    T: float                  # precision threshold
    precision_cert: float     # fresh-seed re-certified precision of final S
    half_width_cert: float    # Hoeffding half-width of the certification
    is_weak_paxp: bool        # precision_cert >= T (point estimate)
    n_per_test: int           # samples per deletion test
    delta: float              # total confidence budget
    n_tests: int              # deletion tests performed
    order: list               # feature traversal order used
    start_size: int           # |initial S| (== d when starting from F)


def _mc_importance(predict_fn, x_bin, S_all, n_score, rng):
    """
    MC heuristic score per feature [Izza §4.3]: estimated precision of
    S_all \\ {j} — HIGH precision means j contributes little (safe to drop
    early). Returned as scores to sort DESCENDING (least contributing first).
    """
    scores = {}
    for j in S_all:
        est = precision_hat(predict_fn, x_bin, [i for i in S_all if i != j],
                            n=n_score, rng=rng, seed=None)
        scores[j] = est.precision
    return scores


def fit_lmpaxp(predict_fn, x_bin: np.ndarray, T: float = 0.95,
               delta: float = 0.05, eps_test: float = 0.05,
               order: str = 'mc', n_score: int = 200,
               init_S=None, seed: int = 42) -> LmPAXp:
    """
    Compute an LmPAXp for (predict_fn, x_bin) at threshold T.

    T=0.95 is the RF regime of [Izza §5] (project decision 2026-07-02);
    delta=0.05 and eps_test=0.05 mirror the existing rc_bt PAC parameters.
    Per-test sample size is the Hoeffding N for (eps_test, delta/m) — the
    strict split of Alg. 1; the plan says: start strict, measure the cost.

    init_S: optional starting subset (e.g. another arm's S); default = all
    features, the pure black-box entry point of Alg. 1.
    """
    x_bin = np.asarray(x_bin).ravel().astype(np.int8)
    d = x_bin.shape[0]
    rng = np.random.default_rng(seed)
    pred = int(np.asarray(predict_fn(x_bin.reshape(1, -1))).ravel()[0])

    S = sorted(set(range(d) if init_S is None else (int(j) for j in init_S)))
    start_size = len(S)
    m_tests = max(len(S), 1)
    n_test = hoeffding_n(eps_test, delta / m_tests)     # delta/m split, Alg. 1 l.5

    if order == 'mc':                                    # [Izza §4.3]
        sc = _mc_importance(predict_fn, x_bin, S, n_score, rng)
        traversal = sorted(S, key=lambda j: -sc[j])      # least contributing first
    elif order == 'index':
        traversal = list(S)
    else:
        raise ValueError("order must be 'mc' or 'index'")

    n_done = 0
    for j in traversal:                                  # single pass, Alg. 1
        cand = [i for i in S if i != j]
        est = precision_hat(predict_fn, x_bin, cand, n=n_test,
                            rng=rng, seed=None)
        n_done += 1
        if est.precision >= T:                           # WeakPAXp(S\{j})? l.7
            S = cand                                     # l.8

    # fresh-seed re-certification of the final S at full delta
    cert = precision_hat(predict_fn, x_bin, S, n=max(n_test, 1000),
                         delta=delta, seed=seed + 10_007)

    return LmPAXp(
        S=S, literals=[(j, int(x_bin[j])) for j in S], head=pred, T=T,
        precision_cert=cert.precision, half_width_cert=cert.half_width,
        is_weak_paxp=cert.precision >= T, n_per_test=n_test, delta=delta,
        n_tests=n_done, order=traversal, start_size=start_size)


# ── Self-tests (python lmpaxp_explainer.py --selftest) ─────────────────────────

def _selftest() -> int:
    failures = []

    def check(name, cond):
        print(f'  [{"PASS" if cond else "FAIL"}] {name}')
        if not cond:
            failures.append(name)

    d, J = 20, [2, 5, 11]
    def f(Z):
        Z = np.atleast_2d(Z)
        return np.all(Z[:, J] == 1, axis=1).astype(int)
    x = np.ones(d, dtype=np.int8)

    # 1. Planted monomial at T=0.95: dropping any j in J collapses precision
    #    to 0.5 < T, dropping any other j keeps 1.0 -> final S == J exactly.
    r = fit_lmpaxp(f, x, T=0.95, seed=0)
    print(f'    T=0.95: S={r.S} start={r.start_size} n/test={r.n_per_test} '
          f'prec_cert={r.precision_cert:.3f}')
    check('planted monomial: LmPAXp == J exactly', sorted(r.S) == J)
    check('certified WeakPAXp at T (fresh seed)',
          r.is_weak_paxp and r.precision_cert >= r.T - r.half_width_cert)
    check('shorter than the start set', len(r.S) < r.start_size)

    # 2. index order gives a WeakPAXp too (order affects size, not validity)
    r_idx = fit_lmpaxp(f, x, T=0.95, order='index', seed=0)
    check('index order also certified', r_idx.is_weak_paxp)

    # 3. size/precision trade-off: at T=0.45, one planted literal becomes
    #    droppable (precision 0.5 >= 0.45) -> strictly smaller S.
    r_low = fit_lmpaxp(f, x, T=0.45, seed=0)
    print(f'    T=0.45: S={r_low.S} prec_cert={r_low.precision_cert:.3f}')
    check('lower T -> strictly shorter S (size/precision trade-off)',
          len(r_low.S) < len(r.S))

    # 4. init_S: starting from a superset of J lands on J
    r_init = fit_lmpaxp(f, x, T=0.95, init_S=J + [0, 7, 13], seed=0)
    check('init_S superset reduces to J', sorted(r_init.S) == J)

    # 5. MLP smoke test — the actual NN-tier loop, black-box end to end
    try:
        from sklearn.neural_network import MLPClassifier
        rng = np.random.default_rng(1)
        Xtr = rng.integers(0, 2, size=(800, d)).astype(float)
        mlp = MLPClassifier(random_state=0, max_iter=400).fit(Xtr, f(Xtr))
        pf = lambda Z: mlp.predict(np.atleast_2d(Z).astype(float))
        r_mlp = fit_lmpaxp(pf, x, T=0.95, seed=0)
        print(f'    mlp: S={r_mlp.S} prec_cert={r_mlp.precision_cert:.3f}')
        check('MLP smoke: certified WeakPAXp at T=0.95', r_mlp.is_weak_paxp)
    except ImportError:
        print('    (sklearn absent — MLP smoke test skipped)')

    print(f'=== lmpaxp_explainer selftest: '
          f'{"ALL PASS" if not failures else "FAILURES: " + str(failures)} ===')
    return 0 if not failures else 1


if __name__ == '__main__':
    import sys
    sys.exit(_selftest())
