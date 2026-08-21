"""
bb_metrics.py — J1 of the NN tier (research/PLAN_NN_TIER.md).

Black-box precision instrument. The ONLY metric family of the NN tier:
everything here touches the model exclusively through `predict_fn` (query
access), per the guiding criterion of the report (§1).

Definitions implemented verbatim from the papers:

- Precision error  eps_{f,x,D}(S) = P_{z~D}[ f(z) != f(x) | z_S = x_S ]
  [Koriche et al., Eq. (1)].  D = uniform over {0,1}^d (their §4.1 setup)
  for the oracle estimator; D = empirical training distribution for the
  CP-2 "training" variant.

- MC-sampling estimator [Izza et al., §4.2]: FIX the features of S to x's
  values and sample the complement uniformly — NOT sample-then-filter.
  (The legacy compute_precision_error_oracle in run_comparison_bt.py:714-742
  samples the full box then filters; for a term with many conditions the
  effective sample count collapses and the Hoeffding bound is applied to a
  random, tiny N. This module is the corrective.)

- Hoeffding bound [Izza et al., Prop. 1]:  N >= (1/(2 eps^2)) ln(2/delta)
  guarantees P(|X - mu| >= eps) <= delta.  half_width(n, delta) is the
  inverse reading used for reporting:  eps = sqrt(ln(2/delta) / (2 n)).

- WeakPAXp test [Izza et al., Eq. (4)]:  Pr(kappa(z)=c | z_S = x_S) >= T.

Conventions:
- x_bin : 1-D numpy array of {0,1}, length d (the binarized instance).
- S     : iterable of 0-based feature indices (the explanation's fixed set).
- predict_fn : callable, (n, d) {0,1} array -> (n,) label array. Batched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    'hoeffding_n', 'hoeffding_half_width', 'sample_conditional',
    'precision_hat', 'precision_training', 'is_weak_paxp', 'PrecisionEstimate',
]


# ── Hoeffding machinery [Izza et al., Prop. 1] ─────────────────────────────────

def hoeffding_n(eps: float, delta: float) -> int:
    """Minimum N such that P(|X_bar - mu| >= eps) <= delta."""
    return int(math.ceil(math.log(2.0 / delta) / (2.0 * eps * eps)))


def hoeffding_half_width(n: int, delta: float) -> float:
    """Half-width eps of the (1-delta) confidence interval after n samples."""
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


# ── Conditional sampler (fix S, sample the complement) [Izza §4.2] ─────────────

def sample_conditional(x_bin: np.ndarray, S, n: int,
                       rng: np.random.Generator) -> np.ndarray:
    """
    n samples z ~ Uniform({0,1}^d | z_S = x_S).

    Every returned row satisfies the condition BY CONSTRUCTION, so all n
    samples are effective (unlike sample-then-filter, whose effective count
    is Binomial(n, 2^-|S|) for uniform bits).
    """
    x_bin = np.asarray(x_bin).ravel()
    d = x_bin.shape[0]
    S = np.asarray(sorted(set(int(j) for j in S)), dtype=int)
    if S.size and (S.min() < 0 or S.max() >= d):
        raise ValueError(f'S contains indices outside [0, {d})')
    Z = rng.integers(0, 2, size=(n, d), dtype=np.int8)
    if S.size:
        Z[:, S] = x_bin[S].astype(np.int8)
    return Z


# ── Precision estimator ────────────────────────────────────────────────────────

@dataclass
class PrecisionEstimate:
    eps_hat: float        # estimated precision error  eps_{f,x}(S)
    precision: float      # 1 - eps_hat  (the Pr of WeakPAXp, Eq. (4) [Izza])
    half_width: float     # Hoeffding half-width at (n, delta)
    n: int                # effective sample count (== requested n, by design)
    delta: float
    pred_class: int       # f(x), the class being explained


def precision_hat(predict_fn, x_bin: np.ndarray, S, n: int = 1000,
                  delta: float = 0.05, seed: int | None = 42,
                  rng: np.random.Generator | None = None) -> PrecisionEstimate:
    """
    (eps, delta)-estimate of eps_{f,x}(S) under D = Uniform({0,1}^d),
    by MC-sampling with S fixed [Izza §4.2; Koriche Eq. (1)].

    n=1000 matches the evaluation protocol of [Koriche §4.1] and satisfies
    the Hoeffding requirement N >= 738 for eps = delta = 0.05.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    x_bin = np.asarray(x_bin).ravel()
    pred = int(np.asarray(predict_fn(x_bin.reshape(1, -1))).ravel()[0])
    Z = sample_conditional(x_bin, S, n, rng)
    labels = np.asarray(predict_fn(Z)).ravel()
    eps_hat = float(np.mean(labels != pred))
    return PrecisionEstimate(
        eps_hat=eps_hat, precision=1.0 - eps_hat,
        half_width=hoeffding_half_width(n, delta), n=n, delta=delta,
        pred_class=pred)


def precision_training(predict_fn, x_bin: np.ndarray, S,
                       X_bin: np.ndarray) -> tuple[float | None, int]:
    """
    CP-2 "training" variant: same conditional, D = empirical distribution of
    the (binarized) training set — the arbitrary-D case Eq. (1) explicitly
    allows [Koriche p.437]. Returns (eps_hat, n_neighbors); (None, 0) when no
    training row agrees with x on S (no bound is claimed here — secondary
    metric, kept for continuity with the BT tier's CP-2 column).
    """
    x_bin = np.asarray(x_bin).ravel()
    X_bin = np.asarray(X_bin)
    S = sorted(set(int(j) for j in S))
    pred = int(np.asarray(predict_fn(x_bin.reshape(1, -1))).ravel()[0])
    mask = np.ones(len(X_bin), dtype=bool)
    for j in S:
        mask &= (X_bin[:, j] == x_bin[j])
    n_nb = int(mask.sum())
    if n_nb == 0:
        return None, 0
    labels = np.asarray(predict_fn(X_bin[mask])).ravel()
    return float(np.mean(labels != pred)), n_nb


def is_weak_paxp(est: PrecisionEstimate, T: float) -> bool:
    """WeakPAXp(S) at threshold T [Izza Eq. (4)], on the point estimate."""
    return est.precision >= T


# ── Self-tests (python bb_metrics.py --selftest) ───────────────────────────────

def _selftest() -> int:
    rng_global = np.random.default_rng(0)
    failures = []

    def check(name, cond):
        print(f'  [{"PASS" if cond else "FAIL"}] {name}')
        if not cond:
            failures.append(name)

    d, J = 20, [2, 5, 11]                       # planted monotone monomial
    def f(Z):                                   # f(z) = AND_{j in J} z_j
        Z = np.atleast_2d(Z)
        return np.all(Z[:, J] == 1, axis=1).astype(int)

    x = np.ones(d, dtype=np.int8)               # f(x) = 1

    # 1. S = J  ->  sufficient by construction  ->  eps_hat exactly 0
    est = precision_hat(f, x, J, n=1000, seed=1)
    check('planted monomial: eps(S=J) == 0', est.eps_hat == 0.0)

    # 2. S = J minus one var -> f(z)=1 iff the free planted bit is 1 -> eps = 0.5
    est2 = precision_hat(f, x, J[:-1], n=4000, seed=2)
    check(f'drop one literal: eps ~ 0.5 (got {est2.eps_hat:.3f})',
          abs(est2.eps_hat - 0.5) <= est2.half_width)

    # 3. S = {} -> eps = P(f(z) != 1) = 1 - 2^-3 = 0.875
    est3 = precision_hat(f, x, [], n=4000, seed=3)
    check(f'empty S: eps ~ 0.875 (got {est3.eps_hat:.3f})',
          abs(est3.eps_hat - 0.875) <= est3.half_width)

    # 4. Hoeffding numbers match the paper-cited values
    check('hoeffding_n(0.05, 0.05) == 738', hoeffding_n(0.05, 0.05) == 738)
    check('half_width(1000, 0.05) ~ 0.0429',
          abs(hoeffding_half_width(1000, 0.05) - 0.0429) < 1e-3)

    # 5. Determinism under a fixed seed
    a = precision_hat(f, x, J[:-1], n=500, seed=7).eps_hat
    b = precision_hat(f, x, J[:-1], n=500, seed=7).eps_hat
    check('deterministic under fixed seed', a == b)

    # 6. Effective-sample argument vs the legacy sample-then-filter scheme:
    #    with |S|=12 fixed bits, a filtered scheme keeps Binomial(n, 2^-12)
    #    samples (~0.24 expected out of 1000); the conditional sampler keeps n.
    S12 = list(range(12))
    Z = sample_conditional(x, S12, 1000, np.random.default_rng(4))
    agree = np.all(Z[:, S12] == 1, axis=1)
    n_filter = int(np.sum(np.all(
        rng_global.integers(0, 2, size=(1000, d))[:, S12] == 1, axis=1)))
    check(f'conditional keeps n=1000 effective (filter kept {n_filter})',
          bool(agree.all()) and n_filter < 5)

    # 7. training-D variant: rows equal to x on S all predict 1 -> eps = 0
    X_tr = rng_global.integers(0, 2, size=(500, d), dtype=np.int8)
    eps_tr, n_nb = precision_training(f, x, J, X_tr)
    check(f'training variant conditional (n_neighbors={n_nb})',
          (eps_tr == 0.0 and n_nb > 0) or n_nb == 0)

    # 8. WeakPAXp helper
    check('weak_paxp at T=0.95', is_weak_paxp(est, 0.95)
          and not is_weak_paxp(est3, 0.95))

    print(f'=== bb_metrics selftest: '
          f'{"ALL PASS" if not failures else "FAILURES: " + str(failures)} ===')
    return 0 if not failures else 1


if __name__ == '__main__':
    import sys
    sys.exit(_selftest())
