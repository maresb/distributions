from math import comb, factorial

import numpy as np
import pytensor.tensor as pt
from pytensor.tensor.special import betaln

from pytensor_distributions.genpareto import (
    _const_like,
    _endpoint_logpdf,
    _gpd_coordinates,
    _gpd_quantile_from_excess,
    _gpd_upper_bound,
    _log1p_div,
)
from pytensor_distributions.helper import ppf_bounds_cont

# Extended Generalized Pareto core. Naveau et al. (2016) extended GPD with
# carrier G(v) = v ** kappa: the CDF is F = H ** kappa with H the GPD CDF, so
# kappa > 0 reshapes the lower tail while xi keeps controlling the upper tail.
# kappa = 1 recovers the plain GPD.


def logpdf(x, mu, sigma, xi, kappa):
    log_s, log_margin, delta, upper, endpoint = _gpd_coordinates(x, mu, sigma, xi)
    interior = (delta > 0) & ~upper
    log_h = pt.log1mexp(pt.switch(interior, log_s, -1))
    value = pt.log(kappa) + (kappa - 1) * log_h + log_s - log_margin - pt.log(sigma)
    lower_value = pt.switch(
        pt.eq(kappa, 1), -pt.log(sigma), pt.switch(pt.lt(kappa, 1), np.inf, -np.inf)
    )
    value = pt.switch(interior, value, pt.switch(pt.eq(delta, 0), lower_value, -np.inf))
    return pt.switch(endpoint, pt.log(kappa) + _endpoint_logpdf(sigma, xi), value)


def _log_probabilities(x, mu, sigma, xi, kappa):
    log_s, _, delta, upper, _ = _gpd_coordinates(x, mu, sigma, xi)
    interior = (delta > 0) & ~upper
    a = pt.switch(interior, log_s, -1.0)
    log_eps = _const_like(np.log(np.finfo(a.dtype).eps), a)
    # log(-log H) tends to log S when S is small. Keep the carrier in logs
    # until after scaling, even when exp(log S) itself would underflow.
    small_s = a < log_eps
    log_h = pt.log1mexp(pt.switch(small_s, -1.0, a))
    log_rate = pt.log(kappa) + pt.switch(small_s, a, pt.log(-log_h))
    logcdf = pt.switch(small_s, -pt.exp(log_rate), kappa * log_h)
    logsf = pt.switch(
        log_rate < log_eps, log_rate, pt.log1mexp(-pt.exp(pt.clip(log_rate, log_eps, 7.0)))
    )
    return (
        pt.switch(upper, 0.0, pt.switch(interior, logcdf, -np.inf)),
        pt.switch(upper, -np.inf, pt.switch(interior, logsf, 0.0)),
    )


def logcdf(x, mu, sigma, xi, kappa):
    return _log_probabilities(x, mu, sigma, xi, kappa)[0]


def logsf(x, mu, sigma, xi, kappa):
    return _log_probabilities(x, mu, sigma, xi, kappa)[1]


def _ext_gpd_excess_from_log_prob(log_q, kappa):
    """GPD excess ``m = -log(1 - F ** (1/kappa))`` from ``log_q = log F``.

    For the carrier ``F = H ** kappa``, the GPD CDF is ``H = exp(log_q / kappa)``
    and its survival ``1 - H``, so ``m = -log(1 - H) = -log1mexp(log_q / kappa)``
    (``pt.log1mexp(a) = log(1 - exp(a))`` for ``a <= 0``). This form stays accurate when ``H``
    rounds to ``1``: for small ``kappa`` the survival ``1 - H`` is tiny, and forming it directly
    would collapse the excess to ``0`` (and the quantile to ``mu``). Shared by
    ``ppf``, ``isf`` and ``rvs`` so the inverses agree. ``log_q`` must be ``<= 0``
    (a log-probability).
    """
    return -pt.log1mexp(log_q / kappa)


def _ext_gpd_excess_from_logit(value, kappa):
    """GPD excess ``m = -log S(x)`` from ``value = logit(F(x))``.

    The ExtGPD CDF is ``F = G ** kappa = sigmoid(value)`` (``G`` the GPD CDF), so the excess
    depends only on ``value`` and ``kappa``:
        m = -log(1 - G) = -log(1 - sigmoid(value) ** (1/kappa)).
    Evaluated stably in two branches split at a large cutoff. Writing
    a := -log G = -log(sigmoid(value)) / kappa gives m = -log1mexp(-a):
      bulk (value < cutoff): m = -log1mexp(log_F / kappa), log_F = log sigmoid(value).
      tail (value >= cutoff, the extreme upper tail where F -> 1): a held in logs as log_a,
        m = -log_a where a < eps, else -log1mexp(-a).
    The unused branch's inputs are clamped (log_F via cutoff, a via log_max) so it stays finite.
    Backs ``ppf_logit``.
    """
    value = pt.true_div(value, 1)
    finfo = np.finfo(value.dtype)
    log_max = float(np.log(finfo.max))  # a = exp(min(log_a, log_max)) stays finite
    log_eps = float(np.log(finfo.eps))  # at or below log_eps, m = -log_a
    # cutoff is the large value where log_F / kappa reaches finfo.tiny. Near the tail
    # log_F = -exp(-value), giving value = -log(tiny) - log(kappa); for kappa <= 1 it is -log(tiny).
    cutoff = np.asarray(-np.log(finfo.tiny), dtype=value.dtype) - pt.maximum(
        np.asarray(0.0, value.dtype), pt.log(kappa)
    )
    t = pt.softplus(value)
    log_F = -pt.softplus(-pt.minimum(value, cutoff))
    m_bulk = _ext_gpd_excess_from_log_prob(log_F, kappa)
    s = pt.exp(-pt.maximum(t, cutoff))  # S_ext, clamped so _log1p_div stays finite
    log_a = -t + pt.log(_log1p_div(-s)) - pt.log(kappa)
    a = pt.exp(pt.minimum(log_a, log_max))
    m_tail = pt.switch(log_a < log_eps, -log_a, -pt.log1mexp(-a))
    return pt.switch(value < cutoff, m_bulk, m_tail)


def ppf(q, mu, sigma, xi, kappa):
    q = pt.as_tensor_variable(q)
    # F = H ** kappa = q  ->  H = q ** (1/kappa); excess m = -log(1 - H), built
    # with log1mexp so a tiny 1 - H (small kappa) is not rounded away to 0.
    excess = _ext_gpd_excess_from_log_prob(pt.log(q), kappa)
    x = _gpd_quantile_from_excess(excess, mu, sigma, xi)
    return ppf_bounds_cont(x, q, mu, _gpd_upper_bound(mu, sigma, xi))


def cdf(x, mu, sigma, xi, kappa):
    return pt.exp(logcdf(x, mu, sigma, xi, kappa))


def pdf(x, mu, sigma, xi, kappa):
    return pt.exp(logpdf(x, mu, sigma, xi, kappa))


def sf(x, mu, sigma, xi, kappa):
    return pt.exp(logsf(x, mu, sigma, xi, kappa))


def isf(x, mu, sigma, xi, kappa):
    x = pt.as_tensor_variable(x)
    # log F = log1p(-x), accurate for tiny x; ppf(1 - x) forms 1 - x first and loses it.
    excess = _ext_gpd_excess_from_log_prob(pt.log1p(-x), kappa)
    quantile = _gpd_quantile_from_excess(excess, mu, sigma, xi)
    return ppf_bounds_cont(quantile, x, _gpd_upper_bound(mu, sigma, xi), mu)


def ppf_logit(y, mu, sigma, xi, kappa):
    """Quantile from the logit-CDF coordinate ``y = logit(F(x)) = logcdf - logsf``.

    Equivalent to ``ppf(expit(y))`` but evaluated in log space (see
    ``_ext_gpd_excess_from_logit``), so it stays accurate when ``F`` saturates to 0 or
    1 deep in either tail. ``y`` is unconstrained (standard ``Logistic`` under the
    model), so no bounds are applied. This is the inverse for a logit-CDF
    (probability-integral) reparametrization, the stable unconstrained transform for
    sampling the extended GPD as a latent variable.
    """
    excess = _ext_gpd_excess_from_logit(y, kappa)
    return _gpd_quantile_from_excess(excess, mu, sigma, xi)


def rvs(mu, sigma, xi, kappa, size=None, random_state=None):
    # Inverse-CDF on a carrier draw u = F; excess = -log(1 - u ** (1/kappa)).
    if size is None:
        size = pt.broadcast_arrays(mu, sigma, xi, kappa)[0].shape
    u = pt.random.uniform(size=size, rng=random_state, return_next_rng=True)[1]
    excess = _ext_gpd_excess_from_log_prob(pt.log(u), kappa)
    return _gpd_quantile_from_excess(excess, mu, sigma, xi)


# Summary statistics. With x = mu + (sigma/xi)(W ** (-xi) - 1), W ~ Beta(1, kappa), the
# r-th central moment is sigma ** r times q_r = E[((W ** (-xi) - 1)/xi - mean) ** r]. Its
# exact Beta-function form is finite (like the plain GPD) only for r xi < 1 -- hence the
# guards -- and cancels catastrophically as xi -> 0. q_r is one function of the order r:
# near xi = 0 a Taylor series in xi (the same idea as _log1p_div / _expm1_div), the exact
# Beta combination otherwise. The series coefficients are generic in r -- joint central
# moments of L = -log(W), each tamed by a 1/(n!) factor so the factorial growth of the raw
# moments does not spoil convergence -- built from L's cumulants
#   kappa_k = (-1) ** (k - 1) [psi ** (k - 1)(kappa + 1) - psi ** (k - 1)(1)].


def _pow_moment(j, xi, kappa):
    """``E[W ** (-j xi)] = kappa * B(1 - j xi, kappa)`` for ``W ~ Beta(1, kappa)``."""
    return pt.exp(pt.log(kappa) + betaln(1 - j * xi, kappa))


def _safe_xi(xi):
    """Nonzero stand-in for ``xi`` used only in the discarded ``xi != 0`` moment branch.

    The Beta-function forms divide by ``xi``; at ``xi = 0`` that branch is not selected,
    but ``pt.switch`` still differentiates it, so a raw ``0/0`` would make the gradient
    ``nan``. The placeholder ``0.1`` lies in ``(0, 1/4)``, keeping every ``B(1 - j xi,
    kappa)`` (``j <= 4``) and ``B(n kappa, 1 - xi)`` finite.
    """
    return pt.switch(pt.eq(xi, 0.0), _const_like(0.1, xi), xi)


_MOMENT_SERIES_TERMS = 8  # Taylor order m = 0..8 in xi.
# |xi| below which the series beats the cancelling exact Beta form: where the series
# truncation error (~ |xi| ** (TERMS + 1)) reaches float64 eps, i.e. eps ** (1 / (TERMS + 1)).
# Both branches use float64 internally, including for float32 inputs.
_MOMENT_SERIES_CUTOFF = 1.8e-2


def _neglogw_raw_moments(kappa):
    """Raw moments ``mu_0 .. mu_{TERMS + 4}`` of ``L = -log(W)``, ``W ~ Beta(1, kappa)``.

    Built once from L's cumulants (polygamma) via the moment-cumulant recursion and shared
    across every moment order.
    """
    n_max = _MOMENT_SERIES_TERMS + 4
    one = _const_like(1.0, kappa)
    cumulants = []
    for k in range(1, n_max + 1):
        d = k - 1
        poly_kappa = pt.psi(kappa + 1) if d == 0 else pt.polygamma(d, kappa + 1)
        cumulants.append((-1.0) ** d * (poly_kappa - pt.polygamma(d, one)))
    raw = [pt.ones_like(kappa)]
    for n in range(1, n_max + 1):
        raw.append(sum(comb(n - 1, i) * cumulants[n - 1 - i] * raw[i] for i in range(n)))
    return raw


def _partitions(total, parts):
    """Non-increasing tuples of ``parts`` positive integers summing to ``total``."""
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(total - parts + 1, 0, -1):
        for rest in _partitions(total - first, parts - 1):
            if rest[0] <= first:
                yield (first, *rest)


def _ordering_count(partition):
    """How many distinct orderings ``partition`` has (its multinomial coefficient)."""
    count = factorial(len(partition))
    for value in set(partition):
        count //= factorial(partition.count(value))
    return count


def _joint_central_moment(orders, raw):
    """Joint central moment ``E[prod_i (L ** o_i - mu_{o_i})]`` by subset inclusion-exclusion."""
    r = len(orders)
    total = 0.0
    for mask in range(1 << r):
        kept = sum(orders[i] for i in range(r) if mask >> i & 1)
        term = raw[kept]
        for i in range(r):
            if not mask >> i & 1:
                term = term * (-raw[orders[i]])
        total = total + term
    return total


def _central_moment_series(r, xi, raw):
    """Taylor series of ``q_r`` in ``xi``; the 1/n! factors tame the factorial raw moments."""
    series = 0.0
    for m in range(_MOMENT_SERIES_TERMS + 1):
        coef = 0.0
        for orders in _partitions(m + r, r):  # orders_i = j_i + 1 >= 1, sum_i = m + r
            denom = 1
            for n in orders:
                denom *= factorial(n)
            jcm = _joint_central_moment(orders, raw)
            coef = coef + _ordering_count(orders) * jcm / _const_like(denom, raw[0])
        series = series + coef * xi**m
    return series


def _central_moment_exact(r, xi, kappa):
    """Exact ``q_r`` from the Beta moments ``E[Y ** i] = M(i xi)``; accurate away from xi = 0."""
    sx = _safe_xi(xi)
    mean_y = _pow_moment(1, sx, kappa)
    central = sum(
        comb(r, i) * (-1.0) ** (r - i) * _pow_moment(i, sx, kappa) * mean_y ** (r - i)
        for i in range(r + 1)
    )
    return central / sx**r


def _scaled_central_moment(r, xi, kappa, raw):
    """``q_r = E[((W ** (-xi) - 1)/xi - mean) ** r]``: series near xi = 0, exact otherwise."""
    return pt.switch(
        pt.lt(pt.abs(xi), _MOMENT_SERIES_CUTOFF),
        _central_moment_series(r, xi, raw),
        _central_moment_exact(r, xi, kappa),
    )


def _central_moments(sigma, xi, kappa):
    """Central moments ``E[(X - mean) ** r]`` for r = 2, 3, 4, computed in float64.

    Both the Beta expressions and the series coefficients cancel in float32.
    Keep the wider precision through normalization, then cast the final statistic.
    Autodiff retains the float64 working precision through these casts.
    """
    sigma, xi, kappa = (pt.cast(p, "float64") for p in (sigma, xi, kappa))
    raw = _neglogw_raw_moments(kappa)
    return tuple(sigma**r * _scaled_central_moment(r, xi, kappa, raw) for r in (2, 3, 4))


def _moment_dtype(*params):
    return pt.true_div(pt.add(*params), 1).dtype


def mean(mu, sigma, xi, kappa):
    dtype = _moment_dtype(mu, sigma, xi, kappa)
    mu, sigma, xi, kappa = (pt.cast(p, "float64") for p in (mu, sigma, xi, kappa))
    raw = _neglogw_raw_moments(kappa)
    # Raw first scaled moment q1 = (M(xi) - 1)/xi = sum_m mu_{1+m}/(1+m)! xi^m near xi = 0.
    series = sum(
        raw[1 + m] / _const_like(factorial(1 + m), raw[1 + m]) * xi**m
        for m in range(_MOMENT_SERIES_TERMS + 1)
    )
    sx = _safe_xi(xi)
    q1 = pt.switch(
        pt.lt(pt.abs(xi), _MOMENT_SERIES_CUTOFF), series, (_pow_moment(1, sx, kappa) - 1) / sx
    )
    return pt.switch(pt.lt(xi, 1), mu + sigma * q1, np.inf).astype(dtype)


def median(mu, sigma, xi, kappa):
    # F = 1/2  ->  H = (1/2) ** (1/kappa); excess m = -log(1 - H).
    log_half = _const_like(np.log(0.5), mu, sigma, xi, kappa)
    excess = _ext_gpd_excess_from_log_prob(log_half, kappa)
    return _gpd_quantile_from_excess(excess, mu, sigma, xi)


def mode(mu, sigma, xi, kappa):
    # In GPD-CDF coordinates the density is proportional to
    # H**(kappa-1) * (1-H)**(1+xi). Its interior maximum has odds
    # H/(1-H) = (kappa-1)/(1+xi).
    interior = pt.gt(kappa, 1) & pt.gt(xi, -1)
    odds = pt.switch(interior, kappa - 1, 0) / pt.switch(interior, 1 + xi, 1)
    value = _gpd_quantile_from_excess(pt.log1p(odds), mu, sigma, xi)
    upper_mode = pt.lt(xi, -1) | (pt.eq(xi, -1) & pt.gt(kappa, 1))
    return pt.switch(upper_mode, _gpd_upper_bound(mu, sigma, xi), value)


def _variance(sigma, xi, kappa):
    variance, _, _ = _central_moments(sigma, xi, kappa)
    return pt.switch(pt.lt(xi, 0.5), variance, np.inf)


def var(mu, sigma, xi, kappa):
    return _variance(sigma, xi, kappa).astype(_moment_dtype(sigma, xi, kappa))


def std(mu, sigma, xi, kappa):
    return pt.sqrt(_variance(sigma, xi, kappa)).astype(_moment_dtype(sigma, xi, kappa))


def skewness(mu, sigma, xi, kappa):
    variance, central3, _ = _central_moments(sigma, xi, kappa)
    value = pt.switch(pt.lt(xi, 1.0 / 3.0), central3 / variance**1.5, np.nan)
    return value.astype(_moment_dtype(mu, sigma, xi, kappa))


def kurtosis(mu, sigma, xi, kappa):
    # Excess kurtosis.
    variance, _, central4 = _central_moments(sigma, xi, kappa)
    value = pt.switch(pt.lt(xi, 0.25), central4 / variance**2 - 3, np.nan)
    return value.astype(_moment_dtype(mu, sigma, xi, kappa))


def entropy(mu, sigma, xi, kappa):
    # -E[log g(X)] in closed form; reduces to the GPD's log(sigma) + xi + 1 at kappa = 1.
    euler_gamma = _const_like(np.euler_gamma, mu, sigma, xi, kappa)
    return (
        pt.log(sigma) - pt.log(kappa) + 1 - 1 / kappa + (1 + xi) * (pt.psi(kappa + 1) + euler_gamma)
    )


# L-moments. With lambda_{r+1} = (sigma kappa / xi) * sum_k p*_{r,k} B(kappa (k+1), 1 - xi)
# (p* the shifted Legendre coefficients), these reduce to the GPD L-moments at kappa = 1
# and are finite for xi < 1. Like the ordinary moments they are a 0/0 at xi = 0 and switch
# to a digamma limit there.


def _lbeta(n, xi, kappa):
    """``B(n kappa, 1 - xi)``."""
    return pt.exp(betaln(n * kappa, 1 - xi))


def lmoment1(mu, sigma, xi, kappa):
    return mean(mu, sigma, xi, kappa)


def lmoment2(mu, sigma, xi, kappa):
    sx = _safe_xi(xi)
    l1, l2 = _lbeta(1, sx, kappa), _lbeta(2, sx, kappa)
    general = sigma * kappa / sx * (2 * l2 - l1)
    limit = sigma * (pt.psi(2 * kappa + 1) - pt.psi(kappa + 1))  # xi -> 0
    value = pt.switch(pt.eq(xi, 0.0), limit, general)
    return pt.switch(pt.lt(xi, 1), value, np.inf)


def lmoment3(mu, sigma, xi, kappa):
    sx = _safe_xi(xi)
    l1, l2, l3 = (_lbeta(n, sx, kappa) for n in (1, 2, 3))
    general = (6 * l3 - 6 * l2 + l1) / (2 * l2 - l1)
    denom = pt.psi(2 * kappa + 1) - pt.psi(kappa + 1)
    limit = (2 * pt.psi(3 * kappa + 1) - 3 * pt.psi(2 * kappa + 1) + pt.psi(kappa + 1)) / denom
    value = pt.switch(pt.eq(xi, 0.0), limit, general)
    return pt.switch(pt.lt(xi, 1), value, np.inf)


def lmoment4(mu, sigma, xi, kappa):
    sx = _safe_xi(xi)
    l1, l2, l3, l4 = (_lbeta(n, sx, kappa) for n in (1, 2, 3, 4))
    general = (20 * l4 - 30 * l3 + 12 * l2 - l1) / (2 * l2 - l1)
    denom = pt.psi(2 * kappa + 1) - pt.psi(kappa + 1)
    limit = (
        5 * pt.psi(4 * kappa + 1)
        - 10 * pt.psi(3 * kappa + 1)
        + 6 * pt.psi(2 * kappa + 1)
        - pt.psi(kappa + 1)
    ) / denom
    value = pt.switch(pt.eq(xi, 0.0), limit, general)
    return pt.switch(pt.lt(xi, 1), value, np.inf)
