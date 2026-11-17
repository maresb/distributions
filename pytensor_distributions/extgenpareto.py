from math import comb

import numpy as np
import pytensor.tensor as pt
from scipy.special import roots_legendre

from pytensor_distributions.genpareto import (
    _const_like,
    _draw_size,
    _endpoint_logpdf,
    _expm1_div,
    _gpd_coordinates,
    _gpd_quantile_from_excess,
    _gpd_upper_bound,
    _log1p_div,
    _moment_dtype,
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
        pt.eq(kappa, 1),
        log_s - log_margin - pt.log(sigma),
        pt.switch(pt.lt(kappa, 1), np.inf, -np.inf),
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
    log_rate = pt.log(pt.cast(kappa, pt.add(kappa, a).dtype)) + pt.switch(
        small_s, a, pt.log(-log_h)
    )
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
    size = _draw_size(size, mu, sigma, xi, kappa)
    u = pt.random.uniform(size=size, rng=random_state, return_next_rng=True)[1]
    excess = _ext_gpd_excess_from_log_prob(pt.log(u), kappa)
    return _gpd_quantile_from_excess(excess, mu, sigma, xi)


# For E ~ Exp(1), W = 1 - exp(-E/kappa) is Beta(1, kappa) and the
# standardized quantile is expm1(-xi * log W) / xi. Near zero, integrate
# this smooth function directly instead of differencing nearly equal Beta
# moments. A fixed rule makes the graph differentiable, including at xi=0.
# With E=min(1,kappa)*exp(s), the lower tail of the fourth-moment integrand
# decays at least as exp((1-4*0.1)*s); the upper tail decays exponentially
# in exp(s), either in the density or in the quantile.
_NODES, _WEIGHTS = roots_legendre(160)
_LOG_E = (5.0 + 90.0) / 2 * _NODES + (5.0 - 90.0) / 2
_E = np.exp(_LOG_E)
_NEG_LOG_W = -np.log(-np.expm1(-_E))
_NEG_LOG_W[_E > 1] = -np.log1p(-np.exp(-_E[_E > 1]))
_WEIGHTS = (5.0 + 90.0) / 2 * _WEIGHTS * _E


def _quadrature(xi, kappa):
    """Weights and quantiles centered at log(max(kappa, 1)) in log-W space."""
    scale = pt.switch(kappa >= 1, 1.0, kappa)[..., None]
    v = _E / pt.switch(kappa >= 1, kappa, 1.0)[..., None]
    # For large kappa subtract log(kappa) analytically, before rounding.
    centered = pt.switch(
        (kappa >= 1)[..., None],
        -_LOG_E - pt.log(_expm1_div(-v)),
        _NEG_LOG_W,
    )
    quantiles = centered * _expm1_div(xi[..., None] * centered)
    return scale * _WEIGHTS * pt.exp(-scale * _E), quantiles


def _stirling_correction(z):
    inv = 1 / z
    return inv * (1 / 12 + inv**2 * (-1 / 360 + inv**2 * (1 / 1260 - inv**2 / 1680)))


def _log_power(t, kappa, anchor):
    """Log E[W**(-t)] - t*anchor, avoiding cancellation for large kappa."""
    z = 1 + kappa
    d = z - t
    direct = pt.gammaln(z) - pt.gammaln(d) - t * anchor
    # Subtract the Stirling expansions analytically. At z,d >= 16 the
    # first omitted term in each log-gamma is below 1.3e-14.
    large = (z >= 16) & (d >= 16)
    zs, ds = pt.switch(large, z, 16.0), pt.switch(large, d, 16.0)
    ratio = (
        t * pt.log1p(1 / pt.maximum(kappa, 1))
        + (ds - 0.5) * pt.log1p(t / ds)
        - t
        + _stirling_correction(zs)
        - _stirling_correction(ds)
    )
    return pt.gammaln(1 - t) + pt.switch(large, ratio, direct)


def _unit_stat(xi, kappa, order, mean_origin=0.0):
    """Mean, variance, skewness or excess kurtosis at unit scale and zero location."""
    xi, kappa = pt.cast(xi, "float64"), pt.cast(kappa, "float64")
    near = pt.abs(xi) <= 0.1
    small_xi = pt.switch(near, xi, 0.0)
    # Both branches of a symbolic switch can be evaluated and differentiated.
    t = pt.switch(~near & (xi < 1 / order), xi, 0.125)
    anchor = pt.log(pt.switch(kappa >= 1, kappa, 1.0))
    weights, q = _quadrature(small_xi, kappa)
    log_first = _log_power(t, kappa, anchor)
    first = (weights * q).sum(axis=-1)

    if order == 1:
        offset = anchor - mean_origin
        approx = offset * _expm1_div(small_xi * offset) + pt.exp(small_xi * offset) * first
        exact = pt.expm1(log_first + t * offset) / t
        return pt.switch(xi < 1, pt.switch(near, approx, exact), np.inf)

    # The exact expressions use moments of W**(-xi) relative to its mean.
    # expm1 removes the common constant from the binomial central moments.
    ratios = [
        pt.expm1(_log_power(j * t, kappa, anchor) - j * log_first) for j in range(2, order + 1)
    ]
    # For kappa < 1, the omitted upper E tail may carry probability mass,
    # but its quantile is zero. Its contribution to the raw moments is negligible.
    raw = [pt.ones_like(first), first]
    raw.extend((weights * q**j).sum(axis=-1) for j in range(2, order + 1))

    var_q = raw[2] - first**2
    if order == 2:
        exact = pt.exp(2 * (log_first + t * anchor)) * ratios[0] / t**2
        return pt.switch(
            xi < 0.5, pt.switch(near, pt.exp(2 * small_xi * anchor) * var_q, exact), np.inf
        )
    exact_central = sum(
        comb(order, j) * (-1) ** (order - j) * ratios[j - 2] for j in range(2, order + 1)
    )
    exact = pt.sign(t) ** order * exact_central / ratios[0] ** (order / 2)
    central = sum(comb(order, j) * (-first) ** (order - j) * raw[j] for j in range(order + 1))
    approx = central / var_q ** (order / 2)
    return pt.switch(
        xi < 1 / order, pt.switch(near, approx, exact) - (3 if order == 4 else 0), np.nan
    )


def mean(mu, sigma, xi, kappa):
    return (mu + sigma * _unit_stat(xi, kappa, 1)).astype(_moment_dtype(mu, sigma, xi, kappa))


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


def var(mu, sigma, xi, kappa):
    return (pt.cast(sigma, "float64") ** 2 * _unit_stat(xi, kappa, 2)).astype(
        _moment_dtype(sigma, xi, kappa)
    )


def std(mu, sigma, xi, kappa):
    return (sigma * pt.sqrt(_unit_stat(xi, kappa, 2))).astype(_moment_dtype(sigma, xi, kappa))


def skewness(mu, sigma, xi, kappa):
    value = pt.broadcast_arrays(_unit_stat(xi, kappa, 3), mu, sigma)[0]
    return value.astype(_moment_dtype(mu, sigma, xi, kappa))


def kurtosis(mu, sigma, xi, kappa):
    value = pt.broadcast_arrays(_unit_stat(xi, kappa, 4), mu, sigma)[0]
    return value.astype(_moment_dtype(mu, sigma, xi, kappa))


def entropy(mu, sigma, xi, kappa):
    # -E[log g(X)] in closed form; reduces to the GPD's log(sigma) + xi + 1 at kappa = 1.
    euler_gamma = _const_like(np.euler_gamma, mu, sigma, xi, kappa)
    return (
        pt.log(sigma) - pt.log(kappa) + 1 - 1 / kappa + (1 + xi) * (pt.psi(kappa + 1) + euler_gamma)
    )


# The maximum of n draws has the same distribution with carrier n*kappa.
# The L-moments are linear combinations of these expected maxima.
def lmoment1(mu, sigma, xi, kappa):
    return mean(mu, sigma, xi, kappa)


def _unit_lmoment(xi, kappa, order):
    xi = pt.cast(xi, "float64")
    kappa = pt.cast(kappa, "float64")
    n = np.arange(1, order + 1, dtype="float64")
    # Strip a common location and scale from the expected maxima. Far into
    # the bounded tail their ordinary means can round to the same endpoint.
    origin = pt.log(pt.switch(kappa >= 1, kappa, 1.0))
    maxima = _unit_stat(
        pt.switch(xi < 1, xi, 0.0)[..., None], kappa[..., None] * n, 1, origin[..., None]
    )
    scale = maxima[..., 1] - maxima[..., 0]
    if order == 2:
        value = pt.exp(xi * origin) * scale
    elif order == 3:
        value = (2 * maxima[..., 2] - 3 * maxima[..., 1] + maxima[..., 0]) / scale
    else:
        value = (
            5 * maxima[..., 3] - 10 * maxima[..., 2] + 6 * maxima[..., 1] - maxima[..., 0]
        ) / scale
    return pt.switch(xi < 1, value, np.inf)


def lmoment2(mu, sigma, xi, kappa):
    return (sigma * _unit_lmoment(xi, kappa, 2)).astype(_moment_dtype(sigma, xi, kappa))


def lmoment3(mu, sigma, xi, kappa):
    return _unit_lmoment(xi, kappa, 3).astype(_moment_dtype(xi, kappa))


def lmoment4(mu, sigma, xi, kappa):
    return _unit_lmoment(xi, kappa, 4).astype(_moment_dtype(xi, kappa))
