import numpy as np
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable

from pytensor_distributions.helper import ppf_bounds_cont


def _const_like(value, *refs):
    """``value`` as a constant in the upcast dtype of ``refs``.

    Numpy scalars and Python floats that are not exactly float32-representable
    (``np.nan``, ``np.log(2)``, ...) are strong float64, so using them directly
    upcasts an otherwise-float32 result to float64. Casting to the inputs' dtype
    keeps each function's output dtype following its arguments while preserving
    full float64 precision when the inputs are float64.
    """
    dtype = pt.true_div(pt.add(*refs), 1).dtype if len(refs) > 1 else pt.true_div(refs[0], 1).dtype
    return np.asarray(value, dtype=dtype)


def _series_cutoff(dtype) -> float:
    """``|u|`` below which the divided-difference helpers switch to their series.

    The exact forms (``log(1 + u) / u``, ``(exp(u) - 1) / u``) are accurate in *value* for every
    ``u != 0``, but autodiff builds their gradient as a difference that cancels with relative
    error ``~ eps / |u|``. Balancing that against the series' ``~ u**4`` gradient-truncation
    error puts the crossover at ``eps ** (1/5)``, so the cutoff tracks the dtype's ``eps`` (a
    constant tuned for float64 is ~50x too small for float32). The crossover optimizes only
    this single gradient (the first derivative), not the value (accurate either way) or higher
    derivatives.
    """
    return float(np.finfo(dtype).eps) ** 0.2


def _log1p_div(u: TensorVariable) -> TensorVariable:
    """``log(1 + u) / u``, analytically continued across ``u = 0``.

    ``safe_u`` keeps the naive branch away from ``u = 0`` where catastrophic cancellation occurs
    for the gradient.
    """
    cutoff = _series_cutoff(u.dtype)
    use_series = pt.lt(pt.abs(u), cutoff)
    series = 1.0 - u / 2.0 + u**2 / 3.0 - u**3 / 4.0 + u**4 / 5.0
    safe_u = pt.switch(use_series, np.asarray(1.0, dtype=u.dtype), u)
    return pt.switch(use_series, series, pt.log1p(safe_u) / safe_u)


def _expm1_div(u: TensorVariable) -> TensorVariable:
    """``(exp(u) - 1) / u``, analytically continued across ``u = 0``.

    Same series and ``safe_u`` construction as ``_log1p_div``.
    """
    cutoff = _series_cutoff(u.dtype)
    use_series = pt.lt(pt.abs(u), cutoff)
    series = 1.0 + u / 2.0 + u**2 / 6.0 + u**3 / 24.0 + u**4 / 120.0
    safe_u = pt.switch(use_series, np.asarray(1.0, dtype=u.dtype), u)
    return pt.switch(use_series, series, pt.expm1(safe_u) / safe_u)


def _gpd_coordinates(x, mu, sigma, xi):
    """Return log survival and log(1 + xi*z) on the interior, plus support masks."""
    delta = pt.sub(x, mu)
    z = delta / sigma
    t = xi * z
    above = pt.lt(xi, 0) & pt.le(1 + t, 0)
    inside = (delta > 0) & ~above & ~pt.isinf(x)
    # A positive shape has a finite log survival even when z or xi*z overflows.
    large = inside & pt.gt(xi, 0) & (pt.isinf(z) | pt.isinf(t))
    regular = inside & ~large
    safe_z = pt.switch(regular, z, 0)
    safe_t = pt.switch(regular, t, 0)
    half_delta = pt.mul(x, 0.5) - pt.mul(mu, 0.5)
    log_delta = pt.switch(
        pt.isinf(delta),
        pt.log(pt.switch(pt.isinf(delta), half_delta, 1)) + _const_like(np.log(2.0), delta),
        pt.log(pt.switch(large, delta, 1)),
    )
    log_product = pt.log(pt.switch(large, xi, 1)) + log_delta - pt.log(sigma)
    log_margin = pt.switch(large, pt.softplus(log_product), pt.log1p(safe_t))
    hazard = pt.switch(large, log_margin / pt.switch(large, xi, 1), safe_z * _log1p_div(safe_t))
    upper_tail = above | pt.eq(x, np.inf)
    return -hazard, log_margin, delta, upper_tail, pt.lt(xi, 0) & pt.eq(1 + t, 0)


def _gpd_quantile_from_excess(excess, mu, sigma, xi):
    """Invert the GPD given ``excess = -log(survival) = m >= 0``.

    ``z = expm1(xi m) / xi = m * expm1_div(xi m)`` -> ``value = mu + sigma z``.
    Reused by the ``icdf`` method (``excess`` from a CDF probability) and by the
    random Op (``excess`` from a uniform survival draw).
    """
    return mu + sigma * excess * _expm1_div(xi * excess)


def _gpd_upper_bound(mu, sigma, xi):
    """Right endpoint of the GPD support: ``mu - sigma / xi`` for xi < 0, else +inf."""
    has_bounded_support = pt.lt(xi, 0)
    div_xi = pt.switch(has_bounded_support, xi, 1.0)  # 1.0: arbitrary finite, discarded below
    return pt.switch(has_bounded_support, mu - sigma / div_xi, np.inf)


def _endpoint_logpdf(sigma, xi):
    return pt.switch(pt.lt(xi, -1), np.inf, pt.switch(pt.eq(xi, -1), -pt.log(sigma), -np.inf))


def logpdf(x, mu, sigma, xi):
    log_s, log_margin, delta, upper, endpoint = _gpd_coordinates(x, mu, sigma, xi)
    value = pt.switch((delta >= 0) & ~upper, log_s - log_margin - pt.log(sigma), -np.inf)
    return pt.switch(endpoint, _endpoint_logpdf(sigma, xi), value)


def logcdf(x, mu, sigma, xi):
    log_s, _, delta, upper, _ = _gpd_coordinates(x, mu, sigma, xi)
    return pt.switch(upper, 0.0, pt.switch(delta > 0, pt.log1mexp(log_s), -np.inf))


def logsf(x, mu, sigma, xi):
    log_s, _, _, upper, _ = _gpd_coordinates(x, mu, sigma, xi)
    return pt.switch(upper, -np.inf, log_s)


def ppf(q, mu, sigma, xi):
    q = pt.as_tensor_variable(q)
    x = _gpd_quantile_from_excess(-pt.log1p(-q), mu, sigma, xi)  # excess = -log(1 - q)
    return ppf_bounds_cont(x, q, mu, _gpd_upper_bound(mu, sigma, xi))


def cdf(x, mu, sigma, xi):
    return pt.exp(logcdf(x, mu, sigma, xi))


def pdf(x, mu, sigma, xi):
    return pt.exp(logpdf(x, mu, sigma, xi))


def sf(x, mu, sigma, xi):
    return pt.exp(logsf(x, mu, sigma, xi))


def isf(x, mu, sigma, xi):
    x = pt.as_tensor_variable(x)
    # m = -log(x) directly; ppf(1 - x) would lose a tiny x to the 1 - x rounding.
    quantile = _gpd_quantile_from_excess(-pt.log(x), mu, sigma, xi)
    return ppf_bounds_cont(quantile, x, _gpd_upper_bound(mu, sigma, xi), mu)


def ppf_logit(y, mu, sigma, xi):
    """Quantile from the logit-CDF coordinate ``y = logit(F(x)) = logcdf - logsf``.

    Equivalent to ``ppf(expit(y))`` but evaluated without forming ``expit(y)``, so it
    stays accurate when ``F`` saturates to 0 or 1 deep in either tail. ``y`` is
    unconstrained (standard ``Logistic`` under the model), so no bounds are applied.
    This is the inverse for a logit-CDF (probability-integral) reparametrization, the
    stable unconstrained transform for sampling the GPD as a latent variable.
    """
    # excess m = -log S = -log(1 - expit(y)) = softplus(y).
    return _gpd_quantile_from_excess(pt.softplus(y), mu, sigma, xi)


def rvs(mu, sigma, xi, size=None, random_state=None):
    # Inverse-CDF on a survival draw: excess = -log(v) avoids the 1 - v cancellation.
    if size is None:
        size = pt.broadcast_arrays(mu, sigma, xi)[0].shape
    v = pt.random.uniform(size=size, rng=random_state, return_next_rng=True)[1]
    return _gpd_quantile_from_excess(-pt.log(v), mu, sigma, xi)


# Summary statistics. The r-th moment is finite only for xi < 1 / r, so each is
# guarded and returns inf / nan past its threshold (matching scipy's genpareto).


def mean(mu, sigma, xi):
    return pt.switch(pt.lt(xi, 1), mu + sigma / (1 - xi), np.inf)


def median(mu, sigma, xi):
    # Quantile at p = 1/2: excess m = -log(1 - 1/2) = log 2.
    return _gpd_quantile_from_excess(_const_like(np.log(2.0), mu, sigma, xi), mu, sigma, xi)


def mode(mu, sigma, xi):
    # Density is monotone decreasing for xi >= -1 (mode at the lower endpoint mu);
    # for xi < -1 it diverges at the finite upper endpoint.
    mu_b, _, xi_b = pt.broadcast_arrays(mu, sigma, xi)
    return pt.switch(pt.ge(xi_b, -1), mu_b, _gpd_upper_bound(mu, sigma, xi))


def var(mu, sigma, xi):
    return pt.switch(pt.lt(xi, 0.5), sigma**2 / ((1 - xi) ** 2 * (1 - 2 * xi)), np.inf)


def std(mu, sigma, xi):
    return pt.sqrt(var(mu, sigma, xi))


def skewness(mu, sigma, xi):
    value = 2 * (1 + xi) * pt.sqrt(1 - 2 * xi) / (1 - 3 * xi)
    return pt.switch(pt.lt(xi, 1.0 / 3.0), value, _const_like(np.nan, mu, sigma, xi))


def kurtosis(mu, sigma, xi):
    # Excess kurtosis.
    value = 3 * (1 - 2 * xi) * (2 * xi**2 + xi + 3) / ((1 - 3 * xi) * (1 - 4 * xi)) - 3
    return pt.switch(pt.lt(xi, 0.25), value, _const_like(np.nan, mu, sigma, xi))


def entropy(mu, sigma, xi):
    return pt.log(sigma) + xi + 1


# L-moments. With Hosking's shape k = -xi, the GPD has lambda_1 = mean,
# lambda_2 = sigma / ((1 - xi)(2 - xi)), tau_3 = (1 + xi)/(3 - xi),
# tau_4 = (1 + xi)(2 + xi)/((3 - xi)(4 - xi)); all finite for xi < 1.


def lmoment1(mu, sigma, xi):
    return mean(mu, sigma, xi)


def lmoment2(mu, sigma, xi):
    return pt.switch(pt.lt(xi, 1), sigma / ((1 - xi) * (2 - xi)), np.inf)


def lmoment3(mu, sigma, xi):
    return pt.switch(pt.lt(xi, 1), (1 + xi) / (3 - xi), np.inf)


def lmoment4(mu, sigma, xi):
    return pt.switch(pt.lt(xi, 1), (1 + xi) * (2 + xi) / ((3 - xi) * (4 - xi)), np.inf)
