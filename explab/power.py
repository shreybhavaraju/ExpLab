# Power and minimum detectable effect, normal approximation.

import numpy as np
from scipy import stats


def mde(var, n_t, n_c, alpha=0.05, power=0.8):
    """Minimum detectable effect: the smallest true difference that a two-sided test at level
    alpha catches with probability `power`. Smaller effects can still come out significant,
    the test just usually misses them.

        MDE = (z_{1-alpha/2} + z_power) * sqrt(var * (1/n_t + 1/n_c))

    var is the per-user variance of the metric, p (1 - p) for a 0/1 metric."""
    se = np.sqrt(var * (1 / n_t + 1 / n_c))
    return (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se


def power_at(effect, var, n_t, n_c, alpha=0.05):
    """Chance the two-sided z-test is significant when the true difference is `effect`:

        P(Z > z - effect/se) + P(Z < -z - effect/se)

    The second term is a 'significant' result in the wrong direction. It only matters near 0,
    where it makes power_at(0) = alpha."""
    se = np.sqrt(var * (1 / n_t + 1 / n_c))
    z = stats.norm.ppf(1 - alpha / 2)
    return stats.norm.sf(z - effect / se) + stats.norm.cdf(-z - effect / se)

