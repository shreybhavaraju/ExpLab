# Effect estimates for one metric: absolute difference, relative lift, CIs.
# Everything works off (estimate, variance of the estimate) for each arm, so the same
# functions handle plain means and anything else that can give an estimate + variance.

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Estimate:
    value: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float


def normal_estimate(value, se, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * stats.norm.sf(np.abs(value / se))
    return Estimate(value, se, value - z * se, value + z * se, p)


def mean_and_var(y):
    """Sample mean and the variance of that mean (s^2 / n)."""
    y = np.asarray(y, dtype=float)
    return y.mean(), y.var(ddof=1) / len(y)


def binary_mean_and_var(successes, n):
    """mean_and_var for a 0/1 metric straight from counts, which is what the SQL views give.
    For 0/1 data s^2 = n p (1 - p) / (n - 1), so s^2 / n = p (1 - p) / (n - 1)."""
    p = successes / n
    return p, p * (1 - p) / (n - 1)


def diff(est_t, var_t, est_c, var_c, alpha=0.05):
    """Treatment minus control. Arms are independent so the variances just add."""
    return normal_estimate(est_t - est_c, np.sqrt(var_t + var_c), alpha)


def compare(est_t, var_t, est_c, var_c, alpha=0.05):
    d = diff(est_t, var_t, est_c, var_c, alpha)
    # relative lift, CI is just the diff CI scaled by the control mean for now
    lift = Estimate(d.value / est_c, d.se / est_c, d.ci_low / est_c, d.ci_high / est_c, d.p_value)
    return {'treatment': est_t, 'control': est_c, 'diff': d, 'lift': lift}


def readout(y_t, y_c, alpha=0.05):
    """Readout for a per-user metric given the raw arrays."""
    return compare(*mean_and_var(y_t), *mean_and_var(y_c), alpha=alpha)
