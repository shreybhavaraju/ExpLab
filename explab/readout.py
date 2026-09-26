# Effect estimates for one metric: absolute difference, relative lift, CIs.
# Everything works off (estimate, variance of the estimate) for each arm, so the same
# functions handle plain means, ratio metrics, and later the CUPAC-adjusted means.

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


def ratio_from_sums(n, sx, sy, sxx, syy, sxy):
    """Ratio metric R = sum(x) / sum(y), e.g. conversions per visit, where the randomized
    unit is the user and x, y are per-user totals. Delta method again:

        R = mx / my,   Var(R_hat) ~ (Var(x) - 2 R Cov(x, y) + R^2 Var(y)) / (n my^2)

    with per-user variances. Treating every visit as its own independent unit would ignore
    that visits from the same user are correlated. Takes sums so it works on the SQL views.
    """
    mx, my = sx / n, sy / n
    r = mx / my
    var_x = (sxx - n * mx**2) / (n - 1)
    var_y = (syy - n * my**2) / (n - 1)
    cov = (sxy - n * mx * my) / (n - 1)
    return r, (var_x - 2 * r * cov + r**2 * var_y) / (n * my**2)


def ratio_and_var(num, den):
    x = np.asarray(num, dtype=float)
    y = np.asarray(den, dtype=float)
    return ratio_from_sums(len(x), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum())


def diff(est_t, var_t, est_c, var_c, alpha=0.05):
    """Treatment minus control. Arms are independent so the variances just add."""
    return normal_estimate(est_t - est_c, np.sqrt(var_t + var_c), alpha)


def lift(est_t, var_t, est_c, var_c, alpha=0.05):
    """Relative lift L = est_t / est_c - 1, with a delta method CI.

    L = g(a, b) = a / b - 1 where a, b are the treatment and control estimates. First order
    Taylor expansion around the true values:

        L_hat - L ~ dg/da (a_hat - a) + dg/db (b_hat - b),   dg/da = 1/b,  dg/db = -a/b^2

    a_hat and b_hat are independent (separate users), so

        Var(L_hat) ~ Var(a_hat) / b^2 + a^2 Var(b_hat) / b^4

    and we plug in the estimates for a and b. The shortcut of dividing the diff CI by the
    control mean treats that denominator as a known constant, so the control noise gets
    weight 1 instead of (a/b)^2. With a big lift and a small control arm (both true here)
    that's a real difference: on visits the shortcut CI was ~20% too narrow.
    """
    value = est_t / est_c - 1
    var = var_t / est_c**2 + est_t**2 * var_c / est_c**4
    return normal_estimate(value, np.sqrt(var), alpha)


def compare(est_t, var_t, est_c, var_c, alpha=0.05):
    return {
        'treatment': est_t,
        'control': est_c,
        'diff': diff(est_t, var_t, est_c, var_c, alpha),
        'lift': lift(est_t, var_t, est_c, var_c, alpha),
    }


def readout(y_t, y_c, alpha=0.05):
    """Readout for a per-user metric given the raw arrays."""
    return compare(*mean_and_var(y_t), *mean_and_var(y_c), alpha=alpha)


def bootstrap(t_cols, c_cols, stat, n_boot=1000, seed=0):
    """Resample users with replacement within each arm and recompute stat(t_cols, c_cols).

    t_cols / c_cols are lists of per-user arrays for each arm, e.g. [visit] for a mean or
    [conversion, visit] for a ratio. stat can return one number or an array of them, so one
    set of resamples can be shared across a few metrics.
    """
    rng = np.random.default_rng(seed)
    n_t, n_c = len(t_cols[0]), len(c_cols[0])
    out = []
    for _ in range(n_boot):
        it = rng.integers(0, n_t, n_t)
        ic = rng.integers(0, n_c, n_c)
        out.append(stat([a[it] for a in t_cols], [a[ic] for a in c_cols]))
    return np.array(out)


def percentile_ci(boot, alpha=0.05):
    return np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=0)


# stats for bootstrap()
def mean_diff(t, c):
    return t[0].mean() - c[0].mean()


def mean_lift(t, c):
    return t[0].mean() / c[0].mean() - 1


def ratio_diff(t, c):
    return t[0].sum() / t[1].sum() - c[0].sum() / c[1].sum()


def ratio_lift(t, c):
    return (t[0].sum() / t[1].sum()) / (c[0].sum() / c[1].sum()) - 1
