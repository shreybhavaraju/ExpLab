# Power and minimum detectable effect. The normal approximation for planning, plus a
# simulation on resampled real users with a known lift injected, which also checks the CIs.

import numpy as np
from scipy import stats

from explab.readout import binary_mean_and_var, compare


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


def inject_lift(y, abs_lift, rng):
    """Plant a known effect in a 0/1 array: each 0 turns into a 1 with prob q = abs_lift / (1 - p),
    p = mean of y. Only the (1 - p) share of users who are 0 can flip, so the mean goes up by
    (1 - p) q = abs_lift in expectation. Flipping with prob abs_lift would undershoot by (1 - p).
    Everything else about the real data (base rate, noise) stays as it was."""
    y = np.asarray(y)
    p = np.count_nonzero(y) / len(y)  # same as y.mean() for 0/1, much faster on int8
    q = abs_lift / (1 - p)
    if not 0 <= q <= 1:
        raise ValueError(f"can't raise a mean of {p:.4f} by {abs_lift}")
    out = y.copy()
    # setting a 1 to 1 does nothing, so no need to pick out the zeros first. float32 uniforms
    # are fine for a flip prob this size and twice as fast to draw for 12M users
    out[rng.random(len(y), dtype=np.float32) < q] = 1
    return out


def resample(y, n, rng):
    # int32 indices: less memory than the default int64 for an 11.9M treatment arm
    return y[rng.integers(0, len(y), n, dtype=np.int32)]


def binary_readout(y_t, y_c, alpha=0.05):
    """readout() for 0/1 arrays, from the counts. Skips the float copy and var() of 12M rows."""
    return compare(*binary_mean_and_var(np.count_nonzero(y_t), len(y_t)),
                   *binary_mean_and_var(np.count_nonzero(y_c), len(y_c)), alpha=alpha)


def simulate_power(y_pool, n_t, n_c, rel_lifts, n_sims=200, alpha=0.05, seed=0):
    """Power by brute force. Each sim draws a control arm (n_c) and a treatment arm (n_t) from
    y_pool with replacement, so there is no real difference between them, then injects
    abs_lift = rel_lift * mean(y_pool) into treatment and runs the normal readout.
    Returns the share of sims with p < alpha for each lift (at lift 0 that's the FPR)."""
    rng = np.random.default_rng(seed)
    y_pool = np.asarray(y_pool)
    base = y_pool.mean()
    hits = np.zeros(len(rel_lifts))
    for _ in range(n_sims):
        # the same two arms get every lift, which makes the curve smoother than fresh arms per
        # point would, and drawing a 12M arm is the slow part
        y_c = resample(y_pool, n_c, rng)
        y_t = resample(y_pool, n_t, rng)
        for j, rel in enumerate(rel_lifts):
            r = binary_readout(inject_lift(y_t, rel * base, rng), y_c, alpha)
            hits[j] += r['diff'].p_value < alpha
    return hits / n_sims


def ci_coverage(y_pool, n_t, n_c, rel_lift, n_sims=1000, alpha=0.05, seed=0):
    """Same fake experiments as simulate_power with one lift, but here the true effect is
    known (diff = abs_lift, lift = rel_lift), so count how often the readout CIs contain it.
    A 95% CI should hit ~95% of the time, both for the diff and the delta method lift."""
    rng = np.random.default_rng(seed)
    y_pool = np.asarray(y_pool)
    abs_lift = rel_lift * y_pool.mean()
    hit_diff = hit_lift = 0
    for _ in range(n_sims):
        y_c = resample(y_pool, n_c, rng)
        y_t = inject_lift(resample(y_pool, n_t, rng), abs_lift, rng)
        r = binary_readout(y_t, y_c, alpha)
        hit_diff += r['diff'].ci_low < abs_lift < r['diff'].ci_high
        hit_lift += r['lift'].ci_low < rel_lift < r['lift'].ci_high
    return {'diff_coverage': float(hit_diff / n_sims), 'lift_coverage': float(hit_lift / n_sims)}
