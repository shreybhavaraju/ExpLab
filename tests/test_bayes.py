import numpy as np
import pytest
from scipy import integrate, stats

from explab.bayes import bayes_readout, beta_posterior
from explab.readout import binary_mean_and_var, compare


def test_beta_posterior_params():
    assert beta_posterior(3, 10) == (4, 8)
    assert beta_posterior(3, 10, prior=(2, 5)) == (5, 12)
    a, b = beta_posterior(30, 1000)
    assert a / (a + b) == pytest.approx(31 / 1002)  # flat prior -> mean (k + 1) / (n + 2)


def test_prob_better_matches_exact_integral():
    # small arms so nothing is normal yet. P(p_t > p_c) = integral of f_c(x) P(p_t > x) dx,
    # done with quad instead of draws. MC SE is ~0.0007 at p ~ 0.88, so 0.004 is ~5 SE
    k_t, n_t, k_c, n_c = 12, 100, 7, 100
    post_t = stats.beta(*beta_posterior(k_t, n_t))
    post_c = stats.beta(*beta_posterior(k_c, n_c))
    exact = integrate.quad(lambda x: post_c.pdf(x) * post_t.sf(x), 0, 1)[0]
    assert bayes_readout(k_t, n_t, k_c, n_c)['prob_better'] == pytest.approx(exact, abs=0.004)


def test_identical_arms():
    # same data in both arms -> same posterior, so P(p_t > p_c) is exactly 0.5.
    # 200k draws -> MC SE ~0.0011 near 0.5, so 0.005 is ~4.5 SE
    res = bayes_readout(4_000, 100_000, 4_000, 100_000)
    assert res['prob_better'] == pytest.approx(0.5, abs=0.005)
    assert res['lift_ci'][0] < 0 < res['lift_ci'][1]

    # p_t - p_c is ~N(0, s^2) here, and E[max(D, 0)] for that is s / sqrt(2 pi).
    # MC SE is ~0.3% of that, 2% leaves room for the normal approximation
    s = np.sqrt(2 * binary_mean_and_var(4_000, 100_000)[1])
    assert res['expected_loss'] == pytest.approx(s / np.sqrt(2 * np.pi), rel=0.02)


def test_lift_ci_close_to_delta_method():
    # 85/15 split like Criteo. At this size both posteriors are basically normal, so the
    # credible interval should land on the delta method CI. MC error on the ends is ~0.2%
    # of the width and the skew of a ratio adds well under 1%, so 5% is plenty
    k_t, n_t, k_c, n_c = 40_800, 850_000, 6_000, 150_000
    res = bayes_readout(k_t, n_t, k_c, n_c)
    lift = compare(*binary_mean_and_var(k_t, n_t), *binary_mean_and_var(k_c, n_c))['lift']
    width = lift.ci_high - lift.ci_low
    assert abs(res['lift_ci'][0] - lift.ci_low) < 0.05 * width
    assert abs(res['lift_ci'][1] - lift.ci_high) < 0.05 * width
    assert abs(res['lift_mean'] - lift.value) < 0.05 * width


@pytest.mark.parametrize('k_t, n_t, k_c, n_c', [
    (10_160, 200_000, 10_000, 200_000),  # z ~ +1.2
    (41_000, 850_000, 7_300, 150_000),   # z ~ -0.7, unequal arms
])
def test_prob_better_matches_frequentist_z(k_t, n_t, k_c, n_c):
    # borderline results so Phi(z) isn't stuck at 0 or 1. MC SE is < 0.001, so 0.01 is loose
    res = bayes_readout(k_t, n_t, k_c, n_c)
    d = compare(*binary_mean_and_var(k_t, n_t), *binary_mean_and_var(k_c, n_c))['diff']
    assert res['prob_better'] == pytest.approx(stats.norm.cdf(d.value / d.se), abs=0.01)


def test_same_seed_same_answer():
    args = (530, 10_000, 480, 10_000)
    assert bayes_readout(*args, seed=7) == bayes_readout(*args, seed=7)
    assert bayes_readout(*args, seed=7) != bayes_readout(*args, seed=8)
