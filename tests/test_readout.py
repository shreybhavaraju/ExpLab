import numpy as np
import pytest
from statsmodels.stats._delta_method import NonlinearDeltaCov
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from explab.readout import (binary_mean_and_var, bootstrap, compare, mean_and_var, mean_diff,
                            mean_lift, percentile_ci, ratio_and_var, ratio_from_sums, readout)


@pytest.fixture
def small():
    rng = np.random.default_rng(0)
    return rng.normal(1.0, 2.0, 60), rng.normal(0.5, 1.0, 90)


def test_diff_matches_statsmodels(small):
    y_t, y_c = small
    r = readout(y_t, y_c)
    cm = CompareMeans(DescrStatsW(y_t), DescrStatsW(y_c))
    lo, hi = cm.zconfint_diff(alpha=0.05, usevar='unequal')
    _, p = cm.ztest_ind(usevar='unequal')
    assert r['diff'].ci_low == pytest.approx(lo, abs=1e-6)
    assert r['diff'].ci_high == pytest.approx(hi, abs=1e-6)
    assert r['diff'].p_value == pytest.approx(p, abs=1e-6)


def test_lift_matches_statsmodels_delta_method(small):
    y_t, y_c = small
    m_t, v_t = mean_and_var(y_t)
    m_c, v_c = mean_and_var(y_c)
    r = compare(m_t, v_t, m_c, v_c)
    # statsmodels does the delta method with numerical derivatives
    sm = NonlinearDeltaCov(lambda p: p[0] / p[1] - 1, np.array([m_t, m_c]), np.diag([v_t, v_c]))
    assert r['lift'].value == pytest.approx(m_t / m_c - 1, abs=1e-12)
    assert r['lift'].se == pytest.approx(float(sm.se_vectorized()), abs=1e-6)


def test_binary_counts_same_as_arrays():
    y = np.r_[np.ones(37), np.zeros(463)]
    m1, v1 = mean_and_var(y)
    m2, v2 = binary_mean_and_var(37, 500)
    assert m1 == pytest.approx(m2, abs=1e-12)
    assert v1 == pytest.approx(v2, abs=1e-12)


def test_ratio_delta_method():
    rng = np.random.default_rng(1)
    visits = rng.poisson(2.0, 400).astype(float)
    convs = rng.binomial(visits.astype(int), 0.1).astype(float)
    r, var = ratio_and_var(convs, visits)
    n = len(visits)
    cov_means = np.cov(convs, visits) / n
    sm = NonlinearDeltaCov(lambda p: p[0] / p[1], np.array([convs.mean(), visits.mean()]), cov_means)
    assert r == pytest.approx(convs.sum() / visits.sum(), abs=1e-12)
    assert np.sqrt(var) == pytest.approx(float(sm.se_vectorized()), abs=1e-6)

    # sums version (what the SQL views feed in) gives the same thing
    r2, var2 = ratio_from_sums(n, convs.sum(), visits.sum(), (convs**2).sum(),
                               (visits**2).sum(), (convs * visits).sum())
    assert (r2, var2) == pytest.approx((r, var), abs=1e-12)


def test_ci_coverage_with_known_lift():
    # 2000 fake experiments with a known true difference, vectorized with binomial counts
    rng = np.random.default_rng(2)
    n_t, n_c, p_t, p_c = 40_000, 8_000, 0.055, 0.05
    k_t = rng.binomial(n_t, p_t, 2000)
    k_c = rng.binomial(n_c, p_c, 2000)
    r = compare(*binary_mean_and_var(k_t, n_t), *binary_mean_and_var(k_c, n_c))
    d, lift = r['diff'], r['lift']
    diff_cov = np.mean((d.ci_low < p_t - p_c) & (p_t - p_c < d.ci_high))
    lift_cov = np.mean((lift.ci_low < p_t / p_c - 1) & (p_t / p_c - 1 < lift.ci_high))
    assert 0.935 < diff_cov < 0.965
    assert 0.935 < lift_cov < 0.965


def test_shortcut_lift_ci_is_too_narrow():
    # dividing the diff CI by the control mean ignores the control mean being noisy
    rng = np.random.default_rng(3)
    n_t, n_c, p_t, p_c = 40_000, 8_000, 0.065, 0.05
    k_t = rng.binomial(n_t, p_t, 4000)
    k_c = rng.binomial(n_c, p_c, 4000)
    r = compare(*binary_mean_and_var(k_t, n_t), *binary_mean_and_var(k_c, n_c))
    true_lift = p_t / p_c - 1
    lo, hi = r['diff'].ci_low / r['control'], r['diff'].ci_high / r['control']
    shortcut_cov = np.mean((lo < true_lift) & (true_lift < hi))
    delta_cov = np.mean((r['lift'].ci_low < true_lift) & (true_lift < r['lift'].ci_high))
    assert shortcut_cov < 0.93
    assert delta_cov > 0.935


def test_bootstrap_agrees_with_normal_ci():
    rng = np.random.default_rng(4)
    y_t = (rng.random(100_000) < 0.06).astype(np.int8)
    y_c = (rng.random(30_000) < 0.05).astype(np.int8)
    boot = bootstrap([y_t], [y_c], lambda t, c: [mean_diff(t, c), mean_lift(t, c)], n_boot=400)
    (d_lo, l_lo), (d_hi, l_hi) = percentile_ci(boot)
    r = readout(y_t, y_c)
    # ends should agree to within ~10% of the CI width
    d_width = r['diff'].ci_high - r['diff'].ci_low
    l_width = r['lift'].ci_high - r['lift'].ci_low
    assert abs(d_lo - r['diff'].ci_low) < 0.1 * d_width
    assert abs(d_hi - r['diff'].ci_high) < 0.1 * d_width
    assert abs(l_lo - r['lift'].ci_low) < 0.1 * l_width
    assert abs(l_hi - r['lift'].ci_high) < 0.1 * l_width
