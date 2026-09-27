import numpy as np
import pytest
from scipy import stats

from explab.trust import srm_test


def test_srm_matches_scipy():
    n_t, n_c = 8612, 1388
    res = srm_test(n_t, n_c, expected_share=0.85)
    chi2, p = stats.chisquare([n_t, n_c], f_exp=[0.85 * 10000, 0.15 * 10000])
    assert res['chi2'] == pytest.approx(chi2, abs=1e-6)
    assert res['p_value'] == pytest.approx(p, abs=1e-6)


def test_srm_passes_on_a_fair_split():
    rng = np.random.default_rng(0)
    t = rng.random(2_000_000) < 0.85
    assert srm_test(t.sum(), (~t).sum())['passed']


def test_srm_false_alarm_rate():
    # fair 85/15 splits should almost never trip the 0.001 threshold
    rng = np.random.default_rng(1)
    n_t = rng.binomial(1_000_000, 0.85, 2000)
    fails = [not srm_test(k, 1_000_000 - k)['passed'] for k in n_t]
    assert np.mean(fails) < 0.005


def test_srm_catches_dropping_2pct_of_one_arm():
    # pass condition: losing 2% of control users (e.g. a logging bug) has to fail the check
    rng = np.random.default_rng(2)
    t = rng.random(2_000_000) < 0.85
    keep = t | (rng.random(len(t)) > 0.02)
    t = t[keep]
    res = srm_test(t.sum(), (~t).sum())
    assert not res['passed']


def test_srm_uses_the_design_ratio():
    # an 85/15 split is fine against 0.85 but obviously fails against 50/50
    assert srm_test(85_000, 15_000, expected_share=0.85)['passed']
    assert not srm_test(85_000, 15_000, expected_share=0.5)['passed']

