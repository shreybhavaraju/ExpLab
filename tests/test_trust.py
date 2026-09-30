import numpy as np
import pytest
from scipy import stats

from explab.trust import aa_summary, aa_test, balance_test, srm_test


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


def test_balance_matches_scipy():
    n_t = np.array([3000, 4100, 2500, 900])
    n_c = np.array([520, 700, 480, 140])
    res = balance_test(n_t, n_c)
    chi2, p, dof, _ = stats.chi2_contingency(np.array([n_t, n_c]), correction=False)
    assert res['chi2'] == pytest.approx(chi2, abs=1e-6)
    assert res['p_value'] == pytest.approx(p, abs=1e-6)
    assert res['dof'] == dof


def test_balance_passes_when_assignment_is_random():
    rng = np.random.default_rng(6)
    bins = rng.integers(0, 10, 2_000_000)
    t = rng.random(len(bins)) < 0.85
    res = balance_test(np.bincount(bins[t]), np.bincount(bins[~t]))
    assert res['passed']


def test_balance_fails_when_share_depends_on_the_feature():
    # overall share still ~85%, but some bins get 87% and others 83%. the plain SRM check
    # passes on this, the balance check shouldn't
    rng = np.random.default_rng(7)
    bins = rng.integers(0, 10, 2_000_000)
    share = np.where(bins < 5, 0.87, 0.83)
    t = rng.random(len(bins)) < share
    assert srm_test(t.sum(), (~t).sum())['passed']
    assert not balance_test(np.bincount(bins[t]), np.bincount(bins[~t]))['passed']


def test_aa_false_positive_rate():
    # 0/1 metric, no real effect. 500 splits -> SE of the rate ~1pt, so 2.5% to 7.5% is a wide band
    rng = np.random.default_rng(3)
    y = (rng.random(20_000) < 0.04).astype(np.int8)
    p = aa_test(y, n_sims=500, seed=4)
    s = aa_summary(p)
    assert 0.025 < s['fpr'] < 0.075
    assert s['ks_p'] > 0.01


def test_aa_with_cupac_covariate():
    # covariate that predicts y well but has nothing to do with the fake split -> still ~5%
    rng = np.random.default_rng(8)
    x = rng.random(20_000) * 0.1
    y = (rng.random(20_000) < x).astype(np.int8)
    s = aa_summary(aa_test(y, n_sims=500, seed=9, x=x))
    assert 0.025 < s['fpr'] < 0.075
    assert s['ks_p'] > 0.01


def test_aa_catches_a_broken_readout():
    # sanity check that the A/A test can actually fail: feed it a fake readout with a CI
    # that's way too narrow and the false positive rate should blow up
    rng = np.random.default_rng(5)
    y = rng.normal(size=5000)
    pvals = []
    for _ in range(300):
        t = rng.random(len(y)) < 0.5
        d = y[t].mean() - y[~t].mean()
        se = np.sqrt(y.var() / len(y))  # pretends the whole sample is in each arm
        pvals.append(2 * stats.norm.sf(abs(d / se)))
    assert aa_summary(np.array(pvals))['fpr'] > 0.15
