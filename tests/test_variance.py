import numpy as np
import pytest
from lightgbm import LGBMClassifier

from explab.readout import readout
from explab.trust import aa_summary
from explab.variance import balance, cuped, cuped_readout, oof_predictions

# tiny LightGBM so the model tests run in a second or two. min 5 rows per leaf on purpose, it
# makes the in-sample overfitting (the thing the leakage tests are about) easy to see
TINY = {'n_estimators': 30, 'num_leaves': 15, 'min_child_samples': 5}


def test_theta_is_ols_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 1.5, 5000)
    y = 0.7 * x + rng.normal(size=5000)
    y_adj, theta = cuped(y, x)
    slope, _ = np.polyfit(x, y, 1)
    assert theta == pytest.approx(slope, abs=1e-6)
    # x is centered, so the adjusted outcome keeps the original mean
    assert y_adj.mean() == pytest.approx(y.mean(), abs=1e-9)


def test_var_reduction_is_corr_squared():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20_000)
    y = 0.5 * x + rng.normal(size=20_000)
    t = rng.random(20_000) < 0.85
    r = cuped_readout(y, x, t)
    assert r['var_reduction'] == pytest.approx(np.corrcoef(x, y)[0, 1] ** 2, abs=1e-9)

    # a covariate with no information gives ~nothing (corr^2 is ~1/n = 5e-5 by chance)
    noise = rng.normal(size=20_000)
    assert cuped_readout(y, noise, t)['var_reduction'] < 1e-3


def test_adjusted_diff_is_raw_diff_minus_theta_times_imbalance():
    # this is why the CUPAC estimate moves when the covariate isn't balanced between arms
    rng = np.random.default_rng(2)
    x = rng.normal(size=3000)
    t = rng.random(3000) < 0.5 + 0.1 * (x > 0)  # deliberately imbalanced
    y = 0.3 * t + x + rng.normal(size=3000)
    raw = readout(y[t], y[~t])['diff'].value
    adj = cuped_readout(y, x, t)
    shift = adj['theta'] * balance(x, t).value
    assert adj['diff'].value == pytest.approx(raw - shift, abs=1e-12)


def test_cuped_unbiased_with_coverage_and_narrower_ci():
    # 2000 fake experiments with a known effect. y = base + effect * t + x + noise, so
    # corr(x, y)^2 = 0.5 and the CI should shrink by a factor of ~sqrt(0.5)
    rng = np.random.default_rng(3)
    n, base, effect, n_sims = 4000, 1.0, 0.1, 2000
    est, diff_cov, lift_cov, ratio = [], [], [], []
    for _ in range(n_sims):
        x = rng.normal(size=n)
        t = rng.random(n) < 0.85
        y = base + effect * t + x + rng.normal(size=n)
        raw = readout(y[t], y[~t])['diff']
        adj = cuped_readout(y, x, t)
        d, l = adj['diff'], adj['lift']
        est.append(d.value)
        diff_cov.append(d.ci_low < effect < d.ci_high)
        lift_cov.append(l.ci_low < effect / base < l.ci_high)
        ratio.append((d.ci_high - d.ci_low) / (raw.ci_high - raw.ci_low))
    est = np.array(est)

    # mean of 2000 estimates is within 4 Monte Carlo SEs of the truth
    assert abs(est.mean() - effect) < 4 * est.std() / np.sqrt(n_sims)
    # coverage from 2000 sims has SE ~0.5pt, so 93.5% to 96.5% is about +-3 SE
    assert 0.935 < np.mean(diff_cov) < 0.965
    assert 0.935 < np.mean(lift_cov) < 0.965
    # width ratio is very stable at this n (sim-to-sim sd ~1%), so 0.02 is loose
    assert np.mean(ratio) == pytest.approx(np.sqrt(0.5), abs=0.02)
    assert np.max(ratio) < 1


def test_aa_with_cuped():
    # A/A on a 0/1 metric (~6% rate) with the true probability as covariate, i.e. a perfect
    # CUPAC model. Same as trust.aa_test but with the adjustment on.
    rng = np.random.default_rng(4)
    n = 20_000
    p = 1 / (1 + np.exp(3 - rng.normal(size=n)))
    y = (rng.random(n) < p).astype(np.int8)
    pvals = np.empty(1000)
    for i in range(1000):
        fake_t = rng.random(n) < 0.5
        pvals[i] = cuped_readout(y, p, fake_t)['diff'].p_value
    s = aa_summary(pvals)
    # 1000 splits -> SE of the rate ~0.7pt, band is the pass condition
    assert 0.035 < s['fpr'] < 0.065
    assert s['ks_p'] > 0.01


def test_in_sample_predictions_leak_but_oof_do_not():
    # y is pure noise, the features know nothing about it. any correlation between the
    # prediction and y means the model has memorized y_i
    rng = np.random.default_rng(5)
    n = 4000
    X = rng.normal(size=(n, 12)).astype(np.float32)
    y = (rng.random(n) < 0.3).astype(np.int8)
    in_sample = LGBMClassifier(**TINY, verbose=-1).fit(X, y).predict_proba(X)[:, 1]
    oof = oof_predictions(X, y, params=TINY)
    assert np.corrcoef(in_sample, y)[0, 1] > 0.2
    # under no signal the sample corr has sd ~1 / sqrt(n) = 0.016
    assert abs(np.corrcoef(oof, y)[0, 1]) < 3 / np.sqrt(n)


def test_in_sample_covariate_eats_the_effect():
    # with a real effect, a memorized y_i carries the treatment, so the covariate ends up
    # correlated with t and the adjustment subtracts part of the effect
    rng = np.random.default_rng(6)
    n, effect = 4000, 0.1
    X = rng.normal(size=(n, 12)).astype(np.float32)
    t = rng.random(n) < 0.5
    y = (rng.random(n) < 0.2 + effect * t).astype(np.int8)
    in_sample = LGBMClassifier(**TINY, verbose=-1).fit(X, y).predict_proba(X)[:, 1]
    oof = oof_predictions(X, y, params=TINY)

    raw = readout(y[t], y[~t])['diff']
    leaky = cuped_readout(y, in_sample, t)
    clean = cuped_readout(y, oof, t)
    # over a few seeds the leaky version lost 25-40% of the effect and showed a ~30% 'variance
    # reduction' from features that know nothing. the oof one stayed within 0.03 SE of raw
    assert balance(in_sample, t).p_value < 0.01
    assert leaky['diff'].value < 0.8 * raw.value
    assert leaky['var_reduction'] > 0.2
    assert abs(clean['diff'].value - raw.value) < 0.1 * raw.se
    assert clean['var_reduction'] < 0.01


def test_oof_subsample_still_finds_signal():
    rng = np.random.default_rng(7)
    n = 3000
    X = rng.normal(size=(n, 3)).astype(np.float32)
    p = 1 / (1 + np.exp(-2 * X[:, 0]))
    y = (rng.random(n) < p).astype(np.int8)
    # each fold has 2000 training rows, so this forces the subsample down to 500
    oof = oof_predictions(X, y, n_folds=3, max_train_rows=500, params=TINY)
    assert np.all((oof > 0) & (oof < 1))
    assert np.corrcoef(oof, p)[0, 1] > 0.8

