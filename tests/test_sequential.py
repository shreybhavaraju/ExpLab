import numpy as np
import pytest

from explab.sequential import (cumulative_fpr, naive_crossings, obf_boundaries, obf_crossings,
                               peeking_sim)


@pytest.fixture(scope='module')
def aa():
    # 0/1 metric with no real effect, 20k users, 1,000 sims of 20 looks (~0.6s)
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.05).astype(np.int8)
    return y, peeking_sim(y, n_sims=1000, n_looks=20, seed=1)


def test_obf_constant_matches_jennison_turnbull():
    # Jennison & Turnbull table for alpha = 0.05 two-sided: C = 1.960 (K = 1), 2.040 (K = 5).
    # SE of the 95% quantile from 200k paths is sqrt(.05 * .95 / 200k) / density ~ 0.004
    # (seeds 0-4 give 1.955-1.965 for K = 1), so 0.015 is ~3.5 SE
    assert obf_boundaries(1)[-1] == pytest.approx(1.960, abs=0.015)
    b = obf_boundaries(5)
    assert b[-1] == pytest.approx(2.040, abs=0.015)
    assert b == pytest.approx(b[-1] * np.sqrt(5 / np.arange(1, 6)), abs=1e-12)


def test_peeking_false_positive_rates(aa):
    y, sim = aa
    naive = cumulative_fpr(naive_crossings(sim['z']))
    obf = cumulative_fpr(obf_crossings(sim['z'], obf_boundaries(20)))
    # same bands as the peeking row of validation/pass_conditions.md. 1,000 sims -> SE of a
    # ~5% rate is ~0.7pt, so 3.5% to 6.5% is about +-2 SE (this seed gives 5.5%)
    assert naive[0] == pytest.approx(0.05, abs=0.025)  # a single look is fine, ~3.5 SE
    assert naive[-1] > 0.10
    assert 0.035 <= obf[-1] <= 0.065


def test_sim_variance_matches_spread_of_diffs():
    # skewed continuous metric to check the y / y^2 sums work beyond 0/1 data. the spread of
    # the diffs across sims should match the average var. with 1,000 sims the ratio has a
    # ~5% SE (checked over 10 seeds), so 0.8 to 1.2 is ~4 SE at each of the 10 looks
    rng = np.random.default_rng(3)
    y = rng.exponential(2.0, 10_000)
    sim = peeking_sim(y, n_sims=1000, n_looks=10, share=0.3, seed=4)
    assert sim['diff'].shape == sim['var'].shape == (1000, 10)
    assert sim['frac'] == pytest.approx(np.arange(1, 11) / 10)
    ratio = sim['diff'].var(axis=0) / sim['var'].mean(axis=0)
    assert np.all((ratio > 0.8) & (ratio < 1.2))
    assert sim['z'] == pytest.approx(sim['diff'] / np.sqrt(sim['var']))


def test_cumulative_fpr_counts_first_crossings():
    crossings = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=bool)
    assert cumulative_fpr(crossings) == pytest.approx([1 / 3, 2 / 3, 2 / 3])
