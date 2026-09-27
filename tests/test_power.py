import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower

from explab.power import ci_coverage, inject_lift, mde, power_at, simulate_power


@pytest.fixture(scope='module')
def pool():
    rng = np.random.default_rng(0)
    return (rng.random(50_000) < 0.05).astype(np.int8)


@pytest.mark.parametrize('effect, var, n_t, n_c, alpha', [
    (0.002, 0.05 * 0.95, 40_000, 8_000, 0.05),
    (0.0004, 0.0382 * 0.9618, 11_882_655, 2_096_937, 0.05),
    (-0.01, 2.5, 1_000, 3_000, 0.01),
    (0.0, 1.0, 500, 500, 0.05),
])
def test_power_matches_statsmodels(effect, var, n_t, n_c, alpha):
    sm = NormalIndPower().power(effect_size=effect / np.sqrt(var), nobs1=n_t, alpha=alpha,
                                ratio=n_c / n_t, alternative='two-sided')
    assert power_at(effect, var, n_t, n_c, alpha) == pytest.approx(sm, abs=1e-6)


@pytest.mark.parametrize('var, n_t, n_c, alpha, power', [
    (0.0382 * 0.9618, 11_882_655, 2_096_937, 0.05, 0.8),
    (0.05 * 0.95, 40_000, 8_000, 0.01, 0.9),
])
def test_mde_matches_statsmodels(var, n_t, n_c, alpha, power):
    d = NormalIndPower().solve_power(effect_size=None, nobs1=n_t, alpha=alpha, power=power,
                                     ratio=n_c / n_t, alternative='two-sided')
    # statsmodels root-finds and keeps the tiny wrong-direction tail that the closed form
    # drops, they agree to ~2e-5 relative, not 1e-6
    assert mde(var, n_t, n_c, alpha, power) == pytest.approx(d * np.sqrt(var), rel=1e-4)


def test_power_at_mde_is_the_target_power():
    var, n_t, n_c = 0.0382 * 0.9618, 11_882_655, 2_096_937
    assert power_at(mde(var, n_t, n_c), var, n_t, n_c) == pytest.approx(0.8, abs=1e-4)
    assert power_at(mde(var, n_t, n_c, alpha=0.01, power=0.9), var, n_t, n_c,
                    alpha=0.01) == pytest.approx(0.9, abs=1e-4)
    # visit rate at the real arm sizes, the README number is ~1.05% relative
    assert mde(var, n_t, n_c) / 0.0382 == pytest.approx(0.0105, abs=0.0002)


def test_inject_lift_raises_mean_by_abs_lift():
    rng = np.random.default_rng(1)
    y = (rng.random(1_000_000) < 0.04).astype(np.int8)
    out = inject_lift(y, 0.002, rng)
    # sd of the increase is sqrt(n q (1 - p)) / n ~ 4.5e-5, so 2e-4 is ~4.5 sd
    assert out.mean() - y.mean() == pytest.approx(0.002, abs=2e-4)
    assert np.all(out >= y)  # only 0 -> 1 flips
    assert out.dtype == y.dtype
    assert np.array_equal(inject_lift(y, 0.0, rng), y)
    with pytest.raises(ValueError):
        inject_lift(y, 0.99, rng)


@pytest.fixture(scope='module')
def sim(pool):
    n_t, n_c = 40_000, 10_000
    var = pool.mean() * (1 - pool.mean())
    # the mid lift has 50% analytic power, the big one ~100%
    rel_mid = mde(var, n_t, n_c, power=0.5) / pool.mean()
    rel_lifts = [0.0, rel_mid, 3 * rel_mid]
    power = simulate_power(pool, n_t, n_c, rel_lifts, n_sims=2000, seed=2)
    analytic = power_at(np.array(rel_lifts) * pool.mean(), var, n_t, n_c)
    return power, analytic


def test_simulated_power_at_zero_lift_is_alpha(sim):
    power, _ = sim
    # 2000 sims -> SE of a 5% rate is ~0.5pt, band is about +-3 SE
    assert 0.035 < power[0] < 0.065


def test_simulated_power_increases_with_lift(sim):
    power, _ = sim
    assert power[0] < power[1] < power[2]
    assert power[2] > 0.95


def test_simulated_power_close_to_analytic(sim):
    power, analytic = sim
    # SE of a ~50% rate from 2000 sims is ~1.1pt, allow ~3.5 SE
    assert power[1] == pytest.approx(analytic[1], abs=0.04)
    assert power[2] == pytest.approx(analytic[2], abs=0.02)


def test_ci_coverage(pool):
    cov = ci_coverage(pool, 40_000, 10_000, rel_lift=0.1, n_sims=1000, seed=3)
    # pass condition band from validation/pass_conditions.md, SE at 1000 reps is ~0.7pt
    assert 0.935 <= cov['diff_coverage'] <= 0.965
    assert 0.935 <= cov['lift_coverage'] <= 0.965
