import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower

from explab.power import mde, power_at


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

