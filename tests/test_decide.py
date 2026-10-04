from explab.decide import DONT_SHIP, INCONCLUSIVE, SHIP, decide
from explab.readout import Estimate

GOOD_SRM = {'passed': True, 'share': 0.85, 'expected_share': 0.85, 'p_value': 0.6}
BAD_SRM = {'passed': False, 'share': 0.853, 'expected_share': 0.85, 'p_value': 1e-9}
GOOD_BALANCE = {'passed': True, 'shares': [0.849, 0.851], 'p_value': 0.4}
BAD_BALANCE = {'passed': False, 'shares': [0.846, 0.877], 'p_value': 1e-50}


def est(value, lo, hi):
    return Estimate(value, (hi - lo) / 3.92, lo, hi, 0.01)


UP = est(0.08, 0.05, 0.11)
DOWN = est(-0.08, -0.11, -0.05)
FLAT = est(0.005, -0.02, 0.03)


def test_ship_when_everything_checks_out():
    d = decide(UP, GOOD_SRM, mde_rel=0.02, balance={'f0': GOOD_BALANCE},
               guardrails={'conversions per visit': FLAT})
    assert d.verdict == SHIP
    assert 'up +8.0%' in d.reasons[0]


def test_srm_failure_blocks_even_a_big_win():
    d = decide(UP, BAD_SRM, mde_rel=0.02)
    assert d.verdict == INCONCLUSIVE
    assert len(d.reasons) == 1 and 'sample ratio mismatch' in d.reasons[0]


def test_imbalance_blocks():
    d = decide(UP, GOOD_SRM, mde_rel=0.02, balance={'f0': GOOD_BALANCE, 'f2': BAD_BALANCE})
    assert d.verdict == INCONCLUSIVE
    assert len(d.reasons) == 1 and 'f2 bins' in d.reasons[0] and 'confounded' in d.reasons[0]


def test_underpowered_blocks():
    d = decide(UP, GOOD_SRM, mde_rel=0.12, min_effect=0.05)
    assert d.verdict == INCONCLUSIVE
    assert 'underpowered' in d.reasons[0]


def test_ci_with_harm_and_benefit_blocks():
    wide = est(0.01, -0.09, 0.11)
    d = decide(wide, GOOD_SRM, mde_rel=0.04, min_effect=0.05)
    assert d.verdict == INCONCLUSIVE
    assert 'meaningful drop and a meaningful gain' in d.reasons[0]


def test_refusals_are_all_listed():
    wide = est(0.01, -0.09, 0.11)
    d = decide(wide, BAD_SRM, mde_rel=0.12, balance={'f0': BAD_BALANCE})
    assert d.verdict == INCONCLUSIVE
    assert len(d.reasons) == 4


def test_harmed_guardrail_means_dont_ship():
    d = decide(UP, GOOD_SRM, mde_rel=0.02, guardrails={'conversions per visit': DOWN})
    assert d.verdict == DONT_SHIP
    assert 'guardrail conversions per visit' in d.reasons[0]


def test_significant_drop_is_dont_ship():
    d = decide(DOWN, GOOD_SRM, mde_rel=0.02)
    assert d.verdict == DONT_SHIP
    assert 'down' in d.reasons[0]


def test_flat_result_on_a_powered_test_is_dont_ship():
    d = decide(FLAT, GOOD_SRM, mde_rel=0.02)
    assert d.verdict == DONT_SHIP
    assert 'no significant change' in d.reasons[0]


def test_ci_touching_zero_is_not_a_ship():
    d = decide(est(0.03, 0.0, 0.06), GOOD_SRM, mde_rel=0.02)
    assert d.verdict == DONT_SHIP
