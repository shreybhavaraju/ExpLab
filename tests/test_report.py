import numpy as np
import pandas as pd
import pytest

from explab.decide import INCONCLUSIVE, SHIP
from explab.readout import binary_mean_and_var, compare
from explab.report import metric_report


def fake_cells(share_by_bin=None, n_per_cell=200_000, seed=0):
    """segment_stats-shaped table: 3 f0 bins x 2 f2 bins x 2 arms, visit rate 4% -> 5%."""
    rng = np.random.default_rng(seed)
    rows = []
    for f0 in range(3):
        share = share_by_bin[f0] if share_by_bin else 0.85
        for f2 in range(2):
            n_t = rng.binomial(n_per_cell, share)
            for treatment, n in [(1, n_t), (0, n_per_cell - n_t)]:
                visits = rng.binomial(n, 0.05 if treatment else 0.04)
                conversions = rng.binomial(visits, 0.06)
                rows.append({'f0_bin': f0, 'f2_bin': f2, 'treatment': treatment, 'n': n,
                             'visits': visits, 'conversions': conversions,
                             'conv_visits': conversions})
    return pd.DataFrame(rows)


def test_report_matches_plain_readout():
    cells = fake_cells()
    r = metric_report(cells, 'visit')
    t, c = cells[cells.treatment == 1], cells[cells.treatment == 0]
    expected = compare(*binary_mean_and_var(t.visits.sum(), t.n.sum()),
                       *binary_mean_and_var(c.visits.sum(), c.n.sum()))
    assert r['readout']['lift'].value == pytest.approx(expected['lift'].value, abs=1e-12)
    assert r['n_t'] + r['n_c'] == cells.n.sum()


def test_clean_experiment_ships():
    r = metric_report(fake_cells(), 'visit', min_effect=0.05)
    assert r['srm']['passed']
    assert all(b['passed'] for b in r['balance'].values())
    assert r['decision'].verdict == SHIP


def test_share_drifting_by_bin_blocks_the_verdict():
    # overall share still averages ~85%, but it depends on f0
    r = metric_report(fake_cells(share_by_bin=[0.83, 0.85, 0.87]), 'visit')
    assert not r['balance']['f0']['passed']
    assert r['decision'].verdict == INCONCLUSIVE


def test_single_bin_slice_skips_that_features_balance_check():
    cells = fake_cells()
    r = metric_report(cells[cells.f0_bin == 1], 'visit')
    assert 'f0' not in r['balance'] and 'f2' in r['balance']


def test_ratio_metric_and_bayes():
    cells = fake_cells()
    r = metric_report(cells, 'conversions_per_visit')
    assert 'bayes' not in r
    assert r['readout']['control'] == pytest.approx(0.06, abs=0.003)
    assert 0.9 < metric_report(cells, 'visit')['bayes']['prob_better'] <= 1
