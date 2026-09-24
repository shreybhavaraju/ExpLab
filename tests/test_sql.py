import numpy as np
import pandas as pd
import pytest

from explab.load import FEATURES, connect


def fake_criteo(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 12)).astype('float32'), columns=FEATURES)
    df['treatment'] = (rng.random(n) < 0.85).astype('int8')
    df['visit'] = (rng.random(n) < 0.05 + 0.02 * df.treatment).astype('int8')
    df['conversion'] = (df.visit * (rng.random(n) < 0.1)).astype('int8')
    df['exposure'] = (df.treatment * (rng.random(n) < 0.04)).astype('int8')
    return df


@pytest.fixture(scope='module')
def fake():
    return fake_criteo()


def test_metric_views_match_pandas(fake):
    con = connect(fake)
    by_arm = fake.groupby('treatment')

    sizes = con.sql('select * from arm_sizes order by treatment').df()
    assert sizes.n.tolist() == by_arm.size().tolist()

    visits = con.sql('select * from visit_rate order by treatment').df()
    np.testing.assert_allclose(visits.visit_rate, by_arm.visit.mean())

    conv = con.sql('select * from conversion_rate order by treatment').df()
    np.testing.assert_allclose(conv.conversion_rate, by_arm.conversion.mean())

    cpv = con.sql('select * from conversions_per_visit order by treatment').df()
    expected = by_arm.conversion.sum() / by_arm.visit.sum()
    np.testing.assert_allclose(cpv.conversions_per_visit, expected)


def test_segment_bins_are_deciles(fake):
    con = connect(fake)
    seg = con.sql('select f0, f0_bin from criteo_segments').df()
    # continuous feature, no ties -> 10 bins of ~10% each, in order of f0
    counts = seg.f0_bin.value_counts().sort_index()
    assert counts.index.tolist() == list(range(10))
    assert counts.min() > 0.09 * len(fake)
    assert seg.groupby('f0_bin').f0.max().is_monotonic_increasing


def test_ties_merge_bins():
    df = fake_criteo()
    df.loc[:1499, 'f0'] = df.f0.min() - 1  # 30% of rows tied at the bottom
    con = connect(df)
    bins = con.sql('select f0_bin, count(*) as n from criteo_segments group by 1 order by 1').df()
    assert bins.f0_bin.iloc[0] == 0 and bins.n.iloc[0] == 1500
    assert 1 not in bins.f0_bin.values and 2 not in bins.f0_bin.values


def test_segment_stats_add_up(fake):
    con = connect(fake)
    stats = con.sql('select * from segment_stats').df()
    assert stats.n.sum() == len(fake)
    assert stats.visits.sum() == fake.visit.sum()
    assert stats.conversions.sum() == fake.conversion.sum()
