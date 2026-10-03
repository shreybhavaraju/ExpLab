import numpy as np
import pytest
from sklift.metrics import qini_auc_score, qini_curve as sklift_qini_curve, uplift_auc_score
from sklift.metrics import uplift_curve as sklift_uplift_curve

from explab.uplift import (_arms, auuc, captured_share, planted_effect_outcomes, qini_auc,
                           qini_curve, t_learner, uplift_by_decile, uplift_curve)

# small trees so the whole file runs in a few seconds
SMALL = {'n_estimators': 60, 'learning_rate': 0.1, 'num_leaves': 7, 'min_child_samples': 100}
EFFECT = 0.15


def planted(n, seed):
    """4 noise features, 85/15 split, base rate depends on x1, +EFFECT only where x0 is in
    its bottom ~20% (x0 < -0.84)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype('float32')
    t = (rng.random(n) < 0.85).astype(np.int8)
    seg = X[:, 0] < -0.84
    base = 0.05 + 0.05 * (X[:, 1] > 0)
    return X, planted_effect_outcomes(base, t, EFFECT, seg, seed + 100), t, seg


@pytest.fixture(scope='module')
def fitted():
    X, y, t, _ = planted(30_000, 0)
    X_te, y_te, t_te, seg_te = planted(10_000, 1)
    preds = {'t': t_learner(X, y, t, X_te, params=SMALL)}
    return preds, y_te, t_te, seg_te


@pytest.mark.parametrize('name', ['t'])
def test_learners_find_the_planted_segment(fitted, name):
    preds, y, t, seg = fitted
    u = preds[name]
    # true uplift is 0.15 inside and 0 outside, ask for at least half the gap
    assert u[seg].mean() - u[~seg].mean() > EFFECT / 2
    top = np.argsort(-u)[:len(u) // 10]
    assert seg[top].mean() > 0.8  # segment is ~20% of users, so the top 10% can all be in it
    # random scores on this test set have a qini auc sd of ~0.027 (200 shuffles), 0.08 is ~3 sd
    assert qini_auc(y, t, u) > 0.08


def test_random_scores_give_qini_near_zero(fitted):
    _, y, t, _ = fitted
    rng = np.random.default_rng(2)
    aucs = np.array([qini_auc(y, t, rng.random(len(y))) for _ in range(100)])
    # mean of 100 draws should be within ~3 standard errors of 0
    assert abs(aucs.mean()) < 3 * aucs.std() / np.sqrt(len(aucs))
    assert aucs.std() < 0.05


@pytest.mark.filterwarnings('ignore::FutureWarning')  # sklift calls a deprecated sklearn helper
@pytest.mark.parametrize('ties', [False, True])
def test_metrics_match_sklift(ties):
    rng = np.random.default_rng(3)
    n = 400
    y = (rng.random(n) < 0.3).astype(int)
    t = (rng.random(n) < 0.85).astype(int)
    u = rng.normal(size=n)
    if ties:
        u = np.round(u, 1)  # ~40 distinct scores, checks that tied users collapse the same way
    k, q = qini_curve(y, t, u)
    k_sk, q_sk = sklift_qini_curve(y, u, t)
    np.testing.assert_allclose(k, k_sk, atol=1e-6)
    np.testing.assert_allclose(q, q_sk, atol=1e-6)
    k, v = uplift_curve(y, t, u)
    k_sk, v_sk = sklift_uplift_curve(y, u, t)
    np.testing.assert_allclose(k, k_sk, atol=1e-6)
    np.testing.assert_allclose(v, v_sk, atol=1e-6)
    assert qini_auc(y, t, u) == pytest.approx(qini_auc_score(y, u, t), abs=1e-6)
    assert auuc(y, t, u) == pytest.approx(uplift_auc_score(y, u, t), abs=1e-6)


def test_captured_share_of_everyone_is_one():
    rng = np.random.default_rng(4)
    y = (rng.random(1000) < 0.2).astype(int)
    t = (rng.random(1000) < 0.85).astype(int)
    assert captured_share(y, t, rng.normal(size=1000), top=1.0) == pytest.approx(1.0, abs=1e-12)


def test_uplift_by_decile_counts():
    rng = np.random.default_rng(5)
    y = (rng.random(2000) < 0.2).astype(int)
    t = (rng.random(2000) < 0.5).astype(int)
    u = rng.normal(size=2000)
    rows = uplift_by_decile(y, t, u)
    assert [r['n'] for r in rows] == [200] * 10
    top = np.argsort(-u)[:200]
    assert rows[0]['predicted'] == pytest.approx(u[top].mean())
    y_top, t_top = y[top], t[top]
    assert rows[0]['observed'] == pytest.approx(y_top[t_top == 1].mean() - y_top[t_top == 0].mean())


def test_planted_outcomes_rates():
    # 200k draws, SE of each rate is < 0.002, so 0.01 is a ~5 SE band
    rng = np.random.default_rng(6)
    t = (rng.random(200_000) < 0.5).astype(int)
    seg = rng.random(200_000) < 0.3
    y = planted_effect_outcomes(np.full(200_000, 0.1), t, 0.2, seg, seed=7)
    assert y[seg & (t == 1)].mean() == pytest.approx(0.3, abs=0.01)
    assert y[seg & (t == 0)].mean() == pytest.approx(0.1, abs=0.01)
    assert y[~seg & (t == 1)].mean() == pytest.approx(0.1, abs=0.01)


def test_max_train_rows_caps_each_arm():
    X, _, t, _ = planted(6000, 8)
    # pass t in as the outcome so it's easy to see which arm each kept row came from
    (X0, y0), (X1, y1) = _arms(X, t, t, 500, seed=0)
    assert len(X0) == len(X1) == 500
    assert np.all(y0 == 0) and np.all(y1 == 1)
    (X0, _), (X1, _) = _arms(X, t, t, None, seed=0)
    assert len(X0) + len(X1) == 6000
