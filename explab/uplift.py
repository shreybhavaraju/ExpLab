# Uplift modeling: which users does the ad actually move? T-learner on LightGBM, plus
# Qini / AUUC to check it on held-out users against random targeting.

import lightgbm as lgb
import numpy as np

# min_child_samples is high on purpose, uplift is a small difference of two noisy rates and
# tiny leaves just fit noise
PARAMS = {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31,
          'min_child_samples': 500, 'verbose': -1}


def _params(params, seed):
    return {**PARAMS, 'random_state': seed, **(params or {})}


def _arms(X, y, t, max_rows, seed):
    """(X0, y0), (X1, y1) for control and treated, each capped at max_rows random rows."""
    rng = np.random.default_rng(seed)
    out = []
    for arm in [0, 1]:
        idx = np.flatnonzero(t == arm)
        if max_rows is not None and len(idx) > max_rows:
            idx = np.sort(rng.choice(idx, max_rows, replace=False))
        out.append((X[idx], y[idx]))
    return out


def t_learner(X, y, t, X_new, params=None, max_train_rows=None, seed=0):
    """Two separate classifiers, p1(x) = P(y = 1 | x, treated) fit on treated rows only and
    p0(x) the same on control rows. Predicted uplift on X_new is p1(x) - p0(x).
    max_train_rows caps each arm's training rows (the treated arm is 8M+ on the real data)."""
    (X0, y0), (X1, y1) = _arms(X, y, t, max_train_rows, seed)
    p = _params(params, seed)
    m0 = lgb.LGBMClassifier(**p).fit(X0, y0)
    m1 = lgb.LGBMClassifier(**p).fit(X1, y1)
    return m1.predict_proba(X_new)[:, 1] - m0.predict_proba(X_new)[:, 1]


# Why not accuracy or ROC AUC like a normal classifier? Those compare each prediction to that
# user's true answer, but the true uplift y(treated) - y(control) is never seen for anyone, each
# user only lands in one arm. So uplift can only be checked on groups: take the top k users by
# predicted uplift and compare treated vs control outcomes inside that group. That's the Qini curve.

def _cum_counts(y, t, uplift):
    """Sort by predicted uplift, highest first, and return cumulative N, N_t, Y_t, Y_c at the
    end of each block of tied scores. There's no right order inside a tie so, like sklift, only
    the last row of each block is kept."""
    uplift = np.asarray(uplift, dtype=float)
    order = np.argsort(-uplift, kind='stable')
    u = uplift[order]
    y = np.asarray(y, dtype=float)[order]
    t = np.asarray(t, dtype=float)[order]
    ends = np.r_[np.flatnonzero(np.diff(u)), len(u) - 1]
    return ends + 1, np.cumsum(t)[ends], np.cumsum(y * t)[ends], np.cumsum(y * (1 - t))[ends]


def qini_curve(y, t, uplift):
    """For the top k users by predicted uplift

        Q(k) = Y_t(k) - Y_c(k) N_t(k) / N_c(k)

    treated responders minus control responders scaled up to the treated count, i.e. the extra
    responders from treating those k. Before the first control user there's nothing to scale, so
    sklift takes the second term as 0 there (Q = Y_t) and so do we. Returns (k, Q) from (0, 0)."""
    n, n_t, y_t, y_c = _cum_counts(y, t, uplift)
    n_c = n - n_t
    ratio = np.divide(n_t, n_c, out=np.zeros_like(n_t), where=n_c > 0)
    return np.r_[0, n], np.r_[0, y_t - y_c * ratio]


def uplift_curve(y, t, uplift):
    """(treated rate - control rate) among the top k, times k. An empty arm counts as rate 0."""
    n, n_t, y_t, y_c = _cum_counts(y, t, uplift)
    n_c = n - n_t
    rate_t = np.divide(y_t, n_t, out=np.zeros_like(y_t), where=n_t > 0)
    rate_c = np.divide(y_c, n_c, out=np.zeros_like(y_c), where=n_c > 0)
    return np.r_[0, n], np.r_[0, (rate_t - rate_c) * n]


def _area(k, q):
    return float(np.sum(np.diff(k) * (q[1:] + q[:-1]) / 2))


def _normalized(model, perfect):
    """(area under model - area under random) / (area under perfect - area under random).
    Random targeting is the straight line to the last point: k random users get k/N of the total.
    0 means no better than random, 1 means the perfect ordering."""
    k, q = perfect
    random_area = k[-1] * q[-1] / 2
    return float((_area(*model) - random_area) / (_area(*perfect) - random_area))


def qini_auc(y, t, uplift):
    """Normalized area under the Qini curve, same as sklift's qini_auc_score. The perfect model
    puts treated responders first and control responders last."""
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    return _normalized(qini_curve(y, t, uplift), qini_curve(y, t, y * t - y * (1 - t)))


def auuc(y, t, uplift):
    """Normalized area under the uplift curve, same as sklift's uplift_auc_score. The perfect
    ordering is copied from sklift: y == t users first, then control responders or treated
    non-responders, whichever group is bigger."""
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    ctrl_responders = np.sum((y == 1) & (t == 0))
    trt_non_responders = np.sum((y == 0) & (t == 1))
    perfect = 2 * (y == t) + (y if ctrl_responders > trt_non_responders else t)
    return _normalized(uplift_curve(y, t, uplift), uplift_curve(y, t, perfect))


def captured_share(y, t, uplift, top=0.2):
    """Q(top N) / Q(N): share of all the incremental outcomes you keep by treating only the top
    fraction by predicted uplift. Random targeting gets ~top. Between curve points (ties) Q is
    read off linearly."""
    k, q = qini_curve(y, t, uplift)
    return float(np.interp(top * k[-1], k, q) / q[-1])


def uplift_by_decile(y, t, uplift, n_bins=10):
    """Mean predicted vs observed uplift (treated rate - control rate) by bin of predicted
    uplift, bin 1 = the top 10%. Tied scores get split between bins by sort order."""
    y, t, uplift = np.asarray(y), np.asarray(t), np.asarray(uplift)
    bins = np.empty(len(uplift), dtype=int)
    bins[np.argsort(-uplift, kind='stable')] = np.arange(len(uplift)) * n_bins // len(uplift)
    rows = []
    for b in range(n_bins):
        m = bins == b
        rate_t, rate_c = y[m & (t == 1)].mean(), y[m & (t == 0)].mean()
        rows.append({'bin': b + 1, 'n': int(m.sum()), 'predicted': float(uplift[m].mean()),
                     'treated_rate': float(rate_t), 'control_rate': float(rate_c),
                     'observed': float(rate_t - rate_c)})
    return rows


def planted_effect_outcomes(base_p, t, effect, in_segment, seed=0):
    """Semi-synthetic outcomes with a known answer, y ~ Bernoulli(base_p + effect * in_segment * t).
    True uplift is `effect` inside the segment and exactly 0 everywhere else."""
    rng = np.random.default_rng(seed)
    p = np.clip(base_p + effect * np.asarray(in_segment) * np.asarray(t), 0, 1)
    return (rng.random(len(p)) < p).astype(np.int8)
