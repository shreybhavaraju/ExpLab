# Variance reduction with CUPAC. Criteo has no pre-period metric to use for CUPED, so the
# covariate is an out-of-fold model prediction of the outcome from the pre-treatment features.

import numpy as np

from explab.readout import compare, diff, mean_and_var

# small and regularized on purpose, the prediction only has to correlate with y, it doesn't
# have to be a great model. min_child_samples is high because conversions are ~0.3% positive.
LGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 200,
    'verbose': -1,
}


def oof_predictions(X, y, n_folds=5, max_train_rows=2_000_000, seed=0, params=None,
                    train_on=None):
    """Out-of-fold P(y = 1 | features) from LightGBM, to use as the CUPAC covariate.

    Each fold's model is trained on the other folds and only predicts the held-out one, so
    user i's prediction never saw y_i. An in-sample prediction has seen y_i, and y_i depends
    on treatment, so the covariate would end up correlated with treatment. The adjustment
    would then subtract part of the real effect and fake a variance reduction.
    X should be the pre-treatment features only, never the treatment flag.

    train_on is an optional bool mask of the rows the models may learn from (everyone still
    gets a prediction). Passing the control group makes the covariate "what this user would
    do without treatment". In a clean experiment it doesn't matter, but when the treatment
    share depends on the features (Criteo), a model fit on both arms partly learns the
    treatment effect and the adjustment would subtract some of it.
    """
    # imported here so the app (which only needs the adjustment) doesn't need lightgbm
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import KFold

    params = {**LGB_PARAMS, 'random_state': seed, **(params or {})}
    rng = np.random.default_rng(seed)
    X, y = np.asarray(X), np.asarray(y)
    preds = np.empty(len(y))
    for train, test in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        if train_on is not None:
            train = train[train_on[train]]
        # the full data has ~11M training rows per fold, 2M is plenty for 12 features and
        # keeps each fit to ~10s
        if len(train) > max_train_rows:
            train = np.sort(rng.choice(train, max_train_rows, replace=False))
        model = LGBMClassifier(**params).fit(X[train], y[train])
        preds[test] = model.predict_proba(X[test])[:, 1]
    return preds


def cuped(y, x):
    """CUPED adjustment y_adj = y - theta (x - mean(x)), with theta = Cov(y, x) / Var(x),
    i.e. the OLS slope of y on x. theta is pooled over all users of both arms: with the same
    theta everywhere the adjustment to the diff is theta (xbar_t - xbar_c), which has mean 0
    when x is a pre-treatment variable, so the estimate stays unbiased. Var(y_adj) is
    Var(y) (1 - corr(x, y)^2), which is where the narrower CI comes from."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    theta = np.dot(xc, y - y.mean()) / np.dot(xc, xc)
    return y - theta * xc, theta


def cuped_readout(y, x, treatment, alpha=0.05):
    """Readout (diff + lift, same as readout.compare) on the CUPED-adjusted outcome.
    Centering x at its pooled mean keeps the adjusted arm means on the original scale, so
    the lift is still relative to the control mean. Also returns theta and var_reduction."""
    t = np.asarray(treatment).astype(bool)
    y_adj, theta = cuped(y, x)
    out = compare(*mean_and_var(y_adj[t]), *mean_and_var(y_adj[~t]), alpha=alpha)
    out['theta'] = float(theta)
    out['var_reduction'] = float(1 - y_adj.var() / np.var(y))
    return out


def balance(x, treatment, alpha=0.05):
    """Mean covariate in treatment minus control, with its SE / CI / p-value. In a randomized
    experiment this is ~0. If it isn't, CUPAC moves the point estimate by -theta times it."""
    t = np.asarray(treatment).astype(bool)
    x = np.asarray(x, dtype=float)
    return diff(*mean_and_var(x[t]), *mean_and_var(x[~t]), alpha=alpha)


def cuped_from_sums(t, c):
    """cuped_readout from per-arm sums instead of per-user arrays, which is all the app has.
    t and c are dicts with n, y, yy, x, xx, xy (sums of y, y^2, x, x^2, x*y over the arm).
    Same pooled theta, then for each arm
        mean = ybar - theta (xbar_arm - xbar_pooled)
        var  = (Var(y) + theta^2 Var(x) - 2 theta Cov(x, y)) / n      (within the arm)
    Returns (est_t, var_t, est_c, var_c) for compare(), and theta."""
    n = t['n'] + c['n']
    sx, sy = t['x'] + c['x'], t['y'] + c['y']
    cov = (t['xy'] + c['xy'] - sx * sy / n) / (n - 1)
    var_x = (t['xx'] + c['xx'] - sx**2 / n) / (n - 1)
    theta = cov / var_x
    out = []
    for a in (t, c):
        my, mx = a['y'] / a['n'], a['x'] / a['n']
        v_y = (a['yy'] - a['n'] * my**2) / (a['n'] - 1)
        v_x = (a['xx'] - a['n'] * mx**2) / (a['n'] - 1)
        c_xy = (a['xy'] - a['n'] * mx * my) / (a['n'] - 1)
        out += [my - theta * (mx - sx / n), (v_y + theta**2 * v_x - 2 * theta * c_xy) / a['n']]
    return out, theta
