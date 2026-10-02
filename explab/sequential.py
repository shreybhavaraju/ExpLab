# Peeking: what happens to the false positive rate if you keep checking the readout as users
# come in, and two ways to fix it (O'Brien-Fleming for planned looks, mSPRT for any looks).
# Criteo has no timestamps, so arrival order is simulated by shuffling users.

import numpy as np
from scipy import stats


def peeking_sim(y, n_sims=1000, n_looks=20, share=0.5, seed=0):
    """A/A test with peeking. Each sim shuffles the users into a random arrival order, splits
    them into two fake arms with P(treatment) = share, and computes the readout after every
    1/n_looks of the users. Running sums of y and y^2 per arm give the mean and s^2 / n at each
    look for any per-user metric. diff, var (= s_t^2 / n_t + s_c^2 / n_c, same as readout.diff)
    and z = diff / sqrt(var) are (n_sims, n_looks), frac is the share of users seen at each look.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    n = len(y)
    ends = np.arange(1, n_looks + 1) * n // n_looks - 1  # index of the last user at each look
    diff = np.empty((n_sims, n_looks))
    var = np.empty((n_sims, n_looks))
    for i in range(n_sims):
        y_arrival = y[rng.permutation(n)]
        t = rng.random(n) < share
        means, var_means = [], []
        for arm in (t, ~t):
            cnt = np.cumsum(arm)[ends]
            s = np.cumsum(y_arrival * arm)[ends]
            ss = np.cumsum(y_arrival**2 * arm)[ends]
            mean = s / cnt
            means.append(mean)
            var_means.append((ss - cnt * mean**2) / (cnt - 1) / cnt)
        diff[i] = means[0] - means[1]
        var[i] = var_means[0] + var_means[1]
    return {'diff': diff, 'var': var, 'z': diff / np.sqrt(var), 'frac': (ends + 1) / n}


def naive_crossings(z, alpha=0.05):
    """Plain two-sided z-test at every look, as if each look were the only one."""
    return np.abs(z) > stats.norm.ppf(1 - alpha / 2)


def obf_boundaries(n_looks, alpha=0.05, n_paths=200_000, seed=0):
    """O'Brien-Fleming boundaries for K equally spaced looks: stop at look k if
    |z_k| > b_k = C sqrt(K / k). Very strict early (with K = 20 the first look needs |z| > ~9.5)
    and only a bit above 1.96 at the end (C ~ 2.13), so the alpha 'spent' on the early looks is
    tiny and the last look keeps nearly all its power.

    C is calibrated by simulating null z paths, z_k = S_k / sqrt(k) with S_k a sum of k iid
    N(0, 1). |z_k| > C sqrt(K / k) is the same as |S_k| > C sqrt(K), so C is the 1 - alpha
    quantile of max_k |S_k| / sqrt(K). This needs the looks fixed up front, the Lan-DeMets
    spending function version gives OBF-like boundaries for unplanned looks.
    """
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((n_paths, n_looks)).cumsum(axis=1)
    c = np.quantile(np.abs(s).max(axis=1) / np.sqrt(n_looks), 1 - alpha)
    k = np.arange(1, n_looks + 1)
    return c * np.sqrt(n_looks / k)


def obf_crossings(z, bounds):
    return np.abs(z) > bounds


def msprt_lr(diff, var, tau2):
    """Mixture SPRT likelihood ratio (Johari et al. 2017, 'always valid p-values'). Treat the
    estimate as diff ~ N(theta, var) and average the likelihood ratio over theta ~ N(0, tau2):

        Lambda = N(diff; 0, var + tau2) / N(diff; 0, var)
               = sqrt(var / (var + tau2)) exp(tau2 diff^2 / (2 var (var + tau2)))

    i.e. a Bayes factor. Under the null it's a martingale, so by Ville's inequality
    P(Lambda ever >= 1 / alpha) <= alpha no matter how often you look. That bound covers
    looking after every user, so with only 20 looks it's conservative. Uses the normal approx
    with the plug-in variance, which is fine for 0/1 data at these sizes.
    """
    # on a real effect with millions of users the exponent can be in the thousands and exp()
    # overflows to inf, which still gives the right answer (crosses, p = 0), so no warning
    with np.errstate(over='ignore'):
        return np.sqrt(var / (var + tau2)) * np.exp(tau2 * diff**2 / (2 * var * (var + tau2)))


def msprt_crossings(diff, var, tau2, alpha=0.05):
    return msprt_lr(diff, var, tau2) >= 1 / alpha


def always_valid_p(diff, var, tau2):
    """p_k = min(1, 1 / Lambda_k), carried forward as a running min so it never goes back up.
    Stopping the first time p_k <= alpha is the same as msprt_crossings."""
    p = np.minimum(1, 1 / msprt_lr(diff, var, tau2))
    return np.minimum.accumulate(p, axis=-1)


def cumulative_fpr(crossings):
    """Share of sims that have crossed at or before each look, i.e. the false positive rate of
    'stop at the first crossing' if the test had been planned to end at that look."""
    return np.logical_or.accumulate(crossings, axis=1).mean(axis=0)
