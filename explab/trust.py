# Trust checks, i.e. can the readout be believed at all.
#   srm_test      - did the users split into arms the way the design says they should?
#   balance_test  - same idea inside bins of a pre-treatment feature
#   aa_test       - does the readout give ~5% false positives when there's nothing to find?

import numpy as np
from scipy import stats

from explab.readout import readout
from explab.variance import cuped_readout

SRM_ALPHA = 0.001


def srm_test(n_t, n_c, expected_share=0.85, alpha=SRM_ALPHA):
    """Sample ratio mismatch: chi-square goodness of fit of the arm sizes against the
    designed split. Criteo is 85/15 so testing against 50/50 would 'fail' for no reason.
    If this fails something upstream (assignment, logging, filtering) is broken and the
    readout can't be trusted no matter what it says."""
    n = n_t + n_c
    observed = np.array([n_t, n_c], dtype=float)
    expected = n * np.array([expected_share, 1 - expected_share])
    chi2 = ((observed - expected) ** 2 / expected).sum()
    p = stats.chi2.sf(chi2, df=1)
    return {
        'share': n_t / n,
        'expected_share': expected_share,
        'chi2': chi2,
        'p_value': p,
        'passed': bool(p >= alpha),
    }


def balance_test(n_t, n_c, alpha=SRM_ALPHA):
    """SRM inside segments. n_t, n_c are the arm sizes in each bin of a pre-treatment feature.
    With real randomization the treatment share is the same in every bin (whatever the overall
    share is), so this is a chi-square test of independence between arm and bin, dof = bins - 1.
    The overall SRM check only looks at the totals, so it can pass while this one fails."""
    table = np.array([n_t, n_c], dtype=float)
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    chi2 = ((table - expected) ** 2 / expected).sum()
    dof = table.shape[1] - 1
    p = stats.chi2.sf(chi2, dof)
    return {
        'shares': (table[0] / table.sum(axis=0)).tolist(),
        'chi2': chi2,
        'dof': dof,
        'p_value': p,
        'passed': bool(p >= alpha),
    }


def aa_test(y, n_sims=1000, share=0.5, seed=0, x=None):
    """A/A test: randomly split one group (control) into two fake arms over and over and run
    the normal readout on each split. There is no real effect, so the p-values should be
    uniform and ~5% of splits should come out 'significant' at 0.05. Returns the p-values.
    With a covariate x (the CUPAC prediction) each split is read with the CUPED adjustment
    instead, to check the adjustment doesn't create false positives of its own."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pvals = np.empty(n_sims)
    for i in range(n_sims):
        fake_t = rng.random(len(y)) < share
        if x is None:
            r = readout(y[fake_t], y[~fake_t])
        else:
            r = cuped_readout(y, x, fake_t)
        pvals[i] = r['diff'].p_value
    return pvals


def aa_summary(pvals, alpha=0.05):
    return {
        'fpr': float(np.mean(pvals < alpha)),
        'ks_p': float(stats.kstest(pvals, 'uniform').pvalue),  # flat histogram <=> uniform
    }
