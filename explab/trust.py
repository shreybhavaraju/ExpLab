# Trust checks, i.e. can the readout be believed at all.
#   srm_test  - did the users split into arms the way the design says they should?

import numpy as np
from scipy import stats

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

