# Glue: readout + trust checks + power + verdict for one metric on one slice of users, starting
# from the segment_stats view (one row per f0 bin x f2 bin x arm). The app and
# scripts/run_decision.py both go through this so they can't disagree.

from explab.bayes import bayes_readout
from explab.decide import decide
from explab.power import mde
from explab.readout import binary_mean_and_var, compare, ratio_from_sums
from explab.trust import balance_test, srm_test
from explab.variance import cuped_from_sums

LABELS = {
    'visit': 'visit rate',
    'conversion': 'conversion rate',
    'conversions_per_visit': 'conversions per visit',
}
COUNT_COL = {'visit': 'visits', 'conversion': 'conversions'}
# metric that must not get worse when the other one is the primary
GUARDRAIL = {
    'visit': 'conversions_per_visit',
    'conversion': 'visit',
    'conversions_per_visit': 'visit',
}
KEYS = ['f0_bin', 'f2_bin', 'treatment']


def arm_estimate(row, metric):
    """(estimate, variance of the estimate) for one arm from its summed counts."""
    if metric == 'conversions_per_visit':
        # visit and conversion are 0/1, so the sums of squares are just the sums
        return ratio_from_sums(row.n, row.conversions, row.visits, row.conversions, row.visits,
                               row.conv_visits)
    return binary_mean_and_var(row[COUNT_COL[metric]], row.n)


def arm_totals(cells):
    sums = cells.drop(columns=KEYS)
    return sums[cells.treatment == 1].sum(), sums[cells.treatment == 0].sum()


def cupac_estimates(t, c, metric):
    """CUPED-adjusted (est_t, var_t, est_c, var_c) from the per-cell prediction sums that
    scripts/build_app_data.py adds. 0/1 outcome so sum of y^2 = sum of y."""
    col = COUNT_COL[metric]
    sums = [{'n': a.n, 'y': a[col], 'yy': a[col], 'x': a[f'x_{metric}'],
             'xx': a[f'xx_{metric}'], 'xy': a[f'xy_{metric}']} for a in (t, c)]
    est, _ = cuped_from_sums(*sums)
    return est


def balance_checks(cells):
    """balance_test across the f0 and f2 bins present in this slice (skips a feature if the
    slice only covers one of its bins)."""
    out = {}
    for f in ['f0', 'f2']:
        by_bin = cells.groupby([f'{f}_bin', 'treatment']).n.sum().unstack(fill_value=0)
        if len(by_bin) > 1:
            out[f] = balance_test(by_bin[1].values, by_bin[0].values)
    return out


def metric_report(cells, metric, min_effect=0.05, expected_share=0.85, alpha=0.05, cupac=False):
    t, c = arm_totals(cells)
    if cupac:
        est_t, var_t, est_c, var_c = cupac_estimates(t, c, metric)
    else:
        est_t, var_t = arm_estimate(t, metric)
        est_c, var_c = arm_estimate(c, metric)
    res = compare(est_t, var_t, est_c, var_c, alpha)

    srm = srm_test(t.n, c.n, expected_share)
    balance = balance_checks(cells)
    # per-user variance from the control arm (var of the mean is var / n). with CUPAC on this
    # is the adjusted variance, so the MDE drops too
    mde_rel = mde(var_c * c.n, t.n, c.n, alpha) / est_c

    g = GUARDRAIL[metric]
    guard = compare(*arm_estimate(t, g), *arm_estimate(c, g), alpha)['lift']
    decision = decide(res['lift'], srm, mde_rel, min_effect, balance, {LABELS[g]: guard})

    out = {
        'metric': metric,
        'n_t': int(t.n),
        'n_c': int(c.n),
        'readout': res,
        'srm': srm,
        'balance': balance,
        'mde_rel': mde_rel,
        'guardrail': (LABELS[g], guard),
        'decision': decision,
    }
    if metric in COUNT_COL:
        col = COUNT_COL[metric]
        out['bayes'] = bayes_readout(t[col], t.n, c[col], c.n, draws=100_000)
    return out
