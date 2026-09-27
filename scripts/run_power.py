# Power and MDE on the real data:
#   - MDE (absolute and relative to the control mean) at the actual arm sizes, for all 3 metrics
#   - simulated power curves for visit and conversion (resampled control users with a known
#     lift injected, 200 runs per point) next to the normal approximation
#   - CI coverage validation: inject a 5% lift, 1,000 runs at full arm sizes, the diff and lift
#     CIs have to cover it 93.5%..96.5% of the time (validation/pass_conditions.md)
# Writes results/power.json and figures/power_curves.png.
# python scripts/run_power.py   (~8 min, every run draws an 11.9M-user treatment arm)

import json
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from explab.load import ROOT, connect
from explab.plots import BLUE, GRAY, ORANGE, save, setup
from explab.power import ci_coverage, mde, power_at, simulate_power
from explab.readout import ratio_from_sums

RESULTS = ROOT / 'results'
# relative lifts for the curves, from 0 to a bit over 2x the MDE
CURVES = {
    'visit_rate': ('visit', np.linspace(0, 0.025, 9)),
    'conversion_rate': ('conversion', np.linspace(0, 0.10, 9)),
}
N_SIMS_POWER = 200
N_SIMS_COVERAGE = 1000
COVERAGE_LIFT = 0.05
COVERAGE_BAND = (0.935, 0.965)


def control_stats(con):
    """Control mean and per-user variance for each metric, from the SQL views."""
    out = {}
    for metric, col in [('visit_rate', 'visits'), ('conversion_rate', 'conversions')]:
        c = con.sql(f'select * from {metric} where treatment = 0').df().iloc[0]
        p = c[col] / c.n
        out[metric] = (float(p), float(p * (1 - p)))

    # visit and conversion are 0/1, so the sums of squares are just the sums
    c = con.sql('select * from conversions_per_visit where treatment = 0').df().iloc[0]
    r, var_r = ratio_from_sums(c.n, c.conversions, c.visits, c.conversions, c.visits,
                               c.conv_visits)
    # var_r is the variance of R_hat over n_c users, times n_c gives it per user
    out['conversions_per_visit'] = (float(r), float(var_r * c.n))
    return out


def plot_curves(curves, mdes, n_t, n_c):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (metric, c) in zip(axes, curves.items()):
        mean, var, m = (mdes[metric][k] for k in ['control', 'per_user_var', 'mde_rel'])
        rel, sim = np.array(c['rel_lifts']), np.array(c['simulated'])
        grid = np.linspace(0, rel[-1], 200)
        analytic = power_at(grid * mean, var, n_t, n_c)

        ax.plot(grid, analytic, color=BLUE)
        se = np.sqrt(sim * (1 - sim) / N_SIMS_POWER)
        ax.errorbar(rel, sim, yerr=2 * se, fmt='o', color=ORANGE, ms=5, elinewidth=1,
                    capsize=0)
        ax.axhline(0.8, color=GRAY, linestyle='--', linewidth=1)
        ax.text(0, 0.82, ' 80% power', color=GRAY, fontsize=8, va='bottom')
        ax.plot([m, m], [0, 0.8], color=GRAY, linestyle='--', linewidth=1)
        ax.text(m, 0.03, f' MDE {m:.2%}', color=GRAY, fontsize=8)

        # direct labels instead of a legend
        x95 = grid[min(np.searchsorted(analytic, 0.95), len(grid) - 1)]
        ax.text(x95, 0.95, 'normal approximation  ', color=BLUE, fontsize=8, ha='right',
                va='center')
        ax.text(rel[-1], 0.6, f'simulated, {N_SIMS_POWER} runs\nper point (±2 SE)',
                color=ORANGE, fontsize=8, ha='right', va='center')

        ax.set_title(f"{metric.replace('_', ' ')}, control {mean:.3%}")
        ax.set_xlabel('true relative lift')
        ax.set_xlim(-0.02 * rel[-1], 1.02 * rel[-1])
        ax.set_ylim(0, 1.03)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylabel('power (share of runs with p < 0.05)')
    fig.suptitle(f'Power at the real arm sizes ({n_t / 1e6:.1f}M treatment, '
                 f'{n_c / 1e6:.1f}M control)', x=0.01, ha='left', fontsize=12,
                 fontweight='bold')
    fig.tight_layout(w_pad=2)
    save(fig, 'power_curves.png')


def main():
    t0 = time.time()
    con = connect()
    sizes = con.sql('select * from arm_sizes order by treatment').df()
    n_c, n_t = int(sizes.n[0]), int(sizes.n[1])

    mdes = {}
    for metric, (mean, var) in control_stats(con).items():
        m = float(mde(var, n_t, n_c))
        mdes[metric] = {'control': mean, 'per_user_var': var, 'mde_abs': m, 'mde_rel': m / mean}
        print(f'{metric:22s} control {mean:.5f}  MDE {m:.6f} abs, {m / mean:.2%} relative')

    ctrl = con.sql('select visit, conversion from criteo where treatment = 0').fetchnumpy()
    curves = {}
    for metric, (col, rel_lifts) in CURVES.items():
        print(f'power curve for {metric}: {len(rel_lifts)} lifts x {N_SIMS_POWER} runs '
              f'({time.time() - t0:.0f}s)')
        mean, var = mdes[metric]['control'], mdes[metric]['per_user_var']
        sim = simulate_power(ctrl[col], n_t, n_c, rel_lifts, n_sims=N_SIMS_POWER, seed=1)
        analytic = power_at(rel_lifts * mean, var, n_t, n_c)
        curves[metric] = {
            'rel_lifts': rel_lifts.tolist(),
            'simulated': sim.tolist(),
            'analytic': analytic.tolist(),
        }
        for rel, s, a in zip(rel_lifts, sim, analytic):
            print(f'  lift {rel:6.2%}  simulated {s:6.1%}  analytic {a:6.1%}')

    lo, hi = COVERAGE_BAND
    coverage = {'rel_lift': COVERAGE_LIFT, 'n_sims': N_SIMS_COVERAGE, 'band': [lo, hi]}
    for metric, (col, _) in CURVES.items():
        print(f'ci coverage for {metric}: {N_SIMS_COVERAGE} runs ({time.time() - t0:.0f}s)')
        cov = ci_coverage(ctrl[col], n_t, n_c, COVERAGE_LIFT, n_sims=N_SIMS_COVERAGE, seed=2)
        cov['passed'] = all(lo <= v <= hi for v in cov.values())
        coverage[metric] = cov
        print(f'  {cov}')

    plot_curves(curves, mdes, n_t, n_c)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / 'power.json', 'w') as f:
        json.dump({'n_t': n_t, 'n_c': n_c, 'mde': mdes, 'power_curves': curves,
                   'ci_coverage': coverage}, f, indent=2)
    print(f'done ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
