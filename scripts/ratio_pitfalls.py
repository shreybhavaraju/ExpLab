# Conversions per visit with the naive variance (every visit an independent Bernoulli trial)
# vs the delta method on per-user sums.
#   part 1: real data, per arm. on Criteo the two SEs are identical, see the algebra below
#   part 2: simulated users with several visits each, where the naive CI covers well under 95%
# Writes results/ratio_pitfalls.json and figures/ratio_pitfalls.png. Takes ~15s.

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from explab.load import ROOT, connect
from explab.plots import BLUE, GRAY, INK, ORANGE, save, setup
from explab.readout import compare, ratio_and_var, ratio_from_sums

RESULTS = ROOT / 'results'

# visits ~ Poisson(lam) with lam ~ Gamma(shape 0.5, mean 3), so most users barely visit and a
# few visit a lot. each user has their own conversion propensity q ~ Beta.
N_USERS = 20_000
N_SIMS = 1000
VISIT_MEAN, VISIT_SHAPE = 3.0, 0.5
Q_C, Q_T = (0.5, 9.5), (0.55, 9.45)  # means 0.05 and 0.055, true lift +10%


def naive_var(r, visits):
    """Every visit an independent Bernoulli(r) trial: Var(r_hat) = r (1 - r) / visits."""
    return r * (1 - r) / visits


# Why naive and delta agree on Criteo: x = conversion and y = visit are both 0/1 with x <= y,
# so x^2 = x, y^2 = y and xy = x. Writing mx = R my:
#     Var(x) = R my (1 - R my),  Var(y) = my (1 - my),  Cov(x, y) = mx - mx my = R my (1 - my)
#     Var(x) - 2R Cov(x, y) + R^2 Var(y)
#         = R my - R^2 my^2 - 2 R^2 my + 2 R^2 my^2 + R^2 my - R^2 my^2
#         = R my (1 - R)
# and dividing by n my^2 gives R (1 - R) / (n my) = R (1 - R) / visits, the naive formula
# (up to the n / (n - 1) in the sample variances). With one visit per user there's nothing
# for visits to be correlated with, so treating them as independent is fine here.
def real_data_ses(con):
    """Naive vs delta method SE of conversions per visit for each arm, from the SQL view."""
    df = con.sql('select * from conversions_per_visit order by treatment').df()
    out = {}
    for row in df.itertuples():
        # 0/1 columns, so the sums of squares are the plain sums (same as run_readout.py)
        r, var = ratio_from_sums(row.n, row.conversions, row.visits, row.conversions,
                                 row.visits, row.conv_visits)
        out['treatment' if row.treatment else 'control'] = {
            'n': int(row.n),
            'visits': int(row.visits),
            'conversions': int(row.conversions),
            'ratio': float(r),
            'naive_se': float(np.sqrt(naive_var(r, row.visits))),
            'delta_se': float(np.sqrt(var)),
        }
    return out


def simulate_arm(rng, n, q_prior):
    """Per-user visits and conversions. All of a user's visits share the same q, so they're
    correlated, and the heavy visitors get counted many times by the naive formula."""
    lam = rng.gamma(VISIT_SHAPE, VISIT_MEAN / VISIT_SHAPE, n)
    visits = rng.poisson(lam)
    convs = rng.binomial(visits, rng.beta(*q_prior, n))
    return convs, visits


def simulate(n_sims=N_SIMS, n_users=N_USERS, seed=0):
    """n_sims fake experiments. Per arm: the ratio, its delta method variance and the naive one."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_sims):
        row = []
        for q_prior in (Q_T, Q_C):
            convs, visits = simulate_arm(rng, n_users, q_prior)
            r, var = ratio_and_var(convs, visits)
            row += [r, var, naive_var(r, visits.sum())]
        rows.append(row)
    keys = ['r_t', 'delta_var_t', 'naive_var_t', 'r_c', 'delta_var_c', 'naive_var_c']
    return dict(zip(keys, np.array(rows).T))


def true_ratio(q_prior):
    # lam and q are independent, so E[conv] / E[visits] = E[lam q] / E[lam] = E[q]
    a, b = q_prior
    return a / (a + b)


def coverage(est, truth):
    return float(np.mean((est.ci_low < truth) & (truth < est.ci_high)))


def plot(cov, true_lift, n_sims=N_SIMS):
    setup()
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for y, key in enumerate(['lift', 'diff']):
        for method, label, color, side in [('naive', 'naive', ORANGE, -1),
                                           ('delta', 'delta method', BLUE, 1)]:
            c = cov[method][key]
            err = 1.96 * np.sqrt(c * (1 - c) / n_sims)  # monte carlo error of the coverage
            ax.errorbar(c, y, xerr=err, fmt='o', color=color, markersize=8, elinewidth=2,
                        capsize=0)
            ax.text(c + side * (err + 0.006), y, f'{label} {c:.1%}', va='center',
                    ha='right' if side < 0 else 'left', color=INK, fontsize=9)
    ax.axvline(0.95, color=GRAY, linestyle='--', linewidth=1)
    ax.text(0.953, 1.45, '95% target', ha='left', va='center', color=GRAY, fontsize=8)
    ax.set_xlim(min(cov['naive'].values()) - 0.09, 1.0)
    ax.set_ylim(-0.6, 1.7)
    ax.set_yticks([0, 1], ['relative lift CI', 'absolute diff CI'])
    ax.grid(axis='y', visible=False)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(f'share of {n_sims:,} simulated experiments where the CI covers the truth '
                  '(bars: monte carlo error)')
    ax.set_title(f'{N_USERS // 1000}k users per arm, ~{VISIT_MEAN:.0f} visits each on average, '
                 f'true lift {true_lift:+.0%}', color=GRAY)
    fig.suptitle('Conversions per visit: the naive CI is too narrow once users visit more '
                 'than once', x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    save(fig, 'ratio_pitfalls.png')


def main():
    real = real_data_ses(connect())
    for arm, s in real.items():
        print(f"{arm:9s} R {s['ratio']:.5f}  naive se {s['naive_se']:.4e}  "
              f"delta se {s['delta_se']:.4e}")

    sim = simulate()
    true_c, true_t = true_ratio(Q_C), true_ratio(Q_T)
    truth = {'diff': true_t - true_c, 'lift': true_t / true_c - 1}
    cov = {}
    for method in ['naive', 'delta']:
        r = compare(sim['r_t'], sim[f'{method}_var_t'], sim['r_c'], sim[f'{method}_var_c'])
        cov[method] = {k: coverage(r[k], truth[k]) for k in truth}
        print(method, cov[method])

    # the actual spread of R_hat across experiments, which a good SE should match
    se_t = {
        'sd_across_sims': float(sim['r_t'].std(ddof=1)),
        'mean_naive_se': float(np.sqrt(sim['naive_var_t']).mean()),
        'mean_delta_se': float(np.sqrt(sim['delta_var_t']).mean()),
    }
    print('treatment se', se_t)

    plot(cov, truth['lift'])
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / 'ratio_pitfalls.json', 'w') as f:
        json.dump({
            'real_data': real,
            'simulation': {
                'n_users_per_arm': N_USERS,
                'n_sims': N_SIMS,
                'visits_mean': VISIT_MEAN,
                'visits_gamma_shape': VISIT_SHAPE,
                'q_beta_control': list(Q_C),
                'q_beta_treatment': list(Q_T),
                'true_ratio_control': true_c,
                'true_ratio_treatment': true_t,
                'true_lift': truth['lift'],
                'coverage': cov,
                'treatment_se': se_t,
            },
        }, f, indent=2)


if __name__ == '__main__':
    main()
