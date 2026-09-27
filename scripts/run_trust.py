# Trust checks on the real data:
#   - SRM against the designed 85/15 split
#   - SRM again after randomly dropping 2% of control (validation, this one should fail)
#   - A/A test: 1,000 random splits of the control group, for visits and conversions
#   - balance: SRM inside the f0 / f2 decile bins (pre-treatment features), and how much the
#     imbalance moves the estimate (raw vs post-stratified on the f0 x f2 cells)
# Writes results/trust.json, results/aa_pvalues.csv, figures/aa_pvalues.png and figures/balance.png.

import json
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explab.load import ROOT, connect
from explab.plots import BLUE, GRAY, save, setup
from explab.readout import binary_mean_and_var, compare, stratified
from explab.trust import aa_summary, aa_test, balance_test, srm_test

RESULTS = ROOT / 'results'


def plot_aa(pvals, fprs):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, (metric, p) in zip(axes, pvals.items()):
        ax.hist(p, bins=bins, color=BLUE, edgecolor='white', linewidth=1.5)
        ax.axhline(len(p) / 20, color=GRAY, linestyle='--', linewidth=1)
        ax.set_title(f'{metric}: {fprs[metric]:.1%} of splits had p < 0.05')
        ax.set_xlabel('p-value')
        ax.set_xlim(0, 1)
    axes[0].set_ylabel('number of A/A splits')
    fig.suptitle('A/A test, 1,000 random splits of the control group', x=0.01, ha='left',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.text(0.01, 0.89, 'dashed line = count per bin if the p-values were perfectly uniform, '
             'which is what a valid test gives', color=GRAY, fontsize=9)
    save(fig, 'aa_pvalues.png')


def bin_table(con, feature):
    return con.sql(f"""
        select {feature}_bin as bin,
               sum(n) filter (where treatment = 1) as n_t,
               sum(n) filter (where treatment = 0) as n_c,
               sum(visits) filter (where treatment = 0) / sum(n) filter (where treatment = 0) as ctrl_visit_rate
        from segment_stats
        group by 1
        order by 1
    """).df()


def imbalance_impact(con):
    cells = con.sql('select * from segment_stats').df()
    t = cells[cells.treatment == 1].set_index(['f0_bin', 'f2_bin'])
    c = cells[cells.treatment == 0].set_index(['f0_bin', 'f2_bin']).loc[t.index]
    out = {}
    for metric, col in [('visit_rate', 'visits'), ('conversion_rate', 'conversions')]:
        raw = compare(*binary_mean_and_var(t[col].sum(), t.n.sum()),
                      *binary_mean_and_var(c[col].sum(), c.n.sum()))
        strat = compare(*stratified(t[col], t.n, c[col], c.n))
        out[metric] = {name: {'diff': asdict(r['diff']), 'lift': asdict(r['lift'])}
                       for name, r in [('raw', raw), ('stratified_f0_f2', strat)]}
        print(f"{metric}: raw diff {raw['diff'].value:+.5f} (lift {raw['lift'].value:+.1%}), "
              f"stratified diff {strat['diff'].value:+.5f} (lift {strat['lift'].value:+.1%})")
    return out


def plot_balance(tables, overall_share):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (feature, df) in zip(axes, tables.items()):
        n = df.n_t + df.n_c
        share = df.n_t / n
        # 99.9% interval for the share if assignment really were 85/15 everywhere
        half = 3.29 * np.sqrt(0.85 * 0.15 / n)
        x = np.arange(len(df))
        ax.fill_between(x, 0.85 - half, 0.85 + half, color=GRAY, alpha=0.15, linewidth=0)
        ax.axhline(0.85, color=GRAY, linestyle='--', linewidth=1)
        ax.plot(x, share, 'o', color=BLUE, markersize=7)
        ax.set_xticks(x, df.bin.astype(int))
        ax.set_xlabel(f'{feature} decile bin')
        ax.set_title(f'{feature}')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1%}'))
    axes[0].set_ylabel('treatment share')
    axes[0].text(0, 0.8515, 'designed 85%, gray band = 99.9% range if it held', color=GRAY, fontsize=8,
                 va='bottom')
    fig.suptitle('Treatment share by feature decile', x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.text(0.01, 0.89, f'overall share is {overall_share:.4%} and passes SRM, but inside the bins it '
             'clearly depends on the features', color=GRAY, fontsize=9)
    save(fig, 'balance.png')


def main():
    con = connect()
    sizes = con.sql('select * from arm_sizes order by treatment').df()
    n_c, n_t = int(sizes.n[0]), int(sizes.n[1])
    srm = srm_test(n_t, n_c)
    print('srm', srm)

    # validation: lose 2% of control users at random (like a logging bug). has to fail.
    rng = np.random.default_rng(0)
    srm_dropped = srm_test(n_t, int(rng.binomial(n_c, 0.98)))
    print('srm with 2% of control dropped', srm_dropped)

    ctrl = con.sql('select visit, conversion from criteo where treatment = 0').fetchnumpy()
    pvals, aa = {}, {}
    for metric in ['visit', 'conversion']:
        p = aa_test(ctrl[metric], n_sims=1000, seed=1)
        pvals[metric] = p
        aa[metric] = aa_summary(p)
        print('aa', metric, aa[metric])

    # overall SRM passes almost too well (share = 0.8500001). check inside segments
    tables, balance = {}, {}
    for feature in ['f0', 'f2']:
        df = bin_table(con, feature)
        tables[feature] = df
        balance[feature] = balance_test(df.n_t.values, df.n_c.values)
        balance[feature]['bins'] = df.bin.astype(int).tolist()
        balance[feature]['ctrl_visit_rate'] = df.ctrl_visit_rate.tolist()
        print('balance', feature, {k: balance[feature][k] for k in ['chi2', 'p_value', 'passed']})
        print(df.assign(share=df.n_t / (df.n_t + df.n_c)).round(4).to_string(index=False))

    impact = imbalance_impact(con)

    plot_aa(pvals, {m: aa[m]['fpr'] for m in aa})
    plot_balance(tables, srm['share'])
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(pvals).to_csv(RESULTS / 'aa_pvalues.csv', index=False)
    with open(RESULTS / 'trust.json', 'w') as f:
        json.dump({'srm': srm, 'srm_2pct_control_dropped': srm_dropped, 'aa': aa, 'balance': balance,
                   'imbalance_impact': impact}, f, indent=2)


if __name__ == '__main__':
    main()
