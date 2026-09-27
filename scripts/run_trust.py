# Trust checks on the real data:
#   - SRM against the designed 85/15 split
#   - SRM again after randomly dropping 2% of control (validation, this one should fail)
#   - A/A test: 1,000 random splits of the control group, for visits and conversions
# Writes results/trust.json, results/aa_pvalues.csv and figures/aa_pvalues.png.

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explab.load import ROOT, connect
from explab.plots import BLUE, GRAY, save, setup
from explab.trust import aa_summary, aa_test, srm_test

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

    plot_aa(pvals, {m: aa[m]['fpr'] for m in aa})
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(pvals).to_csv(RESULTS / 'aa_pvalues.csv', index=False)
    with open(RESULTS / 'trust.json', 'w') as f:
        json.dump({'srm': srm, 'srm_2pct_control_dropped': srm_dropped, 'aa': aa}, f, indent=2)


if __name__ == '__main__':
    main()
