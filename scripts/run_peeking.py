# Peeking on the real data. Criteo has no timestamps, so arrival order is simulated by shuffling
# the 2.1M control users. Each sim splits them 50/50 into fake arms and looks after every 5%.
# Stop at the first p < 0.05 (naive) vs O'Brien-Fleming vs mSPRT, for visits and conversions
# (secondary). Writes results/peeking.json and figures/peeking_fpr.png.
# python scripts/run_peeking.py   (~4 min, ~0.1s per sim on an M4, 1,000 sims x 2 metrics)

import json
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from explab.load import ROOT, connect
from explab.plots import BLUE, GRAY, GREEN, INK, ORANGE, save, setup
from explab.sequential import (cumulative_fpr, msprt_crossings, naive_crossings, obf_boundaries,
                               obf_crossings, peeking_sim)

RESULTS = ROOT / 'results'
N_SIMS, N_LOOKS, ALPHA = 1000, 20, 0.05
TAU_LIFT = 0.05
METHODS = [('naive', 'naive', BLUE), ('obf', "O'Brien-Fleming", ORANGE), ('msprt', 'mSPRT', GREEN)]
TITLES = {'visit': 'visit', 'conversion': 'conversion (secondary)'}


def checks(fpr):
    # validation/pass_conditions.md, peeking row
    return {
        'naive_above_10pct': fpr['naive'] > 0.10,
        'obf_between_3.5_and_6.5pct': 0.035 <= fpr['obf'] <= 0.065,
        'msprt_at_most_6.5pct': fpr['msprt'] <= 0.065,
    }


def run_metric(y, bounds, seed=0):
    sim = peeking_sim(y, n_sims=N_SIMS, n_looks=N_LOOKS, seed=seed)
    # tau is the spread of the mSPRT mixing prior, i.e. the effect size it's tuned to catch:
    # a 5% relative lift. it only changes power (how fast a real effect crosses), the
    # P(ever crossing) <= alpha guarantee holds for any tau
    tau = TAU_LIFT * float(np.mean(y))
    curves = {
        'naive': cumulative_fpr(naive_crossings(sim['z'], ALPHA)),
        'obf': cumulative_fpr(obf_crossings(sim['z'], bounds)),
        'msprt': cumulative_fpr(msprt_crossings(sim['diff'], sim['var'], tau**2, ALPHA)),
    }
    fpr = {k: float(c[-1]) for k, c in curves.items()}
    return sim['frac'], {
        'mean': float(np.mean(y)),
        'tau': tau,
        'fpr': fpr,
        'fpr_se': {k: float(np.sqrt(f * (1 - f) / N_SIMS)) for k, f in fpr.items()},
        'passed': checks(fpr),
        'cumulative_fpr': {k: c.tolist() for k, c in curves.items()},
    }


def label_ends(ax, x, ys, texts, gap):
    # direct labels at the line ends, nudged apart when two lines finish close together
    order = np.argsort(ys)
    pos = np.array(ys, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    for i, y in zip(order, pos):
        ax.annotate(texts[i], (x, y), xytext=(6, 0), textcoords='offset points', va='center',
                    fontsize=8.5, color=INK, annotation_clip=False)


def plot_peeking(frac, res):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    x = 100 * frac
    top = 1.15 * max(max(r['cumulative_fpr']['naive']) for r in res.values())
    for ax, (metric, r) in zip(axes, res.items()):
        ends, texts = [], []
        for key, name, color in METHODS:
            c = r['cumulative_fpr'][key]
            ax.plot(x, c, color=color)
            ends.append(c[-1])
            texts.append(f'{name} {c[-1]:.1%}')
        ax.axhline(ALPHA, color=GRAY, linestyle='--', linewidth=1)
        ax.text(50, ALPHA + 0.01 * top, '5% = what a valid test gives', ha='center',
                va='bottom', color=GRAY, fontsize=8)
        ax.set_ylim(0, top)
        label_ends(ax, x[-1], ends, texts, gap=0.06 * top)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(PercentFormatter(100))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel('users seen so far')
        ax.set_title(TITLES[metric])
    axes[0].set_ylabel('A/A tests already called significant')
    fig.suptitle(f'Peeking after every 5% of users, {N_SIMS:,} A/A splits of the control group',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    save(fig, 'peeking_fpr.png')


def main():
    start = time.time()
    con = connect()
    ctrl = con.sql('select visit, conversion from criteo where treatment = 0').fetchnumpy()
    bounds = obf_boundaries(N_LOOKS, ALPHA)
    print(f'OBF C = {bounds[-1]:.3f}, first look needs |z| > {bounds[0]:.2f}')

    res = {}
    for metric in ['visit', 'conversion']:
        # same seed for both, so it's the same 1,000 fake experiments read on two metrics
        frac, res[metric] = run_metric(ctrl[metric], bounds, seed=2)
        r = res[metric]
        fprs = ', '.join(f"{k} {f:.1%} (se {r['fpr_se'][k]:.1%})" for k, f in r['fpr'].items())
        print(metric, fprs, r['passed'])

    plot_peeking(frac, res)
    runtime = time.time() - start
    print(f'took {runtime / 60:.1f} min')
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / 'peeking.json', 'w') as f:
        json.dump({
            'n_sims': N_SIMS,
            'n_looks': N_LOOKS,
            'alpha': ALPHA,
            'share': 0.5,
            'tau_relative_lift': TAU_LIFT,
            'obf_C': float(bounds[-1]),
            'obf_boundaries': bounds.tolist(),
            'frac': frac.tolist(),
            'metrics': res,
            'runtime_s': runtime,
        }, f, indent=2)


if __name__ == '__main__':
    main()
