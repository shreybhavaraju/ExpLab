# CUPAC on the real data. Criteo has no pre-period metric, so each metric's covariate is an
# out-of-fold LightGBM prediction of it from f0..f11 (see explab/variance.py).
#   - raw vs CUPAC readout for visit and conversion: diff, lift, CI widths, var reduction
#   - balance check: mean prediction in treatment vs control, should be ~0 in a clean test
#   - the A/A test on the control group again, with the adjustment on
# Writes data/cupac_preds.parquet, results/cupac.json and figures/cupac_ci.png.
# python scripts/run_cupac.py   (~5 min on an M4: ~2 min of LightGBM fits, ~2 min of A/A splits)

import json
import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from explab.load import FEATURES, ROOT, connect
from explab.plots import BG, BLUE, GRAY, ORANGE, save, setup
from explab.readout import readout
from explab.trust import aa_summary, aa_test
from explab.variance import balance, cuped_readout, oof_predictions

RESULTS = ROOT / 'results'
PREDS_PATH = ROOT / 'data' / 'cupac_preds.parquet'
METRICS = ['visit', 'conversion']


def load(con):
    # a plain scan keeps the parquet row order (duckdb's preserve_insertion_order is on by
    # default), so row i of the saved predictions lines up with row i of the data
    cols = ', '.join(FEATURES)
    data = con.sql(f'select {cols}, treatment, visit, conversion from criteo').fetchnumpy()
    X = np.column_stack([data.pop(f) for f in FEATURES]).astype(np.float32, copy=False)
    return X, data['treatment'] == 1, {m: data[m] for m in METRICS}


def fit_predictions(X, ys):
    preds = {}
    for m, y in ys.items():
        start = time.time()
        preds[m] = oof_predictions(X, y)
        print(f'oof {m}: {time.time() - start:.0f}s')
    return preds


def width(e):
    return e['ci_high'] - e['ci_low']


def compare_metric(y, x, t):
    raw = readout(y[t], y[~t])
    adj = cuped_readout(y, x, t)
    bal = balance(x, t)
    out = {
        'auc': float(roc_auc_score(y, x)),
        'corr': float(np.corrcoef(x, y)[0, 1]),
        'theta': adj['theta'],
        'var_reduction': adj['var_reduction'],
        'raw': {k: asdict(raw[k]) for k in ['diff', 'lift']},
        'cupac': {k: asdict(adj[k]) for k in ['diff', 'lift']},
    }
    # the lift CI also shrinks when the lift itself moves down (its SE scales with the level),
    # so the diff number is the cleaner read on the variance reduction
    out['ci_width_reduction'] = {
        k: 1 - width(out['cupac'][k]) / width(out['raw'][k]) for k in ['diff', 'lift']}
    # adjusted diff = raw diff - theta * (mean x treated - mean x control), so any imbalance
    # in the prediction moves the point estimate by exactly this much
    out['balance'] = {
        'mean_pred_treatment': float(x[t].mean()),
        'mean_pred_control': float(x[~t].mean()),
        'diff': bal.value,
        'z': bal.value / bal.se,
        'p_value': bal.p_value,
        'estimate_shift': adj['diff'].value - raw['diff'].value,
        'estimate_shift_in_raw_se': (adj['diff'].value - raw['diff'].value) / raw['diff'].se,
    }
    return out


def show(m, r):
    print(f"\n{m}: oof auc {r['auc']:.3f}  corr {r['corr']:.3f}  theta {r['theta']:.3f}  "
          f"var reduction {r['var_reduction']:.1%}")
    for name in ['raw', 'cupac']:
        d, l = r[name]['diff'], r[name]['lift']
        print(f"  {name:5s}  diff {d['value']:+.5f} [{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]  "
              f"lift {l['value']:+.2%} [{l['ci_low']:+.2%}, {l['ci_high']:+.2%}]")
    w, b = r['ci_width_reduction'], r['balance']
    print(f"  CI width: diff {w['diff']:.1%} narrower, lift {w['lift']:.1%} narrower")
    print(f"  balance: mean prediction trt {b['mean_pred_treatment']:.5f} vs ctrl "
          f"{b['mean_pred_control']:.5f}, diff {b['diff']:+.6f} (z = {b['z']:.1f})")
    if abs(b['z']) > 3:
        print('  not balanced. a clean experiment shows z ~ 0 here, this is the treatment share '
              'varying with the features again')
    print(f"  CUPAC estimate moves by {b['estimate_shift']:+.5f} "
          f"({b['estimate_shift_in_raw_se']:+.1f} raw SEs) = -theta * balance diff")


def plot_cis(res):
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    for ax, m in zip(axes, METRICS):
        for row, (name, color) in enumerate([('raw', BLUE), ('cupac', ORANGE)]):
            d = res[m][name]['diff']
            lo, val, hi = 100 * d['ci_low'], 100 * d['value'], 100 * d['ci_high']
            ax.plot([lo, hi], [row, row], color=color, solid_capstyle='butt')
            ax.plot(val, row, 'o', color=color, markersize=6)
            # BG box so the dashed reference line doesn't run through the numbers
            ax.text(val, row - 0.22, f'{val:+.3f}  [{lo:+.3f}, {hi:+.3f}]', ha='center',
                    va='bottom', color=GRAY, fontsize=8,
                    bbox={'facecolor': BG, 'edgecolor': 'none', 'pad': 1})
        raw_val = 100 * res[m]['raw']['diff']['value']
        ax.axvline(raw_val, color=GRAY, linestyle='--', linewidth=1)
        ax.set_yticks([0, 1], ['raw', 'CUPAC'])
        for label, color in zip(ax.get_yticklabels(), [BLUE, ORANGE]):
            label.set_color(color)
        ax.yaxis.grid(False)
        ax.set_ylim(1.5, -0.7)  # raw on top
        ax.margins(x=0.3)
        ax.set_title(f"{m}: CI {res[m]['ci_width_reduction']['diff']:.0%} narrower")
        ax.set_xlabel('treatment - control (percentage points)')
    fig.suptitle('Raw vs CUPAC 95% CI on the absolute difference (dashed = raw estimate)',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    save(fig, 'cupac_ci.png')


def main():
    start = time.time()
    X, t, ys = load(connect())
    print(f'loaded {len(t):,} rows: {time.time() - start:.0f}s')
    preds = fit_predictions(X, ys)
    del X
    # float32 is plenty for a saved probability and halves the file. the in-memory copy stays
    # float64, a float32 mean over 12M rows is only good to ~1e-6
    out = pd.DataFrame({f'cupac_{m}': preds[m].astype(np.float32) for m in METRICS})
    out.to_parquet(PREDS_PATH, index=False)

    res = {}
    for m in METRICS:
        res[m] = compare_metric(ys[m], preds[m], t)
        show(m, res[m])
        # same A/A setup as run_trust.py (control group, 1,000 splits, seed 1) with CUPAC on
        p = aa_test(ys[m][~t], n_sims=1000, seed=1, x=preds[m][~t])
        res[m]['aa'] = aa_summary(p)
        print('aa with cupac', m, res[m]['aa'])

    plot_cis(res)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / 'cupac.json', 'w') as f:
        json.dump(res, f, indent=2)
    print(f'done in {(time.time() - start) / 60:.1f} min')


if __name__ == '__main__':
    main()
