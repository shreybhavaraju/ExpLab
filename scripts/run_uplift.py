# Uplift models on the real data, outcome = visit.
#   - 70/30 train/test split stratified by treatment. T-learner and X-learner fit on the 70%
#     (each model capped at 3M training rows), Qini / AUUC / policy numbers on the held-out 30%
#   - semi-synthetic check: real features, fake outcomes with a +3pt visit effect planted only in
#     the bottom 20% of f8, to see if both learners find it
# Writes results/uplift.json and figures/qini.png. Takes ~1-2 min on the M4 (most of it
# is the six LightGBM fits on up to 3M rows).
#
# The treatment share drifts across feature deciles (84.6% to 87.7%), so the raw treated vs
# control gap is partly imbalance, not the ad (1.03pp raw vs ~0.80pp stratified). Qini scores
# every group the model picks with that same raw comparison, so the incremental visits and Qini
# numbers on the real data are a bit optimistic. The semi-synthetic check uses a fresh 85/15
# assignment, so it doesn't have this problem.

import json
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
from sklearn.model_selection import train_test_split

from explab.load import FEATURES, ROOT, connect
from explab.plots import BLUE, GRAY, ORANGE, save, setup
from explab.uplift import (PARAMS, auuc, captured_share, planted_effect_outcomes, qini_auc,
                           qini_curve, t_learner, uplift_by_decile, x_learner)

RESULTS = ROOT / 'results'
LEARNERS = {'T-learner': t_learner, 'X-learner': x_learner}
MAX_TRAIN_ROWS = 3_000_000
TOPS = [0.1, 0.2, 0.3, 0.5]
SEG_FEATURE, EFFECT = 'f8', 0.03  # planted in the bottom 20% of f8
SEED = 0


def load():
    cols = ', '.join(FEATURES)
    d = connect().sql(f'select {cols}, treatment, visit from criteo').fetchnumpy()
    return np.column_stack([d[f] for f in FEATURES]), d['visit'], d['treatment']


def split(n, t, seed=SEED):
    return train_test_split(np.arange(n), test_size=0.3, stratify=t, random_state=seed)


def evaluate(y, t, uplift):
    return {
        'qini_auc': qini_auc(y, t, uplift),
        'auuc': auuc(y, t, uplift),
        # random targeting would keep ~top of the total here
        'captured_share': {str(top): captured_share(y, t, uplift, top) for top in TOPS},
        'by_decile': uplift_by_decile(y, t, uplift),
    }


def real_data(X, y, t, max_train_rows=MAX_TRAIN_ROWS):
    tr, te = split(len(y), t)
    out = {'n_train': len(tr), 'n_test': len(te), 'max_train_rows': max_train_rows}
    X_tr, y_tr, t_tr, X_te, y_te, t_te = X[tr], y[tr], t[tr], X[te], y[te], t[te]
    curves = {}
    for name, learner in LEARNERS.items():
        # the X-learner keeps g at the design 0.85 even though the cap leaves fewer treated rows
        # for training, any g in [0, 1] is fine, it just shifts weight between tau0 and tau1
        u = learner(X_tr, y_tr, t_tr, X_te, max_train_rows=max_train_rows, seed=SEED)
        out[name] = evaluate(y_te, t_te, u)
        curves[name] = qini_curve(y_te, t_te, u)
        print(name, {k: v for k, v in out[name].items() if k != 'by_decile'})
    noise = np.random.default_rng(SEED).random(len(te))
    out['random'] = {'qini_auc': qini_auc(y_te, t_te, noise), 'auuc': auuc(y_te, t_te, noise)}
    return out, curves


def semi_synthetic(X, y, t, n=1_000_000, seed=SEED):
    """Real features, fake visits. base_p comes from a model fit on a separate control sample,
    then +EFFECT is planted for treated users in the bottom 20% of SEG_FEATURE only."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    rows, rest = perm[:n], perm[n:]
    ctrl = rest[t[rest] == 0][:300_000]
    base = lgb.LGBMClassifier(**PARAMS).fit(X[ctrl], y[ctrl])
    Xs = X[rows]
    base_p = np.clip(base.predict_proba(Xs)[:, 1], 0.001, 0.9)

    f = Xs[:, FEATURES.index(SEG_FEATURE)]
    cuts = np.quantile(f, [0.2, 0.4, 0.6, 0.8])
    # bin = how many cutpoints the value is above, like segments.sql. bin 0 is the planted segment.
    # f8 has one value with ~half the users, so the middle bins merge
    bins = np.searchsorted(cuts, f, side='left')
    seg = bins == 0
    ts = (rng.random(n) < 0.85).astype(np.int8)
    ys = planted_effect_outcomes(base_p, ts, EFFECT, seg, seed)

    tr, te = split(n, ts, seed)
    y_te, t_te, seg_te = ys[te], ts[te], seg[te]
    # qini auc of pure noise scores on this test set, so the models have a scale to be read against
    rand = [qini_auc(y_te, t_te, rng.random(len(te))) for _ in range(50)]
    out = {'segment': f'{SEG_FEATURE} <= {cuts[0]:.4f} (bottom 20%)',
           'segment_share': float(seg.mean()), 'effect': EFFECT, 'n': n,
           'qini_auc_random_mean': float(np.mean(rand)), 'qini_auc_random_sd': float(np.std(rand)),
           # ranking by the true segment, about the best any model can do with this much noise
           'qini_auc_oracle': qini_auc(y_te, t_te, seg_te.astype(float))}
    for name, learner in LEARNERS.items():
        u = learner(Xs[tr], ys[tr], ts[tr], Xs[te], seed=seed)
        by_bin = {int(b): float(u[bins[te] == b].mean()) for b in np.unique(bins[te])}
        top = np.argsort(-u)[:len(te) // 5]
        res = {
            'mean_pred_in_segment': float(u[seg_te].mean()),
            'mean_pred_outside': float(u[~seg_te].mean()),
            'mean_pred_by_bin': by_bin,
            'segment_share_of_top20': float(seg_te[top].mean()),
            'qini_auc': qini_auc(y_te, t_te, u),
        }
        # pass condition from validation/pass_conditions.md
        res['passed'] = bool(max(by_bin, key=by_bin.get) == 0 and res['qini_auc'] > 0)
        out[name] = res
        print('semi-synthetic', name, res)
    return out


def plot_qini(curves, aucs):
    setup()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.linspace(0, 1, 501)
    span = (x >= 0.2) & (x <= 0.45)  # the model labels sit over this stretch
    at = []
    for name, color in zip(curves, [BLUE, ORANGE]):
        k, q = curves[name]
        q = np.interp(x * k[-1], k, q)
        ax.plot(x, q, color=color)
        at.append((q[span].mean(), q[span].max(), q[span].min(), name, color))
    total = q[-1]  # Q(N) is everyone's incremental visits, the same for any ordering
    pad = 0.04 * total
    ax.plot([0, 1], [0, total], color=GRAY, linestyle='--', linewidth=1)
    # near the start the models are far above the random line, so there's room under it
    ax.text(0.1, 0.1 * total - pad, 'random targeting', va='top', color=GRAY, fontsize=8)

    # label the higher curve above its highest point over the span and the lower one below its
    # lowest point, so the labels stay clear of both lines
    (_, _, low, lo, c_lo), (_, high, _, hi, c_hi) = sorted(at)
    ax.text(0.2, high + pad, f'{hi}, qini auc {aucs[hi]:.3f}', color=c_hi, va='bottom', fontsize=9)
    ax.text(0.2, low - pad, f'{lo}, qini auc {aucs[lo]:.3f}', color=c_lo, va='top', fontsize=9)
    ax.set_ylim(min(0, ax.get_ylim()[0]), max(ax.get_ylim()[1], high + 4 * pad))

    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel('users targeted, highest predicted uplift first')
    ax.set_ylabel('incremental visits')
    ax.set_title('held-out 30% of users, models fit on the other 70%')
    fig.suptitle('Qini curves: extra visits from targeting by predicted uplift', x=0.01,
                 ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    save(fig, 'qini.png')


def main():
    start = time.time()
    X, y, t = load()
    out, curves = real_data(X, y, t)
    plot_qini(curves, {name: out[name]['qini_auc'] for name in curves})
    out['semi_synthetic'] = semi_synthetic(X, y, t)
    out['runtime_minutes'] = (time.time() - start) / 60
    print(f"done in {out['runtime_minutes']:.1f} min")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / 'uplift.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
