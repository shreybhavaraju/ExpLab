# The verdicts: overall, and for every f0 / f2 decile segment, for each metric. Goes through
# report.metric_report, same as the app. Writes results/decision.json.
# Needs results/segment_stats.csv (scripts/build_app_data.py). A few seconds.

import json
from collections import Counter

import pandas as pd

from explab.load import ROOT
from explab.report import COUNT_COL, LABELS, metric_report

RESULTS = ROOT / 'results'
MIN_EFFECT = 0.05


def summarize(r):
    lift = r['readout']['lift']
    return {
        'n_t': r['n_t'],
        'n_c': r['n_c'],
        'control': r['readout']['control'],
        'treatment': r['readout']['treatment'],
        'lift': lift.value,
        'lift_ci': [lift.ci_low, lift.ci_high],
        'p_value': lift.p_value,
        'mde_rel': r['mde_rel'],
        'srm_passed': r['srm']['passed'],
        'balance_passed': all(b['passed'] for b in r['balance'].values()),
        'verdict': r['decision'].verdict,
        'reasons': r['decision'].reasons,
    }


def main():
    cells = pd.read_csv(RESULTS / 'segment_stats.csv')
    out = {'min_effect': MIN_EFFECT, 'overall': {}, 'overall_cupac': {}, 'segments': []}

    for metric in LABELS:
        s = summarize(metric_report(cells, metric, MIN_EFFECT))
        out['overall'][metric] = s
        print(f"{LABELS[metric]:22s} lift {s['lift']:+.1%} [{s['lift_ci'][0]:+.1%}, "
              f"{s['lift_ci'][1]:+.1%}]  MDE {s['mde_rel']:.1%}  -> {s['verdict']}")
        for reason in s['reasons']:
            print(f'    {reason}')
        if metric in COUNT_COL:
            out['overall_cupac'][metric] = summarize(metric_report(cells, metric, MIN_EFFECT,
                                                                   cupac=True))

    for f in ['f0', 'f2']:
        for b in sorted(cells[f'{f}_bin'].unique()):
            sub = cells[cells[f'{f}_bin'] == b]
            for metric in LABELS:
                for cupac in [False, True] if metric in COUNT_COL else [False]:
                    s = summarize(metric_report(sub, metric, MIN_EFFECT, cupac=cupac))
                    out['segments'].append({'segment': f'{f} bin {b}', 'metric': metric,
                                            'cupac': cupac, **s})

    counts = Counter((s['metric'], s['cupac'], s['verdict']) for s in out['segments'])
    print('\nsegment verdicts:')
    for (metric, cupac, verdict), n in sorted(counts.items()):
        print(f"  {LABELS[metric]:22s} {'cupac' if cupac else 'raw':5s} {verdict}: {n}")
    for s in out['segments']:
        if s['verdict'] != 'INCONCLUSIVE':
            print(f"  -> {s['segment']}, {LABELS[s['metric']]}, cupac={s['cupac']}: "
                  f"{s['verdict']}, lift {s['lift']:+.1%}, MDE {s['mde_rel']:.1%}")

    with open(RESULTS / 'decision.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
