# Full-data readout for the three metrics. Normal / delta method CIs come from the SQL views,
# bootstrap CIs (resampling users) are there to compare against. Writes results/readout.json.
# python scripts/run_readout.py   (bootstrap takes ~5 min)

import json
from dataclasses import asdict

from explab.load import ROOT, connect
from explab.readout import (binary_mean_and_var, bootstrap, compare, mean_diff, mean_lift,
                            percentile_ci, ratio_diff, ratio_from_sums, ratio_lift)

RESULTS = ROOT / 'results'
METRICS = ['visit_rate', 'conversion_rate', 'conversions_per_visit']


def view_readouts(con):
    out = {}
    for metric, col in [('visit_rate', 'visits'), ('conversion_rate', 'conversions')]:
        df = con.sql(f'select * from {metric} order by treatment').df()
        c, t = df.iloc[0], df.iloc[1]
        out[metric] = compare(*binary_mean_and_var(t[col], t.n), *binary_mean_and_var(c[col], c.n))

    df = con.sql('select * from conversions_per_visit order by treatment').df()
    # visit and conversion are 0/1, so the sums of squares are just the sums
    c, t = [ratio_from_sums(r.n, r.conversions, r.visits, r.conversions, r.visits, r.conv_visits)
            for r in df.itertuples()]
    out['conversions_per_visit'] = compare(*t, *c)
    return out


def all_stats(t, c):
    v_t, conv_t = t
    v_c, conv_c = c
    return [
        mean_diff([v_t], [v_c]), mean_lift([v_t], [v_c]),
        mean_diff([conv_t], [conv_c]), mean_lift([conv_t], [conv_c]),
        ratio_diff([conv_t, v_t], [conv_c, v_c]), ratio_lift([conv_t, v_t], [conv_c, v_c]),
    ]


def main():
    con = connect()
    res = view_readouts(con)

    data = con.sql('select treatment, visit, conversion from criteo').fetchnumpy()
    t = data['treatment'] == 1
    boot = bootstrap([data['visit'][t], data['conversion'][t]],
                     [data['visit'][~t], data['conversion'][~t]], all_stats, n_boot=1000)
    lo, hi = percentile_ci(boot)

    out = {}
    for i, metric in enumerate(METRICS):
        r = res[metric]
        out[metric] = {
            'control': r['control'],
            'treatment': r['treatment'],
            'diff': asdict(r['diff']),
            'lift': asdict(r['lift']),
            'boot_diff_ci': [lo[2 * i], hi[2 * i]],
            'boot_lift_ci': [lo[2 * i + 1], hi[2 * i + 1]],
        }
        d, l = r['diff'], r['lift']
        print(f"{metric:22s} ctrl {r['control']:.5f}  trt {r['treatment']:.5f}  "
              f"diff {d.value:+.5f} [{d.ci_low:+.5f}, {d.ci_high:+.5f}] boot [{lo[2*i]:+.5f}, {hi[2*i]:+.5f}]  "
              f"lift {l.value:+.2%} [{l.ci_low:+.2%}, {l.ci_high:+.2%}] boot [{lo[2*i+1]:+.2%}, {hi[2*i+1]:+.2%}]")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / 'readout.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
