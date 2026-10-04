# Builds results/segment_stats.csv, the only data the Streamlit app needs: the segment_stats view
# (f0 bin x f2 bin x arm) plus per-cell sums of the CUPAC predictions, so the app can do the CUPED
# adjustment from sums without shipping 14M rows. Needs data/cupac_preds.parquet from
# scripts/run_cupac.py. Takes ~1 min.

import numpy as np
import pandas as pd

from explab.load import DATA_DIR, ROOT, connect

METRICS = ['visit', 'conversion']


def main():
    con = connect()
    rows = con.sql('select f0_bin, f2_bin, treatment, visit, conversion from criteo_segments').df()
    # the predictions are in the row order of the plain criteo view, make sure the segments
    # view kept that order before lining them up
    plain = con.sql('select treatment, visit, conversion from criteo').df()
    assert np.array_equal(rows[['treatment', 'visit', 'conversion']].values, plain.values)

    preds = pd.read_parquet(DATA_DIR / 'cupac_preds.parquet')
    sums = {'n': ('visit', 'size'), 'visits': ('visit', 'sum'),
            'conversions': ('conversion', 'sum'), 'conv_visits': ('conv_visit', 'sum')}
    rows['conv_visit'] = rows.conversion * rows.visit
    for m in METRICS:
        x = preds[f'cupac_{m}'].to_numpy()
        rows[f'x_{m}'] = x
        rows[f'xx_{m}'] = x * x
        rows[f'xy_{m}'] = x * rows[m]
        sums.update({c: (c, 'sum') for c in [f'x_{m}', f'xx_{m}', f'xy_{m}']})

    cells = rows.groupby(['f0_bin', 'f2_bin', 'treatment']).agg(**sums).reset_index()
    for c in ['n', 'visits', 'conversions', 'conv_visits']:
        cells[c] = cells[c].astype('int64')

    # same counts as the SQL layer or something went wrong
    sql = con.sql('select * from segment_stats').df()
    check = cells.merge(sql, on=['f0_bin', 'f2_bin', 'treatment'], suffixes=('', '_sql'))
    for c in ['n', 'visits', 'conversions', 'conv_visits']:
        assert (check[c] == check[f'{c}_sql']).all()

    cells.to_csv(ROOT / 'results' / 'segment_stats.csv', index=False)
    print(f'{len(cells)} cells written')


if __name__ == '__main__':
    main()
