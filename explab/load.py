# Download the Criteo uplift data and convert it to parquet.
# The go.criteo.net link from the paper is dead now, same file is on Criteo's Hugging Face page.

import hashlib
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
CSV_PATH = DATA_DIR / 'criteo-research-uplift-v2.1.csv.gz'
PARQUET_PATH = DATA_DIR / 'criteo.parquet'
SQL_DIR = ROOT / 'sql'
SQL_FILES = ['metrics.sql', 'segments.sql']

URL = 'https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz'
SHA256 = '2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc'

FEATURES = [f'f{i}' for i in range(12)]
FLAGS = ['treatment', 'conversion', 'visit', 'exposure']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(force=False):
    if CSV_PATH.exists() and not force:
        return CSV_PATH
    DATA_DIR.mkdir(exist_ok=True)
    print(f'downloading {URL} (~311MB)')
    urllib.request.urlretrieve(URL, CSV_PATH)
    if sha256(CSV_PATH) != SHA256:
        raise ValueError('checksum mismatch, delete the file and try again')
    return CSV_PATH


def to_parquet(force=False):
    if PARQUET_PATH.exists() and not force:
        return PARQUET_PATH
    # features as float32 and the 0/1 columns as tinyint, gets it down to ~150MB
    feats = ', '.join(f'cast({f} as float) as {f}' for f in FEATURES)
    flags = ', '.join(f'cast({c} as tinyint) as {c}' for c in FLAGS)
    duckdb.sql(f"""
        copy (select {feats}, {flags} from read_csv_auto('{CSV_PATH}'))
        to '{PARQUET_PATH}' (format parquet)
    """)
    return PARQUET_PATH


def connect(source=PARQUET_PATH):
    """DuckDB connection with the data as a `criteo` view plus the metric views from sql/.
    `source` can also be a DataFrame (the tests use a small fake one)."""
    con = duckdb.connect()
    if isinstance(source, pd.DataFrame):
        con.register('criteo', source)
    else:
        con.execute(f"create view criteo as select * from read_parquet('{source}')")
    for name in SQL_FILES:
        con.execute((SQL_DIR / name).read_text())
    return con


def summary(con):
    print(con.sql('select count(*) as n_rows, avg(treatment) as treatment_share from criteo'))
    print(con.sql("""
        select treatment, count(*) as n, avg(visit) as visit_rate,
               avg(conversion) as conversion_rate, avg(exposure) as exposure_rate
        from criteo
        group by treatment
        order by treatment
    """))
    print(con.sql('select * from conversions_per_visit order by treatment'))
    print(con.sql('select * from exposed_vs_control order by grp'))


if __name__ == '__main__':
    download()
    to_parquet()
    summary(connect())
