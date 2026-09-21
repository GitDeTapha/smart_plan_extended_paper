#!/usr/bin/env python3
"""
Convert a large CSV file into chunked Parquet files with controlled dtypes.
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LOAD_COLS = [
    'patientsid',
    'activitydatetime',
    'sta3n',
    'stopcode',
    'dta',
    'dts',
    'dtc',
    'non_va',
    'zip',
    'state',
    'disp',
    'AdministrativeFlag',
]

DTYPES = {
    'patientsid': 'uint32',
    'activitydatetime': object,
    'sta3n': 'category',
    'stopcode': 'category',
    'dta': 'float32',
    'dts': 'uint16',
    'dtc': 'uint16',
    'non_va': 'category',
    'zip': 'category',
    'state': 'category',
    'disp': 'category',
    'AdministrativeFlag': 'category',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a large CSV into chunked Parquet files.'
    )
    parser.add_argument(
        '--csv',
        default='data/consult_waits_2024_03_25.csv',
        help='Path to the input CSV file.',
    )
    parser.add_argument(
        '--out-dir',
        default='data_parquet_chunks',
        help='Output directory for Parquet files.',
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        default=500_000,
        help='Number of rows per output Parquet file.',
    )
    parser.add_argument(
        '--compression',
        default='snappy',
        choices=['snappy', 'gzip', 'zstd', 'brotli', 'none'],
        help='Parquet compression codec.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of chunks to write (for testing).',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Reading CSV from {csv_path}')
    print(f'Writing Parquet files to {out_dir}')
    print(f'Using chunksize={args.chunksize}, compression={args.compression}')

    for idx, chunk in enumerate(
        pd.read_csv(
            csv_path,
            usecols=LOAD_COLS,
            dtype=DTYPES,
            chunksize=args.chunksize,
            on_bad_lines='skip',
            low_memory=False,
            engine='c',
            memory_map=True,
        )
    ):
        if args.limit is not None and idx >= args.limit:
            break

        output_file = out_dir / f'part_{idx:04d}.parquet'
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_table(table, output_file, compression=args.compression)
        print(f'Wrote {output_file} ({len(chunk)} rows)')

    print('Done.')


if __name__ == '__main__':
    main()
