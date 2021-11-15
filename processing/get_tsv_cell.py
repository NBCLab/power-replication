"""
Collect native-space preprocessed data from fMRIPrep working directory and
copy into fMRIPrep derivatives directory, in BIDS format.
"""
import argparse
import os.path as op

import pandas as pd


def get_tsv_cell(tsv_file, column_name, row_number):
    """Extract the value for a cell in a TSV file."""
    assert op.isfile(tsv_file), f"File DNE: {tsv_file}"
    df = pd.read_table(tsv_file)
    value = df.loc[row_number, column_name]
    return value


def _get_parser():
    parser = argparse.ArgumentParser(description="Grab cell from TSV file.")
    parser.add_argument(
        dest="tsv_file",
        help="Path to TSV file.",
    )
    parser.add_argument(
        "--column",
        dest="column_name",
        required=True,
        help="Column to index in TSV file.",
    )
    parser.add_argument(
        "--row",
        dest="row_number",
        type=int,
        required=True,
        help="Row number to index in TSV file.",
    )
    return parser


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    value = get_tsv_cell(**kwargs)
    print(value)


if __name__ == "__main__":
    _main()
