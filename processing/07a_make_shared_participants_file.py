"""Combine individual datasets' participants.tsv files into one."""
import os.path as op

import pandas as pd

if __name__ == "__main__":
    PROJECT_DIR = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    out_file = op.join(PROJECT_DIR, "participants.tsv")

    DATASETS = [
        "dset-cambridge",
        "dset-camcan",
        "dset-cohen",
        "dset-dalenberg",
        "dset-dupre",
    ]
    dfs = []
    for dset in DATASETS:
        file_ = op.join(PROJECT_DIR, dset, "participants.tsv")
        df = pd.read_table(file_)
        df = df.loc[df["exclude"] == 0]
        df["dset"] = dset
        df = df[["dset", "participant_id"]]
        dfs.append(df)

    out_df = pd.merge(dfs)
    out_df.to_csv(out_file, sep="\t", index=False)
