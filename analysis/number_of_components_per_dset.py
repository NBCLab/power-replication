""""""
import os.path as op
from glob import glob

#import matplotlib.pyplot as plt
import pandas as pd

#import seaborn as sns

path = (
    "/home/data/nbc/misc-projects/Salo_PowerReplication/dset-*/derivatives/tedana/sub-*/"
    "func/sub-*_desc-tedana_metrics.tsv"
)

out_df = pd.DataFrame(
    columns=["dataset", "participant_id", "classification", "count"],
)

files = sorted(glob(path))
for f in files:
    parts = f.split("/")
    dset = [p for p in parts if p.startswith("dset-")]
    dset = dset[0]
    subject = op.basename(f).split("_")[0]

    df = pd.read_table(f)
    n_accepted = (df["classification"] == "accepted").sum()
    n_rejected = (df["classification"] == "rejected").sum()
    n_ignored = (df["classification"] == "ignored").sum()
    out_df.loc[out_df.shape[0]] = [dset, subject, "accepted", n_accepted]
    out_df.loc[out_df.shape[0]] = [dset, subject, "rejected", n_rejected]
    out_df.loc[out_df.shape[0]] = [dset, subject, "ignored", n_ignored]

out_df.to_csv("component_counts.tsv", sep="\t", index=False)
