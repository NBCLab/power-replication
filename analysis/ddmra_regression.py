import os.path as op
from glob import glob

import numpy as np
import pandas as pd


def get_rank(value, null):
    null = np.sort(null)
    rank = np.searchsorted(null, value, side="left")
    return rank

IN_DIR = "/home/data/nbc/misc-projects/Salo_PowerReplication/analyses/experiment02_group05_analysis01/"
analysis_dirs = sorted(glob(op.join(IN_DIR, "*")))
analysis_dirs = [ad for ad in analysis_dirs if op.isdir(ad)]
n_iters = 10000

for analysis_dir in analysis_dirs:
    analysis_name = op.basename(analysis_dir)
    print(analysis_name, flush=True)

    values_file = op.join(analysis_dir, "analysis_values.tsv.gz")
    values_df = pd.read_table(values_file)
    distances = values_df["distance"].values
    # Set the intercept to 35mm
    distances -= 35
    iv_arr = np.vstack((distances, np.full(distances.shape, 1))).T

    for analysis_type in ["qcrsfc", "highlow", "scrubbing"]:
        print(f"    {analysis_type}", flush=True)
        values = values_df[analysis_type].values
        values = values[:, None]
        if analysis_type == "highlow":
            temp_type = "hl"
        else:
            temp_type = analysis_type

        null_file = op.join(analysis_dir, f"{temp_type}_null.npz")
        null = np.load(null_file)
        null = null["arr_0"]
        test_betas = np.linalg.lstsq(iv_arr, values, rcond=None)[0]
        test_slope, test_intercept = test_betas
        null_betas = np.linalg.lstsq(iv_arr, null.T, rcond=None)[0]
        null_slopes, null_intercepts = null_betas
        slope_rank = get_rank(test_slope, null_slopes)[0]
        intercept_rank = get_rank(test_intercept, null_intercepts)[0]
        # lower (more negative) slopes are more significant, so higher ranks = higher p-values
        slope_p = slope_rank / n_iters
        # higher intercepts are more significant, so higher ranks = lower p-values
        intercept_p = 1 - (intercept_rank / n_iters)
        print(f"\tIntercept (rank={intercept_rank}/{n_iters}, p={intercept_p})", flush=True)
        print(f"\tSlope (rank={slope_rank}/{n_iters}, p={slope_p})", flush=True)
