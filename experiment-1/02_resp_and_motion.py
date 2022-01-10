"""Experiment 1, Analysis Group 2.

Characterizing the relationship between respiration and head motion.

RPV correlated with mean framewise displacement.

RVT correlated with framewise displacement.

RV correlated with framewise displacement.
"""
import os
import os.path as op
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_1samp

sys.path.append("..")

from utils import pearson_r  # noqa: E402


def correlate_rpv_with_mean_fd(project_dir, participants_file, confounds_pattern):
    """Perform analysis 1.

    Correlate RPV with mean FD across participants.
    Perform one-sided test of significance on correlation coefficient to determine if RPV is
    significantly, positively correlated with mean FD.
    """
    print("Experiment 1, Analysis Group 2, Analysis 1", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment01_group02")
    os.makedirs(out_dir, exist_ok=True)

    ALPHA = 0.05

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value
    participants_df = participants_df.dropna(subset=["rpv"])
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    participants_df["mean_fd"] = np.nan
    for i, row in participants_df.iterrows():
        participant_id = row["participant_id"]
        confounds_file = confounds_pattern.format(participant_id=participant_id)
        assert op.isfile(confounds_file), f"{confounds_file} DNE"

        confounds_df = pd.read_table(confounds_file)
        fd_arr = confounds_df["framewise_displacement"].values
        mean_fd = np.mean(fd_arr)
        participants_df.loc[i, "mean_fd"] = mean_fd

    # We are performing a one-sided test to determine if the correlation is
    # statistically significant (alpha = 0.05) and positive.
    corr, p = pearson_r(
        participants_df["rpv"], participants_df["mean_fd"], alternative="greater"
    )
    if p <= ALPHA:
        print(
            "\tRPV and mean FD were found to be positively and statistically "
            "significantly correlated, "
            f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
        )
    else:
        print(
            "\tRPV and mean FD were not found to be statistically significantly "
            "correlated, "
            f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
        )

    g = sns.JointGrid(data=participants_df, x="rpv", y="mean_fd")
    g.plot(sns.scatterplot, sns.histplot)
    g.savefig(op.join(out_dir, "analysis_01.png"), dpi=400)


def correlate_rvt_with_fd(project_dir, participants_file, confounds_pattern):
    """Perform analysis 2.

    Correlate RVT with FD for each participant, then z-transform the correlation coefficients
    and perform a one-sample t-test against zero with the z-values.
    """
    print("Experiment 1, Analysis Group 2, Analysis 2", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment01_group02")
    os.makedirs(out_dir, exist_ok=True)

    ALPHA = 0.05
    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value, since those are ones with good physio data
    participants_df = participants_df.dropna(subset=["rpv"])
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    participants_df["rvt_fd_corr"] = np.nan
    for i, row in participants_df.iterrows():
        participant_id = row["participant_id"]
        confounds_file = confounds_pattern.format(participant_id=participant_id)
        assert op.isfile(confounds_file), f"{confounds_file} DNE"

        confounds_df = pd.read_table(confounds_file)
        corr = confounds_df["RVTRegression_RVT"].corr(
            confounds_df["framewise_displacement"]
        )
        participants_df.loc[i, "rvt_fd_corr"] = corr

    # Now transform correlation coefficients to Z-values
    z_values = np.arctanh(participants_df["rvt_fd_corr"].values)
    mean_z = np.mean(z_values)
    sd_z = np.std(z_values)

    # Now perform one-sample t-test against zero.
    t, p = ttest_1samp(z_values, popmean=0, alternative="greater")

    if p <= ALPHA:
        print(
            "\tCorrelations between RVT and FD "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "\tCorrelations between RVT and FD "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were not significantly higher than "
            "zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=z_values, x="Z-transformed correlation coefficient", ax=ax)
    fig.suptitle("Distribution of correlations between RPV upper envelope and RV")
    fig.savefig(op.join(out_dir, "analysis_02.png", dpi=400))


def correlate_rv_with_fd(project_dir, participants_file, confounds_pattern):
    """Perform analysis 3.

    Correlate RV with FD for each participant, then z-transform the correlation coefficients
    and perform a one-sample t-test against zero with the z-values.
    """
    print("Experiment 1, Analysis Group 2, Analysis 3", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment01_group02")
    os.makedirs(out_dir, exist_ok=True)

    ALPHA = 0.05
    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value, since those are ones with good physio data
    participants_df = participants_df.dropna(subset=["rpv"])
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    participants_df["rv_fd_corr"] = np.nan
    for i, row in participants_df.iterrows():
        participant_id = row["participant_id"]
        confounds_file = confounds_pattern.format(participant_id=participant_id)
        assert op.isfile(confounds_file), f"{confounds_file} DNE"

        confounds_df = pd.read_table(confounds_file)
        corr = confounds_df["RVRegression_RV"].corr(
            confounds_df["framewise_displacement"]
        )
        participants_df.loc[i, "rv_fd_corr"] = corr

    # Now transform correlation coefficients to Z-values
    z_values = np.arctanh(participants_df["rv_fd_corr"].values)
    mean_z = np.mean(z_values)
    sd_z = np.std(z_values)

    # Now perform one-sample t-test against zero.
    t, p = ttest_1samp(z_values, popmean=0, alternative="greater")

    if p <= ALPHA:
        print(
            "\tCorrelations between RV and FD "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "\tCorrelations between RV and FD "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were not significantly higher than "
            "zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=z_values, x="Z-transformed correlation coefficient", ax=ax)
    fig.suptitle("Distribution of correlations between RPV upper envelope and RV")
    fig.savefig(op.join(out_dir, "analysis_03.png", dpi=400))


if __name__ == "__main__":
    print("Experiment 1, Analysis Group 2")
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    in_dir = op.join(project_dir, "dset-dupre/")
    participants_file = op.join(in_dir, "participants.tsv")
    confounds_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/func",
        "{participant_id}_task-rest_run-1_desc-confounds_timeseries.tsv",
    )
    correlate_rpv_with_mean_fd(project_dir, participants_file, confounds_pattern)
    correlate_rvt_with_fd(project_dir, participants_file, confounds_pattern)
    correlate_rv_with_fd(project_dir, participants_file, confounds_pattern)
