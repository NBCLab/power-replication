"""Experiment 1, Analysis Group 1.

Validation of RPV metric.

RPV correlated with mean RV, across participants.

RPV correlated with mean RVT, across participants.

RPV upper envelope (ENV) correlated with RV, then z-transformed and assessed across participants
via t-test.

RPV upper envelope (ENV) correlated with RVT, then z-transformed and assessed across participants
via t-test.
"""
import os
import os.path as op
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import ttest_1samp  # noqa: E402

sys.path.append("..")

from utils import pearson_r  # noqa: E402


def correlate_rpv_with_mean_rv(project_dir, participants_file, confounds_pattern):
    """Perform analysis 1.

    Correlate RPV with mean RV, across participants.
    Perform one-sided test of significance on correlation coefficient to determine if RPV is
    significantly, positively correlated with mean RV.
    """
    print("Experiment 1, Analysis Group 1, Analysis 1", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment01_group01")
    os.makedirs(out_dir, exist_ok=True)

    N_ANALYSES_IN_FAMILY = 2
    ALPHA = 0.05 / N_ANALYSES_IN_FAMILY

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value
    participants_df = participants_df.dropna(subset=["rpv"])
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    participants_df["mean_rv"] = np.nan
    for i, row in participants_df.iterrows():
        participant_id = row["participant_id"]
        confounds_file = confounds_pattern.format(participant_id=participant_id)
        assert op.isfile(confounds_file), f"{confounds_file} DNE"

        confounds_df = pd.read_table(confounds_file)
        rv_arr = confounds_df["RVRegression_RV"].values
        mean_rv = np.mean(rv_arr)
        participants_df.loc[i, "mean_rv"] = mean_rv

    # We are performing a one-sided test to determine if the correlation is
    # statistically significant (alpha = 0.05) and positive.
    corr, p = pearson_r(
        participants_df["rpv"], participants_df["mean_rv"], alternative="greater"
    )
    if p <= ALPHA:
        print(
            "\tRPV and mean RV were found to be positively and statistically "
            "significantly correlated, "
            f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
        )
    else:
        print(
            "\tRPV and mean RV were not found to be statistically significantly "
            "correlated, "
            f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
        )

    g = sns.JointGrid(data=participants_df, x="rpv", y="mean_rv")
    g.plot(sns.regplot, sns.histplot)
    g.savefig(op.join(out_dir, "analysis_01.png"), dpi=400)

    participants_df.to_csv(
        op.join(out_dir, "analysis_01_results.tsv"), sep="\t", index=False
    )


def correlate_rpv_with_mean_rvt(project_dir, participants_file, confounds_pattern):
    """Perform analysis 2.

    Correlate RPV with mean RVT, across participants.
    Perform one-sided test of significance on correlation coefficient to determine if RPV is
    significantly, positively correlated with mean RVT.
    """
    print("Experiment 1, Analysis Group 1, Analysis 2", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment01_group01")
    os.makedirs(out_dir, exist_ok=True)

    N_ANALYSES_IN_FAMILY = 2
    ALPHA = 0.05 / N_ANALYSES_IN_FAMILY

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value
    participants_df = participants_df.dropna(subset=["rpv"])
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    participants_df["mean_rvt"] = np.nan
    for i, row in participants_df.iterrows():
        participant_id = row["participant_id"]
        confounds_file = confounds_pattern.format(participant_id=participant_id)
        assert op.isfile(confounds_file), f"{confounds_file} DNE"

        confounds_df = pd.read_table(confounds_file)
        rv_arr = confounds_df["RVTRegression_RVT"].values
        mean_rvt = np.mean(rv_arr)
        participants_df.loc[i, "mean_rvt"] = mean_rvt

    # We are performing a one-sided test to determine if the correlation is
    # statistically significant (alpha = 0.05) and positive.
    corr, p = pearson_r(
        participants_df["rpv"], participants_df["mean_rvt"], alternative="greater"
    )

    if p <= ALPHA:
        print(
            "\tRPV and mean RVT were found to be positively and statistically "
            "significantly correlated, "
            f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
        )
    else:
        print(
            "\tRPV and mean RVT were not found to be statistically significantly "
            "correlated, "
            f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
        )

    g = sns.JointGrid(data=participants_df, x="rpv", y="mean_rvt")
    g.plot(sns.regplot, sns.histplot)
    g.savefig(op.join(out_dir, "analysis_02.png"), dpi=400)

    participants_df.to_csv(
        op.join(out_dir, "analysis_02_results.tsv"), sep="\t", index=False
    )


def compare_env_with_rv(project_dir, participants_file, confounds_pattern):
    """Perform analysis 3.

    Correlate ENV (upper envelope used to calculate RPV) with RV for each participant,
    then z-transform the correlation coefficients and perform a one-sample t-test against zero
    with the z-values.
    """
    print("Experiment 1, Analysis Group 1, Analysis 3", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment01_group01")
    os.makedirs(out_dir, exist_ok=True)

    N_ANALYSES_IN_FAMILY = 2
    ALPHA = 0.05 / N_ANALYSES_IN_FAMILY

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value, since those are ones with good physio data
    participants_df = participants_df.dropna(subset=["rpv"])
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    participants_df["env_rv_corr"] = np.nan
    for i, row in participants_df.iterrows():
        participant_id = row["participant_id"]
        confounds_file = confounds_pattern.format(participant_id=participant_id)
        assert op.isfile(confounds_file), f"{confounds_file} DNE"

        confounds_df = pd.read_table(confounds_file)
        corr = confounds_df["RPVRegression_Envelope"].corr(
            confounds_df["RVRegression_RV"]
        )
        participants_df.loc[i, "env_rv_corr"] = corr

    # Now transform correlation coefficients to Z-values
    z_values = np.arctanh(participants_df["env_rv_corr"].values)
    mean_z = np.mean(z_values)
    sd_z = np.std(z_values)

    # Now perform one-sample t-test against zero.
    t, p = ttest_1samp(z_values, popmean=0, alternative="greater")

    if p <= ALPHA:
        print(
            "\tCorrelations between the upper envelope used to calculate RPV and RV "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "\tCorrelations between the upper envelope used to calculate RPV and RV "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were not significantly higher than "
            "zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=z_values, ax=ax)
    ax.set_xlabel("Z-transformed correlation coefficient")
    fig.suptitle("Distribution of correlations between RPV upper envelope and RV")
    fig.savefig(op.join(out_dir, "analysis_03.png"), dpi=400)

    participants_df.to_csv(
        op.join(out_dir, "analysis_03_results.tsv"), sep="\t", index=False
    )


def compare_env_with_rvt(project_dir, participants_file, confounds_pattern):
    """Perform analysis 4.

    Correlate ENV (upper envelope used to calculate RPV) with RVT for each participant,
    then z-transform the correlation coefficients and perform a one-sample t-test against zero
    with the z-values.
    """
    print("Experiment 1, Analysis Group 1, Analysis 4", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment01_group01")
    os.makedirs(out_dir, exist_ok=True)

    N_ANALYSES_IN_FAMILY = 2
    ALPHA = 0.05 / N_ANALYSES_IN_FAMILY

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value, since those are ones with good physio data
    participants_df = participants_df.dropna(subset=["rpv"])
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    participants_df["env_rvt_corr"] = np.nan
    for i, row in participants_df.iterrows():
        participant_id = row["participant_id"]
        confounds_file = confounds_pattern.format(participant_id=participant_id)
        assert op.isfile(confounds_file), f"{confounds_file} DNE"

        confounds_df = pd.read_table(confounds_file)
        corr = confounds_df["RPVRegression_Envelope"].corr(
            confounds_df["RVTRegression_RVT"]
        )
        participants_df.loc[i, "env_rvt_corr"] = corr

    # Now transform correlation coefficients to Z-values
    z_values = np.arctanh(participants_df["env_rvt_corr"].values)
    mean_z = np.mean(z_values)
    sd_z = np.std(z_values)

    # Now perform one-sample t-test against zero.
    t, p = ttest_1samp(z_values, popmean=0, alternative="greater")

    if p <= ALPHA:
        print(
            "\tCorrelations between the upper envelope used to calculate RPV and RVT "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "\tCorrelations between the upper envelope used to calculate RPV and RVT "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were not significantly higher than "
            "zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=z_values, ax=ax)
    ax.set_xlabel("Z-transformed correlation coefficient")
    fig.suptitle("Distribution of correlations between RPV upper envelope and RVT")
    fig.savefig(op.join(out_dir, "analysis_04.png"), dpi=400)

    participants_df.to_csv(
        op.join(out_dir, "analysis_04_results.tsv"), sep="\t", index=False
    )


if __name__ == "__main__":
    print("Experiment 1, Analysis Group 1")
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    in_dir = op.join(project_dir, "dset-dupre/")
    participants_file = op.join(in_dir, "participants.tsv")
    confounds_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/func",
        "{participant_id}_task-rest_run-1_desc-confounds_timeseries.tsv",
    )
    correlate_rpv_with_mean_rv(project_dir, participants_file, confounds_pattern)
    correlate_rpv_with_mean_rvt(project_dir, participants_file, confounds_pattern)
    compare_env_with_rv(project_dir, participants_file, confounds_pattern)
    compare_env_with_rvt(project_dir, participants_file, confounds_pattern)
