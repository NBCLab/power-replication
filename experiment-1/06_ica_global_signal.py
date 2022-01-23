"""Experiment 1, Analysis Group 6.

Does TEDANA retain global BOLD signal in BOLD ICA components?

Carpet plots generated for ICA components and OC data, along with line plots for physiological
traces.

ICA components correlated with mean cortical signal of OC dataset.
-   Record the percentage and number of BOLD-like and non-BOLD-like components correlated with the
    cortical signal at r > 0.5 and r > 0.3 across participants.
-   Mean correlation coefficient for BOLD and non-BOLD components with mean cortical signal is
    calculated for each participant, and distributions of correlation coefficients were compared
    to zero and each other with t-tests.


Mean cortical signal from MEDN is correlated with mean cortical signal from OC for each
participant, and distribution of coefficients is compared to zero with one-sampled t-test.
"""
import os
import os.path as op
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from nilearn import masking  # noqa: E402
from scipy.stats import ttest_1samp  # noqa: E402


def plot_components_and_physio():
    """Generate plots for analysis 1."""
    print("Experiment 1, Analysis Group 6, Analysis 1", flush=True)
    ...


def correlate_ica_with_cortical_signal(
    project_dir,
    participants_file,
    ica_pattern,
    ctab_pattern,
    oc_pattern,
    mask_pattern,
):
    """Perform analysis 2.

    Correlate each ICA component's time series with the mean cortical signal of the OC dataset.
    Divide the components into BOLD and non-BOLD, then record the percentage and count of each
    type with r > 0.5 and r > 0.3.
    Also record the mean correlation coefficient (after z-transform) for each type.
    Compare the z-transformed coefficients to zero with t-tests.
    """
    print("Experiment 1, Analysis Group 6, Analysis 2", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment01_group06")
    os.makedirs(out_dir, exist_ok=True)

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with good data
    participants_df = participants_df.loc[participants_df["exclude"] != 1]
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    out_df = pd.DataFrame(
        index=participants_df["participant_id"],
        columns=[
            "total accepted",
            "total rejected",
            "accepted count, r > 0.3",
            "accepted proportion, r > 0.3",
            "rejected count, r > 0.3",
            "rejected proportion, r > 0.3",
            "accepted count, r > 0.5",
            "accepted proportion, r > 0.5",
            "rejected count, r > 0.5",
            "rejected proportion, r > 0.5",
            "accepted mean z",
            "rejected mean z",
        ],
    )

    ALPHA = 0.05
    for i_run, participant_row in participants_df.iterrows():
        subj_id = participant_row["participant_id"]

        mask_file = mask_pattern.format(participant_id=subj_id)
        oc_file = oc_pattern.format(participant_id=subj_id)
        ica_file = ica_pattern.format(participant_id=subj_id)
        ctab_file = ctab_pattern.format(participant_id=subj_id)

        oc_data = masking.apply_mask(oc_file, mask_file)
        oc_data = np.mean(oc_data, axis=1)

        comptable = pd.read_table(ctab_file, index_col="Component")

        out_df.loc[subj_id, "total accepted"] = (
            comptable["classification"] != "rejected"
        ).sum()
        out_df.loc[subj_id, "total rejected"] = (
            comptable["classification"] == "rejected"
        ).sum()

        ica_df = pd.read_table(ica_file)
        ica_df["OC"] = oc_data

        correlations = ica_df.corr()["OC"]

        acc_corrs, rej_corrs = [], []
        for comp in comptable.index:
            row = comptable.loc[comp]
            corr = correlations[comp]
            classification = row["classification"]
            if classification == "rejected":
                rej_corrs.append(corr)
            elif classification != "rejected":
                acc_corrs.append(corr)

        acc_corrs = np.array(acc_corrs)
        rej_corrs = np.array(rej_corrs)

        # Proportion and count of r > 0.3 for each classification
        out_df.loc[subj_id, "accepted count, r > 0.3"] = np.sum(acc_corrs > 0.3)
        out_df.loc[subj_id, "accepted proportion, r > 0.3"] = (
            np.sum(acc_corrs > 0.3) / acc_corrs.size
        )
        out_df.loc[subj_id, "rejected count, r > 0.3"] = np.sum(rej_corrs > 0.3)
        out_df.loc[subj_id, "rejected proportion, r > 0.3"] = (
            np.sum(rej_corrs > 0.3) / rej_corrs.size
        )

        # Proportion and count of r > 0.5 for each classification
        out_df.loc[subj_id, "accepted count, r > 0.5"] = np.sum(acc_corrs > 0.5)
        out_df.loc[subj_id, "accepted proportion, r > 0.5"] = (
            np.sum(acc_corrs > 0.5) / acc_corrs.size
        )
        out_df.loc[subj_id, "rejected count, r > 0.5"] = np.sum(rej_corrs > 0.5)
        out_df.loc[subj_id, "rejected proportion, r > 0.5"] = (
            np.sum(rej_corrs > 0.5) / rej_corrs.size
        )

        # Mean z-transformed correlations
        out_df.loc[subj_id, "accepted mean z"] = np.nanmean(np.arctanh(acc_corrs))
        out_df.loc[subj_id, "rejected mean z"] = np.nanmean(np.arctanh(rej_corrs))

    for clf in ("accepted", "rejected"):
        col = f"{clf} mean z"
        temp_out_df = out_df.dropna(subset=[col])
        z_values = temp_out_df[col].values
        mean_z = np.mean(z_values)
        sd_z = np.std(z_values)

        # And now a significance test!!
        t, p = ttest_1samp(z_values, popmean=0, alternative="greater")
        if p <= ALPHA:
            print(
                "\tCorrelations between the mean cortical signal from optimally combined data and "
                f"{clf} ICA component time series "
                f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were significantly higher than "
                "zero, "
                f"t({temp_out_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
            )
        else:
            print(
                "\tCorrelations between the mean cortical signal from optimally combined data and "
                f"{clf} ICA component time series "
                f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were not significantly higher than "
                "zero, "
                f"t({temp_out_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
            )

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.histplot(data=z_values, ax=ax)
        ax.set_xlabel("Z-transformed correlation coefficient")
        fig.suptitle(
            f"Distribution of correlations between mean cortical OC data and {clf} ICA components"
        )
        fig.savefig(op.join(out_dir, f"analysis_02_{clf}.png"), dpi=400)

    out_df.to_csv(
        op.join(out_dir, "analysis_02_results.tsv"),
        sep="\t",
        index_label="participant_id",
    )


def correlate_medn_with_oc(
    project_dir,
    participants_file,
    medn_pattern,
    oc_pattern,
    mask_pattern,
):
    """Perform analysis 3.

    Correlate mean cortical signal from MEDN with OC equivalent for each participant,
    convert coefficients to z-values, and perform a t-test against zero on the distribution.
    """
    print("Experiment 1, Analysis Group 6, Analysis 3", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment01_group06")
    os.makedirs(out_dir, exist_ok=True)

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with good data
    participants_df = participants_df.loc[participants_df["exclude"] != 1]
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    ALPHA = 0.05
    out_df = pd.DataFrame(
        index=participants_df["participant_id"], columns=["correlation"]
    )

    for i_run, participant_row in participants_df.iterrows():
        subj_id = participant_row["participant_id"]

        mask_file = mask_pattern.format(participant_id=subj_id)
        medn_file = medn_pattern.format(participant_id=subj_id)
        oc_file = oc_pattern.format(participant_id=subj_id)

        medn_data = masking.apply_mask(medn_file, mask_file)
        oc_data = masking.apply_mask(oc_file, mask_file)

        # Average across voxels
        medn_data = np.mean(medn_data, axis=1)
        oc_data = np.mean(oc_data, axis=1)
        corr = np.corrcoef((medn_data, oc_data))
        assert corr.shape == (2, 2), corr.shape
        corr = corr[1, 0]

        out_df.loc[subj_id, "correlation"] = corr

    # Convert r values to normally distributed z values with Fisher's
    # transformation (not test statistics though)
    r_values = out_df["correlation"].values.astype(float)
    z_values = np.arctanh(r_values)
    mean_z = np.mean(z_values)
    sd_z = np.std(z_values)

    # And now a significance test!!
    t, p = ttest_1samp(z_values, popmean=0, alternative="greater")
    if p <= ALPHA:
        print(
            "\tCorrelations between the mean cortical signal from multi-echo denoised signal and "
            "that of optimally combined data "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "\tCorrelations between the mean cortical signal from multi-echo denoised signal and "
            "that of optimally combined data "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were not significantly higher than "
            "zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=z_values, ax=ax)
    ax.set_xlabel("Z-transformed correlation coefficient")
    fig.suptitle(
        "Distribution of correlations between mean cortical signal from MEDN and OC data"
    )
    fig.savefig(op.join(out_dir, "analysis_03.png"), dpi=400)

    out_df.to_csv(
        op.join(out_dir, "analysis_03_results.tsv"),
        sep="\t",
        index_label="participant_id",
    )


if __name__ == "__main__":
    print("Experiment 1, Analysis Group 6")
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    in_dir = op.join(project_dir, "dset-dupre/")
    participants_file = op.join(in_dir, "participants.tsv")
    mask_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/anat",
        "{participant_id}_space-scanner_res-bold_label-CGM_mask.nii.gz",
    )
    medn_pattern = op.join(
        in_dir,
        "derivatives/tedana/{participant_id}/func",
        "{participant_id}_task-rest_run-1_desc-optcomDenoised_bold.nii.gz",
    )
    oc_pattern = op.join(
        in_dir,
        "derivatives/tedana/{participant_id}/func",
        "{participant_id}_task-rest_run-1_desc-optcom_bold.nii.gz",
    )
    ica_pattern = op.join(
        in_dir,
        "derivatives/tedana/{participant_id}/func",
        "{participant_id}_task-rest_run-1_desc-ICA_mixing.tsv",
    )
    ctab_pattern = op.join(
        in_dir,
        "derivatives/tedana/{participant_id}/func",
        "{participant_id}_task-rest_run-1_desc-tedana_metrics.tsv",
    )

    # plot_components_and_physio()
    correlate_ica_with_cortical_signal(
        project_dir,
        participants_file,
        ica_pattern,
        ctab_pattern,
        oc_pattern,
        mask_pattern,
    )
    correlate_medn_with_oc(
        project_dir,
        participants_file,
        medn_pattern,
        oc_pattern,
        mask_pattern,
    )
