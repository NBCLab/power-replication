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
import json
import os
import os.path as op
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from ddmra.utils import r2z  # noqa: E402
from nilearn import masking  # noqa: E402
from scipy.stats import ttest_1samp, ttest_rel  # noqa: E402

sys.path.append("..")

from utils import _plot_components_and_physio, crop_physio_data  # noqa: E402


def plot_components_and_physio(
    project_dir,
    participants_file,
    nss_file,
    ica_pattern,
    ctab_pattern,
    oc_pattern,
    dseg_pattern,
    confounds_pattern,
    physio_pattern,
    physio_metadata_pattern,
):
    """Generate plots for analysis 1."""
    print("Experiment 1, Analysis Group 6, Analysis 1", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment01_group06", "analysis01")
    os.makedirs(out_dir, exist_ok=True)

    participants_df = pd.read_table(participants_file)
    nss_df = pd.read_table(nss_file, index_col="participant_id")
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value
    participants_df = participants_df.dropna(subset=["rpv"])
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    for i_run, participant_row in participants_df.iterrows():
        subj_id = participant_row["participant_id"]
        nss = nss_df.loc[subj_id, "nss_count"]

        out_file = op.join(out_dir, f"{subj_id}.svg")

        dseg_file = dseg_pattern.format(participant_id=subj_id)
        oc_file = oc_pattern.format(participant_id=subj_id)
        confounds_file = confounds_pattern.format(participant_id=subj_id)
        ica_file = ica_pattern.format(participant_id=subj_id)
        ctab_file = ctab_pattern.format(participant_id=subj_id)
        physio_file = physio_pattern.format(participant_id=subj_id)
        physio_metadata = physio_metadata_pattern.format(participant_id=subj_id)

        # Load and process the data
        metadata_file = oc_file.replace(".nii.gz", ".json")
        with open(metadata_file) as fo:
            t_r = json.load(fo)["RepetitionTime"]

        with open(physio_metadata, "r") as fo:
            phys_meta = json.load(fo)
            physio_samplerate = phys_meta["SamplingFrequency"]
            physio_types = phys_meta["Columns"]

        # TODO: Use instantaneous heart rate at original resolution.
        n_vols = nib.load(oc_file).shape[3]
        physio_data = np.loadtxt(physio_file)
        physio_data = crop_physio_data(physio_data, physio_samplerate, t_r, nss, n_vols)
        physio_df = pd.DataFrame(columns=physio_types, data=physio_data)

        confounds_df = pd.read_table(confounds_file)

        ica_df = pd.read_table(ica_file)
        components_arr = ica_df.values.T
        comptable = pd.read_table(ctab_file, index_col="Component")
        classifications = comptable["classification"].tolist()

        _plot_components_and_physio(
            oc_file,
            dseg_file,
            confounds_df,
            physio_df,
            components_arr,
            classifications,
            t_r,
            physio_samplerate,
            out_file,
        )


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
    And to each other.
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
            "total ignored",
            "accepted count, r > 0.3",
            "accepted proportion, r > 0.3",
            "rejected count, r > 0.3",
            "rejected proportion, r > 0.3",
            "ignored count, r > 0.3",
            "ignored proportion, r > 0.3",
            "accepted count, r > 0.5",
            "accepted proportion, r > 0.5",
            "rejected count, r > 0.5",
            "rejected proportion, r > 0.5",
            "ignored count, r > 0.5",
            "ignored proportion, r > 0.5",
            "accepted mean z",
            "rejected mean z",
            "ignored mean z",
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
            comptable["classification"] == "accepted"
        ).sum()
        out_df.loc[subj_id, "total rejected"] = (
            comptable["classification"] == "rejected"
        ).sum()
        out_df.loc[subj_id, "total ignored"] = (
            comptable["classification"] == "ignored"
        ).sum()

        ica_df = pd.read_table(ica_file)
        ica_df["OC"] = oc_data

        correlations = ica_df.corr()["OC"]

        acc_corrs, rej_corrs, ign_corrs = [], [], []
        for comp in comptable.index:
            row = comptable.loc[comp]
            corr = correlations[comp]
            classification = row["classification"]
            if classification == "rejected":
                rej_corrs.append(corr)
            elif classification == "accepted":
                acc_corrs.append(corr)
            else:
                ign_corrs.append(corr)

        acc_corrs = np.array(acc_corrs)
        rej_corrs = np.array(rej_corrs)
        ign_corrs = np.array(ign_corrs)

        # Proportion and count of r > 0.3 for each classification
        out_df.loc[subj_id, "accepted count, r > 0.3"] = np.sum(acc_corrs > 0.3)
        out_df.loc[subj_id, "accepted proportion, r > 0.3"] = (
            np.sum(acc_corrs > 0.3) / acc_corrs.size
        )
        out_df.loc[subj_id, "rejected count, r > 0.3"] = np.sum(rej_corrs > 0.3)
        out_df.loc[subj_id, "rejected proportion, r > 0.3"] = (
            np.sum(rej_corrs > 0.3) / rej_corrs.size
        )
        out_df.loc[subj_id, "ignored count, r > 0.3"] = np.sum(ign_corrs > 0.3)
        out_df.loc[subj_id, "ignored proportion, r > 0.3"] = (
            np.sum(ign_corrs > 0.3) / ign_corrs.size
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
        out_df.loc[subj_id, "ignored count, r > 0.5"] = np.sum(ign_corrs > 0.5)
        out_df.loc[subj_id, "ignored proportion, r > 0.5"] = (
            np.sum(ign_corrs > 0.5) / ign_corrs.size
        )

        # Mean z-transformed correlations
        # I used ddmra's r2z function because it crops extreme correlations that would
        # evaluate to NaNs.
        out_df.loc[subj_id, "accepted mean z"] = np.mean(r2z(acc_corrs))
        out_df.loc[subj_id, "rejected mean z"] = np.mean(r2z(rej_corrs))
        out_df.loc[subj_id, "ignored mean z"] = np.mean(r2z(ign_corrs))

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
                "\tAveraged correlations between the mean cortical signal from optimally combined "
                f"data and {clf} ICA component time series "
                f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were significantly higher than "
                "zero, "
                f"t({temp_out_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
            )
        else:
            print(
                "\tAveraged correlations between the mean cortical signal from optimally combined "
                f"data and {clf} ICA component time series "
                f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were not significantly higher than "
                "zero, "
                f"t({temp_out_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
            )

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.histplot(data=z_values, ax=ax, bins=15)
        ax.set_xlabel("Z-transformed correlation coefficient")
        fig.suptitle(
            f"Distribution of correlations between mean cortical OC data and {clf} ICA components"
        )
        fig.savefig(op.join(out_dir, f"analysis_02_{clf}.png"), dpi=400)

    # Compare accepted and rejected to each other
    col = "accepted mean z"
    # First, we drop any subjects with no rejected or no accepted components
    paired_ttest_df = out_df.dropna(subset=["accepted mean z", "rejected mean z"])
    t, p = ttest_rel(
        a=paired_ttest_df["accepted mean z"].values,
        b=paired_ttest_df["rejected mean z"].values,
        alternative="greater",
    )
    if p <= ALPHA:
        print(
            "\tCorrelations between the mean cortical signal from optimally combined data and "
            "accepted ICA component time series "
            f"(M[Z] = {paired_ttest_df['accepted mean z'].mean():.03f}, "
            f"SD[Z] = {paired_ttest_df['accepted mean z'].std():.03f}) "
            "were significantly higher than those of rejected components "
            f"(M[Z] = {paired_ttest_df['rejected mean z'].mean():.03f}, "
            f"SD[Z] = {paired_ttest_df['rejected mean z'].std():.03f}), "
            f"t({paired_ttest_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "\tCorrelations between the mean cortical signal from optimally combined data and "
            "accepted ICA component time series "
            f"(M[Z] = {paired_ttest_df['accepted mean z'].mean():.03f}, "
            f"SD[Z] = {paired_ttest_df['accepted mean z'].std():.03f}) "
            "were not significantly higher than those of rejected components "
            f"(M[Z] = {paired_ttest_df['rejected mean z'].mean():.03f}, "
            f"SD[Z] = {paired_ttest_df['rejected mean z'].std():.03f}), "
            f"t({paired_ttest_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )

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

    # However, one participant has a roughly perfect correlation because tedana didn't reject
    # any components
    out_df_no_outlier = out_df.loc[out_df["correlation"] < 0.999]
    r_values_no_outlier = out_df_no_outlier["correlation"].values.astype(float)
    z_values_no_outlier = np.arctanh(r_values_no_outlier)
    mean_z_no_outlier = np.mean(z_values_no_outlier)
    sd_z_no_outlier = np.std(z_values_no_outlier)

    t_no_outlier, p_no_outlier = ttest_1samp(z_values_no_outlier, popmean=0, alternative="greater")
    if p_no_outlier <= ALPHA:
        print(
            "\tCorrelations between the mean cortical signal from multi-echo denoised signal and "
            "that of optimally combined data "
            f"(M[Z] = {mean_z_no_outlier:.03f}, SD[Z] = {sd_z_no_outlier:.03f}) were "
            "significantly higher than zero, "
            f"t({out_df_no_outlier.shape[0] - 1}) = {t_no_outlier:.03f}, p = {p_no_outlier:.03f}."
        )
    else:
        print(
            "\tCorrelations between the mean cortical signal from multi-echo denoised signal and "
            "that of optimally combined data "
            f"(M[Z] = {mean_z_no_outlier:.03f}, SD[Z] = {sd_z_no_outlier:.03f}) were not "
            "significantly higher than zero, "
            f"t({out_df_no_outlier.shape[0] - 1}) = {t_no_outlier:.03f}, p = {p_no_outlier:.03f}."
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=z_values_no_outlier, ax=ax)
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
    nss_file = op.join(in_dir, "derivatives", "power", "nss_removed.tsv")
    mask_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/anat",
        "{participant_id}_space-scanner_res-bold_label-CGM_mask.nii.gz",
    )
    dseg_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/anat",
        "{participant_id}_space-scanner_res-bold_desc-totalMaskWithCSF_dseg.nii.gz",
    )
    confounds_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/func",
        "{participant_id}_task-rest_run-1_desc-confounds_timeseries.tsv",
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
    physio_pattern = op.join(
        in_dir,
        "{participant_id}/func/{participant_id}_task-rest_run-01_physio.tsv.gz"
    )
    physio_metadata_pattern = op.join(
        in_dir,
        "{participant_id}/{participant_id}_task-rest_physio.json"
    )

    plot_components_and_physio(
        project_dir,
        participants_file,
        nss_file,
        ica_pattern,
        ctab_pattern,
        oc_pattern,
        dseg_pattern,
        confounds_pattern,
        physio_pattern,
        physio_metadata_pattern,
    )
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
