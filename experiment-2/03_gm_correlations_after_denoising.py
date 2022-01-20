"""Experiment 2, Analysis Group 3.

Evaluating changes in global signal after multi-echo denoising.

Mean cortical signal from OC correlated with mean cortical signal from MEDN.
- One-sample t-test on z-transformed correlation coefficients.

Mean cortical signal from MEDN correlated with mean cortical signal from FIT-R2.
- One-sample t-test on z-transformed correlation coefficients.
"""
import os
import os.path as op
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import masking
from scipy.stats import ttest_1samp

sys.path.append("..")

from utils import get_prefixes  # noqa: E402


def correlate_medn_with_oc(
    project_dir,
    participants_file,
    medn_pattern,
    oc_pattern,
    mask_pattern,
):
    """Correlate mean cortical signal from MEDN files with mean cortical signal from OC files."""
    print("Experiment 2, Analysis Group 3, Analysis 1", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment02_group03")
    os.makedirs(out_dir, exist_ok=True)

    ALPHA = 0.05

    participants_df = pd.read_table(participants_file)

    for i_run, participant_row in participants_df.iterrows():
        subj_id = participant_row["participant_id"]
        dset = participant_row["dataset"]
        dset_prefix = get_prefixes()[dset]
        subj_prefix = dset_prefix.format(participant_id=subj_id)

        medn_file = medn_pattern.format(
            dset=dset, participant_id=subj_id, prefix=subj_prefix
        )
        oc_file = medn_pattern.format(
            dset=dset, participant_id=subj_id, prefix=subj_prefix
        )
        mask_file = mask_pattern.format(dset=dset, participant_id=subj_id)

        medn_data = masking.apply_mask(medn_file, mask_file)
        oc_data = masking.apply_mask(oc_file, mask_file)

        # Average across voxels
        medn_data = np.mean(medn_data, axis=1)
        oc_data = np.mean(oc_data, axis=1)
        corr = np.corrcoef((medn_data, oc_data))
        assert corr.shape == (2, 2), corr.shape
        participants_df.loc[i_run, "correlation"] = corr[1, 0]

    # Convert r values to normally distributed z values with Fisher's
    # transformation (not test statistics though)
    z_values = np.arctanh(participants_df["correlation"])
    mean_z = np.mean(z_values)
    sd_z = np.std(z_values)

    # And now a significance test!!
    t, p = ttest_1samp(z_values, popmean=0, alternative="greater")
    if p <= ALPHA:
        print(
            "\tCorrelations between the mean cortical ribbon signal from the multi-echo "
            "denoised data and the optimally combined data "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "\tCorrelations between the mean cortical ribbon signal from the multi-echo "
            "denoised data and the optimally combined data "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were not significantly higher than "
            "zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=z_values, ax=ax)
    ax.set_xlabel("Z-transformed correlation coefficient")
    fig.suptitle(
        "Distribution of correlations between mean cortical signal from MEDN and OC "
        "data"
    )
    fig.savefig(op.join(out_dir, "analysis_01.png"), dpi=400)

    participants_df.to_csv(
        op.join(out_dir, "analysis_01_results.tsv"), sep="\t", index=False
    )


def correlate_medn_with_fitr2(
    project_dir,
    participants_file,
    medn_pattern,
    fitr2_pattern,
    mask_pattern,
):
    """Correlate mean cortical signal from MEDN files with equivalent from FIT-R2 files."""
    print("Experiment 2, Analysis Group 3, Analysis 2", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment02_group03")
    os.makedirs(out_dir, exist_ok=True)

    ALPHA = 0.05

    participants_df = pd.read_table(participants_file)

    for i_run, participant_row in participants_df.iterrows():
        subj_id = participant_row["participant_id"]
        dset = participant_row["dataset"]
        dset_prefix = get_prefixes()[dset]
        subj_prefix = dset_prefix.format(participant_id=subj_id)

        medn_file = medn_pattern.format(
            dset=dset, participant_id=subj_id, prefix=subj_prefix
        )
        fitr2_file = medn_pattern.format(
            dset=dset, participant_id=subj_id, prefix=subj_prefix
        )
        mask_file = mask_pattern.format(dset=dset, participant_id=subj_id)

        medn_data = masking.apply_mask(medn_file, mask_file)
        fitr2_data = masking.apply_mask(fitr2_file, mask_file)

        # Average across voxels
        medn_data = np.mean(medn_data, axis=1)
        fitr2_data = np.mean(fitr2_data, axis=1)
        corr = np.corrcoef((medn_data, fitr2_data))
        assert corr.shape == (2, 2), corr.shape
        participants_df.loc[i_run, "correlation"] = corr[1, 0]

    # Convert r values to normally distributed z values with Fisher's
    # transformation (not test statistics though)
    z_values = np.arctanh(participants_df["correlation"])
    mean_z = np.mean(z_values)
    sd_z = np.std(z_values)

    # And now a significance test!!
    t, p = ttest_1samp(z_values, popmean=0, alternative="greater")
    if p <= ALPHA:
        print(
            "\tCorrelations between the mean cortical ribbon signal from the multi-echo "
            "denoised data and the FIT-R2 data "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "\tCorrelations between the mean cortical ribbon signal from the multi-echo "
            "denoised data and the FIT-R2 data "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were not significantly higher than "
            "zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=z_values, ax=ax)
    ax.set_xlabel("Z-transformed correlation coefficient")
    fig.suptitle(
        "Distribution of correlations between mean cortical signal from MEDN and FIT-R2 "
        "data"
    )
    fig.savefig(op.join(out_dir, "analysis_02.png"), dpi=400)

    participants_df.to_csv(
        op.join(out_dir, "analysis_02_results.tsv"), sep="\t", index=False
    )


if __name__ == "__main__":
    print("Experiment 2, Analysis Group 3")
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    participants_file = op.join(project_dir, "participants.tsv")
    in_dir = op.join(project_dir, "{dset}")
    medn_pattern = op.join(
        in_dir,
        "derivatives",
        "tedana",
        "{participant_id}",
        "func",
        "{prefix}_desc-optcomDenoised_bold.nii.gz",
    )
    oc_pattern = op.join(
        in_dir,
        "derivatives",
        "tedana",
        "{participant_id}",
        "func",
        "{prefix}_desc-optcom_bold.nii.gz",
    )
    fitr2_pattern = op.join(
        in_dir,
        "derivatives",
        "t2smap",
        "{participant_id}",
        "func",
        "{prefix}_T2starmap.nii.gz",
    )
    mask_pattern = op.join(
        in_dir,
        "derivatives",
        "power",
        "{participant_id}",
        "anat",
        "{participant_id}_space-scanner_res-bold_label-CGM_mask.nii.gz",
    )
    correlate_medn_with_oc(
        project_dir,
        participants_file,
        medn_pattern,
        oc_pattern,
        mask_pattern,
    )
    correlate_medn_with_fitr2(
        project_dir,
        participants_file,
        medn_pattern,
        fitr2_pattern,
        mask_pattern,
    )
