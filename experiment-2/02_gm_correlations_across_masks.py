"""Experiment 2, Analysis Group 2.

Comparing measures of global signal.

Mean cortical signal of MEDN correlated with signal of all gray matter
-   Distribution of Pearson correlation coefficients
-   Page 2, right column, first paragraph

Mean cortical signal of MEDN correlated with signal of whole brain
-   Distribution of Pearson correlation coefficients
-   Page 2, right column, first paragraph
"""
import os
import os.path as op
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import image, masking
from scipy.stats import ttest_1samp

sys.path.append("..")

from utils import get_prefixes  # noqa: E402


def correlate_cort_with_gm(
    project_dir,
    participants_file,
    medn_pattern,
    mask_pattern,
    dseg_pattern,
):
    """Correlate mean cortical signal from MEDN files with signal from all gray matter.

    -   Distribution of Pearson correlation coefficients
    -   Page 2, right column, first paragraph
    """
    print("Experiment 2, Analysis Group 2, Analysis 1", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment02_group02")
    os.makedirs(out_dir, exist_ok=True)

    ALPHA = 0.05

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value
    participants_df = participants_df.loc[participants_df["exclude"] != 1]
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    for i_run, participant_row in participants_df.iterrows():
        subj_id = participant_row["participant_id"]
        dset = participant_row["dataset"]
        dset_prefix = get_prefixes()[dset]
        subj_prefix = dset_prefix.format(participant_id=subj_id)

        medn_file = medn_pattern.format(
            dset=dset, participant_id=subj_id, prefix=subj_prefix
        )
        mask_file = mask_pattern.format(dset=dset, participant_id=subj_id)
        dseg_file = dseg_pattern.format(dset=dset, participant_id=subj_id)

        # Values 1-3 are cortical ribbon, subcortical structures, and cerebellum, respectively.
        gm_mask = image.math_img(
            "np.logical_and(img > 0, img <= 3).astype(int)", img=dseg_file
        )

        cort_data = masking.apply_mask(medn_file, mask_file)
        gm_data = masking.apply_mask(medn_file, gm_mask)

        # Average across voxels
        cort_data = np.mean(cort_data, axis=1)
        gm_data = np.mean(gm_data, axis=1)
        corr = np.corrcoef((cort_data, gm_data))
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
            "\tCorrelations between the mean multi-echo denoised signal extracted from "
            "the cortical ribbon and that extracted from all gray matter "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "\tCorrelations between the mean multi-echo denoised signal extracted from "
            "the cortical ribbon and that extracted from all gray matter "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were not significantly higher than "
            "zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=z_values, ax=ax)
    ax.set_xlabel("Z-transformed correlation coefficient")
    fig.suptitle(
        "Distribution of correlations between mean cortical and mean gray matter signal from MEDN "
        "data"
    )
    fig.savefig(op.join(out_dir, "analysis_01.png", dpi=400))

    participants_df.to_csv(
        op.join(out_dir, "analysis_01_results.tsv"), sep="\t", index=False
    )


def correlate_cort_with_wb(
    project_dir,
    participants_file,
    medn_pattern,
    mask_pattern,
    dseg_pattern,
):
    """Correlate mean cortical signal from MEDN files with signal from whole brain.

    -   Distribution of Pearson correlation coefficients
    -   Page 2, right column, first paragraph
    """
    print("Experiment 2, Analysis Group 2, Analysis 1", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment02_group02")
    os.makedirs(out_dir, exist_ok=True)

    ALPHA = 0.05

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value
    participants_df = participants_df.loc[participants_df["exclude"] != 1]
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    for i_run, participant_row in participants_df.iterrows():
        subj_id = participant_row["participant_id"]
        dset = participant_row["dataset"]
        dset_prefix = get_prefixes()[dset]
        subj_prefix = dset_prefix.format(participant_id=subj_id)

        medn_file = medn_pattern.format(
            dset=dset, participant_id=subj_id, prefix=subj_prefix
        )
        mask_file = mask_pattern.format(dset=dset, participant_id=subj_id)
        dseg_file = dseg_pattern.format(dset=dset, participant_id=subj_id)

        # Values 1+ are brain.
        wb_mask = image.math_img("img > 0", img=dseg_file)

        cort_data = masking.apply_mask(medn_file, mask_file)
        wb_data = masking.apply_mask(medn_file, wb_mask)

        # Average across voxels
        cort_data = np.mean(cort_data, axis=1)
        wb_data = np.mean(wb_data, axis=1)
        corr = np.corrcoef((cort_data, wb_data))
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
            "\tCorrelations between the mean multi-echo denoised signal extracted from "
            "the cortical ribbon and that extracted from the whole brain "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "\tCorrelations between the mean multi-echo denoised signal extracted from "
            "the cortical ribbon and that extracted from the whole brain "
            f"(M[Z] = {mean_z:.03f}, SD[Z] = {sd_z:.03f}) were not significantly higher than "
            "zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=z_values, ax=ax)
    ax.set_xlabel("Z-transformed correlation coefficient")
    fig.suptitle(
        "Distribution of correlations between mean cortical and mean whole brain signal from MEDN "
        "data"
    )
    fig.savefig(op.join(out_dir, "analysis_02.png", dpi=400))

    participants_df.to_csv(
        op.join(out_dir, "analysis_02_results.tsv"), sep="\t", index=False
    )


if __name__ == "__main__":
    print("Experiment 2, Analysis Group 2")
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
    mask_pattern = op.join(
        in_dir,
        "derivatives",
        "power",
        "{participant_id}",
        "anat",
        "{participant_id}_space-scanner_res-bold_label-CGM_mask.nii.gz",
    )
    dseg_pattern = op.join(
        in_dir,
        "derivatives",
        "power",
        "{participant_id}",
        "anat",
        "{participant_id}_space-scanner_res-bold_desc-totalMaskWithCSF_dseg.nii.gz",
    )
    correlate_cort_with_gm(
        project_dir,
        participants_file,
        medn_pattern,
        mask_pattern,
        dseg_pattern,
    )
    correlate_cort_with_wb(
        project_dir,
        participants_file,
        medn_pattern,
        mask_pattern,
        dseg_pattern,
    )
