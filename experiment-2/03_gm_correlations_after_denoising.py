"""Experiment 2, Analysis Group 3.

Evaluating changes in global signal after multi-echo denoising.

Mean cortical signal from OC correlated with mean cortical signal from MEDN.
- One-sample t-test on z-transformed correlation coefficients.

Mean cortical signal from MEDN correlated with mean cortical signal from FIT-R2.
- One-sample t-test on z-transformed correlation coefficients.
"""
import os.path as op

import numpy as np
from nilearn import masking
from scipy.stats import ttest_1samp


def correlate_medn_with_oc(project_dir, participants_df):
    """Correlate mean cortical signal from MEDN files with mean cortical signal from OC files."""
    ALPHA = 0.05

    corrs = []
    for i_run, participant_row in participants_df.iterrows():
        if participant_row["include"] == 0:
            print(f"Skipping {participant_row['participant_id']}.")
            continue

        subj_id = participant_row["participant_id"]
        dset = participant_row["dataset"]

        cort_mask = op.join(
            project_dir,
            dset,
            "derivatives",
            "power",
            subj_id,
            "anat",
            f"{subj_id}_space-scanner_res-bold_label-CGM_mask.nii.gz",
        )
        medn_file = op.join(
            project_dir,
            dset,
            "derivatives",
            "tedana",
            subj_id,
            "func",
            "sub-01_task-rest_run-1_desc-optcomDenoised_bold.nii.gz",
        )
        oc_file = op.join(
            project_dir,
            dset,
            "derivatives",
            "tedana",
            subj_id,
            "func",
            "sub-01_task-rest_run-1_desc-optcom_bold.nii.gz",
        )

        medn_data = masking.apply_mask(medn_file, cort_mask)
        oc_data = masking.apply_mask(oc_file, cort_mask)

        # Average across voxels
        medn_data = np.mean(medn_data, axis=1)  # TODO: CHECK AXIS ORDER
        oc_data = np.mean(oc_data, axis=1)
        corr = np.corrcoef((medn_data, oc_data))
        assert corr.shape == (2, 2), corr.shape
        corr = corr[1, 0]

        corrs.append(corr)

    corrs = np.array(corrs)

    # Convert r values to normally distributed z values with Fisher's
    # transformation (not test statistics though)
    z_values = np.arctanh(corrs)
    mean_z = np.mean(z_values)
    sd_z = np.std(z_values)

    # And now a significance test!!
    # TODO: Should we compute confidence intervals from z-values then
    # convert back to r-values? I think so, but there's so little in the
    # literature about dealing with *distributions* of correlation
    # coefficients.
    t, p = ttest_1samp(z_values, popmean=0, alternative="greater")
    if p <= ALPHA:
        print(
            "ANALYSIS 1: Correlations between the mean cortical ribbon signal from the multi-echo "
            "denoised data and the optimally combined data "
            f"(M[Z] = {mean_z}, SD[Z] = {sd_z}) were significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "ANALYSIS 1: Correlations between the mean cortical ribbon signal from the multi-echo "
            "denoised data and the optimally combined data "
            f"(M[Z] = {mean_z}, SD[Z] = {sd_z}) were not significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )


def correlate_medn_with_fitr2(project_dir, participants_df):
    """Correlate mean cortical signal from MEDN files with equivalent from FIT-R2 files."""
    ALPHA = 0.05

    corrs = []
    for i_run, participant_row in participants_df.iterrows():
        if participant_row["include"] == 0:
            print(f"Skipping {participant_row['participant_id']}.")
            continue

        subj_id = participant_row["participant_id"]
        dset = participant_row["dataset"]

        cort_mask = op.join(
            project_dir,
            dset,
            "derivatives",
            "power",
            subj_id,
            "anat",
            f"{subj_id}_space-scanner_res-bold_label-CGM_mask.nii.gz",
        )
        medn_file = op.join(
            project_dir,
            dset,
            "derivatives",
            "tedana",
            subj_id,
            "func",
            f"{subj_id}_task-rest_run-1_desc-optcomDenoised_bold.nii.gz",
        )
        fitr2_file = op.join(
            project_dir,
            dset,
            "derivatives",
            "t2smap",
            subj_id,
            "func",
            f"{subj_id}_task-rest_run-1_T2starmap.nii.gz",
        )

        medn_data = masking.apply_mask(medn_file, cort_mask)
        fitr2_data = masking.apply_mask(fitr2_file, cort_mask)

        # Average across voxels
        medn_data = np.mean(medn_data, axis=1)  # TODO: CHECK AXIS ORDER
        fitr2_data = np.mean(fitr2_data, axis=1)
        corr = np.corrcoef((medn_data, fitr2_data))
        assert corr.shape == (2, 2), corr.shape
        corr = corr[1, 0]

        corrs.append(corr)

    corrs = np.array(corrs)

    # Convert r values to normally distributed z values with Fisher's
    # transformation (not test statistics though)
    z_values = np.arctanh(corrs)
    mean_z = np.mean(z_values)
    sd_z = np.std(z_values)

    # And now a significance test!!
    # TODO: Should we compute confidence intervals from z-values then
    # convert back to r-values? I think so, but there's so little in the
    # literature about dealing with *distributions* of correlation
    # coefficients.
    t, p = ttest_1samp(z_values, popmean=0, alternative="greater")
    if p <= ALPHA:
        print(
            "ANALYSIS 1: Correlations between the mean cortical ribbon signal from the multi-echo "
            "denoised data and the FIT-R2 data "
            f"(M[Z] = {mean_z}, SD[Z] = {sd_z}) were significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "ANALYSIS 1: Correlations between the mean cortical ribbon signal from the multi-echo "
            "denoised data and the FIT-R2 data "
            f"(M[Z] = {mean_z}, SD[Z] = {sd_z}) were not significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
