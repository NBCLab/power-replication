"""Experiment 2, Analysis Group 2.

Comparing measures of global signal.

Mean cortical signal of MEDN correlated with signal of all gray matter
-   Distribution of Pearson correlation coefficients
-   Page 2, right column, first paragraph

Mean cortical signal of MEDN correlated with signal of whole brain
-   Distribution of Pearson correlation coefficients
-   Page 2, right column, first paragraph
"""
import os.path as op

import numpy as np
from nilearn import image, masking
from scipy.stats import ttest_1samp


def correlate_cort_with_gm(project_dir, participants_df):
    """Correlate mean cortical signal from MEDN files with signal from all gray matter.

    -   Distribution of Pearson correlation coefficients
    -   Page 2, right column, first paragraph
    """
    ALPHA = 0.05

    corrs = []
    for i_run, participant_row in participants_df.iterrows():
        if participant_row["exclude"] == 1:
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
        dseg_file = op.join(
            project_dir,
            dset,
            "derivatives",
            "power",
            subj_id,
            "anat",
            f"{subj_id}_space-scanner_res-bold_desc-totalMaskWithCSF_dseg.nii.gz",
        )
        # Values 1-3 are cortical ribbon, subcortical structures, and cerebellum, respectively.
        gm_mask = image.math_img(
            "np.logical_and(img > 0, img <= 3).astype(int)", img=dseg_file
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

        cort_data = masking.apply_mask(medn_file, cort_mask)
        gm_data = masking.apply_mask(medn_file, gm_mask)

        # Average across voxels
        cort_data = np.mean(cort_data, axis=1)  # TODO: CHECK AXIS ORDER
        gm_data = np.mean(gm_data, axis=1)
        corr = np.corrcoef((cort_data, gm_data))
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
            "ANALYSIS 1: Correlations between the mean multi-echo denoised signal extracted from "
            "the cortical ribbon and that extracted from all gray matter "
            f"(M[Z] = {mean_z}, SD[Z] = {sd_z}) were significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "ANALYSIS 1: Correlations between the mean multi-echo denoised signal extracted from "
            "the cortical ribbon and that extracted from all gray matter "
            f"(M[Z] = {mean_z}, SD[Z] = {sd_z}) were not significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )


def correlate_cort_with_wb(project_dir, participants_df):
    """Correlate mean cortical signal from MEDN files with signal from whole brain.

    -   Distribution of Pearson correlation coefficients
    -   Page 2, right column, first paragraph
    """
    ALPHA = 0.05

    corrs = []
    for i_run, participant_row in participants_df.iterrows():
        if participant_row["exclude"] == 1:
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
        dseg_file = op.join(
            project_dir,
            dset,
            "derivatives",
            "power",
            subj_id,
            "anat",
            f"{subj_id}_space-scanner_res-bold_desc-totalMaskWithCSF_dseg.nii.gz",
        )
        # Values 1+ are brain.
        wb_mask = image.math_img("img > 0", img=dseg_file)
        medn_file = op.join(
            project_dir,
            dset,
            "derivatives",
            "tedana",
            subj_id,
            "func",
            "sub-01_task-rest_run-1_desc-optcomDenoised_bold.nii.gz",
        )

        cort_data = masking.apply_mask(medn_file, cort_mask)
        wb_data = masking.apply_mask(medn_file, wb_mask)

        # Average across voxels
        cort_data = np.mean(cort_data, axis=1)  # TODO: CHECK AXIS ORDER
        wb_data = np.mean(wb_data, axis=1)
        corr = np.corrcoef((cort_data, wb_data))
        assert corr.shape == (2, 2), corr.shape
        corr = corr[1, 0]

        corrs.append(corr)

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
            "ANALYSIS 2: Correlations between the mean multi-echo denoised signal extracted from "
            "the cortical ribbon and that extracted from the whole brain "
            f"(M[Z] = {mean_z}, SD[Z] = {sd_z}) were significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
    else:
        print(
            "ANALYSIS 2: Correlations between the mean multi-echo denoised signal extracted from "
            "the cortical ribbon and that extracted from the whole brain "
            f"(M[Z] = {mean_z}, SD[Z] = {sd_z}) were not significantly higher than zero, "
            f"t({participants_df.shape[0] - 1}) = {t:.03f}, p = {p:.03f}."
        )
