"""
Mean cortical signal correlated with signal of all gray matter
-   Distribution of Pearson correlation coefficients
-   Page 2, right column, first paragraph

Mean cortical signal correlated with signal of whole brain
-   Distribution of Pearson correlation coefficients
-   Page 2, right column, first paragraph
"""
import os.path as op

import nibabel as nib
import numpy as np
from bids.grabbids import BIDSLayout
from nilearn.masking import apply_mask
from scipy.stats import ttest_1samp


def mean_signal_corr_dist(fmri_files, mask1_files, mask2_files):
    """
    Extract timeseries from masked 4D BOLD files and generate/analyze
    distributions of correlation coefficients.

    1)  Take in a list of 4D BOLD files, a list of mask files for first mask,
        and a list of mask files for second mask. Lists are in same order
        according to subject.
    2)  Load data into Nifti1Image objects and extract masked data to VxT
        arrays for each mask. Average across voxels.
    3)  Compute Pearson correlation coefficient between timeseries of the two
        masks. Save subject's correlation coefficient to numpy array.
    4)  Convert correlation coefficients to z-scores (*not* test statistics).
    5)  Perform one-sample t-test on z-scores.
    """
    assert len(fmri_files) == len(mask1_files) == len(mask2_files)
    corrs = np.zeros(len(fmri_files))
    for i, fmri_file in enumerate(fmri_files):
        fmri_img = nib.load(fmri_file)
        mask1_img = nib.load(mask1_files[i])
        mask2_img = nib.load(mask2_files[i])
        mask1_fmri = apply_mask(img, mask1_img)
        mask2_fmri = apply_mask(img, mask2_img)

        # Average across voxels
        mask1_fmri = np.mean(mask1_fmri, axis=1)  # TODO: CHECK AXIS ORDER
        mask2_fmri = np.mean(mask2_fmri, axis=1)
        corr = np.corrcoef(mask1_fmri, mask2_fmri)[1, 0]
        corrs[i] = corr

        # Convert r values to normally distributed z values with Fisher's
        # transformation (not test statistics though)
        z_vals = np.arctanh(corrs)

        # And now a significance test!!
        # TODO: Should we compute confidence intervals from z-values then
        # convert back to r-values? I think so, but there's so little in the
        # literature about dealing with *distributions* of correlation
        # coefficients.
        t, p = ttest_1samp(z_vals, popmean=0)

    return corrs, t, p


def analysis_01(in_dir, dset):
    """
    Mean cortical signal correlated with signal of all gray matter
    -   Distribution of Pearson correlation coefficients
    -   Page 2, right column, first paragraph
    """
    dset_dir = op.join(in_dir, dset)
    deriv_dir = op.join(dset_dir, "derivatives")
    layout = BIDSLayout(dset_dir)
    subjects = layout.get_subjects()

    fmri_files = []
    cort_mask_files = []
    gm_mask_files = []
    for subj in subjects:
        cort_mask_file = op.join(deriv_dir, "power", subj, "cortical_mask.nii.gz")
        gm_mask_file = op.join(deriv_dir, "power", subj, "graymatter_mask.nii.gz")
        fmri_file = op.join(deriv_dir, "fmriprep", subj, "preproc.nii.gz")
        fmri_files.append(fmri_file)
        cort_mask_files.append(cort_mask_file)
        gm_mask_files.append(gm_mask_file)

    corrs, t, p = mean_signal_corr_dist(fmri_files, cort_mask_files, gm_mask_files)


def analysis_02(in_dir, dset):
    """
    Mean cortical signal correlated with signal of whole brain
    -   Distribution of Pearson correlation coefficients
    -   Page 2, right column, first paragraph
    """
    dset_dir = op.join(in_dir, dset)
    deriv_dir = op.join(dset_dir, "derivatives")
    layout = BIDSLayout(dset_dir)
    subjects = layout.get_subjects()

    fmri_files = []
    cort_mask_files = []
    brain_mask_files = []
    for subj in subjects:
        cort_mask_file = op.join(deriv_dir, "power", subj, "cortical_mask.nii.gz")
        brain_mask_file = op.join(deriv_dir, "power", subj, "brain_mask.nii.gz")
        fmri_file = op.join(deriv_dir, "fmriprep", subj, "preproc.nii.gz")
        fmri_files.append(fmri_file)
        cort_mask_files.append(cort_mask_file)
        brain_mask_files.append(brain_mask_file)

    corrs, t, p = mean_signal_corr_dist(fmri_files, cort_mask_files, brain_mask_files)
