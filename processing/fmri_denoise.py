"""
Perform standard denoising (not TE-dependent denoising).

Methods:
- Global signal regression
- Robust PCA
- GODEC
- RVT (with lags) regression
- RV (with lags) regression
- Dynamic global signal regression
- Wavelet denoising (integrated in tedana and thus run in fmri_denoise_me.py)
"""
import os.path as op

import numpy as np
import nibabel as nib
from bids.grabbids import BIDSLayout
from nilearn.masking import apply_mask, unmask


def run_gsr():
    """
    Run global signal regression on input image using GM mask for v2.5.
    """
    pass


def run_rpca():
    pass


def run_godec():
    pass


def run_rvtreg(dset, method, in_dir='/scratch/tsalo006/power-replication/'):
    """
    Generate ICA denoised data after regressing out RVT and RVT convolved with
    RRF (including lags)

    Used for:
    -   Scatter plots with fitted lines and correlations between STD of
        ventilatory envelope and STD of global signal for ICA-denoised data
        after regression of RVT + RVT*RRF regressors (from S5d’s approach)
    -   RPV correlated with SD of global fMRI signal from
        MEICA-denoised+RVT + RVT*RRF data
    """
    dset_dir = op.join(in_dir, dset)
    layout = BIDSLayout(dset_dir)
    subjects = layout.get_subjects()

    power_dir = op.join(dset_dir, 'derivatives/power')

    for subj in subjects:
        power_subj_dir = op.join(power_dir, subj)
        preproc_dir = op.join(power_subj_dir, 'preprocessed')
        physio_dir = op.join(preproc_dir, 'physio')
        anat_dir = op.join(preproc_dir, 'anat')

        func_file = op.join(power_subj_dir, 'denoised', method)
        mask_file = op.join(anat_dir, 'cortical_mask.nii.gz')
        func_img = nib.load(func_file)
        mask_img = nib.load(mask_file)

        rvt_file = op.join(physio_dir,
                           'sub-{0}_task-rest_run-01_rvt.txt'.format(subj))
        rvt_x_rrf_file = op.join(physio_dir,
                                 'sub-{0}_task-rest_run-01_rvtXrrf'
                                 '.txt'.format(subj))
        rvt = np.loadtxt(rvt_file)
        rvt_x_rrf = np.loadtxt(rvt_x_rrf_file)
        rvt = np.hstack((np.ones(func_img.shape[-1]), rvt, rvt_x_rrf))

        func_data = apply_mask(func_img, mask_img)
        lstsq_res = np.linalg.lstsq(rvt, func_data, rcond=None)
        pred = np.dot(rvt, lstsq_res[0])
        residuals = func_data - pred
        resid_img = unmask(residuals, mask_img)


def run_rvreg():
    pass


def run_dgsr():
    pass
