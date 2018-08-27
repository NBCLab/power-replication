"""
Perform standard denoising (not TE-dependent denoising).

Methods:
-   Global signal regression (integrated in tedana, but we do it for real here
    because we want to use the cortical ribbon specifically)
-   Dynamic global signal regression
-   compCor
-   GODEC (integrated in tedana)
-   RVT (with lags) regression
-   RV (with lags) regression
"""
import os.path as op

import numpy as np
import nibabel as nib
from bids.grabbids import BIDSLayout
from nilearn.masking import apply_mask, unmask


def run_rvtreg(dset, task, method, suffix, in_dir='/scratch/tsalo006/power-replication/'):
    """
    Generate ICA denoised data after regressing out RVT and RVT convolved with
    RRF (including lags)

    Parameters
    ----------
    dset : {'ds000210', 'ds000254', 'ds000258'}
    task : {'rest', 'fingertapping'}
    method : {'meica_v2_5', 'fit'}
    suffix : {'dn_ts_OC', 'hik_ts_OC', 't2s'}
    in_dir : str
        Path to analysis folder

    Used for:
    -   Carpet plots of ME-DN after regression of RVT + RVT*RRF (S5)
    -   Carpet plots of FIT-R2 after regression of RVT + RVT*RRF (not in paper)
    -   Carpet plots of ME-HK after regression of RVT + RVT*RRF (not in paper)
    -   Scatter plot of ME-DN-RVT+RVT*RRF SD of global signal against
        SD of ventilatory envelope (RPV) (S8).
    -   Scatter plot of FIT-R2-RVT+RVT*RRF SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    -   Scatter plot of ME-HK-RVT+RVT*RRF SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
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

        func_file = op.join(power_subj_dir, 'denoised', method,
                            'sub-{0}_task-{1}_run-01_{2}'
                            '.nii.gz'.format(subj, task, suffix))
        mask_file = op.join(anat_dir, 'cortical_mask.nii.gz')
        func_img = nib.load(func_file)
        mask_img = nib.load(mask_file)
        if subj == subjects[0]:
            resid_sds = np.empty((len(subjects), func_img.shape[-1]))

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

        # Get volume-wise standard deviation
        resid_sd = np.std(residuals, axis=0)
        resid_sds[subj, :] = resid_sd
        return resid_sds


def run_rvreg():
    """
    Generate ICA denoised data after regressing out RV and RV convolved with
    RRF (including lags)

    Parameters
    ----------
    dset : {'ds000210', 'ds000254', 'ds000258'}
    task : {'rest', 'fingertapping'}
    method : {'meica_v2_5', 'fit'}
    suffix : {'dn_ts_OC', 'hik_ts_OC', 't2s'}
    in_dir : str
        Path to analysis folder

    Used for:
    -   Carpet plots of ME-DN after regression of RV + RV*RRF (S5)
    -   Carpet plots of FIT-R2 after regression of RV + RV*RRF (not in paper)
    -   Carpet plots of ME-HK after regression of RV + RV*RRF (not in paper)
    -   Scatter plot of ME-DN-RV+RV*RRF SD of global signal against
        SD of ventilatory envelope (RPV) (S8).
    -   Scatter plot of FIT-R2-RV+RV*RRF SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    -   Scatter plot of ME-HK-RV+RV*RRF SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    """
    pass


def run_nuisance():
    """
    Generate ICA denoised data after regressing out nuisance regressors
    Regressors include mean white matter, mean CSF, 6 motion parameters, and
    first temporal derivatives of motion parameters

    Parameters
    ----------
    dset : {'ds000210', 'ds000254', 'ds000258'}
    task : {'rest', 'fingertapping'}
    method : {'meica_v2_5', 'fit'}
    suffix : {'dn_ts_OC', 'hik_ts_OC', 't2s'}
    in_dir : str
        Path to analysis folder

    Used for:
    -   Carpet plots of ME-DN after regression of nuisance (S7)
    -   Carpet plots of FIT-R2 after regression of nuisance (S6)
    -   Carpet plots of ME-HK after regression of nuisance (not in paper)
    -   Scatter plot of ME-DN-Nuis SD of global signal against
        SD of ventilatory envelope (RPV) (S8).
    -   Scatter plot of FIT-R2-Nuis SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    -   Scatter plot of ME-HK-Nuis SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    """
    pass


def run_dgsr():
    pass


def run_gsr():
    """
    Run global signal regression.

    Parameters
    ----------
    dset : {'ds000210', 'ds000254', 'ds000258'}
    task : {'rest', 'fingertapping'}
    method : {'meica_v2_5'}
    suffix : {'dn_ts_OC', 'hik_ts_OC', 't2s'}
    in_dir : str
        Path to analysis folder

    Used for:
    -   Carpet plots of ME-DN after GSR (3, S12)
    -   Carpet plots of ME-HK after GSR (not in paper)
    -   QC:RSFC plot of ME-DN after GODEC with motion as QC (4, 5, S10, S13)
        - S10 involves censoring FD>0.2mm
    -   QC:RSFC plot of ME-HK after GODEC with motion as QC (not in paper)
    -   QC:RSFC plot of ME-DN after GODEC with RPV as QC (5)
    -   QC:RSFC plot of ME-HK after GODEC with RPV as QC (not in paper)
    -   High-low motion plot of ME-DN after GSR (4, S10)
        - S10 involves censoring FD>0.2mm
    -   High-low motion plot of ME-HK after GSR (not in paper)
    -   Scrubbing plot of ME-DN after GSR (4)
    -   Scrubbing plot of ME-HK after GSR (not in paper)
    -   Mean correlation matrix and histogram of ME-DN after GSR (S13)
    -   Correlation scatterplot of ME-DN after GSR against other ME-DN
        outputs (S13)
    -   Scatter plot of ME-DN-GSR SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    -   Scatter plot of ME-HK-GSR SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    """
    pass


def run_godec():
    """
    Not to be run

    Parameters
    ----------
    dset : {'ds000210', 'ds000254', 'ds000258'}
    task : {'rest', 'fingertapping'}
    method : {'meica_v2_5'}
    suffix : {'dn_ts_OC', 'hik_ts_OC', 't2s'}
    in_dir : str
        Path to analysis folder

    Used for:
    -   Carpet plots of ME-DN after GODEC (3, S9, S12)
    -   Carpet plots of ME-HK after GODEC (not in paper)
    -   QC:RSFC plot of ME-DN after GODEC with motion as QC (4, 5, S10, S13)
        - S10 involves censoring FD>0.2mm
    -   QC:RSFC plot of ME-HK after GODEC with motion as QC (not in paper)
    -   QC:RSFC plot of ME-DN after GODEC with RPV as QC (5)
    -   QC:RSFC plot of ME-HK after GODEC with RPV as QC (not in paper)
    -   High-low motion plot of ME-DN after GODEC (4, S10)
        - S10 involves censoring FD>0.2mm
    -   High-low motion plot of ME-HK after GODEC (not in paper)
    -   Scrubbing plot of ME-DN after GODEC (4)
    -   Scrubbing plot of ME-HK after GODEC (not in paper)
    -   Mean correlation matrix and histogram of ME-DN after GODEC (S13)
    -   Correlation scatterplot of ME-DN after GODEC against other ME-DN
        outputs (S13)
    -   Scatter plot of ME-DN-GODEC SD of global signal against
        SD of ventilatory envelope (RPV) (2).
    -   Scatter plot of ME-HK-GODEC SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    """
    pass


def run_compcor():
    pass
