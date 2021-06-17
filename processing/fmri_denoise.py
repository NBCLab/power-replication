"""
Perform standard denoising (not TE-dependent denoising).

Methods:
-   Global signal regression with custom code
    (integrated in tedana, but we do it separately here because the approach is very different)
-   Dynamic global signal regression with rapidtide
-   aCompCor with custom code
-   GODEC
-   RVT (with lags) regression
-   RV (with lags) regression
"""
import json
import os
import os.path as op
from glob import glob

import nibabel as nib
import numpy as np
from nilearn import image, masking


def run_rvtreg(
    dset, task, method, suffix, in_dir="/scratch/tsalo006/power-replication/",
):
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

    power_dir = op.join(dset_dir, "derivatives/power")

    for subj in subjects:
        power_subj_dir = op.join(power_dir, subj)
        preproc_dir = op.join(power_subj_dir, "preprocessed")
        physio_dir = op.join(preproc_dir, "physio")
        anat_dir = op.join(preproc_dir, "anat")

        func_file = op.join(
            power_subj_dir,
            "denoised",
            method,
            "sub-{0}_task-{1}_run-01_{2}.nii.gz".format(subj, task, suffix),
        )
        mask_file = op.join(anat_dir, "cortical_mask.nii.gz")
        func_img = nib.load(func_file)
        mask_img = nib.load(mask_file)
        if subj == subjects[0]:
            resid_sds = np.empty((len(subjects), func_img.shape[-1]))

        rvt_file = op.join(physio_dir, "sub-{0}_task-rest_run-01_rvt.txt".format(subj))
        rvt_x_rrf_file = op.join(
            physio_dir, "sub-{0}_task-rest_run-01_rvtXrrf" ".txt".format(subj)
        )
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


def run_dgsr(medn_file, confounds_file, out_dir):
    """Run dynamic global signal regression with rapidtide.

    Parameters
    ----------
    medn_file
    confounds_file
    out_dir

    Used for:
    -   Carpet plots of ME-DN after dGSR (3, S12)
    -   Carpet plots of ME-HK after dGSR (not in paper)
    -   QC:RSFC plot of ME-DN after dGSR with motion as QC (4, 5, S10, S13)
        - S10 involves censoring FD>0.2mm
    -   QC:RSFC plot of ME-HK after dGSR with motion as QC (not in paper)
    -   QC:RSFC plot of ME-DN after dGSR with RPV as QC (5)
    -   QC:RSFC plot of ME-HK after dGSR with RPV as QC (not in paper)
    -   High-low motion plot of ME-DN after dGSR (4, S10)
        - S10 involves censoring FD>0.2mm
    -   High-low motion plot of ME-HK after dGSR (not in paper)
    -   Scrubbing plot of ME-DN after dGSR (4)
    -   Scrubbing plot of ME-HK after dGSR (not in paper)
    -   Mean correlation matrix and histogram of ME-DN after dGSR (S13)
    -   Correlation scatterplot of ME-DN after dGSR against other ME-DN
        outputs (S13)
    -   Scatter plot of ME-DN-dGSR SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    -   Scatter plot of ME-HK-dGSR SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    """
    # I don't trust that tedana will retain the TR in the nifti header,
    # so will extract from json directly.
    medn_json = medn_file.replace(".nii.gz", ".json")
    with open(medn_json, "r") as fo:
        metadata = json.load(fo)
    t_r = metadata["RepetitionTime"]

    medn_name = op.basename(medn_file)
    prefix = medn_name.split("desc-")[0].rstrip("_")
    prefix = op.join(out_dir, prefix)

    dgsr_file = f"{prefix}_desc-lfofilterCleaned_bold.nii.gz"
    dgsr_noise_file = f"{prefix}_desc-noise_bold.nii.gz"

    cmd = f"rapidtide --denoising --datatstep {t_r} --motionfile {confounds_file} {medn_file} {prefix}"
    dgsr_noise_img = image.math_img("img1 - img2", img1=medn_file, img2=dgsr_file)
    dgsr_noise_img.to_filename(dgsr_noise_file)


def run_gsr(medn_file, cgm_mask, out_dir):
    """Run global signal regression.

    Parameters
    ----------
    medn_file
    cgm_mask
    out_dir

    Used for:
    -   Carpet plots of ME-DN after GSR (3, S12)
    -   Carpet plots of ME-HK after GSR (not in paper)
    -   QC:RSFC plot of ME-DN after GSR with motion as QC (4, 5, S10, S13)
        - S10 involves censoring FD>0.2mm
    -   QC:RSFC plot of ME-HK after GSR with motion as QC (not in paper)
    -   QC:RSFC plot of ME-DN after GSR with RPV as QC (5)
    -   QC:RSFC plot of ME-HK after GSR with RPV as QC (not in paper)
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
    medn_name = op.basename(medn_file)
    prefix = medn_name.split("desc-")[0]
    prefix = op.join(out_dir, prefix)
    gsr_file = prefix + "_desc-GSR_bold.nii.gz"
    gsr_noise_file = prefix + "_desc-GSRNoise_bold.nii.gz"

    # Extract global signal from cortical ribbon
    gsr_signal = masking.apply_mask(medn_file, cgm_mask)
    gsr_signal = np.mean(gsr_signal, axis=1)


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


def run_acompcor(medn_file, seg_file, out_dir):
    """Run anatomical compCor.

    Parameters
    ----------
    dset : {'ds000210', 'ds000254', 'ds000258'}
    task : {'rest', 'fingertapping'}
    method : {'meica_v2_5'}
    suffix : {'dn_ts_OC', 'hik_ts_OC', 't2s'}
    in_dir : str
        Path to analysis folder

    Used for:
    -   Carpet plots of ME-DN after aCompCor (3, S9, S12)
    -   Carpet plots of ME-HK after aCompCor (not in paper)
    -   QC:RSFC plot of ME-DN after aCompCor with motion as QC (4, 5, S10, S13)
        - S10 involves censoring FD>0.2mm
    -   QC:RSFC plot of ME-HK after aCompCor with motion as QC (not in paper)
    -   QC:RSFC plot of ME-DN after aCompCor with RPV as QC (5)
    -   QC:RSFC plot of ME-HK after aCompCor with RPV as QC (not in paper)
    -   High-low motion plot of ME-DN after aCompCor (4, S10)
        - S10 involves censoring FD>0.2mm
    -   High-low motion plot of ME-HK after aCompCor (not in paper)
    -   Scrubbing plot of ME-DN after aCompCor (4)
    -   Scrubbing plot of ME-HK after aCompCor (not in paper)
    -   Mean correlation matrix and histogram of ME-DN after aCompCor (S13)
    -   Correlation scatterplot of ME-DN after aCompCor against other ME-DN
        outputs (S13)
    -   Scatter plot of ME-DN-aCompCor SD of global signal against
        SD of ventilatory envelope (RPV) (2).
    -   Scatter plot of ME-HK-aCompCor SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    """
    pass


def main(project_dir, dset):
    dset_dir = op.join(project_dir, dset)
    deriv_dir = op.join(dset_dir, "derivatives")
    tedana_dir = op.join(deriv_dir, "tedana")
    preproc_dir = op.join(deriv_dir, "power")

    dgsr_dir = op.join(deriv_dir, "rapidtide")
    gsr_dir = op.join(deriv_dir, "gsr")

    # Get list of participants with good data
    participants_file = op.join(dset_dir, "participants.tsv")
    participants_df = pd.read_table(participants_file)
    subjects = participants_df.loc[
        participants_df["exclude"] == 0, "participant_id"
    ].tolist()

    for subject in subjects:
        print(f"\t{subject}", flush=True)
        preproc_subj_func_dir = op.join(preproc_dir, subject, "func")
        preproc_subj_anat_dir = op.join(preproc_dir, subject, "anat")
        tedana_subj_dir = op.join(tedana_dir, subject, "func")

        # Collect important files
        confounds_files = glob(op.join(preproc_subj_func_dir, "*_desc-confounds_timeseries.tsv"))
        assert len(confounds_files) == 1
        confounds_file = confounds_files[0]

        seg_files = glob(op.join(preproc_subj_anat_dir, "*_space-T1w_res-bold_desc-totalMaskWithCSF_mask.nii.gz"))
        assert len(seg_files) == 1
        seg_file = seg_files[0]

        cgm_files = glob(op.join(preproc_subj_anat_dir, "*_space-T1w_res-bold_label-CGM_mask.nii.gz"))
        assert len(cgm_files) == 1
        cgm_file = cgm_files[0]

        medn_files = glob(op.join(tedana_subj_dir, "*_desc-optcomDenoised_bold.nii.gz"))
        assert len(medn_files) == 1
        medn_file = medn_files[0]

        # dGSR
        dgsr_subj_dir = op.join(dgsr_dir, subject, "func")
        os.makedirs(dgsr_subj_dir, exist_ok=True)
        run_dgsr(medn_file, confounds_file, dgsr_subj_dir)

        # GSR
        gsr_subj_dir = op.join(gsr_dir, subject, "func")
        os.makedirs(gsr_subj_dir, exist_ok=True)
        run_gsr(medn_file, cgm_file, gsr_subj_dir)



if __name__ == "__main__":
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    dsets = [
        "dset-cambridge",
        "dset-camcan",
        "dset-cohen",
        "dset-dalenberg",
        "dset-dupre",
    ]
    for dset in dsets:
        print(dset, flush=True)
        main(project_dir, dset)