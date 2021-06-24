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
import pandas as pd
import sklearn
from nilearn import image, input_data, masking, signal


def run_rvtreg(
    dset,
    task,
    method,
    suffix,
    in_dir="/scratch/tsalo006/power-replication/",
):
    """Clean MEDN data with regression model including RVT and RVT*RRF (plus lags).

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

        func_data = masking.apply_mask(func_img, mask_img)
        lstsq_res = np.linalg.lstsq(rvt, func_data, rcond=None)
        pred = np.dot(rvt, lstsq_res[0])
        residuals = func_data - pred

        # Get volume-wise standard deviation
        resid_sd = np.std(residuals, axis=0)
        resid_sds[subj, :] = resid_sd
        return resid_sds


def run_rvreg():
    """Clean MEDN data with regression model including RV and RV*RRF (plus lags).

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


def run_nuisance(medn_file, mask_file, seg_file, confounds_file, out_dir):
    """Clean MEDN data with nuisance model.

    Regressors include mean deepest white matter, mean deepest CSF,
    6 motion parameters, and first temporal derivatives of motion parameters.

    Parameters
    ----------
    medn_file
    mask_file
    seg_file
    confounds_file
    out_dir

    Notes
    -----
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
    medn_name = op.basename(medn_file)
    prefix = medn_name.split("desc-")[0].rstrip("_")
    prefix = op.join(out_dir, prefix)
    medn_json_file = medn_file.replace(".nii.gz", ".json")
    denoised_file = prefix + "_desc-NuisReg_bold.nii.gz"
    denoised_json_file = denoised_file.replace(".nii.gz", ".json")
    noise_file = prefix + "_desc-NuisRegNoise_bold.nii.gz"
    noise_json_file = noise_file.replace(".nii.gz", ".json")

    # Calculate mean-centered version of MEDN data
    mean_img = image.mean_img(medn_file)
    medn_mean_centered_img = image.math_img(
        "img - avg_img",
        img=medn_file,
        avg_img=mean_img,
    )

    # Load metadata for writing out later and TR now
    with open(medn_json_file, "r") as fo:
        json_info = json.load(fo)

    wm_img = image.math_img("img == 6", img=seg_file)
    wm_img = image.math_img(
        "wm_mask * brain_mask", wm_mask=wm_img, brain_mask=mask_file
    )
    wm_data = masking.apply_mask(medn_file, wm_img)

    csf_img = image.math_img("img == 8", img=seg_file)
    csf_img = image.math_img(
        "csf_mask * brain_mask", csf_mask=csf_img, brain_mask=mask_file
    )
    csf_data = masking.apply_mask(medn_file, csf_img)

    confounds_df = pd.read_table(confounds_file)
    confounds_df["wm_data"] = wm_data
    confounds_df["csf_data"] = csf_data
    nuisance_regressors = confounds_df[
        [
            "wm_data",
            "csf_data",
            "trans_x",
            "trans_y",
            "trans_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "trans_x_derivative1",
            "trans_y_derivative1",
            "trans_z_derivative1",
            "rot_x_derivative1",
            "rot_y_derivative1",
            "rot_z_derivative1",
        ]
    ].values
    # Mean-center and detrend regressors
    # See "fMRI data: nuisance regressions" section of Power appendix (page 3).
    nuisance_regressors = nuisance_regressors - np.mean(axis=0)
    nuisance_regressors = signal._detrend(nuisance_regressors, type="linear")

    # Regress confounds out of MEDN data
    regression_masker = input_data.NiftiMasker(
        mask_file,
        smoothing_fwhm=None,
        standardize=False,
        standardize_confounds=True,
        high_variance_confounds=False,
        low_pass=None,
        high_pass=None,
        detrend=True,  # linearly detrends both confounds and data
        t_r=json_info["RepetitionTime"],
        reports=False,
    )
    regression_masker.fit(medn_mean_centered_img)
    # Mask + remove confounds
    denoised_data = regression_masker.transform(
        medn_mean_centered_img,
        confounds=nuisance_regressors,
    )
    # Mask without removing confounds
    raw_data = regression_masker.transform(
        medn_mean_centered_img,
        confounds=None,
    )
    noise_data = raw_data - denoised_data
    denoised_img = regression_masker.inverse_transform(denoised_data)
    noise_img = regression_masker.inverse_transform(noise_data)

    # Save output files
    denoised_img.to_filename(denoised_file)
    noise_img.to_filename(noise_file)

    # Create json files with Sources and Description fields.
    json_info["Sources"] = [medn_file, mask_file, seg_file, confounds_file]
    json_info["Description"] = (
        "Multi-echo denoised data further denoised with a nuisance regression model including "
        "signal from mean deepest white matter, mean deepest CSF, 6 motion parameters, and "
        "first temporal derivatives of motion parameters."
    )
    with open(denoised_json_file, "w") as fo:
        json.dump(json_info, fo, sort_keys=True, indent=4)

    json_info["Description"] = (
        "Residuals from nuisance regression model applied to multi-echo denoised data. "
        "The nuisance regression model includes signal from mean deepest white matter, "
        "mean deepest CSF, 6 motion parameters, and first temporal derivatives of motion "
        "parameters."
    )
    with open(noise_json_file, "w") as fo:
        json.dump(json_info, fo, sort_keys=True, indent=4)


def run_dgsr(medn_file, mask_file, confounds_file, out_dir):
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

    cmd = (
        f"rapidtide --denoising --datatstep {t_r} "
        f"--motionfile {confounds_file} {medn_file} {prefix}"
    )
    run_command(cmd)
    assert op.isfile(dgsr_file)
    assert op.isfile(dgsr_noise_file)
    # Will the scale of the denoised data be correct? Or is it mean-centered or something?
    dgsr_noise_img = image.math_img("img1 - img2", img1=medn_file, img2=dgsr_file)
    dgsr_noise_img.to_filename(dgsr_noise_file)

    # TODO: Create json files with Sources field.


def run_gsr(medn_file, mask_file, cgm_mask, out_dir):
    """Run global signal regression.

    Parameters
    ----------
    medn_file
    mask_file
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
    cgm_mask = image.math_img(
        "cgm_mask * brain_mask", cgm_mask=cgm_mask, brain_mask=mask_file
    )
    gsr_signal = masking.apply_mask(medn_file, cgm_mask)
    gsr_signal = np.mean(gsr_signal, axis=1)

    # Mean-center and linearly detrend global signal

    # Mean-center and linearly detrend MEDN data

    # Regress global signal out of MEDN data

    # TODO: Create json files with Sources field.


def run_godec():
    """Not to be run.

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


def run_acompcor(medn_file, mask_file, seg_file, out_dir):
    """Run anatomical compCor.

    Parameters
    ----------
    medn_file
    mask_file
    seg_file
    out_dir

    Notes
    -----
    From the original paper's appendix (page 4):
        > For the purposes of Figures S12 and S13, only white matter signals
        > in the deepest mask were used (for these signals are the most isolated
        > from and distinct from those of the gray matter).

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
    medn_name = op.basename(medn_file)
    prefix = medn_name.split("desc-")[0]
    prefix = op.join(out_dir, prefix)
    acompcor_file = prefix + "_desc-aCompCor_bold.nii.gz"
    acompcor_noise_file = prefix + "_desc-aCompCorNoise_bold.nii.gz"

    # Derive aCompCor components
    wm_img = image.math_img("img == 6", img=seg_file)
    wm_img = image.math_img(
        "wm_mask * brain_mask", wm_mask=wm_img, brain_mask=mask_file
    )
    wm_data = masking.apply_mask(medn_file, wm_img)
    pca = sklearn.decomposition.PCA(n_components=5)
    acompcor_components = pca.fit_transform(wm_data)

    # Regress components out of MEDN data

    # TODO: Create json files with Sources field.


def main(project_dir, dset):
    """TODO: Create dataset_description.json files."""
    dset_dir = op.join(project_dir, dset)
    deriv_dir = op.join(dset_dir, "derivatives")
    tedana_dir = op.join(deriv_dir, "tedana")
    preproc_dir = op.join(deriv_dir, "power")

    acompcor_dir = op.join(deriv_dir, "acompcor")
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
        confounds_files = glob(
            op.join(preproc_subj_func_dir, "*_desc-confounds_timeseries.tsv")
        )
        assert len(confounds_files) == 1
        confounds_file = confounds_files[0]

        seg_files = glob(
            op.join(
                preproc_subj_anat_dir,
                "*_space-T1w_res-bold_desc-totalMaskWithCSF_mask.nii.gz",
            )
        )
        assert len(seg_files) == 1
        seg_file = seg_files[0]

        cgm_files = glob(
            op.join(preproc_subj_anat_dir, "*_space-T1w_res-bold_label-CGM_mask.nii.gz")
        )
        assert len(cgm_files) == 1
        cgm_file = cgm_files[0]

        medn_files = glob(op.join(tedana_subj_dir, "*_desc-optcomDenoised_bold.nii.gz"))
        assert len(medn_files) == 1
        medn_file = medn_files[0]

        mask_files = glob(op.join(tedana_subj_dir, "*_desc-goodSignal_mask.nii.gz"))
        assert len(mask_files) == 1
        mask_file = mask_files[0]

        # aCompCor
        acompcor_subj_dir = op.join(acompcor_dir, subject, "func")
        os.makedirs(acompcor_subj_dir, exist_ok=True)
        run_acompcor(medn_file, mask_file, seg_file, acompcor_subj_dir)

        # dGSR
        # TODO: Check settings with Blaise Frederick
        dgsr_subj_dir = op.join(dgsr_dir, subject, "func")
        os.makedirs(dgsr_subj_dir, exist_ok=True)
        run_dgsr(medn_file, mask_file, confounds_file, dgsr_subj_dir)

        # GSR
        gsr_subj_dir = op.join(gsr_dir, subject, "func")
        os.makedirs(gsr_subj_dir, exist_ok=True)
        run_gsr(medn_file, mask_file, cgm_file, gsr_subj_dir)


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
