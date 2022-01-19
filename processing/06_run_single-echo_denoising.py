"""
Perform standard denoising (not TE-dependent denoising).

Methods:
-   Global signal regression with custom code
    (integrated in tedana, but we do it separately here because the approach is very different)
-   Dynamic global signal regression with rapidtide
-   aCompCor with custom code
-   GODEC with the ME-ICA/godec package
-   RVT (with lags) regression
-   RV (with lags) regression
"""
import argparse
import json
import os
import os.path as op
import sys
from shutil import rmtree

import numpy as np
import pandas as pd
import rapidtide
from nilearn import image

from processing_utils import _generic_regression, run_command

sys.path.append("..")

from utils import get_prefixes  # noqa: E402


def run_rvtreg(medn_file, mask_file, confounds_file, out_dir):
    """Clean MEDN data with regression model including RVT and RVT*RRF (plus lags).

    Parameters
    ----------
    medn_file
    mask_file
    confounds_file
    out_dir

    Notes
    -----
    Used for:
    -   Carpet plots of MEDN after regression of RVT + RVT*RRF (S5)
    -   Scatter plot of MEDN-RVT+RVT*RRF SD of global signal against
        SD of ventilatory envelope (RPV) (S8).
    """
    print("\tRVT", flush=True)
    # Parse input files
    medn_name = op.basename(medn_file)
    prefix = medn_name.split("desc-")[0].rstrip("_")
    medn_json_file = medn_file.replace(".nii.gz", ".json")

    # Determine output files
    denoised_file = op.join(out_dir, f"{prefix}_desc-RVTReg_bold.nii.gz")
    noise_file = op.join(out_dir, f"{prefix}_desc-RVTReg_errorts.nii.gz")

    confounds_df = pd.read_table(confounds_file)

    # Load metadata for writing out later and TR now
    with open(medn_json_file, "r") as fo:
        json_info = json.load(fo)

    nuisance_regressors = confounds_df[
        [
            "RVTRegression_RVT",
            "RVTRegression_RVT+5s",
            "RVTRegression_RVT+10s",
            "RVTRegression_RVT+15s",
            "RVTRegression_RVT+20s",
            "RVTRegression_RVT*RRF",
            "RVTRegression_RVT*RRF+5s",
            "RVTRegression_RVT*RRF+10s",
            "RVTRegression_RVT*RRF+15s",
            "RVTRegression_RVT*RRF+20s",
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

    # Some fMRIPrep nuisance regressors have NaN in the first row (e.g., derivatives)
    nuisance_regressors = np.nan_to_num(nuisance_regressors, 0)

    denoised_img, noise_img = _generic_regression(
        medn_file,
        mask_file,
        nuisance_regressors,
        t_r=json_info["RepetitionTime"],
    )

    # Save output files
    denoised_img.to_filename(denoised_file)
    noise_img.to_filename(noise_file)

    # Create json files with Sources and Description fields
    json_info["Sources"] = [medn_file, mask_file, confounds_file]

    SUFFIXES = {
        "desc-RVTReg_bold": (
            "Multi-echo denoised data further denoised with a respiratory-volume-per-time-based "
            "regression model. This model includes RVT lagged 0 seconds, 5 seconds forward, "
            "10 seconds forward, 15 seconds forward, and 20 seconds forward, "
            "along with those five RVT-based regressors convolved with "
            "the respiratory response function, six realigment parameters, "
            "and the realignment parameters' first derivatives."
        ),
        "desc-RVTReg_errorts": (
            "Residuals from respiratory-volume-per-time-based regression model applied to "
            "multi-echo denoised data. This model includes RVT lagged 0 seconds, "
            "5 seconds forward, 10 seconds forward, 15 seconds forward, and 20 seconds forward, "
            "along with those three RV-based regressors convolved with "
            "the respiratory response function, six realigment parameters, "
            "and the realignment parameters' first derivatives."
        ),
    }
    for suffix, description in SUFFIXES.items():
        nii_file = op.join(out_dir, f"{prefix}_{suffix}.nii.gz")
        assert op.isfile(nii_file)

        suff_json_file = op.join(out_dir, f"{prefix}_{suffix}.json")
        json_info["Description"] = description
        with open(suff_json_file, "w") as fo:
            json.dump(json_info, fo, sort_keys=True, indent=4)


def run_rvreg(medn_file, mask_file, confounds_file, out_dir):
    """Clean MEDN data with regression model including RV and RV*RRF (plus lags).

    Parameters
    ----------
    medn_file
    mask_file
    confounds_file
    out_dir

    Notes
    -----
    Used for:
    -   Carpet plots of MEDN after regression of RV + RV*RRF (S5)
    -   Scatter plot of MEDN-RV+RV*RRF SD of global signal against
        SD of ventilatory envelope (RPV) (S8).
    """
    print("\tRV", flush=True)
    # Parse input files
    medn_name = op.basename(medn_file)
    prefix = medn_name.split("desc-")[0].rstrip("_")
    medn_json_file = medn_file.replace(".nii.gz", ".json")

    # Determine output files
    denoised_file = op.join(out_dir, f"{prefix}_desc-RVReg_bold.nii.gz")
    noise_file = op.join(out_dir, f"{prefix}_desc-RVReg_errorts.nii.gz")

    confounds_df = pd.read_table(confounds_file)

    # Load metadata for writing out later and TR now
    with open(medn_json_file, "r") as fo:
        json_info = json.load(fo)

    nuisance_regressors = confounds_df[
        [
            "RVRegression_RV-3s",
            "RVRegression_RV",
            "RVRegression_RV+3s",
            "RVRegression_RV*RRF-3s",
            "RVRegression_RV*RRF",
            "RVRegression_RV*RRF+3s",
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

    # Some fMRIPrep nuisance regressors have NaN in the first row (e.g., derivatives)
    nuisance_regressors = np.nan_to_num(nuisance_regressors, 0)

    denoised_img, noise_img = _generic_regression(
        medn_file,
        mask_file,
        nuisance_regressors,
        t_r=json_info["RepetitionTime"],
    )

    # Save output files
    denoised_img.to_filename(denoised_file)
    noise_img.to_filename(noise_file)

    # Create json files with Sources and Description fields
    json_info["Sources"] = [medn_file, mask_file, confounds_file]

    SUFFIXES = {
        "desc-RVReg_bold": (
            "Multi-echo denoised data further denoised with a respiratory variance-based "
            "regression model. This model includes RV lagged 3 seconds back, 0 seconds, "
            "3 seconds forward, along with those three RV-based regressors convolved with "
            "the respiratory response function, six realigment parameters, "
            "and the realignment parameters' first derivatives."
        ),
        "desc-RVReg_errorts": (
            "Residuals from respiratory variance-based regression model applied to multi-echo "
            "denoised data. This model includes RV lagged 3 seconds back, 0 seconds, "
            "3 seconds forward, along with those three RV-based regressors convolved with "
            "the respiratory response function, six realigment parameters, "
            "and the realignment parameters' first derivatives."
        ),
    }
    for suffix, description in SUFFIXES.items():
        nii_file = op.join(out_dir, f"{prefix}_{suffix}.nii.gz")
        assert op.isfile(nii_file)

        suff_json_file = op.join(out_dir, f"{prefix}_{suffix}.json")
        json_info["Description"] = description
        with open(suff_json_file, "w") as fo:
            json.dump(json_info, fo, sort_keys=True, indent=4)


def run_dgsr(medn_file, mask_file, confounds_file, out_dir):
    """Run dynamic global signal regression with rapidtide.

    Parameters
    ----------
    medn_file
    mask_file
    confounds_file
    out_dir

    Notes
    -----
    Used for:
    -   Carpet plots of MEDN after dGSR (3, S12)
    -   QC:RSFC plot of MEDN after dGSR with motion as QC (4, 5, S10, S13)
        - S10 involves censoring FD>0.2mm
    -   QC:RSFC plot of MEDN after dGSR with RPV as QC (5)
    -   High-low motion plot of MEDN after dGSR (4, S10)
        - S10 involves censoring FD>0.2mm
    -   Scrubbing plot of MEDN after dGSR (4)
    -   Mean correlation matrix and histogram of MEDN after dGSR (S13)
    -   Correlation scatterplot of MEDN after dGSR against other MEDN
        outputs (S13)
    -   Scatter plot of MEDN-dGSR SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    """
    print("\trapidtide", flush=True)
    # I don't trust that tedana will retain the TR in the nifti header,
    # so will extract from json directly.
    medn_json_file = medn_file.replace(".nii.gz", ".json")
    with open(medn_json_file, "r") as fo:
        json_info = json.load(fo)
    t_r = json_info["RepetitionTime"]

    medn_name = op.basename(medn_file)
    prefix = medn_name.split("desc-")[0].rstrip("_")

    dgsr_file = op.join(out_dir, f"{prefix}_desc-lfofilterCleaned_bold.nii.gz")
    dgsr_noise_file = op.join(out_dir, f"{prefix}_desc-lfofilterCleaned_errorts.nii.gz")

    # Use the standard denoising settings, with a smoothing kernel equal to 1/2 voxel size,
    # per rapidtide's recommendation.
    cmd = (
        f"rapidtide --denoising --datatstep {t_r} "
        f"--motionfile {confounds_file} --denoising --spatialfilt -1 "
        f"{medn_file} {op.join(out_dir, prefix)}"
    )
    run_command(cmd)

    # Per the rapidtide documentation, the lfofilterCleaned data have mean included.
    dgsr_noise_img = image.math_img("img1 - img2", img1=medn_file, img2=dgsr_file)
    dgsr_noise_img.to_filename(dgsr_noise_file)

    # Create json files with Sources and Description fields
    json_info["Sources"] = [medn_file, confounds_file]

    SUFFIXES = {
        "desc-lfofilterCleaned_bold": "Multi-echo denoised data further denoised with rapidtide.",
        "desc-lfofilterCleaned_errorts": (
            "Noise time series retained from further denoising multi-echo denoised data with "
            "rapidtide."
        ),
    }
    for suffix, description in SUFFIXES.items():
        nii_file = op.join(out_dir, f"{prefix}_{suffix}.nii.gz")
        assert op.isfile(nii_file)

        suff_json_file = op.join(out_dir, f"{prefix}_{suffix}.json")
        json_info["Description"] = description
        with open(suff_json_file, "w") as fo:
            json.dump(json_info, fo, sort_keys=True, indent=4)


def run_godec(medn_file, mask_file, out_dir):
    """Still need to test a bit.

    Parameters
    ----------
    medn_file
    mask_file
        Brain mask.
    out_dir

    Notes
    -----
    From the original paper's appendix (page 4):
        > Our implementation of GODEC is Python-based (godec.py), and included a random sampling
        > method to estimate the covariance matrix iteratively with a power method as described
        > below. We also included steps of discrete wavelet transform before and after GODEC to
        > conserve autocorrelation in the final solution, using the Daubechies wavelet.
        > A rank-1 approximation was used, with 100 iterations.
        > ...
        > We used parameters that returned low-rank spaces with rank approximately of 1-4 to
        > minimize removal of signals associated with resting state networks.

    Used for:
    -   Carpet plots of MEDN after GODEC (3, S9, S12)
    -   QC:RSFC plot of MEDN after GODEC with motion as QC (4, 5, S10, S13)
        - S10 involves censoring FD>0.2mm
    -   QC:RSFC plot of MEDN after GODEC with RPV as QC (5)
    -   High-low motion plot of MEDN after GODEC (4, S10)
        - S10 involves censoring FD>0.2mm
    -   Scrubbing plot of MEDN after GODEC (4)
    -   Mean correlation matrix and histogram of MEDN after GODEC (S13)
    -   Correlation scatterplot of MEDN after GODEC against other MEDN
        outputs (S13)
    -   Scatter plot of MEDN-GODEC SD of global signal against
        SD of ventilatory envelope (RPV) (2).
    """
    print("\tgodec", flush=True)
    from godec import godec_fmri

    # Parse input files
    medn_name = op.basename(medn_file)
    prefix = medn_name.split("desc-")[0].rstrip("_")
    medn_json_file = medn_file.replace(".nii.gz", ".json")

    godec_fmri(
        medn_file,
        mask_file,
        out_dir,
        prefix=prefix,
        method="greedy",
        ranks=[4],
        norm_mode="vn",
        rank_step_size=1,
        iterated_power=100,
        wavelet=True,
    )

    # Load metadata
    with open(medn_json_file, "r") as fo:
        json_info = json.load(fo)

    # Create json files with Sources and Description fields
    json_info["Sources"] = [medn_file, mask_file]

    SUFFIXES = {
        "desc-GODEC_rank-4_bold": "Multi-echo denoised data further denoised with GODEC.",
        "desc-GODEC_rank-4_lowrankts": (
            "Low-rank time series retained from further denoising multi-echo denoised data with "
            "GODEC."
        ),
    }
    for suffix, description in SUFFIXES.items():
        nii_file = op.join(out_dir, f"{prefix}_{suffix}.nii.gz")
        assert op.isfile(nii_file)

        suff_json_file = op.join(out_dir, f"{prefix}_{suffix}.json")
        json_info["Description"] = description
        with open(suff_json_file, "w") as fo:
            json.dump(json_info, fo, sort_keys=True, indent=4)


def run_gsr(medn_file, mask_file, confounds_file, out_dir):
    """Run global signal regression.

    Parameters
    ----------
    medn_file
    mask_file
    confounds_file
    out_dir

    Notes
    -----
    Used for:
    -   Carpet plots of MEDN after GSR (3, S12)
    -   QC:RSFC plot of MEDN after GSR with motion as QC (4, 5, S10, S13)
        - S10 involves censoring FD>0.2mm
    -   QC:RSFC plot of MEDN after GSR with RPV as QC (5)
    -   High-low motion plot of MEDN after GSR (4, S10)
        - S10 involves censoring FD>0.2mm
    -   Scrubbing plot of MEDN after GSR (4)
    -   Mean correlation matrix and histogram of MEDN after GSR (S13)
    -   Correlation scatterplot of MEDN after GSR against other MEDN
        outputs (S13)
    -   Scatter plot of MEDN-GSR SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    """
    print("\tGSR", flush=True)
    # Parse input files
    medn_name = op.basename(medn_file)
    prefix = medn_name.split("desc-")[0].rstrip("_")
    medn_json_file = medn_file.replace(".nii.gz", ".json")

    # Determine output files
    denoised_file = op.join(out_dir, f"{prefix}_desc-GSR_bold.nii.gz")
    noise_file = op.join(out_dir, f"{prefix}_desc-GSR_errorts.nii.gz")

    confounds_df = pd.read_table(confounds_file)

    # Load metadata for writing out later and TR now
    with open(medn_json_file, "r") as fo:
        json_info = json.load(fo)

    nuisance_regressors = confounds_df[
        [
            "GSRRegression_CorticalRibbon",
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

    # Some fMRIPrep nuisance regressors have NaN in the first row (e.g., derivatives)
    nuisance_regressors = np.nan_to_num(nuisance_regressors, 0)

    denoised_img, noise_img = _generic_regression(
        medn_file,
        mask_file,
        nuisance_regressors,
        t_r=json_info["RepetitionTime"],
    )

    # Save output files
    denoised_img.to_filename(denoised_file)
    noise_img.to_filename(noise_file)

    # Create json files with Sources and Description fields
    json_info["Sources"] = [medn_file, mask_file, confounds_file]

    SUFFIXES = {
        "desc-GSR_bold": (
            "Multi-echo denoised data further denoised with a GSR regression model including "
            "mean signal from the cortical gray matter ribbon."
        ),
        "desc-GSR_errorts": (
            "Residuals from GSR regression model applied to multi-echo denoised data. "
            "The GSR regression model includes mean signal from the cortical gray matter ribbon."
        ),
    }
    for suffix, description in SUFFIXES.items():
        nii_file = op.join(out_dir, f"{prefix}_{suffix}.nii.gz")
        assert op.isfile(nii_file)

        suff_json_file = op.join(out_dir, f"{prefix}_{suffix}.json")
        json_info["Description"] = description
        with open(suff_json_file, "w") as fo:
            json.dump(json_info, fo, sort_keys=True, indent=4)


def run_acompcor(medn_file, mask_file, confounds_file, out_dir):
    """Run anatomical compCor.

    Parameters
    ----------
    medn_file
    mask_file
    confounds_file
    out_dir

    Notes
    -----
    From the original paper's appendix (page 4):
        > For the purposes of Figures S12 and S13, only white matter signals
        > in the deepest mask were used (for these signals are the most isolated
        > from and distinct from those of the gray matter).

    Used for:
    -   Carpet plots of MEDN after aCompCor (3, S9, S12)
    -   QC:RSFC plot of MEDN after aCompCor with motion as QC (4, 5, S10, S13)
        - S10 involves censoring FD>0.2mm
    -   QC:RSFC plot of MEDN after aCompCor with RPV as QC (5)
    -   High-low motion plot of MEDN after aCompCor (4, S10)
        - S10 involves censoring FD>0.2mm
    -   Scrubbing plot of MEDN after aCompCor (4)
    -   Mean correlation matrix and histogram of MEDN after aCompCor (S13)
    -   Correlation scatterplot of MEDN after aCompCor against other MEDN
        outputs (S13)
    -   Scatter plot of MEDN-aCompCor SD of global signal against
        SD of ventilatory envelope (RPV) (2).
    """
    print("\taCompCor", flush=True)
    # Parse input files
    medn_name = op.basename(medn_file)
    prefix = medn_name.split("desc-")[0].rstrip("_")
    medn_json_file = medn_file.replace(".nii.gz", ".json")

    # Determine output files
    denoised_file = op.join(out_dir, f"{prefix}_desc-aCompCor_bold.nii.gz")
    noise_file = op.join(out_dir, f"{prefix}_desc-aCompCor_errorts.nii.gz")

    confounds_df = pd.read_table(confounds_file)

    # Load metadata for writing out later and TR now
    with open(medn_json_file, "r") as fo:
        json_info = json.load(fo)

    nuisance_regressors = confounds_df[
        [
            "aCompCorRegression_Component00",
            "aCompCorRegression_Component01",
            "aCompCorRegression_Component02",
            "aCompCorRegression_Component03",
            "aCompCorRegression_Component04",
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

    # Some fMRIPrep nuisance regressors have NaN in the first row (e.g., derivatives)
    nuisance_regressors = np.nan_to_num(nuisance_regressors, 0)

    denoised_img, noise_img = _generic_regression(
        medn_file,
        mask_file,
        nuisance_regressors,
        t_r=json_info["RepetitionTime"],
    )

    # Save output files
    denoised_img.to_filename(denoised_file)
    noise_img.to_filename(noise_file)

    # Create json files with Sources and Description fields
    json_info["Sources"] = [medn_file, mask_file, confounds_file]

    SUFFIXES = {
        "desc-aCompCor_bold": (
            "Multi-echo denoised data further denoised with an aCompCor regression model "
            "including 5 PCA components from deepest white matter, 6 motion parameters, and "
            "first temporal derivatives of motion parameters."
        ),
        "desc-aCompCor_errorts": (
            "Residuals from aCompCor regression model applied to multi-echo denoised data. "
            "The aCompCor regression model includes 5 PCA components from deepest white matter, "
            "6 motion parameters, and first temporal derivatives of motion parameters."
        ),
    }
    for suffix, description in SUFFIXES.items():
        nii_file = op.join(out_dir, f"{prefix}_{suffix}.nii.gz")
        assert op.isfile(nii_file)

        suff_json_file = op.join(out_dir, f"{prefix}_{suffix}.json")
        json_info["Description"] = description
        with open(suff_json_file, "w") as fo:
            json.dump(json_info, fo, sort_keys=True, indent=4)


def run_nuisance(medn_file, mask_file, confounds_file, out_dir):
    """Clean MEDN data with nuisance model.

    Regressors include mean deepest white matter, mean deepest CSF,
    6 motion parameters, and first temporal derivatives of motion parameters.

    Parameters
    ----------
    medn_file
    mask_file
    confounds_file
    out_dir

    Notes
    -----
    Used for:
    -   Carpet plots of MEDN after regression of nuisance (S7)
    -   Carpet plots of FIT-R2 after regression of nuisance (S6)
    -   Scatter plot of MEDN-Nuis SD of global signal against
        SD of ventilatory envelope (RPV) (S8).
    -   Scatter plot of FIT-R2-Nuis SD of global signal against
        SD of ventilatory envelope (RPV) (not in paper).
    """
    print("\tnuisance", flush=True)
    # Parse input files
    medn_name = op.basename(medn_file)
    prefix = medn_name.split("desc-")[0].rstrip("_")
    medn_json_file = medn_file.replace(".nii.gz", ".json")

    # Determine output files
    denoised_file = op.join(out_dir, f"{prefix}_desc-NuisReg_bold.nii.gz")
    noise_file = op.join(out_dir, f"{prefix}_desc-NuisReg_errorts.nii.gz")

    confounds_df = pd.read_table(confounds_file)

    # Load metadata for writing out later and TR now
    with open(medn_json_file, "r") as fo:
        json_info = json.load(fo)

    nuisance_regressors = confounds_df[
        [
            "NuisanceRegression_WhiteMatter",
            "NuisanceRegression_CerebrospinalFluid",
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

    # Some fMRIPrep nuisance regressors have NaN in the first row (e.g., derivatives)
    nuisance_regressors = np.nan_to_num(nuisance_regressors, 0)

    denoised_img, noise_img = _generic_regression(
        medn_file,
        mask_file,
        nuisance_regressors,
        t_r=json_info["RepetitionTime"],
    )

    # Save output files
    denoised_img.to_filename(denoised_file)
    noise_img.to_filename(noise_file)

    # Create json files with Sources and Description fields
    json_info["Sources"] = [medn_file, mask_file, confounds_file]

    SUFFIXES = {
        "desc-NuisReg_bold": (
            "Multi-echo denoised data further denoised with a nuisance regression model including "
            "signal from mean deepest white matter, mean deepest CSF, 6 motion parameters, and "
            "first temporal derivatives of motion parameters."
        ),
        "desc-NuisReg_errorts": (
            "Residuals from nuisance regression model applied to multi-echo denoised data. "
            "The nuisance regression model includes signal from mean deepest white matter, "
            "mean deepest CSF, 6 motion parameters, and first temporal derivatives of motion "
            "parameters."
        ),
    }
    for suffix, description in SUFFIXES.items():
        nii_file = op.join(out_dir, f"{prefix}_{suffix}.nii.gz")
        assert op.isfile(nii_file)

        suff_json_file = op.join(out_dir, f"{prefix}_{suffix}.json")
        json_info["Description"] = description
        with open(suff_json_file, "w") as fo:
            json.dump(json_info, fo, sort_keys=True, indent=4)


def main(project_dir, dset, subject):
    """Run single-echo denoising workflows on a given dataset."""
    prefixes = get_prefixes()
    dset_prefix = prefixes[dset]
    dset_prefix = dset_prefix.format(participant_id=subject)

    dset_dir = op.join(project_dir, dset)
    deriv_dir = op.join(dset_dir, "derivatives")
    tedana_dir = op.join(deriv_dir, "tedana")
    preproc_dir = op.join(deriv_dir, "power")

    nuis_dir = op.join(deriv_dir, "nuisance-regressions")
    dgsr_dir = op.join(deriv_dir, "rapidtide")
    godec_dir = op.join(deriv_dir, "godec")

    # Get list of participants with good data
    participants_file = op.join(dset_dir, "participants.tsv")
    participants_df = pd.read_table(participants_file)
    subjects = participants_df.loc[
        participants_df["exclude"] == 0, "participant_id"
    ].tolist()
    first_subject = subjects[0]

    with open(op.join(preproc_dir, "dataset_description.json"), "r") as fo:
        preproc_dset_desc = json.load(fo)

    nuis_dset_desc = preproc_dset_desc.copy()
    dgsr_dset_desc = preproc_dset_desc.copy()
    nuis_dset_desc["Name"] = "Nuisance Regressions"
    dgsr_dset_desc["Name"] = "Dynamic Global Signal Regression"
    dgsr_dset_desc["GeneratedBy"] = [
        {
            "Name": "rapidtide",
            "Description": (
                "Dynamic global signal regression for the removal of systemic low-frequency "
                "oscillations from fMRI data with rapidtide."
            ),
            "Version": rapidtide.__version__,
            "CodeURL": "https://github.com/bbfrederick/rapidtide",
        }
    ] + dgsr_dset_desc["GeneratedBy"]

    os.makedirs(nuis_dir, exist_ok=True)
    with open(op.join(nuis_dir, "dataset_description.json"), "w") as fo:
        json.dump(nuis_dset_desc, fo, sort_keys=True, indent=4)

    os.makedirs(dgsr_dir, exist_ok=True)
    with open(op.join(dgsr_dir, "dataset_description.json"), "w") as fo:
        json.dump(dgsr_dset_desc, fo, sort_keys=True, indent=4)

    preproc_subj_func_dir = op.join(preproc_dir, subject, "func")
    tedana_subj_dir = op.join(tedana_dir, subject, "func")

    # Collect important files
    confounds_file = op.join(
        preproc_subj_func_dir, f"{dset_prefix}_desc-confounds_timeseries.tsv"
    )
    assert op.isfile(confounds_file), confounds_file

    medn_file = op.join(
        tedana_subj_dir, f"{dset_prefix}_desc-optcomDenoised_bold.nii.gz"
    )
    assert op.isfile(medn_file), medn_file

    mask_file = op.join(tedana_subj_dir, f"{dset_prefix}_desc-goodSignal_mask.nii.gz")
    assert op.isfile(mask_file), mask_file

    nuis_subj_dir = op.join(nuis_dir, subject, "func")
    os.makedirs(nuis_subj_dir, exist_ok=True)

    # ###################
    # Nuisance Regression
    # ###################
    run_nuisance(medn_file, mask_file, confounds_file, nuis_subj_dir)

    # ########
    # aCompCor
    # ########
    run_acompcor(medn_file, mask_file, confounds_file, nuis_subj_dir)

    # ###
    # GSR
    # ###
    run_gsr(medn_file, mask_file, confounds_file, nuis_subj_dir)

    # ####
    # dGSR
    # ####
    dgsr_subj_dir = op.join(dgsr_dir, subject, "func")
    # rapidtide will break if the files already exist
    if os.path.isdir(dgsr_subj_dir):
        rmtree(dgsr_subj_dir)

    os.makedirs(dgsr_subj_dir, exist_ok=True)
    run_dgsr(medn_file, mask_file, confounds_file, dgsr_subj_dir)

    # #####
    # GODEC
    # #####
    godec_subj_dir = op.join(godec_dir, subject, "func")
    os.makedirs(godec_subj_dir, exist_ok=True)
    run_godec(medn_file, mask_file, godec_subj_dir)

    # Clean up dataset description files
    if subject == first_subject:
        with open(op.join(godec_subj_dir, "dataset_description.json"), "r") as fo:
            godec_dset_desc = json.load(fo)

        godec_dset_desc["GeneratedBy"] += preproc_dset_desc["GeneratedBy"]

        with open(op.join(godec_dir, "dataset_description.json"), "w") as fo:
            json.dump(godec_dset_desc, fo, sort_keys=True, indent=4)

    os.remove(op.join(godec_subj_dir, "dataset_description.json"))

    # ################
    # Physio Denoising
    # ################
    if dset == "dset-dupre":
        run_rvtreg(medn_file, mask_file, confounds_file, nuis_subj_dir)
        run_rvreg(medn_file, mask_file, confounds_file, nuis_subj_dir)


def _get_parser():
    parser = argparse.ArgumentParser(description="Grab cell from TSV file.")
    parser.add_argument(
        "--dset",
        dest="dset",
        required=True,
        help="Dataset name.",
    )
    parser.add_argument(
        "--subject",
        dest="subject",
        required=True,
        help="Subject identifier, with the sub- prefix.",
    )
    return parser


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    main(project_dir=project_dir, **kwargs)


if __name__ == "__main__":
    print(__file__, flush=True)
    _main()
