"""
Additional, post-fMRIPrep preprocessing.
"""
import os
import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd
from bids.layout import BIDSLayout
from nilearn import image
from scipy.ndimage.morphology import binary_erosion


def preprocess(dset, in_dir="/scratch/tsalo006/power-replication/"):
    """
    Perform additional, post-fMRIPrep preprocessing of structural and
    functional MRI data.
    1) Create GM, WM, and CSF masks and resample to 3mm (functional) resolution
    2) Remove non-steady state volumes from each fMRI image
    """
    # Constants
    PROJECT_DIR = "/home/data/nbc/misc-projects/Salo_PowerReplication/"

    FUNC_FILES = {
        "dset-camcan": [
            "{sub}/func/{sub}_task-movie_echo-1_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-movie_echo-2_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-movie_echo-3_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-movie_echo-4_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-movie_echo-5_space-scanner_desc-partialPreproc_bold.nii.gz",
        ],
        "dset-cambridge": [
            "{sub}/func/{sub}_task-rest_echo-1_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-rest_echo-2_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-rest_echo-3_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-rest_echo-4_space-scanner_desc-partialPreproc_bold.nii.gz",
        ],
        "dset-dupre": [
            "{sub}/func/{sub}_task-rest_run-1_echo-1_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-rest_run-1_echo-2_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-rest_run-1_echo-3_space-scanner_desc-partialPreproc_bold.nii.gz",
        ],
        "dset-dalenberg": [
            "{sub}/func/{sub}_task-images_echo-1_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-images_echo-2_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-images_echo-3_space-scanner_desc-partialPreproc_bold.nii.gz",
        ],
        "dset-cohen": [
            "{sub}/func/{sub}_task-bilateralfingertapping_echo-1_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-bilateralfingertapping_echo-2_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-bilateralfingertapping_echo-3_space-scanner_desc-partialPreproc_bold.nii.gz",
            "{sub}/func/{sub}_task-bilateralfingertapping_echo-4_space-scanner_desc-partialPreproc_bold.nii.gz",
        ],
    }

    # LUT values from
    # https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    CORTICAL_LABELS = [3, 17, 18, 19, 20, 42, 53, 54, 55, 56]
    SUBCORTICAL_LABELS = [9, 10, 11, 12, 13, 26, 48, 49, 50, 52, 58]
    WM_LABELS = [2, 7, 41, 46, 192]
    CSF_LABELS = [4, 5, 14, 15, 24, 43, 44, 72]
    CEREBELLUM_LABELS = [8, 47]

    # TODO: Replace with post-qc subject list
    dset_dir = op.join(in_dir, dset)
    layout = BIDSLayout(dset_dir)
    subjects = layout.get_subjects()

    fp_dir = op.join(dset_dir, "derivatives/fmriprep")
    out_dir = op.join(dset_dir, "derivatives/power")
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    for subject in subjects:
        fp_subj_dir = op.join(fp_dir, subject)
        preproc_dir = op.join(out_dir, subject, "preprocessed")
        if not op.isdir(preproc_dir):
            os.mkdir(preproc_dir)

        anat_dir = op.join(preproc_dir, "anat")
        if not op.isdir(anat_dir):
            os.mkdir(anat_dir)

        func_dir = op.join(preproc_dir, "func")
        if not op.isdir(func_dir):
            os.mkdir(func_dir)

        # TODO: Maybe replace with hardcoded dictionary
        # Get echo times in ms as numpy array or list
        echos = layout.get_echoes(
            subject=subject,
            modality="func",
            type="bold",
            task="rest",
            run=1,
            extensions=["nii", "nii.gz"],
        )
        echos = sorted(echos)  # just to be extra safe
        n_echos = len(echos)

        # Create GM, WM, and CSF masks
        # WM and CSF masks must be created from the high resolution Freesurfer aparc file
        # Then they must be eroded
        aparcaseg_t1wres = op.join(
            fp_subj_dir,
            "anat",
            f"sub-{subject}_desc-aparcaseg_dseg.nii.gz",
        )
        aparcaseg_boldres = op.join(
            fp_subj_dir,
            "func",
            f"sub-{subject}_task-rest_run-1_space-T1w_desc-aparcaseg_dseg.nii.gz",
        )
        wm_img = image.math_img(f"np.isin(img, {WM_LABELS}).astype(int)", img=aparcaseg_t1wres)
        csf_img = image.math_img(f"np.isin(img, {CSF_LABELS}).astype(int)", img=aparcaseg_t1wres)

        # Create GM masks in BOLD resolution
        cort_img = image.math_img(f"np.isin(img, {CORTICAL_LABELS}).astype(int)", img=aparcaseg_boldres)
        subcort_img = image.math_img(f"np.isin(img, {SUBCORTICAL_LABELS}).astype(int)", img=aparcaseg_boldres)
        cereb_img = image.math_img(f"np.isin(img, {CEREBELLUM_LABELS}).astype(int)", img=aparcaseg_boldres)

        # Save cortical mask to file
        # NOTE: Used for most analyses of "global signal"
        cort_img.to_filename(
            op.join(
                anat_dir,
                f"sub-{subject}_space-MNI152NLin6Asym_res-bold_label-CGM_mask.nii.gz",
            )
        )

        # Erode WM mask
        wm_ero0 = wm_img.get_fdata()
        wm_ero2 = binary_erosion(wm_ero0, iterations=2)
        wm_ero4 = binary_erosion(wm_ero0, iterations=4)

        # Subtract WM mask
        wm_ero02 = wm_ero0 - wm_ero2
        wm_ero24 = wm_ero2 - wm_ero4
        wm_ero02 = nib.Nifti1Image(wm_ero02, wm_img.affine, header=wm_img.header)  # aka Superficial WM
        wm_ero24 = nib.Nifti1Image(wm_ero24, wm_img.affine, header=wm_img.header)  # aka Deeper WM
        wm_ero4 = nib.Nifti1Image(wm_ero4, wm_img.affine, header=wm_img.header)  # aka Deepest WM

        # Resample WM masks to 3mm (functional) resolution with NN interp
        res_wm_ero02 = image.resample_to_img(wm_ero02, aparcaseg_boldres, interpolation="nearest")
        res_wm_ero24 = image.resample_to_img(wm_ero24, aparcaseg_boldres, interpolation="nearest")
        res_wm_ero4 = image.resample_to_img(wm_ero4, aparcaseg_boldres, interpolation="nearest")

        # Erode CSF masks
        csf_ero0 = csf_img.get_fdata()
        csf_ero2 = binary_erosion(csf_ero0, iterations=2)

        # Subtract CSF masks
        csf_ero02 = csf_ero0 - csf_ero2
        csf_ero02 = nib.Nifti1Image(csf_ero02, csf_img.affine, header=csf_img.header)  # aka Superficial CSF
        csf_ero2 = nib.Nifti1Image(csf_ero2, csf_img.affine, header=csf_img.header)  # aka Deeper CSF

        # Resample CSF masks to 3mm (functional) resolution with NN interp
        res_csf_ero02 = image.resample_to_img(csf_ero02, aparcaseg_boldres, interpolation="nearest")
        res_csf_ero2 = image.resample_to_img(csf_ero2, aparcaseg_boldres, interpolation="nearest")

        # Combine masks with different values for carpet pltos
        seg_arr = np.zeros(cort_img.shape)
        cort_arr = cort_img.get_fdata()
        seg_arr[cort_arr == 1] = 1
        subcort_arr = subcort_img.get_fdata()
        seg_arr[subcort_arr == 1] = 2
        cereb_arr = cereb_img.get_fdata()
        seg_arr[cereb_arr == 1] = 3
        wm_ero02_arr = res_wm_ero02.get_fdata()
        seg_arr[wm_ero02_arr == 1] = 4
        wm_ero24_arr = res_wm_ero24.get_fdata()
        seg_arr[wm_ero24_arr == 1] = 5
        wm_ero4_arr = res_wm_ero4.get_fdata()

        # For carpet plots
        seg_arr[wm_ero4_arr == 1] = 6
        seg_img = nib.Nifti1Image(seg_arr, cort_img.affine, header=cort_img.header)
        seg_img.to_filename(op.join(
            anat_dir,
            f"sub-{subject}_space-T1w_res-bold_desc-totalMaskNoCSF_dseg.nii.gz",
        ))
        mask_arr = (seg_arr > 0).astype(int)
        mask_img = nib.Nifti1Image(mask_arr, cort_img.affine, header=cort_img.header)
        mask_img.to_filename(op.join(
            anat_dir,
            "sub-{subject}_space-T1w_res-bold_desc-totalMaskNoCSF_mask.nii.gz",
        ))

        # For brain images *under* carpet plots
        csf_ero02_arr = res_csf_ero02.get_fdata()
        seg_arr[csf_ero02_arr == 1] = 7
        csf_ero2_arr = res_csf_ero2.get_fdata()
        seg_arr[csf_ero2_arr == 1] = 8
        seg_img = nib.Nifti1Image(seg_arr, cort_img.affine, header=cort_img.header)
        seg_img.to_filename(op.join(
            anat_dir,
            f"sub-{subject}_space-T1w_res-bold_desc-totalMaskWithCSF_dseg.nii.gz",
        ))
        mask_arr = (seg_arr > 0).astype(int)
        mask_img = nib.Nifti1Image(mask_arr, cort_img.affine, header=cort_img.header)
        mask_img.to_filename(op.join(
            anat_dir,
            "sub-{subject}_space-T1w_res-bold_desc-totalMaskWithCSF_mask.nii.gz",
        ))

        # Remove non-steady state volumes from fMRI runs
        # TODO: Load and use confounds files
        confounds_file = op.join(
            fp_subj_dir,
            "func",
            f"sub-{subject}_task-rest_run-1_desc-confounds_timeseries.tsv",
        )
        confounds_df = pd.read_table(confounds_file)
        nss_cols = [c for c in confounds_df.columns if c.startswith("non_steady_state_outlier")]
        if len(nss_cols):
            nss_vols = confounds_df.loc[confounds_df[nss_cols].sum(axis=1).astype(bool)].index.tolist()
            # Assume non-steady state volumes are (1) at the beginning and (2) contiguous.
            first_kept_vol = nss_vols[-1] + 1
            n_vols = confounds_df.shape[0]
            reduced_confounds_df = confounds_df.loc[first_kept_vol:]
            for echo_file in echo_files:
                reduced_echo_img = image.index_img(
                    echo_file,
                    slice(first_kept_vol, n_vols + 1)
                )
