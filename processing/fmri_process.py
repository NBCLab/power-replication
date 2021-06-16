"""
Additional, post-fMRIPrep preprocessing.
"""
import json
import os
import os.path as op
import shutil
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from scipy.ndimage.morphology import binary_erosion


def preprocess(project_dir, dset):
    """
    Perform additional, post-fMRIPrep preprocessing of structural and
    functional MRI data.
    1) Create GM, WM, and CSF masks and resample to 3mm (functional) resolution
    2) Remove non-steady state volumes from each fMRI image
    """
    # LUT values from
    # https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    CORTICAL_LABELS = [3, 17, 18, 19, 20, 42, 53, 54, 55, 56]
    SUBCORTICAL_LABELS = [9, 10, 11, 12, 13, 26, 48, 49, 50, 52, 58]
    WM_LABELS = [2, 7, 41, 46, 192]
    CSF_LABELS = [4, 5, 14, 15, 24, 43, 44, 72]
    CEREBELLUM_LABELS = [8, 47]

    dset_dir = op.join(project_dir, dset)
    fp_dir = op.join(dset_dir, "derivatives/fmriprep")
    out_dir = op.join(dset_dir, "derivatives/power")

    # Get list of participants with good data
    participants_file = op.join(dset_dir, "participants.tsv")
    participants_df = pd.read_table(participants_file)
    subjects = participants_df.loc[
        participants_df["exclude"] == 0, "participant_id"
    ].tolist()

    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    # Summary information saved to a file
    nss_file = op.join(out_dir, "nss_removed.tsv")
    nss_df = pd.DataFrame(columns=["nss_count"], index=subjects)
    nss_df.index.name = "participant_id"

    for subject in subjects:
        print(f"\t{subject}", flush=True)
        subj_fmriprep_dir = op.join(fp_dir, subject)
        subj_out_dir = op.join(out_dir, subject)
        if not op.isdir(subj_out_dir):
            os.mkdir(subj_out_dir)

        anat_out_dir = op.join(subj_out_dir, "anat")
        if not op.isdir(anat_out_dir):
            os.mkdir(anat_out_dir)

        func_out_dir = op.join(subj_out_dir, "func")
        if not op.isdir(func_out_dir):
            os.mkdir(func_out_dir)

        # Create GM, WM, and CSF masks
        # WM and CSF masks must be created from the high resolution Freesurfer aparc file
        # Then they must be eroded
        aparcaseg_t1wres = op.join(
            subj_fmriprep_dir,
            "anat",
            f"{subject}_desc-aparcaseg_dseg.nii.gz",
        )
        aparcaseg_boldres = sorted(
            glob(
                op.join(
                    subj_fmriprep_dir,
                    "func",
                    f"{subject}_task-*_space-T1w_desc-aparcaseg_dseg.nii.gz",
                )
            )
        )[0]
        wm_img = image.math_img(
            f"np.isin(img, {WM_LABELS}).astype(int)",
            img=aparcaseg_t1wres,
        )
        csf_img = image.math_img(
            f"np.isin(img, {CSF_LABELS}).astype(int)",
            img=aparcaseg_t1wres,
        )

        # Create GM masks in BOLD resolution
        cort_img = image.math_img(
            f"np.isin(img, {CORTICAL_LABELS}).astype(int)",
            img=aparcaseg_boldres,
        )
        subcort_img = image.math_img(
            f"np.isin(img, {SUBCORTICAL_LABELS}).astype(int)",
            img=aparcaseg_boldres,
        )
        cereb_img = image.math_img(
            f"np.isin(img, {CEREBELLUM_LABELS}).astype(int)",
            img=aparcaseg_boldres,
        )

        # Save cortical mask to file
        # NOTE: Used for most analyses of "global signal"
        cort_img.to_filename(
            op.join(
                anat_out_dir,
                f"{subject}_space-T1w_res-bold_label-CGM_mask.nii.gz",
            )
        )

        # Erode WM mask
        wm_ero0 = wm_img.get_fdata()
        wm_ero2 = binary_erosion(wm_ero0, iterations=2)
        wm_ero4 = binary_erosion(wm_ero0, iterations=4)

        # Subtract WM mask
        wm_ero02 = wm_ero0.astype(int) - wm_ero2.astype(int)
        wm_ero24 = wm_ero2.astype(int) - wm_ero4.astype(int)
        wm_ero02 = nib.Nifti1Image(
            wm_ero02, wm_img.affine, header=wm_img.header
        )  # aka Superficial WM
        wm_ero24 = nib.Nifti1Image(
            wm_ero24, wm_img.affine, header=wm_img.header
        )  # aka Deeper WM
        wm_ero4 = nib.Nifti1Image(
            wm_ero4, wm_img.affine, header=wm_img.header
        )  # aka Deepest WM

        # Resample WM masks to 3mm (functional) resolution with NN interp
        res_wm_ero02 = image.resample_to_img(
            wm_ero02,
            aparcaseg_boldres,
            interpolation="nearest",
        )
        res_wm_ero24 = image.resample_to_img(
            wm_ero24,
            aparcaseg_boldres,
            interpolation="nearest",
        )
        res_wm_ero4 = image.resample_to_img(
            wm_ero4,
            aparcaseg_boldres,
            interpolation="nearest",
        )

        # Erode CSF masks
        csf_ero0 = csf_img.get_fdata()
        csf_ero2 = binary_erosion(csf_ero0, iterations=2)

        # Subtract CSF masks
        csf_ero02 = csf_ero0.astype(int) - csf_ero2.astype(int)
        csf_ero02 = nib.Nifti1Image(
            csf_ero02, csf_img.affine, header=csf_img.header
        )  # aka Superficial CSF
        csf_ero2 = nib.Nifti1Image(
            csf_ero2, csf_img.affine, header=csf_img.header
        )  # aka Deeper CSF

        # Resample CSF masks to 3mm (functional) resolution with NN interp
        res_csf_ero02 = image.resample_to_img(
            csf_ero02,
            aparcaseg_boldres,
            interpolation="nearest",
        )
        res_csf_ero2 = image.resample_to_img(
            csf_ero2,
            aparcaseg_boldres,
            interpolation="nearest",
        )

        # Combine masks with different values for carpet plots
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
        seg_arr[wm_ero4_arr == 1] = 6

        # For carpet plots
        seg_img = nib.Nifti1Image(seg_arr, cort_img.affine, header=cort_img.header)
        seg_img.to_filename(
            op.join(
                anat_out_dir,
                f"{subject}_space-T1w_res-bold_desc-totalMaskNoCSF_dseg.nii.gz",
            )
        )
        mask_arr = (seg_arr > 0).astype(int)
        mask_img = nib.Nifti1Image(mask_arr, cort_img.affine, header=cort_img.header)
        mask_img.to_filename(
            op.join(
                anat_out_dir,
                f"{subject}_space-T1w_res-bold_desc-totalMaskNoCSF_mask.nii.gz",
            )
        )

        # For brain images *under* carpet plots
        csf_ero02_arr = res_csf_ero02.get_fdata()
        seg_arr[csf_ero02_arr == 1] = 7
        csf_ero2_arr = res_csf_ero2.get_fdata()
        seg_arr[csf_ero2_arr == 1] = 8
        seg_img = nib.Nifti1Image(seg_arr, cort_img.affine, header=cort_img.header)
        seg_img.to_filename(
            op.join(
                anat_out_dir,
                f"{subject}_space-T1w_res-bold_desc-totalMaskWithCSF_dseg.nii.gz",
            )
        )
        mask_arr = (seg_arr > 0).astype(int)
        mask_img = nib.Nifti1Image(mask_arr, cort_img.affine, header=cort_img.header)
        mask_img.to_filename(
            op.join(
                anat_out_dir,
                f"{subject}_space-T1w_res-bold_desc-totalMaskWithCSF_mask.nii.gz",
            )
        )

        # Remove non-steady state volumes from fMRI runs
        pattern = op.join(
            subj_fmriprep_dir,
            "func",
            f"{subject}_task-*_echo-*_space-scanner_desc-partialPreproc_bold.nii.gz",
        )
        echo_files = sorted(glob(pattern))
        assert len(echo_files) >= 3, pattern
        preproc_json = sorted(
            glob(
                op.join(
                    subj_fmriprep_dir,
                    "func",
                    f"{subject}_task-*_desc-preproc_bold.json",
                )
            )
        )[0]

        # Load and use confounds files
        confounds_file = sorted(
            glob(
                op.join(
                    subj_fmriprep_dir,
                    "func",
                    f"{subject}_task-*_desc-confounds_timeseries.tsv",
                )
            )
        )[0]
        confounds_filename = op.basename(confounds_file)
        out_confounds_file = op.join(func_out_dir, confounds_filename)
        confounds_json_file = confounds_file.replace(".tsv", ".json")
        out_confounds_json_file = out_confounds_file.replace(".tsv", ".json")

        confounds_df = pd.read_table(confounds_file)
        nss_cols = [
            c for c in confounds_df.columns if c.startswith("non_steady_state_outlier")
        ]

        if len(nss_cols):
            nss_vols = confounds_df.loc[
                confounds_df[nss_cols].sum(axis=1).astype(bool)
            ].index.tolist()

            # Assume non-steady state volumes are (1) at the beginning and (2) contiguous.
            first_kept_vol = nss_vols[-1] + 1
            n_vols = confounds_df.shape[0]
            reduced_confounds_df = confounds_df.loc[first_kept_vol:]
            reduced_confounds_df.to_csv(out_confounds_file, sep="\t", index=False)
            nss_df.loc[subject, "nss_count"] = first_kept_vol

            # Copy and update metadata for confounds file
            with open(confounds_json_file, "r") as fo:
                json_info = json.load(fo)
                json_info["Sources"] = [confounds_filename]
                json_info[
                    "Description"
                ] = "fMRIPrep-generated confounds file with non-steady state volumes removed."

            with open(out_confounds_json_file, "w") as fo:
                json.dump(json_info, fo, indent=4, sort_keys=True)

            for echo_file in echo_files:
                reduced_echo_img = image.index_img(
                    echo_file, slice(first_kept_vol, n_vols + 1)
                )
                echo_filename = op.basename(echo_file)
                echo_filename = echo_filename.replace(
                    "_desc-partialPreproc_",
                    "_desc-NSSRemoved_",
                )
                out_echo_file = op.join(func_out_dir, echo_filename)
                reduced_echo_img.to_filename(out_echo_file)

                # Copy and update metadata for imaging files
                out_nii_json_file = out_echo_file.replace(".nii.gz", ".json")
                with open(preproc_json, "r") as fo:
                    json_info = json.load(fo)
                    json_info["Sources"] = [echo_file]
                    json_info["Description"] = (
                        "Echo-wise native-space preprocessed data from fMRIPrep, "
                        f"with {first_kept_vol} non-steady state volume(s) removed."
                    )

                with open(out_nii_json_file, "w") as fo:
                    json.dump(json_info, fo, indent=4, sort_keys=True)
        else:
            shutil.copyfile(confounds_file, out_confounds_file)
            nss_df.loc[subject, "nss_count"] = 0

            # Copy and update metadata for confounds file
            with open(confounds_json_file, "r") as fo:
                json_info = json.load(fo)
                json_info["Sources"] = [confounds_filename]
                json_info[
                    "Description"
                ] = "fMRIPrep-generated confounds file with non-steady state volumes removed."

            with open(out_confounds_json_file, "w") as fo:
                json.dump(json_info, fo, indent=4, sort_keys=True)

            # Copy and update metadata for imaging files
            for echo_file in echo_files:
                echo_filename = op.basename(echo_file)
                echo_filename = echo_filename.replace(
                    "_desc-partialPreproc_",
                    "_desc-NSSRemoved_",
                )
                out_echo_file = op.join(func_out_dir, echo_filename)
                shutil.copyfile(echo_file, out_echo_file)

                # Copy and update metadata
                out_nii_json_file = out_echo_file.replace(".nii.gz", ".json")
                with open(preproc_json, "r") as fo:
                    json_info = json.load(fo)
                    json_info["Sources"] = [echo_file]
                    json_info["Description"] = (
                        "Echo-wise native-space preprocessed data from fMRIPrep, "
                        "with 0 non-steady state volume(s) removed."
                    )

                with open(out_nii_json_file, "w") as fo:
                    json.dump(json_info, fo, indent=4, sort_keys=True)

        nss_df.to_csv(nss_file, sep="\t", index=True, index_label="participant_id")


def compile_metadata(project_dir, dset):
    """Extract metadata from raw BOLD files and add to the preprocessed BOLD file jsons.

    Parameters
    ----------
    project_dir
    dset
    """
    dset_dir = op.join(project_dir, dset)
    power_dir = op.join(dset_dir, "derivatives/power")
    fmriprep_dir = op.join(dset_dir, "derivatives/fmriprep")

    # Get list of participants with good data
    participants_file = op.join(dset_dir, "participants.tsv")
    participants_df = pd.read_table(participants_file)
    subjects = participants_df.loc[
        participants_df["exclude"] == 0, "participant_id"
    ].tolist()

    FROM_RAW_METADATA = ["EchoTime", "RepetitionTime", "FlipAngle", "TaskName"]

    for subject in subjects:
        raw_func_dir = op.join(dset_dir, subject, "func")
        fmriprep_func_dir = op.join(fmriprep_dir, subject, "func")
        power_func_dir = op.join(power_dir, subject, "func")
        raw_files = sorted(glob(op.join(raw_func_dir, "sub-*_bold.nii.gz")))
        a = "sub-04570_task-rest_echo-1_bold.nii.gz"
        b = "sub-04570_task-rest_echo-1_space-scanner_desc-partialPreproc_bold.nii.gz"
        base_filenames = [op.basename(f) for f in raw_files]
        fmriprep_files = [
            op.join(
                fmriprep_func_dir,
                f.replace(
                    "_bold.nii.gz", "_space-scanner_desc-partialPreproc_bold.nii.gz"
                ),
            )
            for f in base_filenames
        ]
        power_files = [
            op.join(
                power_func_dir,
                f.replace("_bold.nii.gz", "_space-scanner_desc-NSSRemoved_bold.nii.gz"),
            )
            for f in base_filenames
        ]
        assert all(op.isfile(f) for f in fmriprep_files), fmriprep_files
        assert all(op.isfile(f) for f in power_files), power_files
        for i_file, raw_file in enumerate(raw_files):
            fmriprep_file = fmriprep_files[i_file]
            power_file = power_files[i_file]
            raw_json = raw_file.replace(".nii.gz", ".json")
            fmriprep_json = fmriprep_file.replace(".nii.gz", ".json")
            power_json = power_file.replace(".nii.gz", ".json")
            if op.isfile(raw_json):
                with open(raw_json, "r") as fo:
                    raw_metadata = json.load(fo)
            else:
                # Inheritance is used in dset-cohen
                raw_json = raw_json.replace(raw_func_dir, dset_dir)
                raw_json = raw_json.replace(f"{subject}_", "")
                with open(raw_json, "r") as fo:
                    raw_metadata = json.load(fo)

            raw_metadata = {k: v for k, v in raw_metadata.items() if k in FROM_RAW_METADATA}

            if op.isfile(fmriprep_json):
                with open(fmriprep_json, "r") as fo:
                    fmriprep_metadata = json.load(fo)
            else:
                fmriprep_metadata = {}

            with open(power_json, "r") as fo:
                power_metadata = json.load(fo)

            # Merge in metadata
            fmriprep_metadata = {**raw_metadata, **fmriprep_metadata}
            power_metadata = {**fmriprep_metadata, **power_metadata}

            fmriprep_metadata["RawSources"] = [raw_file]
            power_metadata["RawSources"] = [raw_file]
            # Already done in preprocess()
            # power_metadata["Sources"] = [fmripep_file]

            with open(fmriprep_json, "w") as fo:
                json.dump(fmriprep_metadata, fo, indent=4, sort_keys=True)

            with open(power_json, "w") as fo:
                json.dump(power_metadata, fo, indent=4, sort_keys=True)


def create_top_level_files(project_dir, dset):
    """Create top-level files describing masks and discrete segmentation values."""
    INFO = {
        "space-T1w_res-bold_label-CGM_mask.json": {
            "Type": "ROI",
            "Resolution": "Native BOLD resolution.",
        },
        "space-T1w_res-bold_desc-totalMaskNoCSF_dseg.json": {
            "Resolution": "Native BOLD resolution.",
        },
        "space-T1w_res-bold_desc-totalMaskNoCSF_dseg.tsv": pd.DataFrame(
            columns=["index", "name", "abbreviation", "mapping"],
            data=[
                [1, "Cortical Ribbon", "CORT", 8],
                [2, "Subcortical Nuclei", "SUBCORT", 9],
                [3, "Cerebellum", "CEREB", 11],
                [4, "Superficial WM", "WMero02", 2],
                [5, "Deeper WM", "WMero24", 2],
                [6, "Deepest WM", "WMero4", 2],
            ],
        ),
        "space-T1w_res-bold_desc-totalMaskWithCSF_dseg.json": {
            "Resolution": "Native BOLD resolution.",
        },
        "space-T1w_res-bold_desc-totalMaskWithCSF_dseg.tsv": pd.DataFrame(
            columns=["index", "name", "abbreviation", "mapping"],
            data=[
                [1, "Cortical Ribbon", "CORT", 8],
                [2, "Subcortical Nuclei", "SUBCORT", 9],
                [3, "Cerebellum", "CEREB", 11],
                [4, "Superficial WM", "WMero02", 2],
                [5, "Deeper WM", "WMero24", 2],
                [6, "Deepest WM", "WMero4", 2],
                [7, "Superficial CSF", "CSFero02", 3],
                [8, "Deeper CSF", "CSFero2", 3],
            ],
        ),
        "space-T1w_res-bold_desc-totalMaskNoCSF_mask.json": {
            "Type": "Brain",
            "Resolution": "Native BOLD resolution.",
        },
        "space-T1w_res-bold_desc-totalMaskWithCSF_mask.json": {
            "Type": "Brain",
            "Resolution": "Native BOLD resolution.",
        },
    }
    out_dir = op.join(project_dir, dset, "derivatives/power")
    for k, v in INFO.items():
        out_file = op.join(out_dir, k)
        print(f"\tCreating {out_file}", flush=True)
        if isinstance(v, dict):
            with open(k, "w") as fo:
                json.dump(v, fo, indent=4, sort_keys=True)
        elif isinstance(v, pd.DataFrame):
            v.to_csv(k, sep="\t", line_terminator="\n", index=False)
        else:
            raise Exception(f"Type {type(v)} not understood.")


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
        print(f"{dset}", flush=True)
        # preprocess(project_dir, dset)
        compile_metadata(project_dir, dset)
        create_top_level_files(project_dir, dset)
