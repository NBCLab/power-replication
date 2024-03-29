"""Additional, post-fMRIPrep preprocessing."""
import json
import os
import os.path as op
import shutil
from glob import glob

import nibabel as nib
import nitransforms as nit
import numpy as np
import pandas as pd
from nilearn import image
from scipy.ndimage.morphology import binary_erosion

from processing_utils import run_command


def create_masks(project_dir, dset):
    """Create GM, WM, and CSF masks and resample to functional resolution."""
    print("\t\tcreate_masks", flush=True)
    dset_dir = op.join(project_dir, dset)
    fmriprep_dir = op.join(dset_dir, "derivatives/fmriprep")
    out_dir = op.join(dset_dir, "derivatives/power")

    # Get list of participants with good data
    participants_file = op.join(dset_dir, "participants.tsv")
    participants_df = pd.read_table(participants_file)
    subjects = participants_df.loc[
        participants_df["exclude"] == 0, "participant_id"
    ].tolist()

    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    for subject in subjects:
        print(f"\t\t\t{subject}", flush=True)
        subj_fmriprep_dir = op.join(fmriprep_dir, subject)
        subj_out_dir = op.join(out_dir, subject)
        if not op.isdir(subj_out_dir):
            os.mkdir(subj_out_dir)

        anat_out_dir = op.join(subj_out_dir, "anat")
        if not op.isdir(anat_out_dir):
            os.mkdir(anat_out_dir)

        # Create GM, WM, and CSF masks
        # WM and CSF masks must be created from the high resolution Freesurfer aseg file
        # Then they must be eroded
        aseg_t1wres_t1wspace = op.join(
            subj_fmriprep_dir,
            "anat",
            f"{subject}_desc-aseg_dseg.nii.gz",
        )

        temp_wm_file = op.join(anat_out_dir, "wm.nii.gz")
        run_command(
            f"mri_binarize --i {aseg_t1wres_t1wspace} --o {temp_wm_file} --all-wm"
        )
        assert op.isfile(temp_wm_file)
        temp_csf_file = op.join(anat_out_dir, "csf.nii.gz")
        run_command(
            f"mri_binarize --i {aseg_t1wres_t1wspace} --o {temp_csf_file} --ventricles"
        )
        assert op.isfile(temp_csf_file)

        # Find the BOLD-res aseg
        aseg_boldres_t1wspace = sorted(
            glob(
                op.join(
                    subj_fmriprep_dir,
                    "func",
                    f"{subject}_task-*_space-T1w_desc-aseg_dseg.nii.gz",
                )
            )
        )
        assert len(aseg_boldres_t1wspace) == 1, aseg_boldres_t1wspace
        aseg_boldres_t1wspace = aseg_boldres_t1wspace[0]

        temp_gm_file = op.join(anat_out_dir, "gm.nii.gz")
        run_command(
            f"mri_binarize --i {aseg_boldres_t1wspace} --o {temp_gm_file} --gm"
        )
        assert op.isfile(temp_gm_file)
        temp_subcort_and_cereb_file = op.join(anat_out_dir, "subcort_and_cereb.nii.gz")
        run_command(
            f"mri_binarize --i {aseg_boldres_t1wspace} "
            f"--o {temp_subcort_and_cereb_file} --subcort-gm"
        )
        assert op.isfile(temp_subcort_and_cereb_file)
        temp_cereb_file = op.join(anat_out_dir, "cereb.nii.gz")
        run_command(
            f"mri_binarize --i {aseg_boldres_t1wspace} "
            f"--o {temp_cereb_file} --match 8 --match 47"
        )
        assert op.isfile(temp_cereb_file)
        temp_stem_file = op.join(anat_out_dir, "stem.nii.gz")
        run_command(
            f"mri_binarize --i {aseg_boldres_t1wspace} "
            f"--o {temp_stem_file} --match 16"
        )
        assert op.isfile(temp_stem_file)

        cort_img = image.math_img(
            "(gm_img - subcort_and_cereb_img) - stem_img",
            gm_img=temp_gm_file,
            subcort_and_cereb_img=temp_subcort_and_cereb_file,
            stem_img=temp_stem_file,
        )
        subcort_img = image.math_img(
            "subcort_and_cereb_img - cereb_img",
            subcort_and_cereb_img=temp_subcort_and_cereb_file,
            cereb_img=temp_cereb_file,
        )
        cereb_img = nib.load(temp_cereb_file)

        # Load T1w-space-to-BOLD-space transform
        xfm_files = sorted(
            glob(
                op.join(
                    subj_fmriprep_dir,
                    "func",
                    "*_from-T1w_to-scanner_mode-image_xfm.txt",
                )
            )
        )
        assert len(xfm_files) == 1
        xfm_file = xfm_files[0]
        xfm = nit.linear.load(xfm_file, fmt="itk")

        # Collect one example scanner-space file to use as a reference
        scanner_files = sorted(
            glob(
                op.join(
                    subj_fmriprep_dir,
                    "func",
                    "*_space-scanner_*_bold.nii.gz",
                )
            )
        )
        assert len(scanner_files) >= 3
        scanner_file = scanner_files[0]

        # Save cortical mask to file
        # NOTE: Used for most analyses of "global signal"
        cort_img.to_filename(
            op.join(
                anat_out_dir,
                f"{subject}_space-T1w_res-bold_label-CGM_mask.nii.gz",
            )
        )

        # Erode WM mask
        wm_img = nib.load(temp_wm_file)
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

        # Resample WM masks to functional resolution with NN interp
        res_wm_ero02 = image.resample_to_img(
            wm_ero02,
            aseg_boldres_t1wspace,
            interpolation="nearest",
        )
        res_wm_ero24 = image.resample_to_img(
            wm_ero24,
            aseg_boldres_t1wspace,
            interpolation="nearest",
        )
        res_wm_ero4 = image.resample_to_img(
            wm_ero4,
            aseg_boldres_t1wspace,
            interpolation="nearest",
        )

        # Erode CSF masks
        csf_img = nib.load(temp_csf_file)
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

        # Resample CSF masks to functional resolution with NN interp
        res_csf_ero02 = image.resample_to_img(
            csf_ero02,
            aseg_boldres_t1wspace,
            interpolation="nearest",
        )
        res_csf_ero2 = image.resample_to_img(
            csf_ero2,
            aseg_boldres_t1wspace,
            interpolation="nearest",
        )

        # Combine masks with different values for carpet plots
        seg_arr = np.zeros(cort_img.shape, dtype=int)
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

        # Apply the transform to the BOLD-resolution, T1w-space output files
        # to produce BOLD-resolution, BOLD-space files
        output_filenames = [
            f"{subject}_space-T1w_res-bold_label-CGM_mask.nii.gz",
            f"{subject}_space-T1w_res-bold_desc-totalMaskNoCSF_dseg.nii.gz",
            f"{subject}_space-T1w_res-bold_desc-totalMaskNoCSF_mask.nii.gz",
            f"{subject}_space-T1w_res-bold_desc-totalMaskWithCSF_dseg.nii.gz",
            f"{subject}_space-T1w_res-bold_desc-totalMaskWithCSF_mask.nii.gz",
        ]
        for output_filename in output_filenames:
            output_file_boldres_t1wspace = op.join(anat_out_dir, output_filename)
            output_file_boldres_boldspace = output_file_boldres_t1wspace.replace(
                "space-T1w", "space-scanner"
            )
            output_img_boldres_boldspace = xfm.apply(
                spatialimage=output_file_boldres_t1wspace,
                reference=scanner_file,
                order=0,
            )
            output_img_boldres_boldspace.to_filename(output_file_boldres_boldspace)

        for temp_file in [
            temp_wm_file,
            temp_csf_file,
            temp_gm_file,
            temp_subcort_and_cereb_file,
            temp_cereb_file,
            temp_stem_file,
        ]:
            os.remove(temp_file)


def remove_nss_vols(project_dir, dset):
    """Remove non-steady state volumes from each fMRI image."""
    print("\t\tremove_nss_vols", flush=True)
    dset_dir = op.join(project_dir, dset)
    fmriprep_dir = op.join(dset_dir, "derivatives/fmriprep")
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
        print(f"\t\t\t{subject}", flush=True)
        subj_fmriprep_dir = op.join(fmriprep_dir, subject)
        subj_out_dir = op.join(out_dir, subject)
        if not op.isdir(subj_out_dir):
            os.mkdir(subj_out_dir)

        func_out_dir = op.join(subj_out_dir, "func")
        if not op.isdir(func_out_dir):
            os.mkdir(func_out_dir)

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
    print("\t\tcompile_metadata", flush=True)
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
        print(f"\t\t\t{subject}", flush=True)
        raw_func_dir = op.join(dset_dir, subject, "func")
        fmriprep_func_dir = op.join(fmriprep_dir, subject, "func")
        power_func_dir = op.join(power_dir, subject, "func")
        raw_files = sorted(glob(op.join(raw_func_dir, "sub-*_bold.nii.gz")))
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
        # For dset-dupre
        fmriprep_files = [f.replace("run-01", "run-1") for f in fmriprep_files]
        power_files = [
            op.join(
                power_func_dir,
                f.replace("_bold.nii.gz", "_space-scanner_desc-NSSRemoved_bold.nii.gz"),
            )
            for f in base_filenames
        ]
        power_files = [f.replace("run-01", "run-1") for f in power_files]
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
                # Inheritance is used in dset-cohen and dset-dupre
                raw_json = raw_json.replace(raw_func_dir, dset_dir)
                raw_json = raw_json.replace(f"{subject}_", "")
                raw_json = raw_json.replace("run-01_", "")
                with open(raw_json, "r") as fo:
                    raw_metadata = json.load(fo)

            raw_metadata = {
                k: v for k, v in raw_metadata.items() if k in FROM_RAW_METADATA
            }

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
            # power_metadata["Sources"] = [fmriprep_file]

            with open(fmriprep_json, "w") as fo:
                json.dump(fmriprep_metadata, fo, indent=4, sort_keys=True)

            with open(power_json, "w") as fo:
                json.dump(power_metadata, fo, indent=4, sort_keys=True)


def create_top_level_files(project_dir, dset):
    """Create top-level files describing masks and discrete segmentation values."""
    print("\t\tcreate_top_level_files", flush=True)
    INFO = {
        "space-scanner_res-bold_label-CGM_mask.json": {
            "Type": "ROI",
            "Resolution": "Native BOLD resolution.",
        },
        "space-scanner_res-bold_desc-totalMaskNoCSF_dseg.json": {
            "Resolution": "Native BOLD resolution.",
        },
        "space-scanner_res-bold_desc-totalMaskNoCSF_dseg.tsv": pd.DataFrame(
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
        "space-scanner_res-bold_desc-totalMaskWithCSF_dseg.json": {
            "Resolution": "Native BOLD resolution.",
        },
        "space-scanner_res-bold_desc-totalMaskWithCSF_dseg.tsv": pd.DataFrame(
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
        "space-scanner_res-bold_desc-totalMaskNoCSF_mask.json": {
            "Type": "Brain",
            "Resolution": "Native BOLD resolution.",
        },
        "space-scanner_res-bold_desc-totalMaskWithCSF_mask.json": {
            "Type": "Brain",
            "Resolution": "Native BOLD resolution.",
        },
    }
    out_dir = op.join(project_dir, dset, "derivatives/power")
    for k, v in INFO.items():
        out_file = op.join(out_dir, k)
        print(f"\tCreating {out_file}", flush=True)
        if isinstance(v, dict):
            with open(out_file, "w") as fo:
                json.dump(v, fo, indent=4, sort_keys=True)
        elif isinstance(v, pd.DataFrame):
            v.to_csv(out_file, sep="\t", line_terminator="\n", index=False)
        else:
            raise Exception(f"Type {type(v)} not understood.")

    fmriprep_data_desc = op.join(
        project_dir,
        dset,
        "derivatives/fmriprep/dataset_description.json",
    )
    out_data_desc = op.join(
        project_dir, dset, "derivatives/power/dataset_description.json"
    )
    with open(fmriprep_data_desc, "r") as fo:
        data_description = json.load(fo)

    data_description["Name"] = "Replication of Power et al. (2018)"
    data_description[
        "HowToAcknowledge"
    ] += " Please cite Salo et al. (2021) (once it's published, of course)."
    data_description["GeneratedBy"] = [
        {
            "Name": "Custom Code",
            "Description": (
                "Postprocessing workflow to "
                "(1) extract echo-wise preprocessed data from the fMRIPrep working directory, "
                "(2) create tissue type masks at functional resolution, "
                "(3) remove non-steady state volumes from each fMRI run, and "
                "(4) calculate nuisance regressors "
                "for the Power et al. (2018) replication."
            ),
            "CodeURL": "https://github.com/NBCLab/power-replication",
        },
    ] + data_description["GeneratedBy"]

    with open(out_data_desc, "w") as fo:
        json.dump(data_description, fo, sort_keys=True, indent=4)


if __name__ == "__main__":
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    dsets = [
        "dset-cambridge",
        "dset-camcan",
        "dset-cohen",
        "dset-dalenberg",
        "dset-dupre",
    ]
    print(op.basename(__file__), flush=True)
    for dset in dsets:
        print(f"\t{dset}", flush=True)
        create_masks(project_dir, dset)
        # remove_nss_vols(project_dir, dset)
        # compile_metadata(project_dir, dset)
        # create_top_level_files(project_dir, dset)
