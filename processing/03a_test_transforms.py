"""Evaluate different approaches to transforms for Power replication.

Perform different combinations of T1w-space-to-BOLD-space transforms,
T1w-res-to-BOLD-res downsampling, and segmentations.

Make sure to load AFNI as a module.
"""
import os
import subprocess

import nibabel as nib
import nitransforms as nit
import numpy as np
from nilearn import image


def run_command(command, env=None):
    """Run a given shell command with certain environment variables set."""
    merged_env = os.environ
    if env:
        merged_env.update(env)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        env=merged_env,
    )
    while True:
        line = process.stdout.readline()
        line = str(line, "utf-8")[:-1]
        print(line)
        if line == "" and process.poll() is not None:
            break

    if process.returncode != 0:
        raise Exception(
            "Non zero return code: {0}\n"
            "{1}\n\n{2}".format(process.returncode, command, process.stdout.read())
        )


def extract_cortical_ribbon(seg_file, out_file):
    """Create a cortical ribbon mask from an ASEG segmentation file."""
    CORTICAL_LABELS = [3, 17, 18, 19, 20, 42, 53, 54, 55, 56]
    cort_img = image.math_img(
        f"np.isin(img, {CORTICAL_LABELS}).astype(np.int32)",
        img=seg_file,
    )
    cort_img.to_filename(out_file)


def run_3dresample(in_file, out_file, target_dims):
    """Downsample a file to target dimensions with 3dresample."""
    target_dims = [np.round(dim, decimals=3) for dim in target_dims]
    dim_str = f"-dxyz {target_dims[0]:.3f} {target_dims[1]:.3f} {target_dims[2]:.3f}"
    cmd_str = f"3dresample -prefix {out_file} -input {in_file} {dim_str} -rmode NN"
    run_command(cmd_str)


def apply_xfm(in_file, out_file, xfm_file, scanner_file):
    """Apply a linear transform to a file with nitransforms."""
    xfm_obj = nit.linear.load(xfm_file, fmt="itk")
    in_img = nib.load(in_file)
    out_img = xfm_obj.apply(
        spatialimage=in_img,
        reference=scanner_file,
        order=0,
    )
    out_img.to_filename(out_file)


def main(in_dir, out_dir, subject):
    """Run the variety of transforms and segmentations on a single subject's files."""
    in_dir = os.path.join(in_dir, subject)

    # Step 1: Identify relevant files
    aseg_t1res_t1space = os.path.join(in_dir, f"anat/{subject}_desc-aseg_dseg.nii.gz")
    aseg_boldres_t1space = os.path.join(
        in_dir, f"func/{subject}_task-movie_space-T1w_desc-aseg_dseg.nii.gz"
    )
    xfm = os.path.join(
        in_dir, f"func/{subject}_task-movie_from-scanner_to-T1w_mode-image_xfm.txt"
    )
    scanner_file = os.path.join(
        in_dir,
        f"func/{subject}_task-movie_echo-1_space-scanner_desc-partialPreproc_bold.nii.gz",
    )
    scanner_img = nib.load(scanner_file)
    target_dims = scanner_img.header.get_zooms()[:3]

    # Step 2: Downsample T1-res version with 3dresample
    aseg_boldres_t1space_from_3dresample = os.path.join(
        out_dir, f"{subject}_desc-asegFrom3dresample_space-T1w_res-bold_dseg.nii.gz"
    )
    run_3dresample(
        aseg_t1res_t1space,
        aseg_boldres_t1space_from_3dresample,
        target_dims,
    )

    # Step 3: Transform T1-space asegs to BOLD space
    aseg_boldres_boldspace_from_t1res = os.path.join(
        out_dir, f"{subject}_desc-asegFromT1wRes_space-bold_res-bold_dseg.nii.gz"
    )
    apply_xfm(aseg_t1res_t1space, aseg_boldres_boldspace_from_t1res, xfm, scanner_file)

    aseg_boldres_boldspace_from_lowres = os.path.join(
        out_dir, f"{subject}_desc-asegFromLowRes_space-bold_res-bold_dseg.nii.gz"
    )
    apply_xfm(
        aseg_boldres_t1space, aseg_boldres_boldspace_from_lowres, xfm, scanner_file
    )

    aseg_boldres_boldspace_from_3dresample = os.path.join(
        out_dir, f"{subject}_desc-asegFrom3dresample_space-bold_res-bold_dseg.nii.gz"
    )
    apply_xfm(
        aseg_boldres_t1space_from_3dresample,
        aseg_boldres_boldspace_from_3dresample,
        xfm,
        scanner_file,
    )

    # Step 4: Extract cortical ribbons
    # Segment the T1-resolution, T1-space aseg
    cort_t1res_t1space = os.path.join(
        out_dir, f"{subject}_desc-cort_space-T1w_res-T1w_mask.nii.gz"
    )
    extract_cortical_ribbon(aseg_t1res_t1space, cort_t1res_t1space)

    # Segment the BOLD-resolution, T1-space aseg
    cort_boldres_t1space = os.path.join(
        out_dir, f"{subject}_desc-cort_space-T1w_res-bold_mask.nii.gz"
    )
    extract_cortical_ribbon(aseg_boldres_t1space, cort_boldres_t1space)

    # OUTPUT1: Apply the transform to the T1-resolution aseg, then segment
    # (1) Transform T1-res (2) Segment
    cort_boldres_boldspace_from_t1res = os.path.join(
        out_dir, f"{subject}_desc-cortFromT1wRes_space-bold_res-bold_mask.nii.gz"
    )
    extract_cortical_ribbon(
        aseg_boldres_boldspace_from_t1res, cort_boldres_boldspace_from_t1res
    )

    # OUTPUT2: Apply the transform to the BOLD-resolution aseg, then segment
    # (1) Transform BOLD-res (2) Segment
    cort_boldres_boldspace_from_lowres = os.path.join(
        out_dir, f"{subject}_desc-cortFromLowRes_space-bold_res-bold_mask.nii.gz"
    )
    extract_cortical_ribbon(
        aseg_boldres_boldspace_from_lowres, cort_boldres_boldspace_from_lowres
    )

    # OUTPUT3: Downsample the T1-res aseg with 3dresample, apply the transform, then segment
    # (1) Downsample T1-res (2) Transform (3) Segment
    cort_boldres_boldspace_from_3dresample = os.path.join(
        out_dir, f"{subject}_desc-cortFrom3dresample_space-bold_res-bold_mask.nii.gz"
    )
    extract_cortical_ribbon(
        aseg_boldres_boldspace_from_3dresample, cort_boldres_boldspace_from_3dresample
    )

    # Step 5: Downsample T1-res cortical ribbon to BOLD-res
    cort_boldres_t1space_from_segmented_t1res_and_3dresample = os.path.join(
        out_dir,
        f"{subject}_desc-cortFromSegT1wRes3dresample_space-T1w_res-bold_mask.nii.gz",
    )
    run_3dresample(
        cort_t1res_t1space,
        cort_boldres_t1space_from_segmented_t1res_and_3dresample,
        target_dims,
    )

    # Step 5: Transform T1-space cortical ribbons to BOLD space
    # OUTPUT4: Segment the T1-res aseg, then apply the transform.
    # (1) Segment T1-res (2) Transform
    cort_boldres_boldspace_from_segmented_t1res = os.path.join(
        out_dir, f"{subject}_desc-cortFromSegT1wRes_space-bold_res-bold_mask.nii.gz"
    )
    apply_xfm(
        cort_t1res_t1space,
        cort_boldres_boldspace_from_segmented_t1res,
        xfm,
        scanner_file,
    )

    # OUTPUT5: Segment the BOLD-res aseg, then apply the transform.
    # (1) Segment BOLD-res (2) Transform
    cort_boldres_boldspace_from_segmented_lowres = os.path.join(
        out_dir, f"{subject}_desc-cortFromSegLowRes_space-bold_res-bold_mask.nii.gz"
    )
    apply_xfm(
        cort_boldres_t1space,
        cort_boldres_boldspace_from_segmented_lowres,
        xfm,
        scanner_file,
    )

    # OUTPUT6: Segment the T1-res aseg, downsample with 3dresample, and then apply the transform.
    # (1) Segment T1-res (2) Downsample (3) Transform
    cort_boldres_boldspace_from_segmented_t1res_and_3dresample = os.path.join(
        out_dir,
        f"{subject}_desc-cortFromSegT1wRes3dresample_space-bold_res-bold_mask.nii.gz",
    )
    apply_xfm(
        cort_boldres_t1space_from_segmented_t1res_and_3dresample,
        cort_boldres_boldspace_from_segmented_t1res_and_3dresample,
        xfm,
        scanner_file,
    )

    # Step 6: Write everything out
    print(f"\tSubject {subject} completed.")


if __name__ == "__main__":
    IN_DIR = "/home/data/nbc/misc-projects/Salo_PowerReplication/dset-camcan/derivatives/fmriprep/"
    TEST_DIR = "/home/tsalo006/xfm_test/"
    SUBJECTS = [
        "sub-CC220284",
        "sub-CC420226",
        "sub-CC520552",
        "sub-CC520673",
        "sub-CC722216",
    ]
    for subject in SUBJECTS:
        main(IN_DIR, TEST_DIR, subject)

    print("Workflow completed.")
