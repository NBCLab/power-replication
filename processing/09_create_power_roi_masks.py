"""Create Power atlas ROI masks in native functional space."""
import argparse
import os.path as op
from glob import glob

import nibabel as nib
import numpy as np
from nilearn import datasets, input_data
from numpy.lib.recfunctions import structured_to_unstructured

from utils import run_command


def transform_to_native_space(
    standard_space_roi_file,
    native_space_file,
    out_file,
    xform_std_to_t1w,
    xform_t1w_to_native,
):
    str_ = (
        f"antsApplyTransforms -d 3 -i {standard_space_roi_file} "
        f"-r {native_space_file} -o {out_file} "
        f"-n NearestNeighbor -t {xform_std_to_t1w} {xform_t1w_to_native}"
    )
    run_command(str_)


def create_standard_space_mask():
    """Create the standard space version of the Power atlas.

    Each coordinate is replaced with a sphere (r=5mm) where the value corresponds to the index
    of the coordinate.
    The standard space mask was downloaded from the TemplateFlow website.

    I ran this separately on the HPC.
    """
    mask_img = nib.load("tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii")

    power_rois = datasets.fetch_coords_power_2011()

    arr = np.zeros(mask_img.shape, dtype=int)
    for i_row in range(power_rois["rois"].shape[0]):
        row = power_rois["rois"][i_row]
        val = row["roi"]
        xyz = structured_to_unstructured(row[["x", "y", "z"]])
        sphere_masker = input_data.NiftiSpheresMasker([xyz], radius=5, mask_img=mask_img)
        sphere_masker.fit(mask_img)
        sphere_img = sphere_masker.inverse_transform(np.array([[1]]))
        sphere_data = np.squeeze(sphere_img.get_fdata())
        arr[sphere_data == 1] = val

    out = nib.Nifti1Image(arr, mask_img.affine, mask_img.header)
    out.to_filename("space-MNI152NLin6Asym_res-01_desc-power_dseg.nii.gz")


def main(project_dir, dset, subject):
    power_dir = op.join(project_dir, dset, "derivatives", "power")
    tedana_dir = op.join(project_dir, dset, "derivatives", "tedana")
    tedana_subj_dir = op.join(tedana_dir, subject, "func")
    fmriprep_subj_dir = op.join(project_dir, dset, "derivatives", "fmriprep", subject)
    out_dir = op.join(power_dir, subject, "func")

    medn_files = glob(op.join(tedana_subj_dir, "*_desc-optcomDenoised_bold.nii.gz"))
    assert len(medn_files) == 1
    medn_file = medn_files[0]

    # Parse input files
    medn_name = op.basename(medn_file)
    prefix = medn_name.split("desc-")[0].rstrip("_")

    # Determine input files
    standard_space_roi_file = "space-MNI152NLin6Asym_res-01_desc-power_dseg.nii.gz"
    native_space_file = medn_file
    out_file = op.join(out_dir, f"{prefix}_space-scanner_res-bold_desc-power_dseg.nii.gz")
    # TODO: Figure out names for these xform files
    xform_std_to_t1w = op.join(fmriprep_subj_dir, "")
    xform_t1w_to_native = op.join(fmriprep_subj_dir, "")

    transform_to_native_space(
        standard_space_roi_file,
        native_space_file,
        out_file,
        xform_std_to_t1w,
        xform_t1w_to_native,
    )


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
