"""Warp the scanner-space target maps to standard space for DDMRA analyses."""
import argparse
import os.path as op
from glob import glob

from processing_utils import run_command

TARGET_FILES = [
    # TE30
    "power/{sub}/func/{prefix}_desc-TE30_bold.nii.gz",
    # OC
    "tedana/{sub}/func/{prefix}_desc-optcom_bold.nii.gz",
    # MEDN
    "tedana/{sub}/func/{prefix}_desc-optcomDenoised_bold.nii.gz",
    # MEDN+MIR
    "tedana/{sub}/func/{prefix}_desc-optcomMIRDenoised_bold.nii.gz",
    # MEDN+MIR Noise
    "tedana/{sub}/func/{prefix}_desc-optcomMIRDenoised_errorts.nii.gz",
    # FIT-R2
    "t2smap/{sub}/func/{prefix}_T2starmap.nii.gz",
    # FIT-S0
    "t2smap/{sub}/func/{prefix}_S0map.nii.gz",
    # MEDN+GODEC (sparse)
    "godec/{sub}/func/{prefix}_desc-GODEC_rank-4_bold.nii.gz",
    # MEDN+GODEC Noise (lowrank)
    "godec/{sub}/func/{prefix}_desc-GODEC_rank-4_lowrankts.nii.gz",
    # MEDN+dGSR
    "rapidtide/{sub}/func/{prefix}_desc-lfofilterCleaned_bold.nii.gz",
    # MEDN+dGSR Noise
    "rapidtide/{sub}/func/{prefix}_desc-lfofilterCleaned_errorts.nii.gz",
    # MEDN+aCompCor
    "nuisance-regressions/{sub}/func/{prefix}_desc-aCompCor_bold.nii.gz",
    # MEDN+aCompCor Noise
    "nuisance-regressions/{sub}/func/{prefix}_desc-aCompCor_errorts.nii.gz",
    # MEDN+GSR
    "nuisance-regressions/{sub}/func/{prefix}_desc-GSR_bold.nii.gz",
    # MEDN+GSR Noise
    "nuisance-regressions/{sub}/func/{prefix}_desc-GSR_errorts.nii.gz",
    # MEDN+Nuis-Reg
    "nuisance-regressions/{sub}/func/{prefix}_desc-NuisReg_bold.nii.gz",
    # MEDN Noise
    "tedana/{sub}/func/{prefix}_desc-optcomDenoised_errorts.nii.gz",
]
PHYSIO_TARGET_FILES = [
    # MEDN+RV-Reg
    "nuisance-regressions/{sub}/func/{prefix}_desc-RVReg_bold.nii.gz",
    # MEDN+RVT-Reg
    "nuisance-regressions/{sub}/func/{prefix}_desc-RVTReg_bold.nii.gz",
]


def transform_to_standard_space(
    native_space_target_file,
    template,
    out_file,
    xform_native_to_t1w,
    xform_t1w_to_std,
):
    print(f"Transforming {native_space_target_file}", flush=True)
    str_ = (
        f"antsApplyTransforms -e 3 -d 4 -i {native_space_target_file} "
        f"-r {template} -o {out_file} "
        f"-n LanczosWindowedSinc -t {xform_native_to_t1w} {xform_t1w_to_std}"
    )
    try:
        run_command(str_)
    except Exception as exc:
        print(exc, flush=True)


def main(project_dir, dset, subject):
    deriv_dir = op.join(project_dir, dset, "derivatives")
    tedana_dir = op.join(deriv_dir, "tedana")
    tedana_subj_dir = op.join(tedana_dir, subject, "func")
    fmriprep_subj_dir = op.join(deriv_dir, "fmriprep", subject)

    # Find native-space map to break down into prefix
    medn_files = glob(op.join(tedana_subj_dir, "*_desc-optcomDenoised_bold.nii.gz"))
    medn_files = [f for f in medn_files if "space-MNI152NLin6Asym" not in f]
    assert len(medn_files) == 1, medn_files
    medn_file = medn_files[0]
    medn_name = op.basename(medn_file)
    prefix = medn_name.split("desc-")[0].rstrip("_")

    # Find BOLD-resolution, standard-space map to use as target in transform
    standard_space_file = op.join(
        fmriprep_subj_dir,
        "func",
        f"{prefix}_space-MNI152NLin6Asym_boldref.nii.gz",
    )
    assert op.isfile(standard_space_file), f"{standard_space_file} DNE!"

    # Identify transform files
    xform_native_to_t1w = op.join(
        fmriprep_subj_dir,
        "func",
        f"{prefix}_from-scanner_to-T1w_mode-image_xfm.txt",
    )
    assert op.isfile(xform_native_to_t1w), f"{xform_native_to_t1w} DNE!"

    xform_t1w_to_std = op.join(
        fmriprep_subj_dir,
        "anat",
        f"{subject}_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5",
    )
    assert op.isfile(xform_t1w_to_std), f"{xform_t1w_to_std} DNE!"

    # Loop through target files and apply xforms
    target_files = TARGET_FILES[:]
    if dset == "dset-dupre":
        target_files += PHYSIO_TARGET_FILES

    for native_space_target_file in target_files:
        temp_name = native_space_target_file.format(sub=subject, prefix=prefix)
        if "space-scanner" in temp_name:
            out_name = temp_name.replace("space-scanner", "space-MNI152NLin6Asym")

        elif "_res-" in temp_name:
            # Some may have res (which is closer to space in the order),
            # but others may only have desc, so we check both and add space
            # before the first one we find.
            # If neither entity is found, we raise an Exception.
            temp1, temp2 = temp_name.split("_res-")
            out_name = temp1 + "_space-MNI152NLin6Asym_res-" + temp2

        elif "_desc-" in temp_name:
            temp1, temp2 = temp_name.split("_desc-")
            out_name = temp1 + "_space-MNI152NLin6Asym_desc-" + temp2

        else:
            # Just add it right before the suffix.
            split_name = temp_name.split("_")
            split_name.insert(-1, "space-MNI152NLin6Asym")
            out_name = "_".join(split_name)

        out_file = op.join(deriv_dir, out_name)
        full_native_space_file = op.join(deriv_dir, temp_name)

        transform_to_standard_space(
            full_native_space_file,
            standard_space_file,
            out_file,
            xform_native_to_t1w,
            xform_t1w_to_std,
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
