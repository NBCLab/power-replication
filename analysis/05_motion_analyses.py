"""Run distance-dependent motion-related artifact analyses.

The rank for the intercept (smoothing curve at 35mm) indexes general dependence
on motion (i.e., a mix of global and focal effects), while the rank for the
slope (difference in smoothing curve at 100mm and 35mm) indexes distance
dependence (i.e., focal effects).

QC:RSFC analysis (QC=FD) for OC, MEDN, MEDN S0, MEDN+GODEC, and MEDN+GSR data
- Figure 4, Figure 5
- To expand with MEDN+dGSR and MEDN+MIR.

High-low motion analysis for OC, MEDN, MEDN S0, MEDN+GODEC, and MEDN+GSR data
- Figure 4
- To expand with MEDN+dGSR and MEDN+MIR.

Scrubbing analysis for OC, MEDN, MEDN S0, MEDN+GODEC, and MEDN+GSR data
- Figure 4
- To expand with MEDN+dGSR and MEDN+MIR.

QC:RSFC analysis (QC=RPV) for OC, MEDN, MEDN S0, MEDN+GODEC, and MEDN+GSR data
- Figure 5
- To expand with MEDN+dGSR and MEDN+MIR.

QC:RSFC analysis (QC=FD) with censored (FD thresh = 0.2) timeseries for OC,
MEDN, MEDN+GODEC, MEDN+GSR, MEDN+RPCA, and MEDN+CompCor data
- Figure S10 (MEDN, MEDN+GODEC, MEDN+GSR)
- Figure S13 (OC, MEDN, MEDN+GODEC, MEDN+GSR, MEDN+RPCA, MEDN+CompCor)
- To expand with MEDN+dGSR and MEDN+MIR.

High-low motion analysis with censored (FD thresh = 0.2) timeseries for MEDN,
MEDN+GODEC, and MEDN+GSR data
- Figure S10
- To expand with MEDN+dGSR and MEDN+MIR.
"""
import argparse
import os.path as op

from ddmra import run_analyses

TARGET_FILES = {
    "TE30": "power/{sub}/func/{prefix}_desc-TE30_bold.nii.gz",
    "OC": "tedana/{sub}/func/{prefix}_desc-optcom_bold.nii.gz",
    "MEDN": "tedana/{sub}/func/{prefix}_desc-optcomDenoised_bold.nii.gz",
    "MEDN+MIR": "tedana/{sub}/func/{prefix}_desc-optcomMIRDenoised_bold.nii.gz",
    "MEDN+MIR Noise": "tedana/{sub}/func/{prefix}_desc-optcomMIRDenoised_errorts.nii.gz",
    "FIT-R2": "t2smap/{sub}/func/{prefix}_T2starmap.nii.gz",
    "FIT-S0": "t2smap/{sub}/func/{prefix}_S0map.nii.gz",
    "MEDN+GODEC (sparse)": "godec/{sub}/func/{prefix}_desc-GODEC_rank-4_bold.nii.gz",
    "MEDN+GODEC Noise (lowrank)": "godec/{sub}/func/{prefix}_desc-GODEC_rank-4_lowrankts.nii.gz",
    "MEDN+dGSR": "rapidtide/{sub}/func/{prefix}_desc-lfofilterCleaned_bold.nii.gz",
    "MEDN+dGSR Noise": "rapidtide/{sub}/func/{prefix}_desc-lfofilterCleaned_errorts.nii.gz",
    "MEDN+aCompCor": "nuisance-regressions/{sub}/func/{prefix}_desc-aCompCor_bold.nii.gz",
    "MEDN+aCompCor Noise": "nuisance-regressions/{sub}/func/{prefix}_desc-aCompCor_errorts.nii.gz",
    "MEDN+GSR": "nuisance-regressions/{sub}/func/{prefix}_desc-GSR_bold.nii.gz",
    "MEDN+GSR Noise": "nuisance-regressions/{sub}/func/{prefix}_desc-GSR_errorts.nii.gz",
    "MEDN+Nuis-Reg": "nuisance-regressions/{sub}/func/{prefix}_desc-NuisReg_bold.nii.gz",
    "MEDN Noise": "tedana/{sub}/func/{prefix}_desc-optcomDenoised_errorts.nii.gz",
}
PHYSIO_TARGET_FILES = {
    "MEDN+RV-Reg": "nuisance-regressions/{sub}/func/{prefix}_desc-RVReg_bold.nii.gz",
    "MEDN+RVT-Reg": "nuisance-regressions/{sub}/func/{prefix}_desc-RVTReg_bold.nii.gz",
}


def main(project_dir, dset, in_type):
    deriv_dir = op.join(project_dir, dset, "derivatives")
    tedana_dir = op.join(deriv_dir, "tedana")
    tedana_subj_dir = op.join(tedana_dir, subject, "func")
    fmriprep_subj_dir = op.join(deriv_dir, "fmriprep", subject)

    file_pattern = TARGET_FILES[in_type]
    run_analyses(file_pattern)


def _get_parser():
    parser = argparse.ArgumentParser(description="Grab cell from TSV file.")
    parser.add_argument(
        "--dset",
        dest="dset",
        required=True,
        help="Dataset name.",
    )
    parser.add_argument(
        "--in_type",
        dest="in_type",
        required=True,
        help="Input name.",
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
