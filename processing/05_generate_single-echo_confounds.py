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
import os.path as op
from glob import glob

import numpy as np
import pandas as pd
import sklearn
from nilearn import image, masking


def compile_nuisance_regressors(
    medn_file,
    mask_file,
    seg_file,
    cgm_file,
    confounds_file,
):
    """Generate regressors for aCompCor, GSR, and the nuisance model and write out to file."""
    confounds_json = confounds_file.replace(".tsv", ".json")

    # #########
    # Load data
    # #########
    confounds_df = pd.read_table(confounds_file)
    with open(confounds_json, "r") as fo:
        confounds_metadata = json.load(fo)

    # #########################
    # Nuisance Regression Model
    # #########################
    # Extract white matter and CSF signals for nuisance regression
    print("\tnuisance", flush=True)
    wm_img = image.math_img("img == 6", img=seg_file)
    wm_img = image.math_img(
        "wm_mask * brain_mask",
        wm_mask=wm_img,
        brain_mask=mask_file,
    )
    wm_data = masking.apply_mask(medn_file, wm_img)

    csf_img = image.math_img("img == 8", img=seg_file)
    csf_img = image.math_img(
        "csf_mask * brain_mask",
        csf_mask=csf_img,
        brain_mask=mask_file,
    )
    csf_data = masking.apply_mask(medn_file, csf_img)

    confounds_df["NuisanceRegression_WhiteMatter"] = np.mean(wm_data, axis=1)
    confounds_df["NuisanceRegression_CerebrospinalFluid"] = np.mean(csf_data, axis=1)
    confounds_metadata["NuisanceRegression_WhiteMatter"] = {
        "Sources": [medn_file, seg_file, mask_file],
        "Description": "Mean signal from deepest white matter mask.",
    }
    confounds_metadata["NuisanceRegression_CerebrospinalFluid"] = {
        "Sources": [medn_file, seg_file, mask_file],
        "Description": "Mean signal from deepest cerebrospinal mask.",
    }

    # ##############
    # aCompCor Model
    # ##############
    # Extract and run PCA on white matter for aCompCor
    print("\taCompCor", flush=True)
    wm_img = image.math_img("img == 6", img=seg_file)
    wm_img = image.math_img(
        "wm_mask * brain_mask",
        wm_mask=wm_img,
        brain_mask=mask_file,
    )
    wm_data = masking.apply_mask(medn_file, wm_img)
    pca = sklearn.decomposition.PCA(n_components=5)
    acompcor_components = pca.fit_transform(wm_data)
    acompcor_columns = [
        "aCompCorRegression_Component00",
        "aCompCorRegression_Component01",
        "aCompCorRegression_Component02",
        "aCompCorRegression_Component03",
        "aCompCorRegression_Component04",
    ]
    confounds_df[acompcor_columns] = acompcor_components
    temp_dict = {
        "aCompCorRegression_Component00": {
            "Sources": [medn_file, seg_file, mask_file],
            "Description": "PCA performed on signal from deepest white matter mask.",
        },
        "aCompCorRegression_Component01": {
            "Sources": [medn_file, seg_file, mask_file],
            "Description": "PCA performed on signal from deepest white matter mask.",
        },
        "aCompCorRegression_Component02": {
            "Sources": [medn_file, seg_file, mask_file],
            "Description": "PCA performed on signal from deepest white matter mask.",
        },
        "aCompCorRegression_Component03": {
            "Sources": [medn_file, seg_file, mask_file],
            "Description": "PCA performed on signal from deepest white matter mask.",
        },
        "aCompCorRegression_Component04": {
            "Sources": [medn_file, seg_file, mask_file],
            "Description": "PCA performed on signal from deepest white matter mask.",
        },
    }
    confounds_metadata = {**temp_dict, **confounds_metadata}

    # ##############################
    # Global Signal Regression Model
    # ##############################
    # Extract mean cortical signal for GSR regression
    print("\tGSR", flush=True)
    cgm_mask = image.math_img(
        "cgm_mask * brain_mask",
        cgm_mask=cgm_file,
        brain_mask=mask_file,
    )
    gsr_signal = masking.apply_mask(medn_file, cgm_mask)
    gsr_signal = np.mean(gsr_signal, axis=1)
    confounds_df["GSRRegression_CorticalRibbon"] = gsr_signal
    confounds_metadata["GSRRegression_CorticalRibbon"] = {
        "Sources": [medn_file, cgm_file, mask_file],
        "Description": "Mean signal from cortical gray matter mask.",
    }

    confounds_df.to_csv(confounds_file, sep="\t", index=False)
    with open(confounds_json, "w") as fo:
        json.dump(confounds_metadata, fo, sort_keys=True, indent=4)

    return confounds_file


def main(project_dir, dset, subject):
    """Run the confound-generation workflow.

    Notes
    -----
    Example physio file: /home/data/nbc/misc-projects/Salo_PowerReplication/dset-dupre/sub-01/\
        func/sub-01_task-rest_run-01_physio.tsv.gz
    Example physio metadata file: /home/data/nbc/misc-projects/Salo_PowerReplication/dset-dupre/\
        sub-01/sub-01_task-rest_physio.json
    """
    dset_dir = op.join(project_dir, dset)
    deriv_dir = op.join(dset_dir, "derivatives")
    tedana_dir = op.join(deriv_dir, "tedana")
    preproc_dir = op.join(deriv_dir, "power")

    preproc_subj_func_dir = op.join(preproc_dir, subject, "func")
    preproc_subj_anat_dir = op.join(preproc_dir, subject, "anat")
    tedana_subj_dir = op.join(tedana_dir, subject, "func")

    # Collect important files
    confounds_files = glob(
        op.join(
            preproc_subj_func_dir,
            "*_desc-confounds_timeseries.tsv",
        )
    )
    assert len(confounds_files) == 1
    confounds_file = confounds_files[0]

    seg_files = glob(
        op.join(
            preproc_subj_anat_dir,
            "*_space-scanner_res-bold_desc-totalMaskWithCSF_dseg.nii.gz",
        )
    )
    assert len(seg_files) == 1
    seg_file = seg_files[0]

    cgm_files = glob(
        op.join(preproc_subj_anat_dir, "*_space-scanner_res-bold_label-CGM_mask.nii.gz")
    )
    assert len(cgm_files) == 1
    cgm_file = cgm_files[0]

    medn_files = glob(op.join(tedana_subj_dir, "*_desc-optcomDenoised_bold.nii.gz"))
    assert len(medn_files) == 1
    medn_file = medn_files[0]

    mask_files = glob(op.join(tedana_subj_dir, "*_desc-goodSignal_mask.nii.gz"))
    assert len(mask_files) == 1
    mask_file = mask_files[0]

    # Generate and compile nuisance regressors for aCompCor, GSR, and the nuisance model
    compile_nuisance_regressors(
        medn_file,
        mask_file,
        seg_file,
        cgm_file,
        confounds_file,
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
