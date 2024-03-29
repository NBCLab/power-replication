"""
Perform multi-echo denoising strategies with tedana.

This includes tedana, MIR, and FIT (via t2smap).
Additionally, we create the TE30 files in this script as well.
"""
import argparse
import json
import logging
import os
import os.path as op
from glob import glob
from shutil import copyfile

import numpy as np
import pandas as pd
from nilearn import image
from tedana import workflows

LGR = logging.getLogger(__file__)


def run_tedana(project_dir, dset, subject):
    """Run multi-echo denoising workflows.

    Run the following tedana workflows:
    - tedana + fittype=curvefit + gscontrol=mir
    - t2smap + fittype=curvefit + fitmode=all

    Notes
    -----
    Should be run *after* fmri_process.py.
    """
    dset_dir = op.join(project_dir, dset)
    deriv_dir = op.join(dset_dir, "derivatives")
    preproc_dir = op.join(deriv_dir, "power")
    t2smap_dir = op.join(deriv_dir, "t2smap")
    tedana_dir = op.join(deriv_dir, "tedana")

    # Get list of participants with good data,
    # just to determine if the subject is the first in the dataset
    participants_file = op.join(dset_dir, "participants.tsv")
    participants_df = pd.read_table(participants_file)
    subjects = participants_df.loc[
        participants_df["exclude"] == 0, "participant_id"
    ].tolist()
    first_subject = subjects[0]

    preproc_subj_dir = op.join(preproc_dir, subject)
    preproc_subj_anat_dir = op.join(preproc_subj_dir, "anat")
    preproc_subj_func_dir = op.join(preproc_subj_dir, "func")
    t2smap_subj_dir = op.join(t2smap_dir, subject, "func")
    tedana_subj_dir = op.join(tedana_dir, subject, "func")
    os.makedirs(t2smap_subj_dir, exist_ok=True)
    os.makedirs(tedana_subj_dir, exist_ok=True)

    preproc_files = sorted(
        glob(op.join(preproc_subj_func_dir, f"{subject}_*_desc-NSSRemoved_bold.nii.gz"))
    )

    # Get prefix from first filename
    first_file = preproc_files[0]
    first_file = op.basename(first_file)
    prefix = first_file.split("_echo")[0]

    # Collect metadata
    json_files = [f.replace(".nii.gz", ".json") for f in preproc_files]
    echo_times, raw_sources = [], []
    for json_file in json_files:
        with open(json_file, "r") as fo:
            metadata = json.load(fo)
            raw_sources += metadata["RawSources"]
            echo_times.append(metadata["EchoTime"] * 1000)

    # Set combined metadata fields (i.e., fields using info from all preproc files)
    metadata["RawSources"] = raw_sources

    # Collect the TE30 files
    TARGET_TE = 30
    echo_times = np.asarray(echo_times)
    te30_idx = (np.abs(echo_times - TARGET_TE)).argmin()
    te30_nii_file = preproc_files[te30_idx]
    te30_json_file = json_files[te30_idx]
    te30_out_nii_file = op.join(
        preproc_subj_func_dir, f"{prefix}_desc-TE30_bold.nii.gz"
    )
    te30_out_json_file = op.join(preproc_subj_func_dir, f"{prefix}_desc-TE30_bold.json")

    with open(te30_json_file, "r") as fo:
        te30_metadata = json.load(fo)

    te30_metadata["Sources"] = [te30_nii_file]
    te30_metadata[
        "Description"
    ] = "Preprocessed data from echo closest to 30ms, with non-steady-state volumes removed."

    with open(te30_out_json_file, "w") as fo:
        json.dump(te30_metadata, fo, sort_keys=True, indent=4)

    copyfile(te30_nii_file, te30_out_nii_file)

    # Remove EchoTime for future use
    metadata.pop("EchoTime")

    # Derive brain mask from discrete segmentation
    dseg_file = op.join(
        preproc_subj_anat_dir,
        f"{subject}_space-scanner_res-bold_desc-totalMaskWithCSF_dseg.nii.gz",
    )
    mask_img = image.math_img("img >= 1", img=dseg_file)

    # Add Sources to metadata that will be reused for all relevant t2smap/tedana outputs
    metadata["Sources"] = preproc_files + [dseg_file]

    # ############
    # FIT denoised
    # ############
    # We retain t2s and s0 timeseries from this method, but do not use
    # optcom or any other derivatives.
    print("\tt2smap", flush=True)
    workflows.t2smap_workflow(
        preproc_files,
        echo_times,
        combmode="t2s",
        fitmode="ts",
        fittype="loglin",
        mask=mask_img,  # The docs say str, but workflows should work fine with an img
        out_dir=t2smap_subj_dir,
        prefix=prefix,
    )

    # Merge metadata into FIT T2/S0 jsons
    SUFFIXES = {
        "T2starmap": "Volume-wise T2* estimated with tedana's t2smap workflow.",
        "S0map": "Volume-wise S0 estimated with tedana's t2smap workflow.",
    }
    for suffix, description in SUFFIXES.items():
        nii_file = op.join(t2smap_subj_dir, f"{prefix}_{suffix}.nii.gz")
        assert op.isfile(nii_file)

        suff_json_file = op.join(t2smap_subj_dir, f"{prefix}_{suffix}.json")
        metadata["Description"] = description
        with open(suff_json_file, "w") as fo:
            json.dump(metadata, fo, sort_keys=True, indent=4)

    t2smap_data_desc = op.join(t2smap_subj_dir, f"{prefix}_dataset_description.json")

    if subject == first_subject:
        # Merge dataset descriptions
        preproc_data_desc = op.join(preproc_dir, "dataset_description.json")
        out_data_desc = op.join(t2smap_dir, "dataset_description.json")

        with open(preproc_data_desc, "r") as fo:
            data_description = json.load(fo)

        with open(t2smap_data_desc, "r") as fo:
            ted_data_description = json.load(fo)

        data_description["Name"] = ted_data_description["Name"]
        data_description["BIDSVersion"] = ted_data_description["BIDSVersion"]
        data_description["GeneratedBy"] = (
            ted_data_description["GeneratedBy"] + data_description["GeneratedBy"]
        )

        with open(out_data_desc, "w") as fo:
            json.dump(data_description, fo, sort_keys=True, indent=4)

    # Remove subject-level dataset descriptions
    os.remove(t2smap_data_desc)

    # ############
    # TEDANA + MIR
    # ############
    # We use MEDN, reconstructed MEDN-noise, MEHK,
    # reconstructed MEHK-noise, optcom, mmix (component timeseries), and
    # comptable (classifications of components)
    print("\ttedana", flush=True)
    workflows.tedana_workflow(
        preproc_files,
        echo_times,
        fittype="curvefit",
        gscontrol="mir",
        tedpca="aic",  # least aggressive maPCA option
        maxit=500,
        maxrestart=100,
        mask=mask_img,  # The docs say str, but workflows should work fine with an img
        out_dir=tedana_subj_dir,
        prefix=prefix,
    )

    # Derive binary mask from adaptive mask
    adaptive_mask = op.join(
        tedana_subj_dir, f"{prefix}_desc-adaptiveGoodSignal_mask.nii.gz"
    )
    updated_mask = image.math_img("img >= 1", img=adaptive_mask)
    updated_mask.to_filename(
        op.join(tedana_subj_dir, f"{prefix}_desc-goodSignal_mask.nii.gz")
    )

    # Derive noise time series for MEDN and MEDN+MIR files
    optcom_file = op.join(tedana_subj_dir, f"{prefix}_desc-optcom_bold.nii.gz")

    medn_file = op.join(tedana_subj_dir, f"{prefix}_desc-optcomDenoised_bold.nii.gz")
    medn_noise_file = op.join(
        tedana_subj_dir, f"{prefix}_desc-optcomDenoised_errorts.nii.gz"
    )
    medn_noise_img = image.math_img("img1 - img2", img1=optcom_file, img2=medn_file)
    medn_noise_img.to_filename(medn_noise_file)

    medn_mir_file = op.join(
        tedana_subj_dir, f"{prefix}_desc-optcomMIRDenoised_bold.nii.gz"
    )
    medn_mir_noise_file = op.join(
        tedana_subj_dir, f"{prefix}_desc-optcomMIRDenoised_errorts.nii.gz"
    )
    medn_mir_noise_img = image.math_img(
        "img1 - img2", img1=optcom_file, img2=medn_mir_file
    )
    medn_mir_noise_img.to_filename(medn_mir_noise_file)

    # Merge metadata into relevant jsons
    SUFFIXES = {
        "desc-goodSignal_mask": "Mask of voxels with good signal in at least the first echo.",
        "desc-optcomDenoised_bold": "Multi-echo denoised data from tedana.",
        "desc-optcom_bold": "Optimally combined data from tedana.",
        "desc-optcomMIRDenoised_bold": (
            "Multi-echo denoised data, further denoised with minimum image regression."
        ),
        "desc-optcomDenoised_errorts": (
            "Noise time series retained from multi-echo denoising the optimally combined data."
        ),
        "desc-optcomMIRDenoised_errorts": (
            "Noise time series retained from multi-echo denoising (with minimum image regression) "
            "the optimally combined data."
        ),
    }
    for suffix, description in SUFFIXES.items():
        nii_file = op.join(tedana_subj_dir, f"{prefix}_{suffix}.nii.gz")
        if not op.isfile(nii_file):
            LGR.warning(f"File not found: {suffix}")

        suff_json_file = op.join(tedana_subj_dir, f"{prefix}_{suffix}.json")
        metadata["Description"] = description
        with open(suff_json_file, "w") as fo:
            json.dump(metadata, fo, sort_keys=True, indent=4)

    tedana_data_desc = op.join(tedana_subj_dir, f"{prefix}_dataset_description.json")

    if subject == first_subject:
        # Merge dataset descriptions
        preproc_data_desc = op.join(preproc_dir, "dataset_description.json")
        out_data_desc = op.join(tedana_dir, "dataset_description.json")

        with open(preproc_data_desc, "r") as fo:
            data_description = json.load(fo)

        with open(tedana_data_desc, "r") as fo:
            ted_data_description = json.load(fo)

        data_description["Name"] = ted_data_description["Name"]
        data_description["BIDSVersion"] = ted_data_description["BIDSVersion"]
        data_description["GeneratedBy"] = (
            ted_data_description["GeneratedBy"] + data_description["GeneratedBy"]
        )

        with open(out_data_desc, "w") as fo:
            json.dump(data_description, fo, sort_keys=True, indent=4)

    # Remove subject-level dataset descriptions
    os.remove(tedana_data_desc)


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
    run_tedana(project_dir=project_dir, **kwargs)


if __name__ == "__main__":
    print(__file__, flush=True)
    _main()
