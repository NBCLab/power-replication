"""
Perform multi-echo denoising strategies with tedana.

This includes tedana, MIR, and FIT (via t2smap).
"""
import json
import os
import os.path as op
from glob import glob

import pandas as pd
import tedana
from nilearn import image


def run_tedana(project_dir, dset):
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

    # Get list of participants with good data
    participants_file = op.join(dset_dir, "participants.tsv")
    participants_df = pd.read_table(participants_file)
    subjects = participants_df.loc[
        participants_df["exclude"] == 0, "participant_id"
    ].tolist()

    for subject in subjects:
        print(f"\t{subject}", flush=True)
        preproc_subj_dir = op.join(preproc_dir, subject)
        preproc_subj_anat_dir = op.join(preproc_subj_dir, "anat")
        preproc_subj_func_dir = op.join(preproc_subj_dir, "func")
        t2smap_subj_dir = op.join(t2smap_dir, subject, "func")
        tedana_subj_dir = op.join(tedana_dir, subject, "func")
        os.makedirs(t2smap_subj_dir, exist_ok=True)
        os.makedirs(tedana_subj_dir, exist_ok=True)

        preproc_files = sorted(
            glob(
                op.join(
                    preproc_subj_func_dir, f"{subject}_*_desc-NSSRemoved_bold.nii.gz"
                )
            )
        )
        json_files = [f.replace(".nii.gz", ".json") for f in preproc_files]
        echo_times = []
        for json_file in json_files:
            with open(json_file, "r") as fo:
                metadata = json.load(fo)
                echo_times.append(metadata["EchoTime"] * 1000)

        # Get prefix from first filename
        first_file = preproc_files[0]
        first_file = op.basename(first_file)
        prefix = first_file.split("_echo")[0]

        # Derive brain mask from discrete segmentation
        dseg_file = op.join(
            preproc_subj_anat_dir,
            f"{subject}_space-T1w_res-bold_desc-totalMaskWithCSF_dseg.nii.gz",
        )
        mask_img = image.math_img("img >= 1", img=dseg_file)

        # ############
        # FIT denoised
        # ############
        # We retain t2s and s0 timeseries from this method, but do not use
        # optcom or any MEICA derivatives.
        print("\t\tt2smap", flush=True)
        tedana.workflows.t2smap_workflow(
            preproc_files,
            echo_times,
            combmode="t2s",
            fitmode="ts",
            fittype="curvefit",
            mask=mask_img,  # The docs say str, but workflows should work fine with an img
            out_dir=t2smap_subj_dir,
            prefix=prefix,
        )
        # TODO: Merge metadata into FIT T2/S0 jsons

        # ############
        # TEDANA + MIR
        # ############
        # We use MEDN, reconstructed MEDN-noise, MEHK,
        # reconstructed MEHK-noise, optcom, mmix (component timeseries), and
        # comptable (classifications of components)
        print("\t\ttedana", flush=True)
        tedana.workflows.tedana_workflow(
            preproc_files,
            echo_times,
            fittype="curvefit",
            gscontrol="mir",
            maxit=500,
            maxrestart=100,
            mask=mask_img,  # The docs say str, but workflows should work fine with an img
            out_dir=tedana_subj_dir,
            prefix=prefix,
        )

        # Derive binary mask from adaptive mask
        adaptive_mask = op.join(
            tedana_subj_dir, prefix + "_desc-adaptiveGoodSignal_mask.nii.gz"
        )
        updated_mask = image.math_img("img >= 1", img=adaptive_mask)
        updated_mask.to_filename(
            op.join(tedana_subj_dir, prefix + "_desc-goodSignal_mask.nii.gz")
        )

        # TODO: Merge metadata into MEDN/OC jsons
        # TODO: Move dataset_description.json to top level and remove from subject folders.


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
        print(dset, flush=True)
        run_tedana(project_dir, dset)
