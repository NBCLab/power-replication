"""Experiment 2, Analysis Group 1.

Evaluating T2* dependence of global signal.

Carpet plots with motion line plots of:
- OC
- FIT-S0
- FIT-R2
"""
import json
import os
import os.path as op
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd  # noqa: E402

sys.path.append("..")

from utils import _plot_three_carpets_and_confounds  # noqa: E402
from utils import get_bad_subjects_nonphysio  # noqa: E402
from utils import get_prefixes  # noqa: E402
from utils import get_target_files  # noqa: E402


def plot_denoised_and_motion(
    project_dir,
    participants_file,
    oc_pattern,
    fitr2_pattern,
    fits0_pattern,
    dseg_pattern,
    confounds_pattern,
):
    """Generate carpet plots of desired derivatives with motion line plots above."""
    out_dir = op.join(project_dir, "analyses", "experiment02_group01")
    os.makedirs(out_dir, exist_ok=True)

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    participants_df = participants_df.loc[
        participants_df["dset"].isin(["dset-camcan", "dset-cambridge", "dset-dupre"])
    ]
    subjects_to_drop = get_bad_subjects_nonphysio()
    for sub_to_drop in subjects_to_drop:
        participants_df = participants_df.loc[
            ~(
                (participants_df["dset"] == sub_to_drop[0])
                & (participants_df["participant_id"] == sub_to_drop[1])
            )
        ]
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    for i_run, participant_row in participants_df.iterrows():
        subj_id = participant_row["participant_id"]
        dset = participant_row["dset"]
        dset_prefix = get_prefixes()[dset]
        subj_prefix = dset_prefix.format(participant_id=subj_id)
        out_file = op.join(out_dir, f"{dset}_{subj_id}.svg")

        dseg_file = dseg_pattern.format(dset=dset, participant_id=subj_id)
        oc_file = oc_pattern.format(dset=dset, participant_id=subj_id, prefix=subj_prefix)
        fitr2_file = fitr2_pattern.format(dset=dset, participant_id=subj_id, prefix=subj_prefix)
        fits0_file = fits0_pattern.format(dset=dset, participant_id=subj_id, prefix=subj_prefix)
        confounds_file = confounds_pattern.format(
            dset=dset, participant_id=subj_id, prefix=subj_prefix
        )

        # Load and process the data
        metadata_file = oc_file.replace(".nii.gz", ".json")
        with open(metadata_file) as fo:
            t_r = json.load(fo)["RepetitionTime"]

        confounds_df = pd.read_table(confounds_file)

        _plot_three_carpets_and_confounds(
            oc_file,
            fitr2_file,
            fits0_file,
            dseg_file,
            confounds_df,
            t_r,
            out_file,
        )

    print("Done.")


if __name__ == "__main__":
    print("Experiment 2, Analysis Group 1")
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    participants_file = op.join(project_dir, "participants.tsv")
    in_dir = op.join(project_dir, "{dset}")
    target_files = get_target_files()
    oc_pattern = op.join(in_dir, "derivatives", target_files["OC"])
    fitr2_pattern = op.join(in_dir, "derivatives", target_files["FIT-R2"])
    fits0_pattern = op.join(in_dir, "derivatives", target_files["FIT-S0"])
    dseg_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/anat",
        "{participant_id}_space-scanner_res-bold_desc-totalMaskWithCSF_dseg.nii.gz",
    )
    confounds_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/func",
        "{prefix}_desc-confounds_timeseries.tsv",
    )
    plot_denoised_and_motion(
        project_dir,
        participants_file,
        oc_pattern,
        fitr2_pattern,
        fits0_pattern,
        dseg_pattern,
        confounds_pattern,
    )
