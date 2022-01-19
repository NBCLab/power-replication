"""Experiment 2, Analysis Group 5.

Evaluating the ability of denoising methods to ameliorate focal and global motion-induced changes
to functional connectivity.

DDMRA analyses on the following inputs:
- OC
- MEDN
- MEDN Noise
- MEDN+GODEC
- MEDN+GSR
- MEDN+GODEC Noise
- MEDN+GSR Noise
- MEDN+dGSR
- MEDN+dGSR Noise
- MEDN+aCompCor
- MEDN+aCompCor Noise
- MEDN+MIR
- MEDN+MIR Noise
"""
import os
import os.path as op
import sys

import pandas as pd
from ddmra import run_analyses

sys.path.append("..")

from utils import get_prefixes_mni  # noqa: E402
from utils import get_target_files  # noqa: E402


def run_ddmra_analyses(
    project_dir,
    participants_file,
    target_file_patterns,
    confounds_pattern,
):
    """Run DDMRA analyses on each of the required derivatives, across datasets."""
    print("Experiment 2, Analysis Group 5, Analysis 1", flush=True)
    out_dir = op.join(project_dir, "analyses/experiment02_group05_analysis01")

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants that aren't excluded
    participants_df = participants_df.loc[participants_df["exclude"] != 1]
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    for filetype, pattern in target_file_patterns.items():
        print(f"\t{filetype}", flush=True)
        filetype_out_dir = op.join(out_dir, filetype.replace(" ", "_"))
        os.makedirs(filetype_out_dir, exist_ok=True)
        target_files, fd_all = [], []
        for i, row in participants_df.iterrows():
            dset_prefix = get_prefixes_mni()[row["dset"]]
            participant_id = row["participant_id"]
            subj_prefix = dset_prefix.format(participant_id=participant_id)

            # Select the target file
            filename = pattern.format(participant_id=participant_id, prefix=subj_prefix)
            target_files.append(filename)

            # Select and load the appropriate confound (FD)
            confounds_file = confounds_pattern.format(
                dset=row["dset"], participant_id=participant_id, prefix=subj_prefix
            )
            confounds_df = pd.read_table(confounds_file)
            fd_all.append(confounds_df["FramewiseDisplacement"].values)

        run_analyses(
            target_files,
            fd_all,
            out_dir=filetype_out_dir,
            n_iters=10000,
            n_jobs=4,
            qc_thresh=0.2,
        )


if __name__ == "__main__":
    print("Experiment 2, Analysis Group 5")
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    in_dir = op.join(project_dir, "dset-dupre/")
    participants_file = op.join(in_dir, "participants.tsv")
    confounds_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/func",
        "{participant_id}_task-rest_run-1_desc-confounds_timeseries.tsv",
    )
    TARGET_FILE_PATTERNS = get_target_files()
    TARGETS = [
        "OC",
        "MEDN",
        "MEDN Noise",
        "MEDN+GODEC",
        "MEDN+GODEC Noise",
        "MEDN+GSR",
        "MEDN+GSR Noise",
        "MEDN+dGSR",
        "MEDN+dGSR Noise",
        "MEDN+aCompCor",
        "MEDN+aCompCor Noise",
        "MEDN+MIR",
        "MEDN+MIR Noise",
    ]
    target_file_patterns = {
        k: op.join(in_dir, "derivatives", v)
        for k, v in TARGET_FILE_PATTERNS.items()
        if k in TARGETS
    }

    run_ddmra_analyses(project_dir, participants_file, target_file_patterns, confounds_pattern)
