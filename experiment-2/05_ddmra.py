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
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from ddmra import run_analyses  # noqa: E402

sys.path.append("..")

from utils import get_bad_subjects_nonphysio  # noqa: E402
from utils import get_prefixes  # noqa: E402
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
    out_dir_outlier = op.join(project_dir, "analyses/experiment02_group05_analysis01_no_outliers")
    out_dir_low_fd = op.join(project_dir, "analyses/experiment02_group05_analysis01_low_fd")

    participants_df = pd.read_table(participants_file)
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
    participants_df = participants_df.reset_index(drop=True)

    for filetype, pattern in target_file_patterns.items():
        print(f"\t{filetype}", flush=True)
        filetype_out_dir = op.join(out_dir, filetype.replace(" ", "_"))
        os.makedirs(filetype_out_dir, exist_ok=True)
        filetype_out_dir_outlier = op.join(out_dir_outlier, filetype.replace(" ", "_"))
        os.makedirs(filetype_out_dir_outlier, exist_ok=True)
        filetype_out_dir_low_fd = op.join(out_dir_low_fd, filetype.replace(" ", "_"))
        os.makedirs(filetype_out_dir_low_fd, exist_ok=True)

        target_files, fd_all = [], []
        target_files_low_fd, fd_all_low_fd = [], []
        for i, row in participants_df.iterrows():
            dset_prefix_mni = get_prefixes_mni()[row["dset"]]
            dset_prefix = get_prefixes()[row["dset"]]
            participant_id = row["participant_id"]
            subj_prefix_mni = dset_prefix_mni.format(participant_id=participant_id)
            subj_prefix = dset_prefix.format(participant_id=participant_id)

            # Select the target file
            filename = pattern.format(
                dset=row["dset"],
                participant_id=participant_id,
                prefix=subj_prefix_mni,
            )
            target_files.append(filename)

            # Select and load the appropriate confound (FD)
            confounds_file = confounds_pattern.format(
                dset=row["dset"], participant_id=participant_id, prefix=subj_prefix
            )
            confounds_df = pd.read_table(confounds_file)
            fd_arr = confounds_df["framewise_displacement"].values
            fd_arr[np.isnan(fd_arr)] = 0
            fd_all.append(fd_arr)

            if np.mean(fd_arr) <= 0.2:
                target_files_low_fd.append(filename)
                fd_all_low_fd.append(fd_all)

        run_analyses(
            target_files,
            fd_all,
            out_dir=filetype_out_dir,
            n_iters=10000,
            n_jobs=4,
            qc_thresh=0.2,
            verbose=True,
            pca_threshold=None,
            outlier_threshold=None,
        )

        run_analyses(
            target_files_low_fd,
            fd_all_low_fd,
            out_dir=filetype_out_dir_low_fd,
            n_iters=10000,
            n_jobs=4,
            qc_thresh=0.2,
            verbose=True,
            pca_threshold=None,
            outlier_threshold=None,
        )

        run_analyses(
            target_files,
            fd_all,
            out_dir=filetype_out_dir_outlier,
            n_iters=10000,
            n_jobs=4,
            qc_thresh=0.2,
            verbose=True,
            pca_threshold=0.95,
            outlier_threshold=0.001,
        )


if __name__ == "__main__":
    print("Experiment 2, Analysis Group 5")
    selected_target = sys.argv[1]
    selected_target = selected_target.replace("_", " ")
    selected_target = selected_target.replace("sparse", "(sparse)")
    selected_target = selected_target.replace("lowrank", "(lowrank)")

    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    in_dir = op.join(project_dir, "{dset}")
    participants_file = op.join(project_dir, "participants.tsv")
    confounds_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/func",
        "{prefix}_desc-confounds_timeseries.tsv",
    )
    TARGET_FILE_PATTERNS = get_target_files()
    TARGETS = [
        "OC",
        "MEDN",
        "MEDN Noise",
        "MEDN+GODEC (sparse)",
        "MEDN+GODEC Noise (lowrank)",
        "MEDN+GSR",
        "MEDN+GSR Noise",
        "MEDN+dGSR",
        "MEDN+dGSR Noise",
        "MEDN+aCompCor",
        "MEDN+aCompCor Noise",
        "MEDN+MIR",
        "MEDN+MIR Noise",
    ]
    assert selected_target in TARGETS, f"{selected_target} not in {', '.join(TARGETS)}"
    target_file_pattern = {
        selected_target: op.join(
            in_dir, "derivatives", TARGET_FILE_PATTERNS[selected_target]
        )
    }

    run_ddmra_analyses(
        project_dir, participants_file, target_file_pattern, confounds_pattern
    )
