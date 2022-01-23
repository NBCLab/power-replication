"""Experiment 1, Analysis Group 5.

Characterizing the relationship between respiration and functional connectivity with and without
denoising.

QC:RSFC and High-Low Motion DDMRA analyses performed on:
- OC
- MEDN Noise
- MEDN
- MEDN+GODEC
- MEDN+Nuis-Reg
- MEDN+RVT-Reg
- MEDN+RV-Reg
- MEDN+aCompCor
- MEDN+dGSR
- MEDN+MIR
- MEDN+GSR

with each of the following used as the quality measure:
- RPV
- mean RV
- mean RVT
"""
import os
import os.path as op
import sys

import numpy as np
import pandas as pd
from ddmra import run_analyses

sys.path.append("..")

from utils import get_prefixes_mni, get_target_files  # noqa: E402


def run_ddmra_of_rpv(project_dir, participants_file, target_file_patterns):
    """Run QC:RSFC and high-low analyses on derivatives against RPV."""
    print("Experiment 1, Analysis Group 5, Analysis 1", flush=True)
    out_dir = op.join(project_dir, "analyses/experiment01_group05_analysis01")

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value
    participants_df = participants_df.dropna(subset=["rpv"])
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")
    prefix = get_prefixes_mni()["dset-dupre"]

    rpv_confounds = participants_df["rpv"].values

    for filetype, pattern in target_file_patterns.items():
        print(f"\t{filetype}", flush=True)
        filetype_out_dir = op.join(out_dir, filetype.replace(" ", "_"))
        os.makedirs(filetype_out_dir, exist_ok=True)
        target_files = []
        for i, row in participants_df.iterrows():
            participant_id = row["participant_id"]
            subj_prefix = prefix.format(participant_id=participant_id)

            filename = pattern.format(participant_id=participant_id, prefix=subj_prefix)
            target_files.append(filename)

        run_analyses(
            target_files,
            rpv_confounds,
            out_dir=filetype_out_dir,
            n_iters=10000,
            n_jobs=4,
            qc_thresh=0.2,
            analyses=("qcrsfc", "highlow")
        )


def run_ddmra_of_mean_rv(
    project_dir, participants_file, confounds_pattern, target_file_patterns
):
    """Run QC:RSFC and high-low analyses on derivatives against mean RV."""
    print("Experiment 1, Analysis Group 5, Analysis 2", flush=True)
    out_dir = op.join(project_dir, "analyses/experiment01_group05_analysis02")

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value
    participants_df = participants_df.dropna(subset=["rpv"])
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")
    prefix = get_prefixes_mni()["dset-dupre"]

    rv_confounds = []
    for i, row in participants_df.iterrows():
        participant_id = row["participant_id"]
        subj_prefix = prefix.format(participant_id=participant_id)

        confounds_file = confounds_pattern.format(participant_id=participant_id)
        confounds_df = pd.read_table(confounds_file)

        mean_rv = np.mean(confounds_df["RVRegression_RV"].values)
        rv_confounds.append(mean_rv)

    for filetype, pattern in target_file_patterns.items():
        print(f"\t{filetype}", flush=True)
        filetype_out_dir = op.join(out_dir, filetype.replace(" ", "_"))
        os.makedirs(filetype_out_dir, exist_ok=True)
        target_files = []
        for i, row in participants_df.iterrows():
            participant_id = row["participant_id"]
            subj_prefix = prefix.format(participant_id=participant_id)
            filename = pattern.format(participant_id=participant_id, prefix=subj_prefix)
            target_files.append(filename)

        run_analyses(
            target_files,
            rv_confounds,
            out_dir=filetype_out_dir,
            n_iters=10000,
            n_jobs=4,
            qc_thresh=0.2,
            analyses=("qcrsfc", "highlow"),
        )


def run_ddmra_of_mean_rvt(
    project_dir, participants_file, confounds_pattern, target_file_patterns
):
    """Run QC:RSFC and high-low analyses on derivatives against mean RVT."""
    print("Experiment 1, Analysis Group 5, Analysis 3", flush=True)
    out_dir = op.join(project_dir, "analyses/experiment01_group05_analysis03")

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value
    participants_df = participants_df.dropna(subset=["rpv"])
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")
    prefix = get_prefixes_mni()["dset-dupre"]

    rvt_confounds = []
    for i, row in participants_df.iterrows():
        participant_id = row["participant_id"]
        subj_prefix = prefix.format(participant_id=participant_id)

        confounds_file = confounds_pattern.format(participant_id=participant_id)
        confounds_df = pd.read_table(confounds_file)

        mean_rvt = np.mean(confounds_df["RVTRegression_RVT"].values)
        rvt_confounds.append(mean_rvt)

    for filetype, pattern in target_file_patterns.items():
        print(f"\t{filetype}", flush=True)
        filetype_out_dir = op.join(out_dir, filetype.replace(" ", "_"))
        os.makedirs(filetype_out_dir, exist_ok=True)
        target_files = []
        for i, row in participants_df.iterrows():
            participant_id = row["participant_id"]
            subj_prefix = prefix.format(participant_id=participant_id)
            filename = pattern.format(participant_id=participant_id, prefix=subj_prefix)
            target_files.append(filename)

        run_analyses(
            target_files,
            rvt_confounds,
            out_dir=filetype_out_dir,
            n_iters=10000,
            n_jobs=4,
            qc_thresh=0.2,
            analyses=("qcrsfc", "highlow"),
        )


if __name__ == "__main__":
    print("Experiment 1, Analysis Group 5")
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
        "MEDN+Nuis-Reg",
        "OC",
        "MEDN",
        "MEDN Noise",
        "MEDN+GODEC (sparse)",
        "MEDN+RVT-Reg",
        "MEDN+RV-Reg",
        "MEDN+aCompCor",
        "MEDN+dGSR",
        "MEDN+MIR",
        "MEDN+GSR",
    ]
    target_file_patterns = {
        t: op.join(in_dir, "derivatives", TARGET_FILE_PATTERNS[t])
        for t in TARGETS
    }

    run_ddmra_of_rpv(project_dir, participants_file, target_file_patterns)
    run_ddmra_of_mean_rv(
        project_dir,
        participants_file,
        confounds_pattern,
        target_file_patterns,
    )
    run_ddmra_of_mean_rvt(
        project_dir,
        participants_file,
        confounds_pattern,
        target_file_patterns,
    )
