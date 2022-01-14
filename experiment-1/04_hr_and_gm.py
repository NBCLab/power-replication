"""Experiment 1, Analysis Group 4.

Characterizing the relationship between heart rate and global BOLD signal with and without
denoising.

HRV correlated with SD of mean cortical signal from:
- TE30
- FIT-R2
- MEDN
"""
import os
import os.path as op
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import masking

sys.path.append("..")

from utils import get_prefixes, get_target_files, pearson_r  # noqa: E402


def correlate_hrv_with_cortical_sd(
    project_dir,
    participants_file,
    target_file_patterns,
    mask_pattern,
    confounds_pattern,
):
    """Perform analysis 1.

    Correlate HRV with standard deviation of mean cortical signal from each of the derivatives,
    across participants.
    Perform one-sided test of significance on correlation coefficient to determine if HRV is
    significantly correlated with SD of mean cortical signal after each denoising approach.
    """
    print("Experiment 1, Analysis Group 4, Analysis 1", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment01_group04")
    os.makedirs(out_dir, exist_ok=True)

    ALPHA = 0.05

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Drop excluded subjects
    participants_df = participants_df.loc[participants_df["exclude"] != 1]
    prefix = get_prefixes()["dset-dupre"]

    participants_df["hrv"] = np.nan
    for col in target_file_patterns.keys():
        participants_df[col] = np.nan

    for i, row in participants_df.iterrows():
        participant_id = row["participant_id"]
        subj_prefix = prefix.format(participant_id=participant_id)

        confounds_file = confounds_pattern.format(participant_id=participant_id)
        assert op.isfile(confounds_file), f"{confounds_file} DNE"

        confounds_df = pd.read_table(confounds_file)
        if "IHRRegression_IHR_Interpolated" not in confounds_df.columns:
            print(f"{participant_id} has bad cardiac data. Skipping.")
            continue

        participants_df.loc[i, "hrv"] = np.std(
            confounds_df["IHRRegression_IHR_Interpolated"]
        )

        mask = mask_pattern.format(participant_id=participant_id)

        for filetype, pattern in target_file_patterns.items():
            filename = pattern.format(participant_id=participant_id, prefix=subj_prefix)
            cortical_signal = masking.apply_mask(filename, mask)
            mean_cortical_signal = np.mean(cortical_signal, axis=0)
            participants_df.loc[i, filetype] = np.std(mean_cortical_signal)

    participants_df = participants_df.dropna(subset=["hrv"])
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    for filetype in target_file_patterns.keys():
        # We are performing a one-sided test to determine if the correlation is
        # statistically significant (alpha = 0.05) and positive.
        corr, p = pearson_r(
            participants_df["hrv"], participants_df[filetype], alternative="greater"
        )
        if p <= ALPHA:
            print(
                f"\tHRV and standard deviation of mean cortical {filetype} signal "
                "were found to be positively and statistically significantly correlated, "
                f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
            )
        else:
            print(
                f"\tHRV and standard deviation of mean cortical {filetype} signal "
                "were not found to be statistically significantly correlated, "
                f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
            )

        g = sns.JointGrid(data=participants_df, x="hrv", y=filetype)
        g.plot(sns.regplot, sns.histplot)
        g.savefig(op.join(out_dir, f"analysis_01_{filetype}.png"), dpi=400)

    participants_df.to_csv(
        op.join(out_dir, "analysis_01_results.tsv"), sep="\t", index=False
    )


if __name__ == "__main__":
    print("Experiment 1, Analysis Group 4")
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    in_dir = op.join(project_dir, "dset-dupre/")
    participants_file = op.join(in_dir, "participants.tsv")
    confounds_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/func",
        "{participant_id}_task-rest_run-1_desc-confounds_timeseries.tsv",
    )
    mask_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/anat",
        "{participant_id}_space-scanner_res-bold_label-CGM_mask.nii.gz",
    )
    TARGET_FILE_PATTERNS = get_target_files()
    TARGETS = ["TE30", "FIT-R2", "MEDN"]
    target_file_patterns = {
        k: op.join(in_dir, "derivatives", v)
        for k, v in TARGET_FILE_PATTERNS.items()
        if k in TARGETS
    }

    correlate_hrv_with_cortical_sd(
        project_dir,
        participants_file,
        target_file_patterns,
        mask_pattern,
        confounds_pattern,
    )
