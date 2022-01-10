"""Experiment 1, Analysis Group 4.

Characterizing the relationship between heart rate and global BOLD signal with and without
denoising.

HRV correlated with SD of mean cortical signal from:
- TE30
- FIT-R2
- MEDN
"""
import os.path as op
import sys

import numpy as np
import pandas as pd
from nilearn import masking

sys.path.append("..")

from utils import get_prefixes, get_target_files, pearson_r  # noqa: E402


def correlate_hrv_with_cortical_sd(
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
    ALPHA = 0.05

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value
    participants_df = participants_df.dropna(subset="rpv")
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")
    prefix = get_prefixes()["dset-dupre"]

    participants_df["TE30"] = np.nan
    participants_df["FIT-R2"] = np.nan
    participants_df["MEDN"] = np.nan
    participants_df["hrv"] = np.nan
    for i, row in participants_df.iterrows():
        participant_id = row["participant_id"]
        confounds_file = confounds_pattern.format(participant_id=participant_id)
        assert op.isfile(confounds_file), f"{confounds_file} DNE"

        confounds_df = pd.read_table(confounds_file)
        participants_df.loc[i, "hrv"] = np.std(confounds_df["IHRRegression_IHR_Interpolated"])

        mask = mask_pattern.format(participant_id=participant_id)

        for filetype, pattern in target_file_patterns.items():
            filename = pattern.format(participant_id=participant_id, prefix=prefix)
            cortical_signal = masking.apply_mask(filename, mask)
            mean_cortical_signal = np.mean(cortical_signal, axis=0)
            participants_df.loc[i, filetype] = np.std(mean_cortical_signal)

    for filetype in target_file_patterns.keys():
        # We are performing a one-sided test to determine if the correlation is
        # statistically significant (alpha = 0.05) and positive.
        corr, p = pearson_r(
            participants_df["hrv"], participants_df["mean_fd"], alternative="greater"
        )
        if p <= ALPHA:
            print(
                "ANALYSIS 1: HRV and standard deviation of mean cortical signal "
                "were found to be positively and statistically significantly correlated, "
                f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
            )
        else:
            print(
                "ANALYSIS 1: HRV and standard deviation of mean cortical signal "
                "were not found to be statistically significantly correlated, "
                f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
            )


if __name__ == "__main__":
    print("Experiment 1, Analysis Group 4")
    in_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/dset-dupre/"
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
    target_file_patterns = {k: v for k, v in TARGET_FILE_PATTERNS.items() if k in TARGETS}
    target_file_patterns = {
        k: op.join(in_dir, "derivatives", v) for k, v in target_file_patterns.items()
    }

    correlate_hrv_with_cortical_sd(
        participants_file,
        target_file_patterns,
        mask_pattern,
        confounds_pattern,
    )
