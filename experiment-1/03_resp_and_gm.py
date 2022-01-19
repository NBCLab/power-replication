"""Experiment 1, Analysis Group 3.

Characterizing the relationship between respiration and global BOLD signal with and without
denoising.

RPV correlated with SD of mean cortical signal from:
- TE30
- FIT-R2
- MEDN
- MEDN+GODEC
- MEDN+Nuis-Reg
- MEDN+RVT-Reg
- MEDN+RV-Reg
- MEDN+aCompCor
- MEDN+dGSR
- MEDN+MIR
- MEDN+GSR

Plot respiration time series for deep breaths against mean signal from:
- OC
- MEDN
- MEDN+GODEC
- TE30
- FIT-R2
- MEDN+Nuis-Reg
- MEDN+RVT-Reg
- MEDN+RV-Reg
- MEDN+aCompCor
- MEDN+dGSR
- MEDN+MIR
- MEDN+GSR
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


def correlate_rpv_with_cortical_sd(
    project_dir,
    participants_file,
    target_file_patterns,
    mask_pattern,
):
    """Perform analysis 1.

    Correlate RPV with standard deviation of mean cortical signal from each of the derivatives,
    across participants.
    Perform one-sided test of significance on correlation coefficient to determine if RPV is
    significantly, positively correlated with SD of mean cortical signal after each denoising
    approach.
    """
    print("Experiment 1, Analysis Group 3, Analysis 1", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment01_group03")
    os.makedirs(out_dir, exist_ok=True)

    ALPHA = 0.05

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value
    participants_df = participants_df.dropna(subset=["rpv"])
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")
    prefix = get_prefixes()["dset-dupre"]

    for col in target_file_patterns.keys():
        participants_df[col] = np.nan

    for i, row in participants_df.iterrows():
        participant_id = row["participant_id"]
        subj_prefix = prefix.format(participant_id=participant_id)

        mask = mask_pattern.format(participant_id=participant_id)

        for filetype, pattern in target_file_patterns.items():
            filename = pattern.format(participant_id=participant_id, prefix=subj_prefix)
            cortical_signal = masking.apply_mask(filename, mask)
            mean_cortical_signal = np.mean(cortical_signal, axis=1)
            assert mean_cortical_signal.size == cortical_signal.shape[0]
            participants_df.loc[i, filetype] = np.std(mean_cortical_signal)

    for filetype in target_file_patterns.keys():
        # We are performing a one-sided test to determine if the correlation is
        # statistically significant (alpha = 0.05) and positive.
        corr, p = pearson_r(
            participants_df["rpv"], participants_df[filetype], alternative="greater"
        )
        if p <= ALPHA:
            print(
                "\tRPV and standard deviation of mean cortical signal "
                f"of {filetype} data "
                "were found to be positively and statistically significantly correlated, "
                f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
            )
        else:
            print(
                "\tRPV and standard deviation of mean cortical signal "
                f"of {filetype} data "
                "were not found to be statistically significantly correlated, "
                f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
            )

        g = sns.JointGrid(data=participants_df, x="rpv", y=filetype)
        g.plot(sns.regplot, sns.histplot)
        g.savefig(op.join(out_dir, f"analysis_01_{filetype}.png"), dpi=400)

    participants_df.to_csv(
        op.join(out_dir, "analysis_01_results.tsv"), sep="\t", index=False
    )


def plot_deep_breath_cortical_signal(
    participants_file,
    deep_breath_indices,
    target_file_patterns,
    dseg_pattern,
):
    """Generate plots for analysis 2.

    Use visually-identified indices of deep breaths from the respiratory trace to extract
    time series from 30 seconds before to 40 seconds after the breath from each of the derivatives,
    then plot the mean signals.
    """
    ...


if __name__ == "__main__":
    print("Experiment 1, Analysis Group 3")
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
    TARGETS = [
        "TE30",
        "FIT-R2",
        "MEDN",
        "MEDN+GODEC",
        "MEDN+Nuis-Reg",
        "MEDN+RVT-Reg",
        "MEDN+RV-Reg",
        "MEDN+aCompCor",
        "MEDN+dGSR",
        "MEDN+MIR",
        "MEDN+GSR",
    ]
    target_file_patterns = {
        k: op.join(in_dir, "derivatives", v)
        for k, v in TARGET_FILE_PATTERNS.items()
        if k in TARGETS
    }

    correlate_rpv_with_cortical_sd(
        project_dir,
        participants_file,
        target_file_patterns,
        mask_pattern,
    )
