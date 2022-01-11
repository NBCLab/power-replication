"""Experiment 2, Analysis Group 4.

Multiple Methods to Remove Brain-Wide Signals.

Carpet plots with motion line plots for:
- MEDN+GODEC
- MEDN+GSR
- MEDN+dGSR
- MEDN+MIR
- MEDN+aCompCor

Correlation of variance removed by GODEC and variance removed by GSR across participants.
"""
import os
import os.path as op
import sys

import matplotlib.pyplot as plt
import pandas as pd
from nilearn.plotting import plot_carpet

sys.path.append("..")

from utils import get_prefixes, plot_confounds  # noqa: E402


def plot_denoised_with_motion(
    project_dir,
    participants_file,
    target_file_patterns,
    confounds_pattern,
    dseg_pattern,
):
    """Generate carpet plots with associated line plots for denoising derivatives."""
    print("Experiment 2, Analysis Group 4, Analysis 1", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment02_group04", "analysis_01")
    os.makedirs(out_dir, exist_ok=True)

    participants_df = pd.read_table(participants_file)
    n_subs_all = participants_df.shape[0]
    # Limit to participants with RPV value
    participants_df = participants_df.loc[participants_df["exclude"] != 1]
    print(f"{participants_df.shape[0]}/{n_subs_all} participants retained.")

    for i_run, participant_row in participants_df.iterrows():
        subj_id = participant_row["participant_id"]
        dset = participant_row["dataset"]
        dset_prefix = get_prefixes()[dset]
        subj_prefix = dset_prefix.format(participant_id=subj_id)

        confounds_file = confounds_pattern.format(
            dset=dset, participant_id=subj_id, prefix=subj_prefix
        )
        dseg_file = dseg_pattern.format(dset=dset, participant_id=subj_id)
        for filetype, pattern in target_file_patterns.items():
            filetype_dir = op.join(out_dir, filetype)
            os.makedirs(filetype_dir, exist_ok=True)

            filename = pattern.format(
                dset=dset, participant_id=subj_id, prefix=subj_prefix
            )

            fig, axes = plt.subplots(figsize=(12, 18), nrows=2)
            plot_confounds(confounds_file, figure=fig, axes=axes[0])
            plot_carpet(filename, dseg_file, figure=fig, axes=axes[1])
            fig.savefig(
                op.join(filetype_dir, "{dset}_{subj_id}_{filetype}.png"), dpi=400
            )


def correlate_variance_removed():
    """Calculate and correlate variance removed by GODEC and GSR across participants."""
    ...


if __name__ == "__main__":
    ...
