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
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from nilearn import masking  # noqa: E402

sys.path.append("..")

from utils import calculate_variance_explained  # noqa: E402
from utils import get_bad_subjects_nonphysio  # noqa: E402
from utils import get_prefixes  # noqa: E402
from utils import get_target_files  # noqa: E402
from utils import pearson_r  # noqa: E402


def correlate_variance_removed(
    project_dir,
    participants_file,
    medn_pattern,
    godec_pattern,
    gsr_pattern,
    brain_mask_pattern,
):
    """Calculate and correlate variance removed by GODEC and GSR across participants."""
    print("Experiment 2, Analysis Group 4, Analysis 2", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment02_group04")
    os.makedirs(out_dir, exist_ok=True)

    ALPHA = 0.05

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

    for i_run, participant_row in participants_df.iterrows():
        subj_id = participant_row["participant_id"]
        dset = participant_row["dset"]
        dset_prefix = get_prefixes()[dset]
        subj_prefix = dset_prefix.format(participant_id=subj_id)

        medn_file = medn_pattern.format(
            dset=dset, participant_id=subj_id, prefix=subj_prefix
        )
        godec_file = godec_pattern.format(
            dset=dset, participant_id=subj_id, prefix=subj_prefix
        )
        gsr_file = gsr_pattern.format(
            dset=dset, participant_id=subj_id, prefix=subj_prefix
        )
        mask_file = brain_mask_pattern.format(dset=dset, participant_id=subj_id)

        medn_data = masking.apply_mask(medn_file, mask_file)
        godec_data = masking.apply_mask(godec_file, mask_file)
        gsr_data = masking.apply_mask(gsr_file, mask_file)
        assert (
            medn_data.shape == godec_data.shape == gsr_data.shape
        ), f"{medn_data.shape}, {godec_data.shape}, {gsr_data.shape}"

        godec_varex = calculate_variance_explained(medn_data, godec_data)
        gsr_varex = calculate_variance_explained(medn_data, gsr_data)
        godec_varrem = 1 - godec_varex
        gsr_varrem = 1 - gsr_varex
        participants_df.loc[i_run, "godec variance removed"] = godec_varrem
        participants_df.loc[i_run, "gsr variance removed"] = gsr_varrem

    # We are performing a one-sided test to determine if the correlation is
    # statistically significant (alpha = 0.05) and positive.
    corr, p = pearson_r(
        participants_df["godec variance removed"],
        participants_df["gsr variance removed"],
        alternative="greater",
    )

    if p <= ALPHA:
        print(
            "\tVariance removed by GODEC and GSR were found to be positively and statistically "
            "significantly correlated, "
            f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
        )
    else:
        print(
            "\tVariance removed by GODEC and GSR were not found to be statistically significantly "
            "correlated, "
            f"r({participants_df.shape[0] - 2}) = {corr:.02f}, p = {p:.03f}"
        )

    g = sns.JointGrid(
        data=participants_df, x="godec variance removed", y="gsr variance removed"
    )
    g.plot(sns.regplot, sns.histplot)
    g.savefig(op.join(out_dir, "analysis_02.png"), dpi=400)

    participants_df.to_csv(
        op.join(out_dir, "analysis_02_results.tsv"), sep="\t", index=False
    )


if __name__ == "__main__":
    print("Experiment 2, Analysis Group 4")
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    in_dir = op.join(project_dir, "{dset}")
    participants_file = op.join(project_dir, "participants.tsv")

    confounds_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/func",
        "{prefix}_desc-confounds_timeseries.tsv",
    )
    dseg_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/anat",
        "{participant_id}_space-scanner_res-bold_desc-totalMaskWithCSF_dseg.nii.gz",
    )
    brain_mask_pattern = op.join(
        in_dir,
        "derivatives/power/{participant_id}/anat",
        "{participant_id}_space-scanner_res-bold_desc-totalMaskNoCSF_mask.nii.gz",
    )

    TARGET_FILE_PATTERNS = get_target_files()

    # Analysis 2 targets
    medn_pattern = op.join(in_dir, "derivatives", TARGET_FILE_PATTERNS["MEDN"])
    godec_pattern = op.join(
        in_dir, "derivatives", TARGET_FILE_PATTERNS["MEDN+GODEC (sparse)"]
    )
    gsr_pattern = op.join(in_dir, "derivatives", TARGET_FILE_PATTERNS["MEDN+GSR"])

    correlate_variance_removed(
        project_dir,
        participants_file,
        medn_pattern,
        godec_pattern,
        gsr_pattern,
        brain_mask_pattern,
    )
