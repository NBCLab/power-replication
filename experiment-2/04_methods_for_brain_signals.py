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
import json
import os
import os.path as op
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from nilearn import masking  # noqa: E402

sys.path.append("..")

from utils import _plot_denoised_and_confounds  # noqa: E402
from utils import calculate_variance_explained  # noqa: E402
from utils import get_bad_subjects_nonphysio  # noqa: E402
from utils import get_prefixes  # noqa: E402
from utils import get_target_files  # noqa: E402
from utils import pearson_r  # noqa: E402


def plot_denoised_with_motion(
    project_dir,
    participants_file,
    target_file_patterns,
    ax_titles,
    confounds_pattern,
    dseg_pattern,
):
    """Generate carpet plots with associated line plots for denoising derivatives."""
    print("Experiment 2, Analysis Group 4, Analysis 1", flush=True)
    out_dir = op.join(project_dir, "analyses", "experiment02_group04", "analysis01")
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

    for i_run, participant_row in participants_df.iloc[:1].iterrows():
        subj_id = participant_row["participant_id"]
        dset = participant_row["dset"]
        dset_prefix = get_prefixes()[dset]
        subj_prefix = dset_prefix.format(participant_id=subj_id)

        confounds_file = confounds_pattern.format(
            dset=dset, participant_id=subj_id, prefix=subj_prefix
        )
        confounds_df = pd.read_table(confounds_file)
        dseg_file = dseg_pattern.format(dset=dset, participant_id=subj_id)

        for filegroup_name, group_file_patterns in target_file_patterns.items():
            filetype_dir = op.join(out_dir, filegroup_name)
            os.makedirs(filetype_dir, exist_ok=True)
            out_file = op.join(filetype_dir, f"{dset}_{subj_id}.svg")

            filegroup_filenames = [
                group_file_pattern.format(
                    dset=dset, participant_id=subj_id, prefix=subj_prefix
                )
                for group_file_pattern in group_file_patterns
            ]

            # MEDN is the first image for each
            # Get the repetition time from the metadata file
            metadata_file = filegroup_filenames[0].replace(".nii.gz", ".json")
            with open(metadata_file) as fo:
                t_r = json.load(fo)["RepetitionTime"]

            _plot_denoised_and_confounds(
                filegroup_filenames,
                filegroup_name,
                ax_titles,
                dseg_file,
                confounds_df,
                t_r,
                out_file,
            )


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

    # Analysis 1 targets
    A1_TARGETS = {
        "GODEC": [
            "MEDN",
            "MEDN+GODEC (sparse)",
            "MEDN+GODEC Noise (lowrank)",
        ],
        "GSR": [
            "MEDN",
            "MEDN+GSR",
            "MEDN+GSR Noise",
        ],
        "dGSR": [
            "MEDN",
            "MEDN+dGSR",
            "MEDN+dGSR Noise",
        ],
        "MIR": [
            "MEDN",
            "MEDN+MIR",
            "MEDN+MIR Noise",
        ],
        "aCompCor": [
            "MEDN",
            "MEDN+aCompCor",
            "MEDN+aCompCor Noise",
        ],
    }
    ax_titles = ["Multi-echo denoised", "Retained", "Removed"]
    a1_target_file_patterns = {}
    for k, v in A1_TARGETS.items():
        group_target_file_patterns = [
            op.join(in_dir, "derivatives", TARGET_FILE_PATTERNS[t]) for t in v
        ]
        a1_target_file_patterns[k] = group_target_file_patterns

    # Analysis 2 targets
    medn_pattern = op.join(in_dir, "derivatives", TARGET_FILE_PATTERNS["MEDN"])
    godec_pattern = op.join(
        in_dir, "derivatives", TARGET_FILE_PATTERNS["MEDN+GODEC (sparse)"]
    )
    gsr_pattern = op.join(in_dir, "derivatives", TARGET_FILE_PATTERNS["MEDN+GSR"])

    plot_denoised_with_motion(
        project_dir,
        participants_file,
        a1_target_file_patterns,
        ax_titles,
        confounds_pattern,
        dseg_pattern,
    )
    correlate_variance_removed(
        project_dir,
        participants_file,
        medn_pattern,
        godec_pattern,
        gsr_pattern,
        brain_mask_pattern,
    )
