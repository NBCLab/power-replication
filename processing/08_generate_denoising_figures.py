"""Identify any subjects without all expected denoising derivatives."""
import os.path as op
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nilearn import plotting

sns.set_style("whitegrid")

FIGURES_CONFIG = {
    # Based on Fig. 1.
    "denoising_plot.png":
        {
            "carpet": [
                {
                    "ref": "OC",
                    "title": "Optimally combined (not denoised)",
                },
                {
                    "ref": "MEDN Noise",
                    "title": "Noise components (S0-dependent; non-BOLD-like; discarded)",
                },
                {
                    "ref": "MEDN",
                    "title": "Multi-echo ICA denoised (R2*-dependent; BOLD-like; retained)",
                },
            ],
            "timeseries": [
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "framewise_displacement",
            ],
        },
    # Based on Fig. 3A.
    "godec_plot.png":
        {
            "carpet": [
                {
                    "ref": "MEDN",
                    "title": "Multi-echo ICA denoised",
                },
                {
                    "ref": "MEDN+GODEC Noise (lowrank)",
                    "title": "Low-rank components",
                },
                {
                    "ref": "MEDN+GODEC (sparse)",
                    "title": "Sparse components",
                },
            ],
            "timeseries": [
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "framewise_displacement",
            ],
        },
    # Based on Fig. 3B.
    "gsr_plot.png":
        {
            "carpet": [
                {
                    "ref": "MEDN",
                    "title": "Multi-echo ICA denoised",
                },
                {
                    "ref": "MEDN+GSR Noise",
                    "title": "Removed by global signal regression",
                },
                {
                    "ref": "MEDN+GSR",
                    "title": "Residual signal",
                },
            ],
            "timeseries": [
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "framewise_displacement",
            ],
        },
    # Based on Fig. S3B.
    "fit_plot.png":
        {
            "carpet": [
                {
                    "ref": "TE30",
                    "title": "TE30",  # was "TE2 (28ms)" in original paper
                },
                {
                    "ref": "FIT-S0",
                    "title": "S0 estimate",
                },
                {
                    "ref": "FIT-R2",
                    "title": "T2* estimate",  # was "R2* estimate" in original paper
                },
            ],
            "timeseries": [
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "framewise_displacement",
            ],
        },

    # Novel, but based on Fig. 3B.
    "dgsr_plot.png":
        {
            "carpet": [
                {
                    "ref": "MEDN",
                    "title": "Multi-echo ICA denoised",
                },
                {
                    "ref": "MEDN+dGSR Noise",
                    "title": "Removed by dynamic global signal regression (rapidtide)",
                },
                {
                    "ref": "MEDN+dGSR",
                    "title": "Residual signal",
                },
            ],
            "timeseries": [
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "framewise_displacement",
            ],
        },
    # Novel, but based on Fig. 3B.
    "acompcor_plot.png":
        {
            "carpet": [
                {
                    "ref": "MEDN",
                    "title": "Multi-echo ICA denoised",
                },
                {
                    "ref": "MEDN+aCompCor Noise",
                    "title": "Removed by aCompCor",
                },
                {
                    "ref": "MEDN+aCompCor",
                    "title": "Residual signal",
                },
            ],
            "timeseries": [
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "framewise_displacement",
            ],
        },
}

TIMESERIES_RENAMER = {
    "trans_x": "X",
    "trans_y": "Y",
    "trans_z": "Z",
    "rot_x": "P",
    "rot_y": "R",
    "rot_z": "Ya",
    "framewise_displacement": "FD",
}

TARGET_FILES = {
    "confounds": "fmriprep/{sub}/func/{prefix}_desc-confounds_timeseries.tsv",
    "TE30": "power/{sub}/func/{prefix}_desc-TE30_bold.nii.gz",
    "OC": "tedana/{sub}/func/{prefix}_desc-optcom_bold.nii.gz",
    "FIT-R2": "t2smap/{sub}/func/{prefix}_T2starmap.nii.gz",
    "FIT-S0": "t2smap/{sub}/func/{prefix}_S0map.nii.gz",
    "MEDN": "tedana/{sub}/func/{prefix}_desc-optcomDenoised_bold.nii.gz",
    "MEDN Noise": "tedana/{sub}/func/{prefix}_desc-optcomDenoised_errorts.nii.gz",
    "MEDN+MIR": "tedana/{sub}/func/{prefix}_desc-optcomMIRDenoised_bold.nii.gz",
    "MEDN+MIR Noise": "tedana/{sub}/func/{prefix}_desc-optcomMIRDenoised_errorts.nii.gz",
    "MEDN+GODEC (sparse)": "godec/{sub}/func/{prefix}_desc-GODEC_rank-4_bold.nii.gz",
    "MEDN+GODEC Noise (lowrank)": "godec/{sub}/func/{prefix}_desc-GODEC_rank-4_lowrankts.nii.gz",
    "MEDN+dGSR": "rapidtide/{sub}/func/{prefix}_desc-lfofilterCleaned_bold.nii.gz",
    "MEDN+dGSR Noise": "rapidtide/{sub}/func/{prefix}_desc-lfofilterCleaned_errorts.nii.gz",
    "MEDN+aCompCor": "nuisance-regressions/{sub}/func/{prefix}_desc-aCompCor_bold.nii.gz",
    "MEDN+aCompCor Noise": "nuisance-regressions/{sub}/func/{prefix}_desc-aCompCor_errorts.nii.gz",
    "MEDN+GSR": "nuisance-regressions/{sub}/func/{prefix}_desc-GSR_bold.nii.gz",
    "MEDN+GSR Noise": "nuisance-regressions/{sub}/func/{prefix}_desc-GSR_errorts.nii.gz",
    "MEDN+Nuis-Reg": "nuisance-regressions/{sub}/func/{prefix}_desc-NuisReg_bold.nii.gz",
    "MEDN+RV-Reg": "nuisance-regressions/{sub}/func/{prefix}_desc-RVReg_bold.nii.gz",
    "MEDN+RVT-Reg": "nuisance-regressions/{sub}/func/{prefix}_desc-RVTReg_bold.nii.gz",
}


def plot_three_part_carpet(out_file, seg_file, config, sub, prefix):
    """Based on Figure 1."""
    timeseries = config["timeseries"]
    confounds_file = TARGET_FILES["confounds"].format(sub=sub, prefix=prefix)

    # Get confounds
    confounds_df = pd.read_table(confounds_file)
    new_columns = [TIMESERIES_RENAMER.get(k, k) for k in timeseries]
    confounds_df = confounds_df.rename(columns=TIMESERIES_RENAMER)
    confounds_df = confounds_df[new_columns]

    fig, axes = plt.subplots(figsize=(16, 8), nrows=4, gridspec_kw={"height_ratios": [1, 3, 3, 3]})

    for i_carpet, carpet_target in enumerate(config["carpet"]):
        target_file = TARGET_FILES[carpet_target["ref"]].format(sub=sub, prefix=prefix)
        display = plotting.plot_carpet(target_file, mask_img=seg_file, figure=fig, axes=axes[i_carpet + 1])

    fig.savefig(out_file)


def plot_two_part_carpet():
    """Based on Figure S7. Similar to three-part carpet, but without a "noise" subplot."""
    ...


def plot_three_part_carpet_with_physio():
    """Based on Figure S5, for DuPre dataset only."""
    ...


def plot_many_carpets(out_file, seg_file, config):
    """Based on Figure S12."""
    ...


def plot_components():
    """Based on Figure S2."""
    ...


def plot_global_signal_removal_summary():
    """Based on Figure S13."""
    ...


if __name__ == "__main__":
    PROJECT_DIR = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    participants_file = op.join(PROJECT_DIR, "participants.tsv")
    bad_subs = []
    df = pd.read_table(participants_file)
    for _, row in df.iterrows():
        dataset = row["dset"]
        sub = row["participant_id"]
        deriv_dir = op.join(PROJECT_DIR, dataset, "derivatives")

        medn_files = sorted(
            glob(
                op.join(
                    deriv_dir,
                    "tedana",
                    sub,
                    "func",
                    "*_desc-optcomDenoised_bold.nii.gz",
                )
            )
        )
        if len(medn_files) != 1:
            bad_subs.append((dataset, sub))
            continue

        medn_file = medn_files[0]
        medn_name = op.basename(medn_file)
        prefix = medn_name.split("desc-")[0].rstrip("_")

        for target_file in TARGET_FILES:
            full_file_pattern = op.join(
                deriv_dir, target_file.format(sub=sub, prefix=prefix)
            )
            matching_files = sorted(glob(full_file_pattern))
            if len(matching_files) == 0:
                print(f"Dataset {dataset} subject {sub} missing {full_file_pattern}")
                bad_subs.append((dataset, sub))

        if dataset == "dset-dupre":
            for target_file in PHYSIO_TARGET_FILES:
                full_file_pattern = op.join(
                    deriv_dir, target_file.format(sub=sub, prefix=prefix)
                )
                matching_files = sorted(glob(full_file_pattern))
                if len(matching_files) == 0:
                    print(
                        f"Dataset {dataset} subject {sub} missing {full_file_pattern}"
                    )
                    bad_subs.append((dataset, sub))

    bad_subs = sorted(list(set(bad_subs)))
    print(bad_subs)
