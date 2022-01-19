"""Identify any subjects without all expected denoising derivatives."""
import os.path as op
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec as mgs
from nilearn import image, masking, plotting
from scipy import stats

sns.set_style("whitegrid")

FIGURES_CONFIG = {
    # Based on Fig. 1.
    "denoising_plot.png": {
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
    "godec_plot.png": {
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
    "gsr_plot.png": {
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
    "fit_plot.png": {
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
    "dgsr_plot.png": {
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
    "acompcor_plot.png": {
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
    "confounds": "power/{sub}/func/{prefix}_desc-confounds_timeseries.tsv",
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


def _plot_timeseries(df, x_arr, fig, ax):
    assert df.shape[0] == x_arr.size, f"{df.shape[0]} != {x_arr.shape}"

    palette = {
        "X": "#ea2807",
        "Y": "#66ff6b",
        "Z": "#2e35fd",
        "P": "#5ffffc",
        "R": "#fe4fff",
        "Ya": "#e9e65d",
        "FD": "#fe2a32",
        "Heart rate": "#a6fe96",
        "Resp. belt": "#625afd",
    }

    ax_right = None
    if "FD" in df.columns:
        fd_arr = df["FD"].values
        df = df[[c for c in df.columns if c != "FD"]]
        ax_right = ax.twinx()
        ax_right.plot(x_arr, fd_arr, color=palette["FD"], label="FD", linewidth=2)
        ax_right.set_ylim(np.floor(np.min(fd_arr)), np.ceil(np.max(fd_arr)))
        ax_right.set_yticks((np.floor(np.min(fd_arr)), np.ceil(np.max(fd_arr))))
        ax_right.tick_params(axis="y", which="both", colors=palette["FD"])
        ax_right.set_ylabel(
            "FD\n(mm)",
            color=palette["FD"],
            rotation=270,
            labelpad=30,
            fontsize=14,
        )
        ax_right.legend()
        ax_right.tick_params(axis="y", which="both", length=0)

    all_min, all_max = 0, 0
    for i_col, label in enumerate(df.columns):
        arr = df[label].values
        ax.plot(x_arr, arr, color=palette[label], label=label)
        all_min = np.minimum(all_min, np.min(arr))
        all_max = np.maximum(all_max, np.max(arr))

    ax.set_ylim(np.floor(all_min), np.ceil(all_max))
    ax.set_yticks((np.floor(all_min), np.ceil(all_max)))
    ax.set_xlim(x_arr[0], x_arr[-1])
    ax.tick_params(axis="y", which="both", length=0)
    ax.xaxis.set_visible(False)
    ax.set_ylabel("Position\n(mm)", fontsize=14)
    ax.legend(ncol=3)
    ax.set_title("Head position & motion", fontsize=16)

    return fig, ax, ax_right


def _plot_carpet(dseg, bold, title, fig, ax):
    palette = [
        "#002d65",
        "#014c93",
        "#0365cb",
        "#02cd00",
        "#018901",
        "#006601",
        "#e7ef13",
        "#c2c503",
    ]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("Power", palette, len(palette))

    mask_img = image.math_img("img > 0", img=dseg)
    bold_data = masking.apply_mask(bold, mask_img)
    modes = stats.mode(bold_data, axis=0)  # voxelwise mode
    modes = np.squeeze(modes[0])
    scalars = 1000 - modes
    bold_mode1000 = bold_data + scalars[None, :]
    bold_mode1000 = masking.unmask(bold_mode1000, mask_img)

    dseg_vals = nib.load(dseg).get_fdata()
    n_gm_voxels = np.sum(np.logical_and(dseg_vals <= 3, dseg_vals > 0)) + 1
    n_brain_voxels = np.sum(dseg_vals > 0)

    display = plotting.plot_carpet(
        bold_mode1000,
        mask_img=dseg,
        detrend=False,
        figure=fig,
        axes=ax,
        cmap=cmap,
    )

    display.axes[-1].xaxis.set_visible(False)
    display.axes[-1].yaxis.set_visible(False)
    display.axes[-1].spines["top"].set_visible(False)
    display.axes[-1].spines["bottom"].set_visible(False)
    display.axes[-1].spines["left"].set_visible(False)
    display.axes[-1].spines["right"].set_visible(False)
    display.axes[-1].axhline(n_gm_voxels, color="#0ffb03", linewidth=2)
    display.axes[-1].set_title(title, fontsize=16)
    display.axes[-2].annotate(
        "Gray\nMatter",
        xy=(-1.6, n_gm_voxels / 2),
        xycoords="data",
        rotation=90,
        fontsize=14,
        annotation_clip=False,
        va="center",
        ha="center",
    )
    display.axes[-2].annotate(
        "White\nMatter / CSF",
        xy=(-1.6, n_gm_voxels + ((n_brain_voxels - n_gm_voxels) / 2)),
        xycoords="data",
        rotation=90,
        fontsize=14,
        annotation_clip=False,
        va="center",
        ha="center",
    )
    return display


def plot_three_part_carpet(out_file, seg_file, t_r, config, sub, prefix):
    """Based on Figure 1."""
    timeseries = config["timeseries"]
    confounds_file = TARGET_FILES["confounds"].format(sub=sub, prefix=prefix)
    carpet_config = config["carpet"]
    target_files = [
        TARGET_FILES[f["ref"]].format(sub=sub, prefix=prefix) for f in carpet_config
    ]
    titles = [f["title"] for f in carpet_config]

    # Get confounds
    confounds_df = pd.read_table(confounds_file)
    new_columns = [TIMESERIES_RENAMER.get(k, k) for k in timeseries]
    confounds_df = confounds_df.rename(columns=TIMESERIES_RENAMER)
    confounds_df = confounds_df[new_columns]

    x_arr = np.linspace(0, (confounds_df.shape[0] - 1) * t_r, confounds_df.shape[0])

    fig, axes = plt.subplots(
        figsize=(16, 16), nrows=4, gridspec_kw={"height_ratios": [1, 3, 3, 3]}
    )

    _plot_carpet(seg_file, target_files[0], titles[0], fig, axes[1])
    _plot_carpet(seg_file, target_files[1], titles[1], fig, axes[2])
    display = _plot_carpet(seg_file, target_files[2], titles[2], fig, axes[3])

    last_carpet_ax = display.axes[1]
    width_ratio = last_carpet_ax.get_subplotspec().get_gridspec().get_width_ratios()

    gs = mgs.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=axes[0],
        width_ratios=width_ratio,
        wspace=0.0,
    )
    ax0 = plt.subplot(gs[1])

    _plot_timeseries(confounds_df, x_arr, fig, ax0)
    display.axes[-3].xaxis.set_visible(True)
    display.axes[-3].set_xlabel("Time (min)", fontsize=14, labelpad=-10)
    display.axes[-3].set_xticks([0, len(x_arr)])
    display.axes[-3].set_xticklabels(
        [0, f"{int(x_arr[-1] // 60)}:{str(int(x_arr[-1] % 60)).zfill(2)}"],
    )
    display.axes[-3].tick_params(axis="x", which="both", length=0, labelsize=12)
    display.axes[-3].spines["bottom"].set_position(("outward", 0))

    fig.savefig(out_file)


def plot_two_part_carpet(out_file, seg_file, t_r, config, sub, prefix):
    """Based on Figure S7. Similar to three-part carpet, but without a "noise" subplot."""
    timeseries = config["timeseries"]
    confounds_file = TARGET_FILES["confounds"].format(sub=sub, prefix=prefix)
    carpet_config = config["carpet"]
    target_files = [
        TARGET_FILES[f["ref"]].format(sub=sub, prefix=prefix) for f in carpet_config
    ]
    titles = [f["title"] for f in carpet_config]

    # Get confounds
    confounds_df = pd.read_table(confounds_file)
    new_columns = [TIMESERIES_RENAMER.get(k, k) for k in timeseries]
    confounds_df = confounds_df.rename(columns=TIMESERIES_RENAMER)
    confounds_df = confounds_df[new_columns]

    x_arr = np.linspace(0, (confounds_df.shape[0] - 1) * t_r, confounds_df.shape[0])

    fig, axes = plt.subplots(
        figsize=(16, 11.2), nrows=3, gridspec_kw={"height_ratios": [1, 3, 3]}
    )

    _plot_carpet(seg_file, target_files[0], titles[0], fig, axes[1])
    display = _plot_carpet(seg_file, target_files[1], titles[1], fig, axes[2])

    last_carpet_ax = display.axes[1]
    width_ratio = last_carpet_ax.get_subplotspec().get_gridspec().get_width_ratios()

    gs = mgs.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=axes[0],
        width_ratios=width_ratio,
        wspace=0.0,
    )
    ax0 = plt.subplot(gs[1])

    _plot_timeseries(confounds_df, x_arr, fig, ax0)
    display.axes[-3].xaxis.set_visible(True)
    display.axes[-3].set_xlabel("Time (min)", fontsize=14, labelpad=-10)
    display.axes[-3].set_xticks([0, len(x_arr)])
    display.axes[-3].set_xticklabels(
        [0, f"{int(x_arr[-1] // 60)}:{str(int(x_arr[-1] % 60)).zfill(2)}"],
    )
    display.axes[-3].tick_params(axis="x", which="both", length=0, labelsize=12)
    display.axes[-3].spines["bottom"].set_position(("outward", 0))

    fig.savefig(out_file)


def plot_three_part_carpet_with_physio():
    """Based on Figure S5, for DuPre dataset only."""
    ...


def plot_many_carpets(out_file, seg_file, config):
    """Based on Figure S12."""
    targets = config["carpet"]
    assert isinstance(targets, list)
    assert isinstance(targets[0], list)

    fig, axes = plt.subplots(
        figsize=(16, 16), nrows=len(targets[0]), ncols=len(targets)
    )
    for i_col, group in enumerate(targets):
        target_files = [
            TARGET_FILES[f["ref"]].format(sub=sub, prefix=prefix) for f in group
        ]
        titles = [f["title"] for f in group]
        for j_row, target_file in enumerate(target_files):
            _plot_carpet(seg_file, target_file, titles[j_row], fig, axes[j_row, i_col])
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
