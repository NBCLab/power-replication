"""Miscellaneous functions used for analyses."""
import logging
import os.path as op

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import gridspec as mgs  # noqa: E402
from nilearn import plotting
from nilearn.signal import clean
from scipy import stats

LGR = logging.getLogger("utils")


def _plot_denoised_and_confounds(
    files,
    title,
    ax_titles,
    dseg_file,
    confounds_df,
    t_r,
    out_file,
):
    """Plot raw, denoised, and noise data with confounds."""
    # Get confounds
    confound_names = {
        "trans_x": "X",
        "trans_y": "Y",
        "trans_z": "Z",
        "rot_x": "P",
        "rot_y": "R",
        "rot_z": "Ya",
        "framewise_displacement": "FD",
    }
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

    confounds_df = confounds_df[confound_names.keys()]
    confounds_df = confounds_df.rename(columns=confound_names)

    assert len(files) == len(ax_titles), f"{len(files)} != {len(ax_titles)}"

    x_arr = np.linspace(0, (confounds_df.shape[0] - 1) * t_r, confounds_df.shape[0])

    fig, axes = plt.subplots(
        figsize=(16, 14),
        nrows=4,
        gridspec_kw={"height_ratios": [1, 3, 3, 3]},
    )
    fig.suptitle(title, fontsize=24)

    _plot_carpet(dseg_file, files[0], ax_titles[0], fig, axes[1])
    _plot_carpet(dseg_file, files[1], ax_titles[1], fig, axes[2])
    display = _plot_carpet(dseg_file, files[2], ax_titles[2], fig, axes[3])

    last_carpet_ax = display.axes[-1]
    width_ratio = last_carpet_ax.get_subplotspec().get_gridspec().get_width_ratios()

    # First timeseries plot: confounds
    confounds_gs = mgs.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=axes[0],
        width_ratios=width_ratio,
        wspace=0.0,
    )
    confounds_ax = plt.subplot(confounds_gs[1])
    fd_arr = confounds_df["FD"].values
    # In case there are no non-steady state volumes, replace the NaNs with zero.
    fd_arr[np.isnan(fd_arr)] = 0
    confounds_df = confounds_df[[c for c in confounds_df.columns if c != "FD"]]

    # Plot FD on the right
    confounds_ax_right = confounds_ax.twinx()
    confounds_ax_right.plot(x_arr, fd_arr, color=palette["FD"], label="FD", linewidth=2)
    confounds_ax_right.set_ylim(np.floor(np.min(fd_arr)), np.ceil(np.max(fd_arr)))
    confounds_ax_right.set_yticks((np.floor(np.min(fd_arr)), np.ceil(np.max(fd_arr))))
    confounds_ax_right.tick_params(axis="y", which="both", colors=palette["FD"])
    confounds_ax_right.set_ylabel(
        "FD\n(mm)",
        color=palette["FD"],
        rotation=270,
        labelpad=30,
        fontsize=14,
    )
    confounds_ax_right.legend()
    confounds_ax_right.tick_params(axis="y", which="both", length=0)

    # Plot everything else on the left
    all_min, all_max = 0, 0
    for i_col, label in enumerate(confounds_df.columns):
        arr = confounds_df[label].values
        confounds_ax.plot(x_arr, arr, color=palette[label], label=label)
        all_min = np.minimum(all_min, np.min(arr))
        all_max = np.maximum(all_max, np.max(arr))

    confounds_ax.set_ylim(np.floor(all_min), np.ceil(all_max))
    confounds_ax.set_yticks((np.floor(all_min), np.ceil(all_max)))
    confounds_ax.set_xlim(x_arr[0], x_arr[-1])
    confounds_ax.tick_params(axis="y", which="both", length=0)
    confounds_ax.xaxis.set_visible(False)
    confounds_ax.set_ylabel("Position\n(mm)", fontsize=14)
    confounds_ax.legend(ncol=3)
    confounds_ax.set_title("Head position & motion", fontsize=16)
    # Add subject/dset info to the top ax
    confounds_ax.text(
        -0.06,
        1.1,
        op.splitext(op.basename(out_file))[0].replace("_", " "),
        transform=confounds_ax.transAxes,
        size=16,
    )

    # Add x-axis labels to bottom ax
    # First, find the last carpet plot
    sel_ax = None
    for i_ax, temp in enumerate(display.axes):
        if temp.get_title() == ax_titles[-1]:
            sel_ax = i_ax

    if sel_ax is None:
        raise Exception(display.axes)

    display.axes[sel_ax].xaxis.set_visible(True)
    display.axes[sel_ax].set_xlabel("Time (min)", fontsize=14, labelpad=-10)
    display.axes[sel_ax].set_xticks([0, len(x_arr)])
    display.axes[sel_ax].set_xticklabels(
        [0, f"{int(x_arr[-1] // 60)}:{str(int(x_arr[-1] % 60)).zfill(2)}"],
    )
    display.axes[sel_ax].tick_params(axis="x", which="both", length=0, labelsize=12)
    display.axes[sel_ax].spines["bottom"].set_position(("outward", 0))

    fig.savefig(out_file)
    fig.close()


def _plot_oc_and_fit(
    oc_file,
    fitr2_file,
    fits0_file,
    dseg_file,
    confounds_df,
    t_r,
    out_file,
):
    """Plot OC and FIT-R2/S0 data with confounds."""
    # Get confounds
    confound_names = {
        "trans_x": "X",
        "trans_y": "Y",
        "trans_z": "Z",
        "rot_x": "P",
        "rot_y": "R",
        "rot_z": "Ya",
        "framewise_displacement": "FD",
    }
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

    confounds_df = confounds_df[confound_names.keys()]
    confounds_df = confounds_df.rename(columns=confound_names)

    x_arr = np.linspace(0, (confounds_df.shape[0] - 1) * t_r, confounds_df.shape[0])

    fig, axes = plt.subplots(
        figsize=(16, 14),
        nrows=4,
        gridspec_kw={"height_ratios": [1, 3, 3, 3]},
    )

    _plot_carpet(dseg_file, oc_file, "Optimally Combined", fig, axes[1])
    _plot_carpet(dseg_file, fitr2_file, "FIT-R2", fig, axes[2])
    display = _plot_carpet(dseg_file, fits0_file, "FIT-S0", fig, axes[3])

    last_carpet_ax = display.axes[-1]
    width_ratio = last_carpet_ax.get_subplotspec().get_gridspec().get_width_ratios()

    # First timeseries plot: confounds
    confounds_gs = mgs.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=axes[0],
        width_ratios=width_ratio,
        wspace=0.0,
    )
    confounds_ax = plt.subplot(confounds_gs[1])
    fd_arr = confounds_df["FD"].values
    # In case there are no non-steady state volumes, replace the NaNs with zero.
    fd_arr[np.isnan(fd_arr)] = 0
    confounds_df = confounds_df[[c for c in confounds_df.columns if c != "FD"]]

    # Plot FD on the right
    confounds_ax_right = confounds_ax.twinx()
    confounds_ax_right.plot(x_arr, fd_arr, color=palette["FD"], label="FD", linewidth=2)
    confounds_ax_right.set_ylim(np.floor(np.min(fd_arr)), np.ceil(np.max(fd_arr)))
    confounds_ax_right.set_yticks((np.floor(np.min(fd_arr)), np.ceil(np.max(fd_arr))))
    confounds_ax_right.tick_params(axis="y", which="both", colors=palette["FD"])
    confounds_ax_right.set_ylabel(
        "FD\n(mm)",
        color=palette["FD"],
        rotation=270,
        labelpad=30,
        fontsize=14,
    )
    confounds_ax_right.legend()
    confounds_ax_right.tick_params(axis="y", which="both", length=0)

    # Plot everything else on the left
    all_min, all_max = 0, 0
    for i_col, label in enumerate(confounds_df.columns):
        arr = confounds_df[label].values
        confounds_ax.plot(x_arr, arr, color=palette[label], label=label)
        all_min = np.minimum(all_min, np.min(arr))
        all_max = np.maximum(all_max, np.max(arr))

    confounds_ax.set_ylim(np.floor(all_min), np.ceil(all_max))
    confounds_ax.set_yticks((np.floor(all_min), np.ceil(all_max)))
    confounds_ax.set_xlim(x_arr[0], x_arr[-1])
    confounds_ax.tick_params(axis="y", which="both", length=0)
    confounds_ax.xaxis.set_visible(False)
    confounds_ax.set_ylabel("Position\n(mm)", fontsize=14)
    confounds_ax.legend(ncol=3)
    confounds_ax.set_title("Head position & motion", fontsize=16)
    # Add subject/dset info to the top ax
    confounds_ax.text(
        -0.06,
        1.1,
        op.splitext(op.basename(out_file))[0].replace("_", " "),
        transform=confounds_ax.transAxes,
        size=16,
    )

    # Add x-axis labels to bottom ax
    # First, find the last carpet plot
    sel_ax = None
    for i_ax, temp in enumerate(display.axes):
        if temp.get_title() == "FIT-S0":
            sel_ax = i_ax

    if sel_ax is None:
        raise Exception(display.axes)

    display.axes[sel_ax].xaxis.set_visible(True)
    display.axes[sel_ax].set_xlabel("Time (min)", fontsize=14, labelpad=-10)
    display.axes[sel_ax].set_xticks([0, len(x_arr)])
    display.axes[sel_ax].set_xticklabels(
        [0, f"{int(x_arr[-1] // 60)}:{str(int(x_arr[-1] % 60)).zfill(2)}"],
    )
    display.axes[sel_ax].tick_params(axis="x", which="both", length=0, labelsize=12)
    display.axes[sel_ax].spines["bottom"].set_position(("outward", 0))

    fig.savefig(out_file)


def _plot_components_and_physio(
    oc_file,
    dseg_file,
    confounds_df,
    physio_df,
    components_arr,
    classifications,
    t_r,
    physio_samplerate,
    out_file,
):
    """Components array must be TxS."""
    # Get confounds
    confound_names = {
        "trans_x": "X",
        "trans_y": "Y",
        "trans_z": "Z",
        "rot_x": "P",
        "rot_y": "R",
        "rot_z": "Ya",
        "framewise_displacement": "FD",
    }
    physio_names = {"cardiac": "Heart rate", "respiratory": "Resp. belt"}
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
    clf_palette = {
        "accepted": "#da5e10",
        "ignored": "#d3a3a3",
        "rejected": "#abd7cc",
    }

    confounds_df = confounds_df[confound_names.keys()]
    confounds_df = confounds_df.rename(columns=confound_names)
    physio_df = physio_df.rename(columns=physio_names)

    x_arr = np.linspace(0, (confounds_df.shape[0] - 1) * t_r, confounds_df.shape[0])
    x_arr_physio = np.linspace(
        0, (physio_df.shape[0] - 1) / physio_samplerate, physio_df.shape[0]
    )

    n_components = len(classifications)
    fig, axes = plt.subplots(
        figsize=(16, 10 + (n_components * 0.1)),
        nrows=4,
        gridspec_kw={"height_ratios": [1, 1, 3, (n_components * 0.1)]},
    )

    display = _plot_carpet(dseg_file, oc_file, "Optimally Combined", fig, axes[2])

    # Components!
    # Sort components by classification
    clf_xformer = {"accepted": 0, "ignored": 1, "rejected": 2}
    clf_xformer_inv = {v: k for k, v in clf_xformer.items()}
    classification_ints = np.array([clf_xformer[clf] for clf in classifications])
    order = np.argsort(classification_ints)
    classification_ints = classification_ints[order]
    components_arr = components_arr[:, order]
    # Detrend the component time series to better identify banding
    components_arr = clean(components_arr.T, t_r=t_r, detrend=True, standardize="zscore").T

    # Determine the colormap
    temp_clf_palette = [clf_palette[clf_xformer_inv[c]] for c in np.unique(classification_ints)]
    # The colormap needs >1 color so I add one to the end just in case. It's bright blue.
    temp_clf_palette = temp_clf_palette + ["#0000FF"]
    clf_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Classification", temp_clf_palette, len(temp_clf_palette)
    )

    # Determine vmin and vmax based on the full data
    std = np.mean(components_arr.std(axis=0))
    vmin = components_arr.mean() - (2 * std)
    vmax = components_arr.mean() + (2 * std)

    # Define nested GridSpec
    legend = False
    wratios = [2, 100, 20]
    components_gs = mgs.GridSpecFromSubplotSpec(
        1,
        2 + int(legend),
        subplot_spec=axes[3],
        width_ratios=wratios[: 2 + int(legend)],
        wspace=0.0,
    )

    clf_ax = plt.subplot(components_gs[0])
    clf_ax.set_xticks([])
    clf_ax.imshow(
        classification_ints[:, np.newaxis],
        interpolation="none",
        aspect="auto",
        cmap=clf_cmap,
    )
    # Add labels to middle of each associated band
    mask_labels_inv = {v: k for k, v in clf_xformer.items()}
    ytick_locs = [
        np.mean(np.where(classification_ints == i)[0])
        for i in np.unique(classification_ints)
    ]
    clf_ax.set_yticks(ytick_locs)
    clf_ax.set_yticklabels([mask_labels_inv[i] for i in np.unique(classification_ints)])
    clf_ax.spines["top"].set_visible(False)
    clf_ax.spines["bottom"].set_visible(False)
    clf_ax.spines["left"].set_visible(False)
    clf_ax.spines["right"].set_visible(False)
    clf_ax.tick_params(axis="y", which="both", length=0)

    # Carpet plot
    components_ax = plt.subplot(components_gs[1])
    components_ax.imshow(
        components_arr.T,
        interpolation="nearest",
        aspect="auto",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    components_ax.xaxis.set_visible(False)
    components_ax.yaxis.set_visible(False)
    components_ax.spines["top"].set_visible(False)
    components_ax.spines["bottom"].set_visible(False)
    components_ax.spines["left"].set_visible(False)
    components_ax.spines["right"].set_visible(False)
    components_ax.set_title("ICA component timeseries", fontsize=16)

    last_carpet_ax = display.axes[3]
    width_ratio = last_carpet_ax.get_subplotspec().get_gridspec().get_width_ratios()

    # First timeseries plot: confounds
    confounds_gs = mgs.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=axes[0],
        width_ratios=width_ratio,
        wspace=0.0,
    )
    confounds_ax = plt.subplot(confounds_gs[1])
    fd_arr = confounds_df["FD"].values
    confounds_df = confounds_df[[c for c in confounds_df.columns if c != "FD"]]

    # Plot FD on the right
    confounds_ax_right = confounds_ax.twinx()
    confounds_ax_right.plot(x_arr, fd_arr, color=palette["FD"], label="FD", linewidth=2)
    confounds_ax_right.set_ylim(np.floor(np.min(fd_arr)), np.ceil(np.max(fd_arr)))
    confounds_ax_right.set_yticks((np.floor(np.min(fd_arr)), np.ceil(np.max(fd_arr))))
    confounds_ax_right.tick_params(axis="y", which="both", colors=palette["FD"])
    confounds_ax_right.set_ylabel(
        "FD\n(mm)",
        color=palette["FD"],
        rotation=270,
        labelpad=30,
        fontsize=14,
    )
    confounds_ax_right.legend()
    confounds_ax_right.tick_params(axis="y", which="both", length=0)

    # Plot everything else on the left
    all_min, all_max = 0, 0
    for i_col, label in enumerate(confounds_df.columns):
        arr = confounds_df[label].values
        confounds_ax.plot(x_arr, arr, color=palette[label], label=label)
        all_min = np.minimum(all_min, np.min(arr))
        all_max = np.maximum(all_max, np.max(arr))

    confounds_ax.set_ylim(np.floor(all_min), np.ceil(all_max))
    confounds_ax.set_yticks((np.floor(all_min), np.ceil(all_max)))
    confounds_ax.set_xlim(x_arr[0], x_arr[-1])
    confounds_ax.tick_params(axis="y", which="both", length=0)
    confounds_ax.xaxis.set_visible(False)
    confounds_ax.set_ylabel("Position\n(mm)", fontsize=14)
    confounds_ax.legend(ncol=3)
    confounds_ax.set_title("Head position & motion", fontsize=16)

    # Second timeseries plot: physio
    physio_gs = mgs.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=axes[1],
        width_ratios=width_ratio,
        wspace=0.0,
    )
    physio_ax = plt.subplot(physio_gs[1])
    for i_col, label in enumerate(physio_df.columns):
        arr = physio_df[label].values
        physio_ax.plot(x_arr_physio, arr, color=palette[label], label=label)

    physio_ax.set_xlim(x_arr_physio[0], x_arr_physio[-1])
    physio_ax.xaxis.set_visible(False)
    physio_ax.set_yticks([])
    physio_ax.legend(ncol=1, loc="upper right")
    physio_ax.set_ylabel("Physiological\ntraces", fontsize=14, labelpad=20)

    # Add x-axis labels
    display.axes[-4].xaxis.set_visible(True)
    display.axes[-4].set_xlabel("Time (min)", fontsize=14, labelpad=-10)
    display.axes[-4].set_xticks([0, len(x_arr)])
    display.axes[-4].set_xticklabels(
        [0, f"{int(x_arr[-1] // 60)}:{str(int(x_arr[-1] % 60)).zfill(2)}"],
    )
    display.axes[-4].tick_params(axis="x", which="both", length=0, labelsize=12)
    display.axes[-4].spines["bottom"].set_position(("outward", 0))

    confounds_ax.text(
        -0.06,
        1.1,
        op.splitext(op.basename(out_file))[0],
        transform=confounds_ax.transAxes,
        size=16,
    )

    fig.savefig(out_file)


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

    dseg_vals = nib.load(dseg).get_fdata()
    n_gm_voxels = np.sum(np.logical_and(dseg_vals <= 3, dseg_vals > 0)) + 1
    n_brain_voxels = np.sum(dseg_vals > 0)

    display = plotting.plot_carpet(
        bold,
        mask_img=dseg,
        detrend=True,
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


def crop_physio_data(physio_arr, samplerate, t_r, nss_count, n_vols):
    """Select portion of physio that corresponds to the fMRI scan."""
    sec_to_drop = nss_count * t_r
    data_start = int(sec_to_drop * samplerate)
    data_end = int((n_vols + nss_count) * t_r * samplerate)
    physio_arr = physio_arr[data_start:data_end, :]
    return physio_arr


def calculate_variance_explained(full_data, reduced_data):
    """Calculate variance explained by denoising method.

    Use in Experiment 2, Analysis Group 4.
    """
    full_dm = full_data - np.mean(full_data, axis=0, keepdims=True)
    red_dm = reduced_data - np.mean(reduced_data, axis=0, keepdims=True)

    # get variance explained by the reduced data
    varexpl = 1 - ((full_dm - red_dm) ** 2.0).sum() / (full_dm ** 2.0).sum()
    return varexpl


def pearson_r(arr1, arr2, alternative="two-sided"):
    """Calculate Pearson correlation coefficient, but allow a specific tailed test.

    Notes
    -----
    Based on
    https://towardsdatascience.com/one-tailed-or-two-tailed-test-that-is-the-question-1283387f631c.

    "alternative" argument from scipy's ttest_1samp.
    """
    assert arr1.ndim == arr2.ndim == 1, f"{arr1.shape} != {arr2.shape}"
    assert arr1.size == arr2.size, f"{arr1.size} != {arr2.size}"
    assert alternative in ("two-sided", "less", "greater")

    r, p = stats.pearsonr(arr1, arr2)

    if alternative == "greater":
        if r > 0:
            p = p / 2
        else:
            p = 1 - (p / 2)
    elif alternative == "less":
        if r < 0:
            p = p / 2
        else:
            p = 1 - (p / 2)

    return r, p


def get_prefixes():
    """Get the prefixes used for each dataset's functional runs."""
    DATASET_PREFIXES = {
        "dset-cambridge": "{participant_id}_task-rest",
        "dset-camcan": "{participant_id}_task-movie",
        "dset-cohen": "{participant_id}_task-bilateralfingertapping",
        "dset-dalenberg": "{participant_id}_task-images",
        "dset-dupre": "{participant_id}_task-rest_run-1",
    }
    return DATASET_PREFIXES


def get_prefixes_mni():
    """Get the prefixes used for each dataset's functional runs."""
    DATASET_PREFIXES = {
        "dset-cambridge": "{participant_id}_task-rest_space-MNI152NLin6Asym",
        "dset-camcan": "{participant_id}_task-movie_space-MNI152NLin6Asym",
        "dset-cohen": "{participant_id}_task-bilateralfingertapping_space-MNI152NLin6Asym",
        "dset-dalenberg": "{participant_id}_task-images_space-MNI152NLin6Asym",
        "dset-dupre": "{participant_id}_task-rest_run-1_space-MNI152NLin6Asym",
    }
    return DATASET_PREFIXES


def get_target_files():
    TARGET_FILES = {
        "TE30": "power/{participant_id}/func/{prefix}_desc-TE30_bold.nii.gz",
        "OC": "tedana/{participant_id}/func/{prefix}_desc-optcom_bold.nii.gz",
        "MEDN": "tedana/{participant_id}/func/{prefix}_desc-optcomDenoised_bold.nii.gz",
        "MEDN Noise": "tedana/{participant_id}/func/{prefix}_desc-optcomDenoised_errorts.nii.gz",
        "MEDN+MIR": "tedana/{participant_id}/func/{prefix}_desc-optcomMIRDenoised_bold.nii.gz",
        "MEDN+MIR Noise": (
            "tedana/{participant_id}/func/{prefix}_desc-optcomMIRDenoised_errorts.nii.gz"
        ),
        "FIT-R2": "t2smap/{participant_id}/func/{prefix}_T2starmap.nii.gz",
        "FIT-S0": "t2smap/{participant_id}/func/{prefix}_S0map.nii.gz",
        "MEDN+GODEC (sparse)": (
            "godec/{participant_id}/func/{prefix}_desc-GODEC_rank-4_bold.nii.gz"
        ),
        "MEDN+GODEC Noise (lowrank)": (
            "godec/{participant_id}/func/{prefix}_desc-GODEC_rank-4_lowrankts.nii.gz"
        ),
        "MEDN+dGSR": "rapidtide/{participant_id}/func/{prefix}_desc-lfofilterCleaned_bold.nii.gz",
        "MEDN+dGSR Noise": (
            "rapidtide/{participant_id}/func/{prefix}_desc-lfofilterCleaned_errorts.nii.gz"
        ),
        "MEDN+aCompCor": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-aCompCor_bold.nii.gz"
        ),
        "MEDN+aCompCor Noise": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-aCompCor_errorts.nii.gz"
        ),
        "MEDN+GSR": "nuisance-regressions/{participant_id}/func/{prefix}_desc-GSR_bold.nii.gz",
        "MEDN+GSR Noise": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-GSR_errorts.nii.gz"
        ),
        "MEDN+Nuis-Reg": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-NuisReg_bold.nii.gz"
        ),
        "MEDN+RV-Reg": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-RVReg_bold.nii.gz"
        ),
        "MEDN+RVT-Reg": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-RVTReg_bold.nii.gz"
        ),
    }
    return TARGET_FILES


def get_bad_subjects_nonphysio():
    BAD_SUBJECTS = (
        ("dset-camcan", "sub-CC110187"),  # SVD failure in tedana
        ("dset-camcan", "sub-CC110411"),  # SVD failure in tedana
        ("dset-camcan", "sub-CC310142"),  # SVD failure in tedana
        ("dset-camcan", "sub-CC420587"),  # SVD failure in tedana
        ("dset-camcan", "sub-CC620026"),  # SVD failure in tedana
    )
    return BAD_SUBJECTS
