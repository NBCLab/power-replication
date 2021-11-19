"""
Generate distance-dependent motion-related artifact plots
The rank for the intercept (smoothing curve at 35mm) indexes general dependence
on motion (i.e., a mix of global and focal effects), while the rank for the
slope (difference in smoothing curve at 100mm and 35mm) indexes distance
dependence (i.e., focal effects).

QC:RSFC analysis (QC=FD) for OC, ME-DN, ME-DN S0, ME-DN+GODEC, and ME-DN+GSR
data
- Figure 4, Figure 5
- To expand with ME-DN+dGSR and ME-DN+MIR.

High-low motion analysis for OC, ME-DN, ME-DN S0, ME-DN+GODEC, and ME-DN+GSR
data
- Figure 4
- To expand with ME-DN+dGSR and ME-DN+MIR.

Scrubbing analysis for OC, ME-DN, ME-DN S0, ME-DN+GODEC, and ME-DN+GSR
data
- Figure 4
- To expand with ME-DN+dGSR and ME-DN+MIR.

QC:RSFC analysis (QC=RPV) for OC, ME-DN, ME-DN S0, ME-DN+GODEC, and ME-DN+GSR
data
- Figure 5
- To expand with ME-DN+dGSR and ME-DN+MIR.

QC:RSFC analysis (QC=FD) with censored (FD thresh = 0.2) timeseries for OC,
ME-DN, ME-DN+GODEC, ME-DN+GSR, ME-DN+RPCA, and ME-DN+CompCor data
- Figure S10 (ME-DN, ME-DN+GODEC, ME-DN+GSR)
- Figure S13 (OC, ME-DN, ME-DN+GODEC, ME-DN+GSR, ME-DN+RPCA, ME-DN+CompCor)
- To expand with ME-DN+dGSR and ME-DN+MIR.

High-low motion analysis with censored (FD thresh = 0.2) timeseries for ME-DN,
ME-DN+GODEC, and ME-DN+GSR data
- Figure S10
- To expand with ME-DN+dGSR and ME-DN+MIR.
"""
import logging
import os.path as op
from os import makedirs

import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from ddmra import analysis, plotting, utils
from nilearn import input_data
from scipy.spatial.distance import pdist, squareform

LGR = logging.getLogger("workflows")
sns.set_style("white")


def run_ddmra_in_native_space(
    files,
    mask_files,
    standard_space_coordinates,
    qc,
    out_dir=".",
    confounds=None,
    n_iters=10000,
    n_jobs=1,
    qc_thresh=0.2,
    window=1000,
):
    """Run scrubbing, high-low motion, and QCRSFC analyses.

    This is a copy of the ddmra package's workflow with changes to support
    native-space analysis.

    Parameters
    ----------
    files : (N,) list of nifti files
        List of 4D (X x Y x Z x T) images in MNI space.
    qc : (N,) list of array_like
        List of 1D (T) numpy arrays with QC metric values per img (e.g., FD or respiration).
    out_dir : str, optional
        Output directory. Default is current directory.
    confounds : None or (N,) list of array-like, optional
        List of 2D (T) numpy arrays with confounds per img.
        Default is None (no confounds are removed).
    n_iters : int, optional
        Number of iterations to run to generate null distributions. Default is 10000.
    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means 'all CPUs'. Default is 1.
    qc_thresh : float, optional
        Threshold for QC metric used in scrubbing analysis. Default is 0.2 (for FD).
    window : int, optional
        Number of units (pairs of ROIs) to include when averaging to generate smoothing curve.
        Default is 1000.
    """
    makedirs(out_dir, exist_ok=True)

    # create LGR with 'spam_application'
    LGR.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(op.join(out_dir, "log.tsv"))
    fh.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s\t%(name)-12s\t%(levelname)-8s\t%(message)s")
    fh.setFormatter(formatter)
    # add the handlers to the LGR
    LGR.addHandler(fh)

    LGR.info("Preallocating matrices")
    n_subjects = len(files)
    assert n_subjects == len(mask_files), f"{n_subjects} != {len(mask_files)}"

    # Load atlas and associated masker
    n_rois = standard_space_coordinates.shape[0]
    triu_idx = np.triu_indices(n_rois, k=1)
    distances = squareform(pdist(standard_space_coordinates))
    distances = distances[triu_idx]

    # Sorting index for distances
    edge_sorting_idx = distances.argsort()
    distances = distances[edge_sorting_idx]

    # prep for qcrsfc and high-low motion analyses
    mean_qc = np.array([np.mean(subj_qc) for subj_qc in qc])
    z_corr_mats = np.zeros((n_subjects, distances.size))

    # Get correlation matrices
    ts_all = []
    LGR.info("Building correlation matrices")
    if confounds:
        LGR.info("Regressing confounds out of data.")

    for i_subj in range(n_subjects):
        mask_file = mask_files[i_subj]
        mask_data = nib.load(mask_file).get_fdata()
        assert len(np.unique(mask_data)) == n_rois + 1, (
            f"{len(np.unique(mask_data))} != {n_rois + 1}"
        )
        del mask_data

        subj_space_masker = input_data.NiftiLabelsMasker(
            labels_img=mask_file,
            t_r=None,
            smoothing_fwhm=4.0,
            detrend=False,
            standardize=False,
            low_pass=None,
            high_pass=None,
        )

        if confounds:
            raw_ts = subj_space_masker.fit_transform(files[i_subj], confounds=confounds[i_subj]).T
        else:
            raw_ts = subj_space_masker.fit_transform(files[i_subj]).T

        assert raw_ts.shape[0] == n_rois, f"{raw_ts.shape[0]} != {n_rois}"

        ts_all.append(raw_ts)
        raw_corrs = np.corrcoef(raw_ts)
        raw_corrs = raw_corrs[triu_idx]
        raw_corrs = raw_corrs[edge_sorting_idx]  # Sort from close to far ROI pairs
        z_corr_mats[i_subj, :] = np.arctanh(raw_corrs)

        del subj_space_masker

    del (raw_corrs, raw_ts)

    analysis_values = pd.DataFrame(columns=["qcrsfc", "highlow", "scrubbing"], index=distances)
    analysis_values.index.name = "distance"

    # QC:RSFC r analysis
    LGR.info("Performing QC:RSFC analysis")
    qcrsfc_values = analysis.qcrsfc_analysis(mean_qc, z_corr_mats)
    analysis_values["qcrsfc"] = qcrsfc_values
    qcrsfc_smoothing_curve = utils.moving_average(qcrsfc_values, window)
    qcrsfc_smoothing_curve, smoothing_curve_distances = utils.average_across_distances(
        qcrsfc_smoothing_curve,
        distances,
    )

    # Quick interlude to create the smoothing_curves DataFrame
    smoothing_curves = pd.DataFrame(
        columns=["qcrsfc", "highlow", "scrubbing"],
        index=smoothing_curve_distances,
    )
    smoothing_curves.index.name = "distance"

    smoothing_curves.loc[smoothing_curve_distances, "qcrsfc"] = qcrsfc_smoothing_curve
    del qcrsfc_values, qcrsfc_smoothing_curve

    # High-low motion analysis
    LGR.info("Performing high-low motion analysis")
    highlow_values = analysis.highlow_analysis(mean_qc, z_corr_mats)
    analysis_values["highlow"] = highlow_values
    hl_smoothing_curve = utils.moving_average(highlow_values, window)
    hl_smoothing_curve, smoothing_curve_distances = utils.average_across_distances(
        hl_smoothing_curve,
        distances,
    )
    smoothing_curves.loc[smoothing_curve_distances, "highlow"] = hl_smoothing_curve
    del highlow_values, hl_smoothing_curve

    # Scrubbing analysis
    LGR.info("Performing scrubbing analysis")
    scrub_values = analysis.scrubbing_analysis(qc, ts_all, edge_sorting_idx, qc_thresh, perm=False)
    analysis_values["scrubbing"] = scrub_values
    scrub_smoothing_curve = utils.moving_average(scrub_values, window)
    scrub_smoothing_curve, smoothing_curve_distances = utils.average_across_distances(
        scrub_smoothing_curve,
        distances,
    )
    smoothing_curves.loc[smoothing_curve_distances, "scrubbing"] = scrub_smoothing_curve
    del scrub_values, scrub_smoothing_curve

    analysis_values.reset_index(inplace=True)
    smoothing_curves.reset_index(inplace=True)

    analysis_values.to_csv(
        op.join(out_dir, "analysis_values.tsv.gz"),
        sep="\t",
        line_terminator="\n",
        index=False,
    )
    smoothing_curves.to_csv(
        op.join(out_dir, "smoothing_curves.tsv.gz"),
        sep="\t",
        line_terminator="\n",
        index=False,
    )

    # Null distributions
    LGR.info("Building null distributions with permutations")
    qcrsfc_null_smoothing_curves, hl_null_smoothing_curves = analysis.other_null_distributions(
        qc,
        z_corr_mats,
        distances,
        window=window,
        n_iters=n_iters,
        n_jobs=n_jobs,
    )
    scrub_null_smoothing_curves = analysis.scrubbing_null_distribution(
        qc,
        ts_all,
        distances,
        qc_thresh,
        edge_sorting_idx,
        window=window,
        n_iters=n_iters,
        n_jobs=n_jobs,
    )

    np.savez_compressed(
        op.join(out_dir, "null_smoothing_curves.npz"),
        qcrsfc=qcrsfc_null_smoothing_curves,
        highlow=hl_null_smoothing_curves,
        scrubbing=scrub_null_smoothing_curves,
    )

    del qcrsfc_null_smoothing_curves, hl_null_smoothing_curves, scrub_null_smoothing_curves

    plotting.plot_results(out_dir)
