"""Miscellaneous functions used for analyses."""
import logging
import os.path as op
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ddmra import analysis, plotting, utils
from nilearn import datasets, input_data
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

LGR = logging.getLogger("utils")


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

    r, p = pearsonr(arr1, arr2)

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
        "MEDN Noise": "tedana/{participant_id}/func/{prefix}_desc-optcomDenoised_errorts.nii.gz",
        "MEDN+RV-Reg": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-RVReg_bold.nii.gz"
        ),
        "MEDN+RVT-Reg": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-RVTReg_bold.nii.gz"
        ),
    }
    return TARGET_FILES


def run_reduced_analyses(
    files,
    qc,
    out_dir=".",
    confounds=None,
    n_iters=10000,
    n_jobs=1,
    qc_thresh=0.2,
    window=1000,
):
    """Run high-low motion and QCRSFC analyses.

    Parameters
    ----------
    files : (N,) list of nifti files
        List of 4D (X x Y x Z x T) images in MNI space.
    qc : (N,) array_like
        List of summary QC values per img (e.g., mean FD).
    out_dir : str, optional
        Output directory. Default is current directory.
    confounds : None or (N,) list of array-like, optional
        List of 2D (T) numpy arrays with confounds per img.
        Default is None (no confounds are removed).
    n_iters : int, optional
        Number of iterations to run to generate null distributions. Default is 10000.
    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means 'all CPUs'. Default is 1.
    window : int, optional
        Number of units (pairs of ROIs) to include when averaging to generate smoothing curve.
        Default is 1000.

    Notes
    -----
    This function writes out several files to out_dir:
    - ``analysis_values.tsv.gz``: Raw analysis values for analyses.
        Has three columns: distance, qcrsfc, and highlow.
    - ``smoothing_curves.tsv.gz``: Smoothing curve information for analyses.
        Has three columns: distance, qcrsfc, and highlow.
    - ``null_smoothing_curves.npz``:
        Null smoothing curves from each analysis.
        Contains two 2D arrays, where number of columns is same
        size and order as distance column in ``smoothing_curves.tsv.gz``
        and number of rows is number of iterations for permutation analysis.
        The two arrays' keys are 'qcrsfc' and 'highlow'.
    - ``[analysis]_analysis.png``: Figure for each analysis.
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

    # Load atlas and associated masker
    atlas = datasets.fetch_coords_power_2011()
    coords = np.vstack((atlas.rois["x"], atlas.rois["y"], atlas.rois["z"])).T
    n_rois = coords.shape[0]
    triu_idx = np.triu_indices(n_rois, k=1)
    distances = squareform(pdist(coords))
    distances = distances[triu_idx]

    # Sorting index for distances
    edge_sorting_idx = distances.argsort()
    distances = distances[edge_sorting_idx]

    LGR.info("Creating masker")
    spheres_masker = input_data.NiftiSpheresMasker(
        seeds=coords,
        radius=5.0,
        t_r=None,
        smoothing_fwhm=None,
        detrend=False,
        standardize=False,
        low_pass=None,
        high_pass=None,
    )

    # prep for qcrsfc and high-low motion analyses
    mean_qc = np.array([np.mean(subj_qc) for subj_qc in qc])
    z_corr_mats = np.zeros((n_subjects, distances.size))

    # Get correlation matrices
    ts_all = []
    LGR.info("Building correlation matrices")
    if confounds:
        LGR.info("Regressing confounds out of data.")

    good_subjects = []
    for i_subj in range(n_subjects):
        skip_subject = False
        if confounds:
            raw_ts = spheres_masker.fit_transform(files[i_subj], confounds=confounds[i_subj]).T
        else:
            raw_ts = spheres_masker.fit_transform(files[i_subj]).T

        assert raw_ts.shape[0] == n_rois

        if np.any(np.isnan(raw_ts)):
            LGR.warning(f"Time series of {files[i_subj]} contains NaNs. Dropping from analysis.")
            skip_subject = True

        roi_variances = np.var(raw_ts, axis=1)
        if any(roi_variances == 0):
            bad_rois = np.where(roi_variances == 0)[0]
            LGR.warning(
                f"ROI(s) {bad_rois} for {files[i_subj]} have variance of 0. "
                "Dropping from analysis."
            )
            skip_subject = True

        if skip_subject:
            continue

        ts_all.append(raw_ts)
        raw_corrs = np.corrcoef(raw_ts)
        raw_corrs = raw_corrs[triu_idx]
        raw_corrs = raw_corrs[edge_sorting_idx]  # Sort from close to far ROI pairs
        z_corr_mats[i_subj, :] = np.arctanh(raw_corrs)
        good_subjects.append(i_subj)

    del (raw_corrs, raw_ts, spheres_masker, atlas, coords)

    good_subjects = np.array(good_subjects)
    z_corr_mats = z_corr_mats[good_subjects, :]
    qc = [qc[i] for i in good_subjects]
    mean_qc = mean_qc[good_subjects]
    LGR.info(f"Retaining {len(good_subjects)}/{n_subjects} for analysis.")
    if len(good_subjects) < 10:
        raise ValueError("Too few subjects remaining for analysis.")

    analysis_values = pd.DataFrame(columns=["qcrsfc", "highlow"], index=distances)
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
        columns=["qcrsfc", "highlow"],
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

    np.savez_compressed(
        op.join(out_dir, "null_smoothing_curves.npz"),
        qcrsfc=qcrsfc_null_smoothing_curves,
        highlow=hl_null_smoothing_curves,
    )

    del qcrsfc_null_smoothing_curves, hl_null_smoothing_curves

    plot_ddmra_results(out_dir)

    LGR.info("Workflow completed")


def plot_ddmra_results(in_dir):
    """Plot the results for all three analyses from a workflow run and save to a file.

    This function leverages the output file structure of :func:`workflows.run_analyses`.
    It writes out an image (analysis_results.png) to the output directory.

    Parameters
    ----------
    in_dir : str
        Path to the output directory of a ``run_analyses`` run.
    """
    METRIC_LABELS = {
        "qcrsfc": r"QC:RSFC $z_{r}$" + "\n(QC = mean FD)",
        "highlow": "High-low motion\n" + r"${\Delta}z_{r}$",
    }
    YLIMS = {
        "qcrsfc": (-1.0, 1.0),
        "highlow": (-1.0, 1.0),
    }
    analysis_values = pd.read_table(op.join(in_dir, "analysis_values.tsv.gz"))
    smoothing_curves = pd.read_table(op.join(in_dir, "smoothing_curves.tsv.gz"))
    null_curves = np.load(op.join(in_dir, "null_smoothing_curves.npz"))

    fig, axes = plt.subplots(figsize=(8, 24), nrows=len(METRIC_LABELS))

    for i_analysis, (analysis_type, label) in enumerate(METRIC_LABELS.items()):
        values = analysis_values[analysis_type].values
        smoothing_curve = smoothing_curves[analysis_type].values

        fig, axes[i_analysis] = plotting.plot_analysis(
            values,
            analysis_values["distance"],
            smoothing_curve,
            smoothing_curves["distance"],
            null_curves[analysis_type],
            n_lines=50,
            ylim=YLIMS[analysis_type],
            metric_name=label,
            fig=fig,
            ax=axes[i_analysis],
        )

    fig.savefig(op.join(in_dir, "analysis_results.png"), dpi=100)


def get_bad_subjects_nonphysio():
    BAD_SUBJECTS = (
        ("dset-camcan", "sub-CC110187"),  # SVD failure in tedana
        ("dset-camcan", "sub-CC110411"),  # SVD failure in tedana
        ("dset-camcan", "sub-CC310142"),  # SVD failure in tedana
        ("dset-camcan", "sub-CC420587"),  # SVD failure in tedana
        ("dset-camcan", "sub-CC620026"),  # SVD failure in tedana
    )
    return BAD_SUBJECTS
