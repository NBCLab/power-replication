"""
Generate plots.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import nibabel as nib
import pandas as pd
import statsmodels.api as sm
from nilearn import input_data, datasets
from scipy import stats
from scipy.spatial.distance import pdist, squareform

sns.set_style('whitegrid')


def fast_pearson(X, y):
    """Fast correlations between y and each row of X.
    From http://qr.ae/TU1B9C
    """
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    assert y.shape[0] == X.shape[1]
    y_bar = np.mean(y)
    y_intermediate = y - y_bar
    X_bar = np.mean(X, axis=1)[:, np.newaxis]
    X_intermediate = X - X_bar
    nums = X_intermediate.dot(y_intermediate)
    y_sq = np.sum(np.square(y_intermediate))
    X_sqs = np.sum(np.square(X_intermediate), axis=1)
    denoms = np.sqrt(y_sq * X_sqs)
    pearsons = nums / denoms
    return pearsons


def get_fd(motion):
    # assuming rotations in degrees
    motion[:, :3] = motion[:, :3] * (np.pi/180.) * 50
    motion = np.vstack((np.array([[0, 0, 0, 0, 0, 0]]),
                        np.diff(motion, axis=0)))
    fd = np.sum(np.abs(motion), axis=1)
    return fd


def qcrsfc(imgs, qc, masker):
    assert len(imgs) == len(qc)
    n_subjects = len(imgs)
    idx = np.triu_indices(masker.seeds.shape[0], k=1)
    corr_mats = np.zeros((n_subjects, len(idx[0])))
    for i in range(n_subjects):
        raw_corrs = np.corrcoef(masker.fit_transform(imgs[i]).T)
        corr_mats[i, :] = raw_corrs[idx]

    qcrsfc_rs = fast_pearson(corr_mats.T, qc)
    return qcrsfc_rs


def get_smoothing_curves(corr_mats, fds, dists, masker):
    pass



def run(imgs, fds):
    assert len(imgs) == len(fds)
    n_subjects = len(imgs)
    atlas = datasets.fetch_coords_power_2011()
    coords = np.vstack((atlas.rois['x'], atlas.rois['y'], atlas.rois['z'])).T
    idx = np.triu_indices(coords.shape[0], k=1)
    dists = squareform(pdist(coords))
    dists = dists[idx]

    tr = imgs[0].header.get_zooms()[-1]
    spheres_masker = input_data.NiftiSpheresMasker(
        seeds=coords, radius=5., t_r=tr,
        smoothing_fwhm=None, detrend=False, standardize=False,
        low_pass=None, high_pass=None)

    # prep for qcrsfc and high-low motion analyses
    raw_corr_mats = np.zeros((n_subjects, len(idx[0])))
    high_motion_idx = fds >= np.median(fds)
    low_motion_idx = fds < np.median(fds)

    # prep for scrubbing analysis
    scrub_X = dists[:, None]
    scrub_X = sm.add_constant(scrub_X, prepend=False)
    scrubbing_slopes = []
    scrubbing_inters = []
    diff_corr_mats = np.zeros((n_subjects, len(idx[0])))

    for i in range(n_subjects):
        raw_ts = masker.fit_transform(imgs[i]).T
        raw_corrs = np.corrcoef(raw_ts)
        raw_corrs = raw_corrs[idx]
        raw_corr_mats[i, :] = raw_corrs

        # Scrubbing analysis
        scrubbed_ts = ts[:, fd <= fd_thresh]
        scrubbed_corrs = np.corrcoef(scrubbed_ts)
        scrubbed_corrs = scrubbed_corrs[idx]
        corrs_diff = scrubbed_corrs - raw_corrs
        diff_corr_mats[i, :] = corrs_diff

        # Fit line for scrubbing analysis
        scrub_mod = sm.OLS(corrs_diff, scrub_X)
        scrub_res = scrub_mod.fit()
        m, b = scrub_res.params
        scrubbing_slopes.append(m)
        scrubbing_inters.append(b)

    # QC:RSFC r analysis
    qcrsfc_rs = fast_pearson(raw_corr_mats.T, fds)

    # High-low motion analysis
    high_motion_corr = np.mean(raw_corr_mats[high_motion_idx, :], axis=0)
    low_motion_corr = np.mean(raw_corr_mats[low_motion_idx, :], axis=0)
    hl_corr_diff = high_motion_corr - low_motion_corr

    # Scrubbing analysis
    mean_diff_corrs = np.mean(diff_corr_mats, axis=0)
    slope_t, slope_p = stats.ttest_1samp(scrubbing_slopes, popmean=0)
    inter_t, inter_p = stats.ttest_1samp(scrubbing_inters, popmean=0)
