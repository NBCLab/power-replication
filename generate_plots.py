"""
Generate plots.
"""
import numpy as np
from nilearn import input_data, datasets
from scipy.spatial.distance import pdist, squareform


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


def moving_average(values, window):
    """Calculate running average along values array
    """
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    buff = np.zeros(int(window / 2)) * np.nan
    sma = np.hstack((buff, sma, buff))[:-1]
    return sma


def scrubbing_analysis(group_timeseries, fds, dists, mat_idx, fd_thresh=0.2,
                       window=1000):
    """
    group_timeseries : list
        List of n_roisXn_timepoints arrays

    """
    n_subjects = len(group_timeseries)
    delta_rs = np.zeros((n_subjects, len(dists)))
    for subj in range(n_subjects):
        ts_arr = group_timeseries[subj]
        fd_arr = fds[subj]
        scrubbed_ts = ts_arr[:, fd_arr <= fd_thresh]
        raw_corrs = np.corrcoef(ts_arr)
        raw_corrs = raw_corrs[mat_idx]
        scrubbed_corrs = np.corrcoef(scrubbed_ts)
        scrubbed_corrs = scrubbed_corrs[mat_idx]
        delta_rs[subj, :] = raw_corrs - scrubbed_corrs  # opposite of Power
    mean_delta_r = np.mean(delta_rs, axis=0)
    sort_idx = dists.argsort()
    sorted_delta_r = mean_delta_r[sort_idx]
    smoothing_curve = moving_average(sorted_delta_r, window)
    return smoothing_curve


def run(imgs, fds, n_iters=100, fd_thresh=0.2):
    assert len(imgs) == len(fds)
    window = 1000
    n_subjects = len(imgs)
    atlas = datasets.fetch_coords_power_2011()
    coords = np.vstack((atlas.rois['x'], atlas.rois['y'], atlas.rois['z'])).T
    mat_idx = np.triu_indices(coords.shape[0], k=1)
    dists = squareform(pdist(coords))
    dists = dists[mat_idx]
    sort_idx = dists.argsort()
    sorted_dists = dists[sort_idx]

    t_r = imgs[0].header.get_zooms()[-1]
    spheres_masker = input_data.NiftiSpheresMasker(
        seeds=coords, radius=5., t_r=t_r,
        smoothing_fwhm=4, detrend=False, standardize=False,
        low_pass=None, high_pass=None)

    # prep for qcrsfc and high-low motion analyses
    mean_fds = np.array([np.mean(fd) for fd in fds])
    raw_corr_mats = np.zeros((n_subjects, len(mat_idx[0])))
    high_motion_idx = mean_fds >= np.median(mean_fds)
    low_motion_idx = mean_fds < np.median(mean_fds)

    # prep for scrubbing analysis
    diff_corr_mats = np.zeros((n_subjects, len(mat_idx[0])))

    ts_all = []
    for i_sub in range(n_subjects):
        raw_ts = spheres_masker.fit_transform(imgs[i_sub]).T
        ts_all.append(raw_ts)
        raw_corrs = np.corrcoef(raw_ts)
        raw_corrs = raw_corrs[mat_idx]
        raw_corr_mats[i_sub, :] = raw_corrs

        # Scrubbing analysis
        scrubbed_ts = raw_ts[:, fds[i_sub] <= fd_thresh]
        scrubbed_corrs = np.corrcoef(scrubbed_ts)
        scrubbed_corrs = scrubbed_corrs[mat_idx]
        corrs_diff = scrubbed_corrs - raw_corrs
        diff_corr_mats[i_sub, :] = corrs_diff

    # QC:RSFC r analysis
    qcrsfc_rs = fast_pearson(raw_corr_mats.T, mean_fds)
    qcrsfc_rs = qcrsfc_rs[sort_idx]
    qcrsfc_smc = moving_average(qcrsfc_rs, window)

    # High-low motion analysis
    high_motion_corr = np.mean(raw_corr_mats[high_motion_idx, :], axis=0)
    low_motion_corr = np.mean(raw_corr_mats[low_motion_idx, :], axis=0)
    hl_corr_diff = high_motion_corr - low_motion_corr
    hl_corr_diff = hl_corr_diff[sort_idx]
    hl_smc = moving_average(hl_corr_diff, window)

    # Scrubbing analysis
    mean_diff_corrs = np.mean(diff_corr_mats, axis=0)
    mean_diff_corrs = mean_diff_corrs[sort_idx]
    scrub_smc = scrubbing_analysis(ts_all, fds, dists, mat_idx, fd_thresh,
                                   window)

    # Null distributions
    fds_copy = [fd.copy() for fd in fds]
    perm_scrub_smc = np.zeros((n_iters, len(dists)))
    perm_qcrsfc_smc = np.zeros((n_iters, len(dists)))
    perm_hl_smc = np.zeros((n_iters, len(dists)))
    for i in range(n_iters):
        # Scrubbing analysis
        perm_fds = [np.random.permutation(perm_fd) for perm_fd in fds_copy]
        temp = scrubbing_analysis(ts_all, perm_fds, dists, mat_idx,
                                  fd_thresh, window)
        perm_scrub_smc[i, :] = temp

        # QC:RSFC analysis
        perm_mean_fds = np.random.permutation(mean_fds)
        perm_qcrsfc_rs = fast_pearson(raw_corr_mats.T, perm_mean_fds)
        perm_qcrsfc_smc[i, :] = moving_average(perm_qcrsfc_rs, window)

        # High-low analysis
        perm_hm_idx = perm_mean_fds >= np.median(perm_mean_fds)
        perm_lm_idx = perm_mean_fds < np.median(perm_mean_fds)
        perm_hm_corr = np.mean(raw_corr_mats[perm_hm_idx, :], axis=0)
        perm_lm_corr = np.mean(raw_corr_mats[perm_lm_idx, :], axis=0)
        perm_hl_smc[i, :] = moving_average(perm_hm_corr - perm_lm_corr,
                                           window)

    results = {'sorted_dists': sorted_dists,
               'qcrsfc_y': qcrsfc_rs,
               'qcrsfc_smc': qcrsfc_smc,
               'qcrsfc_null': perm_qcrsfc_smc,
               'hl_y': hl_corr_diff,
               'hl_smc': hl_smc,
               'hl_null': perm_hl_smc,
               'scrub_y': mean_diff_corrs,
               'scrub_smc': scrub_smc,
               'scrub_null': perm_scrub_smc}
    return results
