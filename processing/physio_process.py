"""
- Instantaneous heart rate from pulse oximeter traces (not yet implemented)
- Respiratory volume per time (RVT)
- Respiratory pattern variability (RPV)
- Respiratory variance (RV)
"""
import os
import os.path as op

import numpy as np
import pandas as pd
from bids.grabbids import BIDSLayout
from scipy.stats import zscore
from scipy.interpolate import interp1d
from scipy.signal import convolve, resample, detrend

# Local install of metco2, adapted to make RVT more flexible
from metco2.utils import physio


def generate_physio_regs(dset, in_dir='/scratch/tsalo006/power-replication/'):
    tr = 3.  # seconds
    samplerate = 50  # Hz
    sr_sec = 1 / samplerate  # sample rate in seconds

    dset_dir = op.join(in_dir, dset)
    layout = BIDSLayout(dset_dir)
    subjects = layout.get_subjects()

    power_dir = op.join(dset_dir, 'derivatives/power')
    if not op.isdir(power_dir):
        os.mkdir(power_dir)

    for subj in subjects:
        power_subj_dir = op.join(power_dir, subj)
        if not op.isdir(power_subj_dir):
            os.mkdir(power_subj_dir)

        preproc_dir = op.join(power_subj_dir, 'preprocessed')
        if not op.isdir(preproc_dir):
            os.mkdir(preproc_dir)

        physio_dir = op.join(preproc_dir, 'physio')
        if not op.isdir(physio_dir):
            os.mkdir(physio_dir)

        func_dir = op.join(dset_dir, subj, 'func')
        physio_tsv = op.join(func_dir,
                             'sub-{0}_task-rest_run-01_physio'
                             '.tsv.gz'.format(subj))
        df = pd.read_csv(physio_tsv, sep='\t', header=None,
                         names=['cardiac', 'respiratory'])
        card = df['cardiac'].values
        n_trs = int((df.shape[0] * sr_sec) / tr)

        # RV
        rv_regs, rv_x_rrf_regs = compute_rv(df, tr, samplerate)
        rv_file = op.join(physio_dir,
                          'sub-{0}_task-rest_run-01_rv.txt'.format(subj))
        rv_x_rrf_file = op.join(physio_dir,
                                'sub-{0}_task-rest_run-01_rvXrrf'
                                '.txt'.format(subj))
        np.savetxt(rv_file, rv_regs)
        np.savetxt(rv_x_rrf_file, rv_x_rrf_regs)

        # RVT
        rvt_regs, rvt_x_rrf_regs = compute_rvt(df, physio_dir, subj, tr,
                                               samplerate)
        rvt_file = op.join(physio_dir,
                           'sub-{0}_task-rest_run-01_rvt.txt'.format(subj))
        rvt_x_rrf_file = op.join(physio_dir,
                                 'sub-{0}_task-rest_run-01_rvtXrrf'
                                 '.txt'.format(subj))
        np.savetxt(rvt_file, rvt_regs)
        np.savetxt(rvt_x_rrf_file, rvt_x_rrf_regs)

        # RPV
        rpv = compute_rpv(df)
        rpv_file = op.join(physio_dir,
                           'sub-{0}_task-rest_run-01_rpv.txt'.format(subj))
        np.savetxt(rpv_file, rpv)


def compute_rpv(df):
    """
    Compute respiratory pattern variability (RPV)
    """
    resp = df['respiratory'].values
    # First, z-score respiratory traces
    resp_z = zscore(resp)
    rpv_lower_env, rpv_upper_env = envelope(resp_z)
    rpv_env = np.hstack((rpv_lower_env[:, None], rpv_upper_env[:, None]))
    rpv = np.std(rpv_env, axis=1)
    return rpv


def compute_rvt(df, physio_dir, subject, tr=3., samplerate=50):
    """
    Compute respiratory-volume-per-time (RVT) regressors
    """
    resp = df['respiratory'].values
    sr_sec = 1 / samplerate
    n_trs = int((resp.shape[0] * sr_sec) / tr)

    # Get respiratory response function (RRF)
    rrf = physio.rrf(tr=tr)

    resp_1d_file = op.join(physio_dir,
                           'sub-{0}_task-rest_run-01_respiratory'
                           '.1d'.format(subject))
    np.savetxt(resp_1d_file, resp)
    rvt = physio.rvt(resp_1d_file, samplerate=samplerate, tr=tr)

    # Resample RVT to 1s from TR
    rvt = resample(rvt, int(n_trs*tr))
    rvt_plus5 = np.hstack((np.zeros(5), rvt[:-5]))
    rvt_plus10 = np.hstack((np.zeros(10), rvt[:-10]))
    rvt_plus15 = np.hstack((np.zeros(15), rvt[:-15]))
    rvt_plus20 = np.hstack((np.zeros(20), rvt[:-20]))

    # Resample back to TR
    rvt = resample(rvt, n_trs)
    rvt_plus5 = resample(rvt_plus5, n_trs)
    rvt_plus10 = resample(rvt_plus10, n_trs)
    rvt_plus15 = resample(rvt_plus15, n_trs)
    rvt_plus20 = resample(rvt_plus20, n_trs)
    rvt_all = np.hstack((rvt[:, None],
                         rvt_plus5[:, None],
                         rvt_plus10[:, None],
                         rvt_plus15[:, None],
                         rvt_plus20[:, None]))

    # Convolve RVT with RRF
    rvt_x_rrf = convolve(rvt, rrf)
    rvt_x_rrf_plus5 = convolve(rvt_plus5, rrf)
    rvt_x_rrf_plus10 = convolve(rvt_plus10, rrf)
    rvt_x_rrf_plus15 = convolve(rvt_plus15, rrf)
    rvt_x_rrf_plus20 = convolve(rvt_plus20, rrf)
    rvt_x_rrf_all = np.hstack((rvt_x_rrf[:, None],
                               rvt_x_rrf_plus5[:, None],
                               rvt_x_rrf_plus10[:, None],
                               rvt_x_rrf_plus15[:, None],
                               rvt_x_rrf_plus20[:, None]))
    rvt_x_rrf_all = rvt_x_rrf_all[:n_trs, :]

    # Remove mean and linear trend terms
    rvt_all = rvt_all - np.mean(rvt_all, axis=0)
    rvt_all = detrend(rvt_all, axis=0)
    rvt_all = zscore(rvt_all, axis=0)

    rvt_x_rrf_all = rvt_x_rrf_all - np.mean(rvt_x_rrf_all, axis=0)
    rvt_x_rrf_all = detrend(rvt_x_rrf_all, axis=0)
    rvt_x_rrf_all = zscore(rvt_x_rrf_all, axis=0)
    return rvt_all, rvt_x_rrf_all


def compute_rv(df, tr=3., samplerate=50):
    """
    Compute respiratory variance regressors
    """
    sr_sec = 1 / samplerate
    n_trs = int((df.shape[0] * sr_sec) / tr)

    # Get respiratory response function (RRF)
    rrf = physio.rrf(tr=tr)

    rv = df['respiratory'].rolling(window=6, center=True).std()
    rv[np.isnan(rv)] = 0.

    # Add 3-second delays
    delay_3 = int(3 / sr_sec)
    rv_minus3 = np.hstack((rv[delay_3:], np.zeros(delay_3)))
    rv_plus3 = np.hstack((np.zeros(delay_3), rv[:-delay_3]))

    # Downsample RV to TR
    rv_ds = resample(rv, n_trs)
    rv_minus3_ds = resample(rv_minus3, n_trs)
    rv_plus3_ds = resample(rv_plus3, n_trs)
    rv_all = np.hstack((rv_minus3_ds[:, None],
                        rv_ds[:, None],
                        rv_plus3_ds[:, None]))

    # Convolve RV with RRF
    rv_x_rrf = convolve(rv_ds, rrf)
    rv_x_rrf_minus3 = convolve(rv_minus3_ds, rrf)
    rv_x_rrf_plus3 = convolve(rv_plus3_ds, rrf)
    rv_x_rrf_all = np.hstack((rv_x_rrf_minus3[:, None],
                              rv_x_rrf[:, None],
                              rv_x_rrf_plus3[:, None]))
    rv_x_rrf_all = rv_x_rrf_all[:n_trs, :]

    # Remove mean and linear trend terms
    rv_all = rv_all - np.mean(rv_all, axis=0)
    rv_all = detrend(rv_all, axis=0)
    rv_all = zscore(rv_all, axis=0)

    rv_x_rrf_all = rv_x_rrf_all - np.mean(rv_x_rrf_all, axis=0)
    rv_x_rrf_all = detrend(rv_x_rrf_all, axis=0)
    rv_x_rrf_all = zscore(rv_x_rrf_all, axis=0)
    return rv_all, rv_x_rrf_all


def envelope(arr):
    """
    Compute envelope of timeseries.

    From https://stackoverflow.com/a/34245942/2589328
    With some adjustments by Taylor Salo
    """
    # Prepend the first value of (s) to the interpolating values.
    # This forces the model to use the same starting point for
    # both the upper and lower envelope models.
    u_x = [0]
    u_y = [arr[0]]

    l_x = [0]
    l_y = [arr[0]]

    # Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y
    # respectively.
    for k in range(1, len(arr)-1):
        if (np.sign(arr[k]-arr[k-1]) == 1) and (np.sign(arr[k]-arr[k+1]) == 1):
            u_x.append(k)
            u_y.append(arr[k])

        if (np.sign(arr[k]-arr[k-1]) == -1) and ((np.sign(arr[k]-arr[k+1])) == -1):
            l_x.append(k)
            l_y.append(arr[k])

    # Append the last value of (s) to the interpolating values.
    # This forces the model to use the same ending point for both the upper and
    # lower envelope models.
    u_x.append(len(arr)-1)
    u_y.append(arr[-1])

    l_x.append(len(arr)-1)
    l_y.append(arr[-1])

    # Fit suitable models to the data.
    u_p = interp1d(u_x, u_y, kind='linear', bounds_error=False, fill_value=0.0)
    l_p = interp1d(l_x, l_y, kind='linear', bounds_error=False, fill_value=0.0)

    upper_env = u_p(range(0, len(arr)))
    lower_env = l_p(range(0, len(arr)))

    return lower_env, upper_env
