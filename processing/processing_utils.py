"""Utility functions for processing data."""
import os
import subprocess

import numpy as np
import pandas as pd
from nilearn import image, input_data
from phys2denoise.metrics.utils import mirrorpad_1d
from scipy.signal import savgol_filter
from scipy.special import erfcinv


def run_command(command, env=None):
    """Run a given shell command with certain environment variables set."""
    merged_env = os.environ
    if env:
        merged_env.update(env)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        env=merged_env,
    )
    while True:
        line = process.stdout.readline()
        line = str(line, "utf-8")[:-1]
        print(line)
        if line == "" and process.poll() is not None:
            break

    if process.returncode != 0:
        raise Exception(
            "Non zero return code: {0}\n"
            "{1}\n\n{2}".format(process.returncode, command, process.stdout.read())
        )


def _generic_regression(medn_file, mask_file, nuisance_regressors, t_r):
    # Calculate mean-centered version of MEDN data
    mean_img = image.mean_img(medn_file)
    medn_mean_centered_img = image.math_img(
        "img - avg_img[..., None]",
        img=medn_file,
        avg_img=mean_img,
    )

    # Regress confounds out of MEDN data
    # Note that confounds should be mean-centered and detrended, which the masker will do.
    # See "fMRI data: nuisance regressions" section of Power appendix (page 3).
    regression_masker = input_data.NiftiMasker(
        mask_file,
        smoothing_fwhm=None,
        standardize=False,
        standardize_confounds=True,  # will mean-center them too
        high_variance_confounds=False,
        low_pass=None,
        high_pass=None,
        detrend=True,  # linearly detrends both confounds and data
        t_r=t_r,
        reports=False,
    )
    regression_masker.fit(medn_mean_centered_img)
    # Mask + remove confounds
    denoised_data = regression_masker.transform(
        medn_mean_centered_img,
        confounds=nuisance_regressors,
    )
    # Mask without removing confounds
    raw_data = regression_masker.transform(
        medn_mean_centered_img,
        confounds=None,
    )
    # Calculate residuals (both raw and denoised should have same scale)
    noise_data = raw_data - denoised_data
    denoised_img = regression_masker.inverse_transform(denoised_data)
    noise_img = regression_masker.inverse_transform(noise_data)
    return denoised_img, noise_img


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    From https://stackoverflow.com/a/6520696/2589328

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def median_linear_fill(data):
    """Replicate MATLAB's filloutliers' movmedian+linear combination."""
    data = np.array(data)
    med = np.median(data)
    c = -1 / (np.sqrt(2) * erfcinv(3 / 2))
    mad = c * np.median(np.abs(data - med))
    # Outliers are defined as elements more than three scaled MAD from the median.
    # The scaled MAD is defined as c*median(abs(A-median(A))), where c=-1/(sqrt(2)*erfcinv(3/2)).
    outlier_thresh = mad * 3
    data[np.abs(data) > outlier_thresh] = np.nan
    nans, x = nan_helper(data)
    data[nans] = np.interp(x(nans), x(~nans), data[~nans])
    # Return the center data point of the window
    return data[data.size // 2]


def filloutliers(data, samplerate=50):
    """Replicate MATLAB's filloutliers.

    Notes
    -----
    Replicates the following command:
    R = filloutliers(R,'linear','movmedian',100);
    """
    assert isinstance(samplerate, int), "samplerate must be an integer"
    window = 2 * samplerate

    # Use mirror-padding to work around values at the beginning and end of the array.
    # NOTE: I don't know if MATLAB does this.
    buf = int(window / 2)
    data = mirrorpad_1d(data, buffer=buf)

    # Run the rolling outlier detection/interpolation.
    filled_data = (
        pd.Series(data).rolling(window=window, center=True).apply(median_linear_fill)
    )

    # Trim out the buffer.
    filled_data = filled_data[buf: filled_data.size - buf]

    return filled_data


def smoothdata(data, samplerate=50):
    """Replicate R = smoothdata(R,'sgolay',50).

    Notes
    -----
    MATLAB's documentation of sgolay option:
    Savitzky-Golay filter, which smooths according to a quadratic polynomial that is fitted over
    each window of A. This method can be more effective than other methods when the data varies
    rapidly.
    """
    smoothed_data = savgol_filter(
        data, window_length=samplerate + 1, polyorder=2, mode="mirror"
    )
    return smoothed_data
