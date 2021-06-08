"""
Utility functions for processing data.
"""
import numpy as np


def mirrorpad_1d(arr, buffer=250):
    """
    Pad both sides of array with flipped values from
    array of length window.
    """
    mirror = np.flip(arr, axis=0)
    idx = range(arr.shape[0] - buffer, arr.shape[0])
    pre_mirror = np.take(mirror, idx, axis=0)
    idx = range(0, buffer)
    post_mirror = np.take(mirror, idx, axis=0)
    arr_out = np.concatenate((pre_mirror, arr, post_mirror), axis=0)
    return arr_out


def rms_envelope_1d(arr, window=500):
    """
    Conceptual translation of MATLAB 2017b's envelope(X, x, 'rms') function.
    https://www.mathworks.com/help/signal/ref/envelope.html

    Returns
    -------
    rms_env : numpy.ndarray
        The upper envelope.
    """
    assert arr.ndim == 1, "Input data must be 1D"
    assert window % 2 == 0, "Window must be even"
    n_t = arr.shape[0]
    buf = int(window / 2)

    # Pad array at both ends
    arr = np.copy(arr).astype(float)
    mean = np.mean(arr)
    arr -= mean
    arr = mirrorpad_1d(arr, buffer=buf)
    rms_env = np.empty(n_t)
    for i in range(n_t):
        # to match matlab
        start_idx = i + buf
        stop_idx = i + buf + window

        # but this is probably more appropriate
        # start_idx = i + buf - 1
        # stop_idx = i + buf + window
        window_arr = arr[start_idx:stop_idx]
        rms = np.sqrt(np.mean(window_arr ** 2))
        rms_env[i] = rms
    rms_env += mean
    return rms_env
