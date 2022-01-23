"""Plot physio confounds."""
import json
import os.path as op

import numpy as np
import pandas as pd
from peakdet import load_physio, operations
from phys2denoise.metrics import cardiac, chest_belt
from phys2denoise.metrics import utils as physutils
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import zscore


def plot_confounds(confounds_file):
    ...


def plot_peaks(
    medn_file,
    confounds_file,
    physio_files,
    participants_file,
    nss_file,
    subject,
):
    """Generate and save physio-based regressors, including RPV, RV, RVT, and HRV."""
    medn_json = medn_file.replace(".nii.gz", ".json")
    confounds_json = confounds_file.replace(".tsv", ".json")

    # #########
    # Load data
    # #########
    participants_df = pd.read_table(participants_file)
    confounds_df = pd.read_table(confounds_file)
    nss_df = pd.read_table(nss_file)
    resp_data = np.loadtxt(physio_files["respiratory-data"])
    card_data = np.loadtxt(physio_files["cardiac-data"])

    with open(confounds_json, "r") as fo:
        confounds_metadata = json.load(fo)

    with open(medn_json, "r") as fo:
        medn_metadata = json.load(fo)

    n_vols = confounds_df.shape[0]
    t_r = medn_metadata["RepetitionTime"]
    nss_count = nss_df.loc[nss_df["participant_id"] == subject, "nss_count"].values[0]

    # Load metadata
    with open(physio_files["respiratory-metadata"], "r") as fo:
        resp_metadata = json.load(fo)

    with open(physio_files["cardiac-metadata"], "r") as fo:
        card_metadata = json.load(fo)

    resp_samplerate = resp_metadata["SamplingFrequency"]
    card_samplerate = card_metadata["SamplingFrequency"]

    # Load peaks and troughs
    resp_peaks = np.loadtxt(physio_files["respiratory-peaks"])
    resp_troughs = np.loadtxt(physio_files["respiratory-troughs"])
    card_peaks = np.loadtxt(physio_files["cardiac-peaks"])

    # Normally we'd offset the data by the start time, but in this dataset that's 0
    assert resp_metadata["StartTime"] == 0

    # Account for dropped non-steady state volumes in regressors
    sec_to_drop = nss_count * t_r
    resp_data_start = int(sec_to_drop * resp_samplerate)
    resp_data_end = int((n_vols + nss_count) * t_r * resp_samplerate)
    assert resp_data.shape[0] >= resp_data_end
    card_data_start = int(sec_to_drop * card_samplerate)
    card_data_end = int((n_vols + nss_count) * t_r * card_samplerate)
    assert card_data.shape[0] >= card_data_end

    time = np.linspace(resp_data.shape[0], resp_samplerate, n_vols * resp_samplerate)

    # Adjust the peaks based on NSS volumes as well
    resp_peaks = resp_peaks[resp_peaks < resp_data_end]
    resp_peaks -= resp_data_start
    resp_peaks = resp_peaks[resp_peaks > 0]
    resp_peaks = resp_peaks.astype(int)
    resp_troughs = resp_troughs[resp_troughs < resp_data_end]
    resp_troughs -= resp_data_start
    resp_troughs = resp_troughs[resp_troughs > 0]
    resp_troughs = resp_troughs.astype(int)
    card_peaks = card_peaks[card_peaks < card_data_end]
    card_peaks -= card_data_start
    card_peaks = card_peaks[card_peaks > 0]
    card_peaks = card_peaks.astype(int)

    fig, axes = plt.subplots(figsize=(16, 12), nrows=2)
    axes[0].plot(resp_data, time)


def _main():
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    main(project_dir=project_dir)


if __name__ == "__main__":
    print(__file__, flush=True)
    _main()
