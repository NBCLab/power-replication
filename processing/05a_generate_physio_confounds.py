"""Calculate the physio confounds for the DuPre dataset.

NOTE: I originally included this in 05_generate_single-echo_confounds,
but realized later that I have (1) made a mistake in the output filenames,
(2) failed to calculate and save the upper envelope regressor, and
(3) by running in parallel across subjects, overwrote some of the RPV values in the
participants.tsv file.
"""
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


def calculate_envelope(resp, window):
    """Derive the upper envelope used for calculating RPV.

    This code is adapted directly from
    https://github.com/tsalo/phys2denoise/blob/cd5a6297dc2860063ad971430986bb4c3166c10e/\
        phys2denoise/metrics/chest_belt.py#L15
    """
    # First, z-score respiratory traces
    resp_z = zscore(resp)

    # Collect upper envelope
    rpv_upper_env = physutils.rms_envelope_1d(resp_z, window)

    return rpv_upper_env


def run_peakdet(physio_file, physio_metadata, out_dir):
    """Run peakdet to (1) identify peaks and troughs in data and (2) calculate HRV.

    Notes
    -----
    Per discussion with Ross Markello:
    -   40-50 Hz is sufficient for respiratory data, but upsampling is necessary for HRV
        calculations.
    -   Upsampling may introduce artifacts to cardiac data, so that is a big caveat that we should
        include in our limitations.
    """
    physio_filename = op.basename(physio_file)
    print("\tpeakdet", flush=True)

    # ######################
    # Determine output files
    # ######################
    resp_filename = physio_filename.split("_")
    resp_filename.insert(-1, "desc-respiratory")
    resp_filename = "_".join(resp_filename)
    resp_file = op.join(out_dir, resp_filename)
    resp_metadata_file = resp_file.replace(".tsv.gz", ".json")
    resp_peaks_file = resp_file.replace("_physio.tsv.gz", "_peaks.txt")
    resp_troughs_file = resp_file.replace("_physio.tsv.gz", "_troughs.txt")

    card_filename = physio_filename.split("_")
    card_filename.insert(-1, "desc-cardiac")
    card_filename = "_".join(card_filename)
    card_file = op.join(out_dir, card_filename)
    card_metadata_file = card_file.replace(".tsv.gz", ".json")
    card_peaks_file = card_file.replace("_physio.tsv.gz", "_peaks.txt")
    card_troughs_file = card_file.replace("_physio.tsv.gz", "_troughs.txt")

    out_files = {
        "respiratory-data": resp_file,
        "respiratory-metadata": resp_metadata_file,
        "respiratory-peaks": resp_peaks_file,
        "respiratory-troughs": resp_troughs_file,
        "cardiac-data": card_file,
        "cardiac-metadata": card_metadata_file,
        "cardiac-peaks": card_peaks_file,
        "cardiac-troughs": card_troughs_file,
    }

    # #########
    # Load data
    # #########
    # Split raw data into respiratory and cardiac arrays
    sampling_rate = physio_metadata["SamplingFrequency"]
    resp_idx = physio_metadata["Columns"].index("respiratory")
    card_idx = physio_metadata["Columns"].index("cardiac")
    physio_data = np.loadtxt(physio_file)
    resp_data = physio_data[:, resp_idx]
    card_data = physio_data[:, card_idx]

    # Mean-center data prior to feeding it to peakdet
    # Necessary for the respiratory data in order to detect peaks/troughs
    resp_data = resp_data - np.mean(resp_data)

    card_physio = load_physio(card_data, fs=sampling_rate)
    resp_physio = load_physio(resp_data, fs=sampling_rate)

    # #####################################
    # Perform processing and peak detection
    # #####################################
    # Upsample cardiac signal to 250 Hz
    # Respiratory data will remain at original resolution (40 Hz for 15 subs, 50 Hz for 16)
    TARGET_SAMPLING_RATE = 250
    card_physio = operations.interpolate_physio(
        card_physio,
        target_fs=TARGET_SAMPLING_RATE,
    )

    # Temporal filtering
    resp_physio = operations.filter_physio(resp_physio, cutoffs=1.0, method="lowpass")
    card_physio = operations.filter_physio(card_physio, cutoffs=1.0, method="lowpass")

    # Peak/trough detection
    resp_physio = operations.peakfind_physio(resp_physio, thresh=0.1, dist=100)
    card_physio = operations.peakfind_physio(card_physio, thresh=0.1, dist=100)

    # ###############
    # Update metadata
    # ###############
    resp_metadata = physio_metadata.copy()
    resp_metadata["Columns"] = ["respiratory"]
    resp_metadata["RawSources"] = [physio_file]
    resp_metadata["Description"] = (
        "Respiratory recording data were extracted from the BIDS physio file, "
        "after which low-pass filtering was applied and peaks and troughs were automatically "
        "detected using `peakdet`."
    )
    resp_metadata["PeakDetHistory"] = resp_physio.history

    card_metadata = physio_metadata.copy()
    card_metadata["Columns"] = ["cardiac"]
    card_metadata["SamplingFrequency"] = TARGET_SAMPLING_RATE
    card_metadata["RawSources"] = [physio_file]
    card_metadata["Description"] = (
        "Cardiac recording data were extracted from the BIDS physio file, "
        "after which the data were upsampled to 250 Hz, low-pass filtering was applied, "
        "and peaks and troughs were automatically detected using `peakdet`."
    )
    card_metadata["PeakDetHistory"] = card_physio.history

    # #################
    # Save output files
    # #################
    # Respiratory data
    np.savetxt(resp_file, resp_physio.data, delimiter="\t", newline="\n")
    with open(resp_metadata_file, "w") as fo:
        json.dump(resp_metadata, fo, sort_keys=True, indent=4)

    np.savetxt(resp_peaks_file, resp_physio.peaks)
    np.savetxt(resp_troughs_file, resp_physio.troughs)

    # Cardiac data
    np.savetxt(card_file, card_physio.data, delimiter="\t", newline="\n")
    with open(card_metadata_file, "w") as fo:
        json.dump(card_metadata, fo, sort_keys=True, indent=4)

    np.savetxt(card_peaks_file, card_physio.peaks)
    np.savetxt(card_troughs_file, card_physio.troughs)

    assert all(op.isfile(v) for v in out_files.values())

    return out_files


def compile_physio_regressors(
    medn_file,
    mask_file,
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

    # ####################
    # Respiratory Variance
    # ####################
    # Calculate RV and RV*RRF regressors
    # This includes -3 second and +3 second lags.
    print("\tRV", flush=True)
    rv_regressors = chest_belt.rv(
        resp_data,
        samplerate=resp_samplerate,
        window=6,
    )

    # Apply lags
    rv_regressors_all = []
    lags_in_sec = [-3, 0, 3]
    for i_col in range(rv_regressors.shape[1]):
        rv_regressors_lagged = physutils.apply_lags(
            rv_regressors[:, i_col],
            lags=[lag * resp_samplerate for lag in lags_in_sec],
        )
        rv_regressors_all.append(rv_regressors_lagged)
    rv_regressors_all = np.hstack(rv_regressors_all)
    assert rv_regressors_all.shape[1] == 6

    # Crop out non-steady-state volumes *and* any trailing time
    rv_regressors_all = rv_regressors_all[resp_data_start:resp_data_end, :]

    # Resample to TR
    rv_regressors_all = signal.resample(rv_regressors_all, num=n_vols, axis=0)

    # Add regressors to confounds and update metadata
    rv_regressor_names = [
        "RVRegression_RV-3s",
        "RVRegression_RV",
        "RVRegression_RV+3s",
        "RVRegression_RV*RRF-3s",
        "RVRegression_RV*RRF",
        "RVRegression_RV*RRF+3s",
    ]
    confounds_df[rv_regressor_names] = rv_regressors_all
    temp_dict = {
        "RVRegression_RV-3s": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
            ],
            "Description": (
                "Respiratory variance time-shifted 3 seconds backward and "
                "downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVRegression_RV": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
            ],
            "Description": (
                "Respiratory variance downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVRegression_RV+3s": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
            ],
            "Description": (
                "Respiratory variance time-shifted 3 seconds forward and "
                "downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVRegression_RV*RRF-3s": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
            ],
            "Description": (
                "Respiratory variance convolved with the respiratory response function, "
                "time-shifted 3 seconds backward, "
                "and downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVRegression_RV*RRF": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
            ],
            "Description": (
                "Respiratory variance convolved with the respiratory response function "
                "and downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVRegression_RV*RRF+3s": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
            ],
            "Description": (
                "Respiratory variance convolved with the respiratory response function, "
                "time-shifted 3 seconds forward, "
                "and downsampled to the repetition time of the fMRI data."
            ),
        },
    }
    confounds_metadata = {**temp_dict, **confounds_metadata}

    # ###########################
    # Respiratory Volume-per-Time
    # ###########################
    print("\tRVT", flush=True)
    rvt_regressors = chest_belt.rvt(
        resp_data,
        resp_peaks,
        resp_troughs,
        resp_samplerate,
    )

    # Apply lags
    rvt_regressors_all = []
    lags_in_sec = [0, 5, 10, 15, 20]
    for i_col in range(rvt_regressors.shape[1]):
        rvt_regressors_lagged = physutils.apply_lags(
            rvt_regressors[:, i_col],
            lags=[lag * resp_samplerate for lag in lags_in_sec],
        )
        rvt_regressors_all.append(rvt_regressors_lagged)
    rvt_regressors_all = np.hstack(rvt_regressors_all)
    assert rvt_regressors_all.shape[1] == 10

    # Crop out non-steady-state volumes *and* any trailing time
    rvt_regressors_all = rvt_regressors_all[resp_data_start:resp_data_end, :]

    # Resample to TR
    rvt_regressors_all = signal.resample(rvt_regressors_all, num=n_vols, axis=0)

    # Add regressors to confounds and update metadata
    rvt_regressor_names = [
        "RVTRegression_RVT",
        "RVTRegression_RVT+5s",
        "RVTRegression_RVT+10s",
        "RVTRegression_RVT+15s",
        "RVTRegression_RVT+20s",
        "RVTRegression_RVT*RRF",
        "RVTRegression_RVT*RRF+5s",
        "RVTRegression_RVT*RRF+10s",
        "RVTRegression_RVT*RRF+15s",
        "RVTRegression_RVT*RRF+20s",
    ]
    confounds_df[rvt_regressor_names] = rvt_regressors_all
    temp_dict = {
        "RVTRegression_RVT": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
                physio_files["respiratory-peaks"],
                physio_files["respiratory-troughs"],
            ],
            "Description": (
                "Respiratory volume-per-time downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVTRegression_RVT+5s": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
                physio_files["respiratory-peaks"],
                physio_files["respiratory-troughs"],
            ],
            "Description": (
                "Respiratory volume-per-time time-shifted 5 seconds forward and "
                "downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVTRegression_RVT+10s": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
                physio_files["respiratory-peaks"],
                physio_files["respiratory-troughs"],
            ],
            "Description": (
                "Respiratory volume-per-time time-shifted 10 seconds forward and "
                "downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVTRegression_RVT+15s": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
                physio_files["respiratory-peaks"],
                physio_files["respiratory-troughs"],
            ],
            "Description": (
                "Respiratory volume-per-time time-shifted 15 seconds forward and "
                "downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVTRegression_RVT+20s": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
                physio_files["respiratory-peaks"],
                physio_files["respiratory-troughs"],
            ],
            "Description": (
                "Respiratory volume-per-time time-shifted 20 seconds forward and "
                "downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVTRegression_RVT*RRF": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
                physio_files["respiratory-peaks"],
                physio_files["respiratory-troughs"],
            ],
            "Description": (
                "Respiratory volume-per-time convolved with the respiratory response function "
                "and downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVTRegression_RVT*RRF+5s": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
                physio_files["respiratory-peaks"],
                physio_files["respiratory-troughs"],
            ],
            "Description": (
                "Respiratory volume-per-time convolved with the respiratory response function, "
                "time-shifted 5 seconds forward, "
                "and downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVTRegression_RVT*RRF+10s": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
                physio_files["respiratory-peaks"],
                physio_files["respiratory-troughs"],
            ],
            "Description": (
                "Respiratory volume-per-time convolved with the respiratory response function, "
                "time-shifted 10 seconds forward, "
                "and downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVTRegression_RVT*RRF+15s": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
                physio_files["respiratory-peaks"],
                physio_files["respiratory-troughs"],
            ],
            "Description": (
                "Respiratory volume-per-time convolved with the respiratory response function, "
                "time-shifted 15 seconds forward, "
                "and downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVTRegression_RVT*RRF+20s": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
                physio_files["respiratory-peaks"],
                physio_files["respiratory-troughs"],
            ],
            "Description": (
                "Respiratory volume-per-time convolved with the respiratory response function, "
                "time-shifted 20 seconds forward, "
                "and downsampled to the repetition time of the fMRI data."
            ),
        },
    }
    confounds_metadata = {**temp_dict, **confounds_metadata}

    # ################################
    # Respiratory Pattern Variability
    # ################################
    # Calculate RPV values and add to participants tsv
    print("\tRPV", flush=True)
    window = resp_samplerate * 10  # window should be 10s
    resp_data_from_scan = resp_data[resp_data_start:resp_data_end]
    rpv = chest_belt.rpv(resp_data_from_scan, window=window)
    participants_df.loc[participants_df["participant_id"] == subject, "rpv"] = rpv

    # ###############################################################
    # Upper Envelope from Respiratory Pattern Variability Calculation
    # ###############################################################
    print("\tUpper Envelope from RPV", flush=True)
    envelope_regressor = calculate_envelope(resp_data_from_scan, window=window)

    # Resample to TR
    envelope_regressor = signal.resample(envelope_regressor, num=n_vols, axis=0)

    # Add regressor to confounds and update metadata
    confounds_df["RPVRegression_Envelope"] = envelope_regressor
    temp_dict = {
        "RPVRegression_Envelope": {
            "Sources": [
                physio_files["respiratory-data"],
                physio_files["respiratory-metadata"],
            ],
            "Description": (
                "The upper envelope used to calculate respiratory pattern variability downsampled "
                "to the repetition time of the fMRI data."
            ),
        },
    }
    confounds_metadata = {**temp_dict, **confounds_metadata}

    # ########################
    # Instantaneous Heart Rate
    # ########################
    # Calculate IHR
    print("\tIHR", flush=True)
    ihr = cardiac.ihr(card_data, card_peaks, card_samplerate)

    # Identify "suspicious periods" (bpm change > 25) and interpolate
    bpm_change = np.diff(ihr, prepend=ihr[0])

    # From https://stackoverflow.com/a/45795263/2589328
    value_idx = np.arange(ihr.shape[0])
    unsus_idx = np.where(bpm_change <= 0)
    sus_interpolator = interp1d(value_idx[unsus_idx], ihr[unsus_idx])
    new_ihr = sus_interpolator(value_idx)

    # Crop out non-steady-state volumes *and* any trailing time
    ihr_regressor = ihr[card_data_start:card_data_end]
    new_ihr_regressor = new_ihr[card_data_start:card_data_end]

    # Resample to TR
    ihr_regressor = signal.resample(ihr_regressor, num=n_vols, axis=0)
    new_ihr_regressor = signal.resample(new_ihr_regressor, num=n_vols, axis=0)

    # Add regressors to confounds and update metadata
    confounds_df["IHRRegression_IHR"] = ihr_regressor
    confounds_df["IHRRegression_IHR_Interpolated"] = new_ihr_regressor

    temp_dict = {
        "IHRRegression_IHR": {
            "Sources": [
                physio_files["cardiac-data"],
                physio_files["cardiac-metadata"],
                physio_files["cardiac-peaks"],
            ],
            "Description": (
                "Instantaneous heart rate calculated from cardiac data upsampled to "
                f"{card_samplerate} Hertz, then downsampled to the repetition time of the "
                "fMRI data."
            ),
        },
        "IHRRegression_IHR_Interpolated": {
            "Sources": [
                physio_files["cardiac-data"],
                physio_files["cardiac-metadata"],
                physio_files["cardiac-peaks"],
            ],
            "Description": (
                "Instantaneous heart rate calculated from cardiac data upsampled to "
                f"{card_samplerate} Hertz, then denoised by linearly interpolating "
                "across any periods where instantaneous change in beats-per-minute was "
                "greater than 25, and then downsampled to the repetition time of the "
                "fMRI data."
            ),
        },
    }
    confounds_metadata = {**temp_dict, **confounds_metadata}

    # ###############
    # Write out files
    # ###############
    participants_df.to_csv(participants_file, sep="\t", index=False)
    confounds_df.to_csv(confounds_file, sep="\t", index=False)
    with open(confounds_json, "w") as fo:
        json.dump(confounds_metadata, fo, sort_keys=True, indent=4)


def main(project_dir):
    """Run the confound-generation workflow.

    Notes
    -----
    Example physio file: /home/data/nbc/misc-projects/Salo_PowerReplication/dset-dupre/sub-01/\
        func/sub-01_task-rest_run-01_physio.tsv.gz
    Example physio metadata file: /home/data/nbc/misc-projects/Salo_PowerReplication/dset-dupre/\
        sub-01/sub-01_task-rest_physio.json
    """
    dset = "dset-dupre"
    dset_dir = op.join(project_dir, dset)
    deriv_dir = op.join(dset_dir, "derivatives")
    tedana_dir = op.join(deriv_dir, "tedana")
    preproc_dir = op.join(deriv_dir, "power")

    # Get list of participants with good data,
    # just to determine if the subject is the first in the dataset
    participants_file = op.join(dset_dir, "participants.tsv")
    participants_df = pd.read_table(participants_file)
    subjects = participants_df["participant_id"].tolist()

    # Also get non-steady-state volume information
    nss_file = op.join(preproc_dir, "nss_removed.tsv")

    for subject in subjects:
        dset_subj_dir = op.join(dset_dir, subject)
        dset_subj_func_dir = op.join(dset_subj_dir, "func")
        preproc_subj_func_dir = op.join(preproc_dir, subject, "func")
        tedana_subj_dir = op.join(tedana_dir, subject, "func")

        # Collect important files
        confounds_file = op.join(
            preproc_subj_func_dir,
            f"{subject}_task-rest_run-1_desc-confounds_timeseries.tsv",
        )
        medn_file = op.join(
            tedana_subj_dir,
            f"{subject}_task-rest_run-1_desc-optcomDenoised_bold.nii.gz",
        )
        mask_file = op.join(
            tedana_subj_dir,
            f"{subject}_task-rest_run-1_desc-goodSignal_mask.nii.gz",
        )

        # Get physio data
        physio_metadata_file = op.join(dset_subj_dir, f"{subject}_task-rest_physio.json")
        assert op.isfile(physio_metadata_file)
        physio_file = op.join(
            dset_subj_func_dir,
            f"{subject}_task-rest_run-01_physio.tsv.gz",
        )
        assert op.isfile(physio_file)

        with open(physio_metadata_file, "r") as fo:
            physio_metadata = json.load(fo)

        # Run peakdet
        new_physio_files = run_peakdet(
            physio_file,
            physio_metadata,
            preproc_subj_func_dir,
        )

        # Create the physio regressors
        compile_physio_regressors(
            medn_file,
            mask_file,
            confounds_file,
            new_physio_files,
            participants_file,
            nss_file,
            subject,
        )

        if subject == subjects[0]:
            data_desc_file = op.join(preproc_dir, "dataset_description.json")
            with open(data_desc_file, "r") as fo:
                dataset_description = json.load(fo)

            dataset_description["GeneratedBy"] = [
                {
                    "Name": "phys2denoise",
                    "Description": (
                        "Physiological metric calculation with phys2denoise. "
                        "Metrics calculated with this include RVT, RV, RPV, Envelope, and IHR."
                    ),
                    "CodeURL": (
                        "https://github.com/tsalo/phys2denoise.git@"
                        "be8251db24b157c9a7717f3e2e41eca60ed23649"
                    ),
                },
                {
                    "Name": "peakdet",
                    "Description": (
                        "Low-pass filtering, and peak detection of physiological data with "
                        "peakdet."
                    ),
                    "CodeURL": (
                        "https://github.com/physiopy/peakdet.git@"
                        "f6908e3cebf2fdc31ba73f2b3d3370bf7dfae89c"
                    ),
                },
            ] + dataset_description["GeneratedBy"]

            with open(data_desc_file, "w") as fo:
                json.dump(dataset_description, fo, sort_keys=True, indent=4)


def _main():
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    main(project_dir=project_dir)


if __name__ == "__main__":
    print(__file__, flush=True)
    _main()
