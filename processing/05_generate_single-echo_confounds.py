"""
Perform standard denoising (not TE-dependent denoising).

Methods:
-   Global signal regression with custom code
    (integrated in tedana, but we do it separately here because the approach is very different)
-   Dynamic global signal regression with rapidtide
-   aCompCor with custom code
-   GODEC
-   RVT (with lags) regression
-   RV (with lags) regression
"""
import json
import os.path as op
from glob import glob

import numpy as np
import pandas as pd
import sklearn
from nilearn import image, masking
from peakdet import load_physio, operations
from phys2denoise.metrics import chest_belt
from scipy import signal


def compile_nuisance_regressors(
    medn_file,
    mask_file,
    seg_file,
    cgm_file,
    confounds_file,
):
    """Generate regressors for aCompCor, GSR, and the nuisance model and write out to file."""
    confounds_json = confounds_file.replace(".tsv", ".json")

    confounds_df = pd.read_table(confounds_file)
    with open(confounds_json, "r") as fo:
        confounds_metadata = json.load(fo)

    # Extract white matter and CSF signals for nuisance regression
    wm_img = image.math_img("img == 6", img=seg_file)
    wm_img = image.math_img(
        "wm_mask * brain_mask", wm_mask=wm_img, brain_mask=mask_file
    )
    wm_data = masking.apply_mask(medn_file, wm_img)

    csf_img = image.math_img("img == 8", img=seg_file)
    csf_img = image.math_img(
        "csf_mask * brain_mask", csf_mask=csf_img, brain_mask=mask_file
    )
    csf_data = masking.apply_mask(medn_file, csf_img)

    confounds_df["NuisanceRegression_WhiteMatter"] = wm_data
    confounds_df["NuisanceRegression_CerebrospinalFluid"] = csf_data
    confounds_metadata["NuisanceRegression_WhiteMatter"] = {
        "Sources": [medn_file, seg_file, mask_file],
        "Description": "Mean signal from deepest white matter mask.",
    }
    confounds_metadata["NuisanceRegression_CerebrospinalFluid"] = {
        "Sources": [medn_file, seg_file, mask_file],
        "Description": "Mean signal from deepest cerebrospinal mask.",
    }

    # Extract and run PCA on white matter for aCompCor
    wm_img = image.math_img("img == 6", img=seg_file)
    wm_img = image.math_img(
        "wm_mask * brain_mask", wm_mask=wm_img, brain_mask=mask_file
    )
    wm_data = masking.apply_mask(medn_file, wm_img)
    pca = sklearn.decomposition.PCA(n_components=5)
    acompcor_components = pca.fit_transform(wm_data)
    acompcor_columns = [
        "aCompCorRegression_Component00",
        "aCompCorRegression_Component01",
        "aCompCorRegression_Component02",
        "aCompCorRegression_Component03",
        "aCompCorRegression_Component04",
    ]
    confounds_df[acompcor_columns] = acompcor_components
    temp_dict = {
        "aCompCorRegression_Component00": {
            "Sources": [medn_file, seg_file, mask_file],
            "Description": "PCA performed on signal from deepest white matter mask.",
        },
        "aCompCorRegression_Component01": {
            "Sources": [medn_file, seg_file, mask_file],
            "Description": "PCA performed on signal from deepest white matter mask.",
        },
        "aCompCorRegression_Component02": {
            "Sources": [medn_file, seg_file, mask_file],
            "Description": "PCA performed on signal from deepest white matter mask.",
        },
        "aCompCorRegression_Component03": {
            "Sources": [medn_file, seg_file, mask_file],
            "Description": "PCA performed on signal from deepest white matter mask.",
        },
        "aCompCorRegression_Component04": {
            "Sources": [medn_file, seg_file, mask_file],
            "Description": "PCA performed on signal from deepest white matter mask.",
        },
    }
    confounds_metadata = {**temp_dict, **confounds_metadata}

    # Extract mean cortical signal for GSR regression
    cgm_mask = image.math_img(
        "cgm_mask * brain_mask",
        cgm_mask=cgm_file,
        brain_mask=mask_file,
    )
    gsr_signal = masking.apply_mask(medn_file, cgm_mask)
    gsr_signal = np.mean(gsr_signal, axis=1)
    confounds_df["GSRRegression_CorticalRibbon"] = gsr_signal
    confounds_metadata["GSRRegression_CorticalRibbon"] = {
        "Sources": [medn_file, cgm_file, mask_file],
        "Description": "Mean signal from cortical gray matter mask.",
    }

    confounds_df.to_csv(confounds_file, sep="\t", index=False)
    with open(confounds_json, "w") as fo:
        json.dump(confounds_metadata, fo, sort_keys=True, indent=4)

    return confounds_file


def compile_physio_regressors(
    medn_file,
    mask_file,
    confounds_file,
    physio_file,
    participants_file,
    subject,
):
    """Generate and save physio-based regressors, including RPV, RV, RVT, and HRV."""
    confounds_json = confounds_file.replace(".tsv", ".json")

    participants_df = pd.read_table(participants_file)
    confounds_df = pd.read_table(confounds_file)
    with open(confounds_json, "r") as fo:
        confounds_metadata = json.load(fo)

    physio_json_file = physio_file.replace(".tsv.gz", ".json")
    physio_data = np.genfromtxt(physio_file)

    n_vols = confounds_df.shape[0]

    # Load metadata
    with open(physio_json_file, "r") as fo:
        physio_metadata = json.load(fo)

    respiratory_column = physio_metadata["Columns"].index("respiratory")
    respiratory_data = physio_data[:, respiratory_column]
    physio_samplerate = physio_metadata["SamplingFrequency"]

    # Normally we'd offset the data by the start time, but in this dataset that's 0
    assert physio_metadata["StartTime"] == 0
    # TODO: Account for dropped non-steady state volumes in regressors!
    # This should require the image's metadata
    # (to extract number of volumes dropped from description) and the TR
    # OR just grab # vols from the tsv I wrote out!

    # Calculate RV regressors and add to confounds file
    # Calculate RV and RV*RRF regressors
    # This includes -3 second and +3 second lags.
    rv_regressors = chest_belt.rv(
        respiratory_data,
        samplerate=physio_samplerate,
        out_samplerate=1 / physio_samplerate,
        window=6,
        lags=[-3 * physio_samplerate, 0, 3 * physio_samplerate],
    )
    n_physio_datapoints = n_vols * physio_samplerate
    rv_regressors = rv_regressors[:n_physio_datapoints, :]
    rv_regressors = signal.resample(rv_regressors, num=n_vols, axis=0)
    rv_regressor_names = [
        "RVRegression_RV-3s",
        "RVRegression_RV",
        "RVRegression_RV+3s",
        "RVRegression_RV*RRF-3s",
        "RVRegression_RV*RRF",
        "RVRegression_RV*RRF+3s",
    ]
    confounds_df[rv_regressor_names] = rv_regressors
    temp_dict = {
        "RVRegression_RV-3s": {
            "Sources": [physio_file],
            "Description": (
                "Respiratory variance time-shifted 3 seconds backward and "
                "downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVRegression_RV": {
            "Sources": [physio_file],
            "Description": (
                "Respiratory variance downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVRegression_RV+3s": {
            "Sources": [physio_file],
            "Description": (
                "Respiratory variance time-shifted 3 seconds forward and "
                "downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVRegression_RV*RRF-3s": {
            "Sources": [physio_file],
            "Description": (
                "Respiratory variance convolved with the respiratory response function, "
                "time-shifted 3 seconds backward, "
                "and downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVRegression_RV*RRF": {
            "Sources": [physio_file],
            "Description": (
                "Respiratory variance convolved with the respiratory response function "
                "and downsampled to the repetition time of the fMRI data."
            ),
        },
        "RVRegression_RV*RRF+3s": {
            "Sources": [physio_file],
            "Description": (
                "Respiratory variance convolved with the respiratory response function, "
                "time-shifted 3 seconds forward, "
                "and downsampled to the repetition time of the fMRI data."
            ),
        },
    }
    confounds_metadata = {**temp_dict, **confounds_metadata}

    # Calculate RVT regressors and add to confounds file

    # Calculate RPV values and add to participants tsv
    window = physio_samplerate * 10  # window should be 10s
    rpv = chest_belt.rpv(respiratory_data, window=window)
    participants_df.loc[participants_df["participant_id"] == subject, "rpv"] = rpv

    # Write out files
    participants_df.to_csv(participants_file, sep="\t", index=False)
    confounds_df.to_csv(confounds_file, sep="\t", index=False)
    with open(confounds_json, "w") as fo:
        json.dump(confounds_metadata, fo, sort_keys=True, indent=4)


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
    sampling_rate = physio_metadata["SamplingRate"]
    card_idx = physio_metadata["Columns"].index("cardiac")
    resp_idx = physio_metadata["Columns"].index("respiratory")
    physio_data = np.loadtxt(physio_file)
    card_data = physio_data[:, card_idx]
    resp_data = physio_data[:, resp_idx]

    card_physio = load_physio(card_data, fs=sampling_rate)
    resp_physio = load_physio(resp_data, fs=sampling_rate)

    # Upsample cardiac signal to 250 Hz
    # Respiratory data will remain at original resolution (40 Hz for 15 subs, 50 Hz for 16)
    target_hz = 250
    card_physio = operations.interpolate_physio(card_physio, target_fs=target_hz)
    card_metadata = physio_metadata.copy()
    card_metadata["SamplingRate"] = target_hz

    # Temporal filtering
    card_physio = operations.filter_physio(card_physio, cutoffs=1.0, method="lowpass")
    resp_physio = operations.filter_physio(resp_physio, cutoffs=1.0, method="lowpass")

    # Peak/trough detection
    card_physio = operations.peakfind_physio(card_physio, thresh=0.1, dist=100)
    resp_physio = operations.peakfind_physio(resp_physio, thresh=0.1, dist=100)

    # Save processed physio to files

    # Save peaks and troughs to files

    # Save history


def main(project_dir, dset):
    """TODO: Create dataset_description.json files."""
    dset_dir = op.join(project_dir, dset)
    deriv_dir = op.join(dset_dir, "derivatives")
    tedana_dir = op.join(deriv_dir, "tedana")
    preproc_dir = op.join(deriv_dir, "power")

    # Get list of participants with good data
    participants_file = op.join(preproc_dir, "participants.tsv")
    participants_df = pd.read_table(participants_file)
    subjects = participants_df.loc[
        participants_df["exclude"] == 0, "participant_id"
    ].tolist()

    for subject in subjects:
        print(f"\t{subject}", flush=True)
        preproc_subj_func_dir = op.join(preproc_dir, subject, "func")
        preproc_subj_anat_dir = op.join(preproc_dir, subject, "anat")
        tedana_subj_dir = op.join(tedana_dir, subject, "func")

        # Collect important files
        confounds_files = glob(
            op.join(preproc_subj_func_dir, "*_desc-confounds_timeseries.tsv")
        )
        assert len(confounds_files) == 1
        confounds_file = confounds_files[0]

        seg_files = glob(
            op.join(
                preproc_subj_anat_dir,
                "*_space-T1w_res-bold_desc-totalMaskWithCSF_mask.nii.gz",
            )
        )
        assert len(seg_files) == 1
        seg_file = seg_files[0]

        cgm_files = glob(
            op.join(preproc_subj_anat_dir, "*_space-T1w_res-bold_label-CGM_mask.nii.gz")
        )
        assert len(cgm_files) == 1
        cgm_file = cgm_files[0]

        medn_files = glob(op.join(tedana_subj_dir, "*_desc-optcomDenoised_bold.nii.gz"))
        assert len(medn_files) == 1
        medn_file = medn_files[0]

        mask_files = glob(op.join(tedana_subj_dir, "*_desc-goodSignal_mask.nii.gz"))
        assert len(mask_files) == 1
        mask_file = mask_files[0]

        # Generate and compile nuisance regressors for aCompCor, GSR, and the nuisance model
        compile_nuisance_regressors(
            medn_file,
            mask_file,
            seg_file,
            cgm_file,
            confounds_file,
        )
        if dset == "dset-dupre":
            compile_physio_regressors(
                medn_file,
                mask_file,
                confounds_file,
                physio_file,
                participants_file,
                subject,
            )


if __name__ == "__main__":
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    dsets = [
        "dset-cambridge",
        "dset-camcan",
        "dset-cohen",
        "dset-dalenberg",
        "dset-dupre",
    ]
    for dset in dsets:
        print(dset, flush=True)
        main(project_dir, dset)
