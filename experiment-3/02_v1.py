"""Experiment 3, Analysis 2.

ROI analysis of primary visual cortex after denoising.

Mean CNR from V1 ROI for:
- OC
- MEDN
- MEDN+GODEC
- MEDN+GSR
- MEDN+aCompCor
- MEDN+dGSR
- MEDN+MIR
"""
import json

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from nilearn import image, masking
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from scipy import stats


def compare_cnr(
    participants_file,
    target_file_patterns,
    roi_file_pattern,
    metadata_file_pattern,
    events_file_pattern,
):
    participants_df = pd.read_table(participants_file)
    n_subjects_total = participants_df.shape[0]
    participants_df = participants_df.loc[participants_df["exclude"] == 0]
    print(f"Retaining {participants_df.shape[0]}/{n_subjects_total} subjects.")

    cnr_results_df = pd.DataFrame(
        columns=["participant_id", "denoising_method", "cnr", "peak_lag"],
    )

    for i_run, participant_row in participants_df.iterrows():
        subj_id = participant_row["participant_id"]
        metadata_file = metadata_file_pattern.format(participant_id=subj_id)
        with open(metadata_file, "r") as fo:
            metadata = json.load(fo)

        t_r = metadata["RepetitionTime"]

        events_file = events_file_pattern.format(participant_id=subj_id)
        events_df = pd.read_table(events_file)

        roi_file = roi_file_pattern.format(participant_id=subj_id)

        for denoising_method, target_file_pattern in target_file_patterns.items():
            target_file = target_file_pattern.format(participant_id=subj_id)

            target_img = nib.load(target_file)
            n_scans = target_img.shape[3]
            frame_times = np.arange(n_scans) * t_r

            # Fit GLM
            # The images task was modeled using a finite impulse response model-based general
            # linear model, in which the image-presentation portion of the task was modeled with
            # a set of seven impulse regressors set at lags of 0, 1, 2, 3, 4, 5, and 6 volumes
            # after the initial presentation of the image.
            # A finite impulse response-based modeling approach was employed to account for
            # potential deviations from the canonical hemodynamic response function within and
            # across individuals, which are more impactful in event-related designs (such as the
            # images task) than block designs (such as the finger-tapping task).
            # The other stages of the task- particularly the image-rating screens, were modeled
            # using response time as the trial duration and convolved with a double gamma
            # hemodynamic response function.
            # Motion parameters were not included in the general linear model due to the focus of
            # the analyses on the effects of motion, although a high-pass filter of 0.01 Hertz was
            # applied as part of the general linear model.
            # The lag-specific contrast and variance images were then used for each participant in
            # subsequent analyses.
            fir_events = events_df.loc[events_df["trial_type"] == "images"]
            fir_design_matrix = make_first_level_design_matrix(
                frame_times,
                fir_events,
                hrf_model="fir",
                drift_model=None,
                fir_delays=np.arange(0, 7),
            )
            other_events = events_df.loc[events_df["trial_type"] != "images"]
            std_design_matrix = make_first_level_design_matrix(
                frame_times,
                other_events,
                hrf_model="spm",
                drift_model="cosine",
                high_pass=0.01,
            )
            full_design_matrix = pd.concat(
                (fir_design_matrix, std_design_matrix),
                axis=1,
            )

            model = FirstLevelModel(
                t_r=metadata["RepetitionTime"],
                slice_time_ref=metadata["SliceTiming"],  # TODO: Set the real reference slice time
                hrf_model="spm",
                drift_model="cosine",
                high_pass=0.01,
                smoothing_fwhm=None,
                standardize=False,
                signal_scaling=False,
                noise_model="ar1",
                minimize_memory=False,
            )
            model.fit(target_file, design_matrices=full_design_matrix)

            # Grab the appropriate outputs
            fir_delay_conditions = [
                "images_delay_0",
                "images_delay_1",
                "images_delay_2",
                "images_delay_3",
                "images_delay_4",
                "images_delay_5",
                "images_delay_6",
            ]
            cope_imgs, varcope_imgs = [], []
            for fir_delay in fir_delay_conditions:
                outputs = model.compute_contrast(
                    fir_delay, stat_type="t", output_type="all"
                )
                cope_img = outputs["effect_size"]
                varcope_img = outputs["effect_variance"]
                cope_imgs.append(cope_img)
                varcope_imgs.append(varcope_img)

            # Determine the peak lag
            cope_arr = masking.apply_mask(cope_imgs, roi_file)
            voxelwise_peak_lags = np.argmax(cope_arr, axis=0)
            mode = stats.mode(voxelwise_peak_lags)
            modal_peak_lag = mode.mode[0]
            perc = (mode.count[0] / voxelwise_peak_lags.size) * 100
            print(
                f"Peak lag across voxels is {modal_peak_lag} ({perc}% of voxels in ROI)."
            )

            # Select the appropriate FIR delays
            if modal_peak_lag == 0:
                selected_fir_delays = [0, 1, 2]
            elif modal_peak_lag == len(fir_delay_conditions) - 1:
                selected_fir_delays = [
                    len(fir_delay_conditions) - 3,
                    len(fir_delay_conditions) - 2,
                    len(fir_delay_conditions) - 1,
                ]
            else:
                selected_fir_delays = [
                    modal_peak_lag - 1,
                    modal_peak_lag,
                    modal_peak_lag + 1,
                ]

            cope_imgs = [cope_imgs[delay] for delay in selected_fir_delays]
            varcope_imgs = [varcope_imgs[delay] for delay in selected_fir_delays]

            # Calculate CNR
            # Contrast-to-noise was operationalized as the contrast parameter estimates associated
            # with the peak lag, and the two closest lags, across voxels in the region of interest
            # divided by parameter estimate variances and averaged across lags.
            # To determine peak lag, the lag with the highest parameter estimate was identified
            # for each voxel.
            # The most common peak lag across voxels within the ROI was then selected along with
            # the two closest lags in time, although the peak lag was allowed to differ across
            # derivatives.
            # For this peak lag and the surrounding lags, the parameter estimate and parameter
            # estimate variance were selected for each voxel within the ROI.
            # Each voxel’s selected parameter estimate was then divided by its associated parameter
            # estimate variance to produce three voxel-specific CNR estimates (one for each lag)
            # and were then averaged across lags.
            # These estimates were averaged across voxels within the ROI, before being subjected to
            # analysis.
            cnr_imgs = []
            for i_delay, cope_img in enumerate(cope_imgs):
                varcope_img = varcope_imgs[i_delay]
                cnr_img = image.math_img(
                    "cope / varcope", cope=cope_img, varcope=varcope_img
                )
                cnr_imgs.append(cnr_img)

            mean_cnr_img = image.mean_img(cnr_imgs)

            # Extract ROI values
            cnr_arr = masking.apply_mask(mean_cnr_img, roi_file)
            cnr_val = np.mean(cnr_arr)
            cnr_results_df.append([subj_id, denoising_method, cnr_val, modal_peak_lag])

    # Save DataFrame to file
    cnr_results_df.to_csv("analysis02_v1_cnr.tsv", sep="\t", index=False)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.boxplot(x="denoising_method", y="cnr", data=cnr_results_df, ax=ax)
    fig.savefig("analysis02_v1_cnr.png", dpi=400)

    # Perform analysis
    # The group-level distributions of contrast-to-noise values were compared across denoising
    # approaches with an ANOVA in order to determine if any denoising strategies improved or
    # reduced the ability of the model to detect task-related BOLD activity.
    # Post-hoc analyses were performed to identify subsequent patterns in performance.
    res = pg.rm_anova(
        dv="cnr",
        within="denoising_method",
        subject="participant_id",
        data=cnr_results_df,
        detailed=True,
    )
    print(res)

    # Assess significance of interactions and main effect of denoising method,
    # then perform post-hoc analyses as necessary.
    ...
