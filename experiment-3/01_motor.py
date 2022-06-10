"""Experiment 3, Analysis 1.

ROI analysis of motor cortices after denoising.

Mean CNR from left and right finger ROIs for:
- OC
- MEDN
- MEDN+GODEC
- MEDN+GSR
- MEDN+aCompCor
- MEDN+dGSR
- MEDN+MIR

NOTE: Check out the following link for more information on STC:
https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/.
If you're using nilearn (which is used within fitlins to estimate the model) and you would like to
ensure that the model and data are aligned, you can simply shift the values in the frame_times by
+TR/2.
- My solution is to set the slice_time_ref to 0.5, which is the middle of the TR.
"""
import json
import os.path as op
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pingouin as pg  # noqa: E402
import seaborn as sns  # noqa: E402
from nilearn import image, masking  # noqa: E402
from nilearn.glm.first_level import FirstLevelModel  # noqa: E402

sys.path.append("..")

from utils import get_prefixes, get_prefixes_mni, get_target_files  # noqa: E402


def compare_cnr(
    participants_file,
    target_file_patterns,
    roi_file_patterns,
    metadata_file_pattern,
    events_file_pattern,
):
    participants_df = pd.read_table(participants_file)
    participants_df = participants_df.loc[participants_df["dset"] == "dset-cohen"]
    dset_prefix = get_prefixes()["dset-cohen"]
    dset_prefix_mni = get_prefixes_mni()["dset-cohen"]

    n_subjects_total = participants_df.shape[0]
    participants_df = participants_df.loc[participants_df["exclude"] == 0]
    print(f"Retaining {participants_df.shape[0]}/{n_subjects_total} subjects.")

    cnr_results_df = pd.DataFrame(
        columns=["participant_id", "roi", "denoising_method", "cnr"],
    )

    for i_run, participant_row in participants_df.iterrows():
        subj_id = participant_row["participant_id"]
        subj_prefix = dset_prefix.format(participant_id=subj_id)
        subj_prefix_mni = dset_prefix_mni.format(participant_id=subj_id)
        metadata_file = metadata_file_pattern.format(prefix=subj_prefix)
        with open(metadata_file, "r") as fo:
            metadata = json.load(fo)

        t_r = metadata["RepetitionTime"]
        del t_r

        events_file = events_file_pattern.format(prefix=subj_prefix)
        events_df = pd.read_table(events_file)

        for denoising_method, target_file_pattern in target_file_patterns.items():
            target_file = target_file_pattern.format(prefix=subj_prefix_mni)

            # Fit the GLM
            # Each finger tapping block was convolved with a double gamma hemodynamic response
            # function, after which a general linear model was used to fit the predicted BOLD
            # response time series to the denoised data for each participant.
            # Motion parameters were not included in the general linear model due to the focus
            # of the analyses on the effects of motion, although a high-pass filter of 0.01 Hertz
            # was applied as part of the general linear model.
            # The contrast and variance images associated with the finger tapping condition were
            # then used for each participant in subsequent analyses.
            model = FirstLevelModel(
                t_r=metadata["RepetitionTime"],
                slice_time_ref=0.5,
                hrf_model="spm",
                drift_model="cosine",
                high_pass=0.01,
                smoothing_fwhm=None,
                standardize=False,
                signal_scaling=False,
                noise_model="ar1",
                minimize_memory=False,
            )
            model.fit(target_file, events_df)

            # Grab the appropriate outputs
            outputs = model.compute_contrast(
                "fingertapping",
                stat_type="t",
                output_type="all",
            )
            cope_img = outputs["effect_size"]
            varcope_img = outputs["effect_variance"]

            # Calculate CNR
            # Contrast-to-noise was operationalized as contrast parameter estimate
            # (from the cope image) divided by contrast variance (from the varcope image),
            # averaged over the ROI, in a similar manner to Lombardo et al. (2016) [27],
            # but applied at the subject level instead of the group level.
            cnr_img = image.math_img(
                "cope / varcope",
                cope=cope_img,
                varcope=varcope_img,
            )

            # Extract ROI values
            for roi_name, roi_file_pattern in roi_file_patterns.items():
                roi_file = roi_file_pattern.format(dset_prefix_mni=subj_prefix_mni)
                cnr_arr = masking.apply_mask(cnr_img, roi_file)
                cnr_val = np.mean(cnr_arr)
                cnr_results_df.append([subj_id, roi_name, denoising_method, cnr_val])

    # Save DataFrame to file
    cnr_results_df.to_csv("analysis01_motor_cnr.tsv", sep="\t", index=False)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.boxplot(x="denoising_method", y="cnr", hue="roi", data=cnr_results_df, ax=ax)
    fig.savefig("analysis01_motor_cnr.png", dpi=400)

    # Perform analysis
    # The group-level distributions of contrast-to-noise values were compared across denoising
    # approaches with an ANOVA in order to determine if any denoising strategies improved or
    # reduced the ability of the model to detect task-related BOLD activity.
    # Post-hoc analyses were performed to identify subsequent patterns in performance.
    res = pg.rm_anova(
        dv="cnr",
        within=["denoising_method", "roi"],
        subject="participant_id",
        data=cnr_results_df,
        detailed=True,
    )
    print(res)

    # Assess significance of interactions and main effects,
    # then perform post-hoc analyses as necessary.
    ...


if __name__ == "__main__":
    print("Experiment 3, Analysis Group 1")
    project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    in_dir = op.join(project_dir, "{dset}")
    participants_file = op.join(project_dir, "participants.tsv")

    roi_file_patterns = {
        "left": op.join(
            in_dir,
            "derivatives/power/{participant_id}/anat/{prefix_mni}_desc-leftFinger_mask.nii.gz",
        ),
        "right": op.join(
            in_dir,
            "derivatives/power/{participant_id}/anat/{prefix_mni}_desc-rightFinger_mask.nii.gz",
        ),
    }
    metadata_file_pattern = op.join(in_dir, "{participant_id}/func/{prefix}_bold.json")
    events_file_pattern = op.join(in_dir, "{participant_id}/func/{prefix}_events.tsv")

    TARGET_FILE_PATTERNS = get_target_files()
    TARGETS = [
        "OC",
        "MEDN",
        "MEDN+GODEC (sparse)",
        "MEDN+GSR",
        "MEDN+aCompCor",
        "MEDN+dGSR",
        "MEDN+MIR",
    ]
    target_file_patterns = {
        target: op.join(in_dir, "derivatives", TARGET_FILE_PATTERNS[target]) for target in TARGETS
    }
    compare_cnr(
        participants_file,
        target_file_patterns,
        roi_file_patterns,
        metadata_file_pattern,
        events_file_pattern,
    )
