"""Miscellaneous functions used for analyses."""
from scipy.stats import pearsonr


def pearson_r(arr1, arr2, alternative="two-sided"):
    """Calculate Pearson correlation coefficient, but allow a specific tailed test.

    Notes
    -----
    Based on
    https://towardsdatascience.com/one-tailed-or-two-tailed-test-that-is-the-question-1283387f631c.

    "alternative" argument from scipy's ttest_1samp.
    """
    assert arr1.ndim == arr2.ndim == 1, f"{arr1.shape} != {arr2.shape}"
    assert arr1.size == arr2.size, f"{arr1.size} != {arr2.size}"
    assert alternative in ("two-sided", "less", "greater")

    r, p = pearsonr(arr1, arr2)

    if alternative == "greater":
        if r > 0:
            p = p / 2
        else:
            p = 1 - (p / 2)
    elif alternative == "less":
        if r < 0:
            p = p / 2
        else:
            p = 1 - (p / 2)

    return r, p


def get_prefixes():
    """Get the prefixes used for each dataset's functional runs."""
    DATASET_PREFIXES = {
        "dset-cambridge": "{participant_id}_task-rest",
        "dset-camcan": "{participant_id}_task-movie",
        "dset-cohen": "{participant_id}_task-bilateralfingertapping",
        "dset-dalenberg": "{participant_id}_task-images",
        "dset-dupre": "{participant_id}_task-rest_run-1",
    }
    return DATASET_PREFIXES


def get_target_files():
    TARGET_FILES = {
        "TE30": "power/{participant_id}/func/{prefix}_desc-TE30_bold.nii.gz",
        "OC": "tedana/{participant_id}/func/{prefix}_desc-optcom_bold.nii.gz",
        "MEDN": "tedana/{participant_id}/func/{prefix}_desc-optcomDenoised_bold.nii.gz",
        "MEDN+MIR": "tedana/{participant_id}/func/{prefix}_desc-optcomMIRDenoised_bold.nii.gz",
        "MEDN+MIR Noise": (
            "tedana/{participant_id}/func/{prefix}_desc-optcomMIRDenoised_errorts.nii.gz"
        ),
        "FIT-R2": "t2smap/{participant_id}/func/{prefix}_T2starmap.nii.gz",
        "FIT-S0": "t2smap/{participant_id}/func/{prefix}_S0map.nii.gz",
        "MEDN+GODEC (sparse)": (
            "godec/{participant_id}/func/{prefix}_desc-GODEC_rank-4_bold.nii.gz"
        ),
        "MEDN+GODEC Noise (lowrank)": (
            "godec/{participant_id}/func/{prefix}_desc-GODEC_rank-4_lowrankts.nii.gz"
        ),
        "MEDN+dGSR": "rapidtide/{participant_id}/func/{prefix}_desc-lfofilterCleaned_bold.nii.gz",
        "MEDN+dGSR Noise": (
            "rapidtide/{participant_id}/func/{prefix}_desc-lfofilterCleaned_errorts.nii.gz"
        ),
        "MEDN+aCompCor": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-aCompCor_bold.nii.gz"
        ),
        "MEDN+aCompCor Noise": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-aCompCor_errorts.nii.gz"
        ),
        "MEDN+GSR": "nuisance-regressions/{participant_id}/func/{prefix}_desc-GSR_bold.nii.gz",
        "MEDN+GSR Noise": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-GSR_errorts.nii.gz"
        ),
        "MEDN+Nuis-Reg": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-NuisReg_bold.nii.gz"
        ),
        "MEDN Noise": "tedana/{participant_id}/func/{prefix}_desc-optcomDenoised_errorts.nii.gz",
        "MEDN+RV-Reg": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-RVReg_bold.nii.gz"
        ),
        "MEDN+RVT-Reg": (
            "nuisance-regressions/{participant_id}/func/{prefix}_desc-RVTReg_bold.nii.gz"
        ),
    }
    return TARGET_FILES
