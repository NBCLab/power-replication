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
