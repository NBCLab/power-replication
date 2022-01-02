"""Experiment 1, Analysis Group 1.

Validation of RPV metric.

RPV correlated with mean RV, across participants.

RPV correlated with mean RVT, across participants.

RPV upper envelope (ENV) correlated with RV, then z-transformed and assessed across participants
via t-test.

RPV upper envelope (ENV) correlated with RVT, then z-transformed and assessed across participants
via t-test.
"""


def correlate_rpv_with_mean_rv():
    """Perform analysis 1.

    Correlate RPV with mean RV, across participants.
    Perform one-sided test of significance on correlation coefficient to determine if RPV is
    significantly, positively correlated with mean RV.
    """
    ...


def correlate_rpv_with_mean_rvt():
    """Perform analysis 2.

    Correlate RPV with mean RVT, across participants.
    Perform one-sided test of significance on correlation coefficient to determine if RPV is
    significantly, positively correlated with mean RVT.
    """
    ...


def compare_env_with_rv():
    """Perform analysis 3.

    Correlate ENV (upper envelope used to calculate RPV) with RV for each participant,
    then z-transform the correlation coefficients and perform a one-sample t-test against zero
    with the z-values.
    """
    ...


def compare_env_with_rvt():
    """Perform analysis 4.

    Correlate ENV (upper envelope used to calculate RPV) with RVT for each participant,
    then z-transform the correlation coefficients and perform a one-sample t-test against zero
    with the z-values.
    """
    ...
