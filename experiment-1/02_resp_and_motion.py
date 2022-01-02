"""Experiment 1, Analysis Group 2.

Characterizing the relationship between respiration and head motion.

RPV correlated with mean framewise displacement.

RVT correlated with framewise displacement.

RV correlated with framewise displacement.
"""


def correlate_rpv_with_mean_fd():
    """Perform analysis 1.

    Correlate RPV with mean FD across participants.
    Perform one-sided test of significance on correlation coefficient to determine if RPV is
    significantly, positively correlated with mean FD.
    """
    ...


def correlate_rvt_with_fd():
    """Perform analysis 2.

    Correlate RVT with FD for each participant, then z-transform the correlation coefficients
    and perform a one-sample t-test against zero with the z-values.
    """
    ...


def correlate_rv_with_fd():
    """Perform analysis 3.

    Correlate RV with FD for each participant, then z-transform the correlation coefficients
    and perform a one-sample t-test against zero with the z-values.
    """
    ...
