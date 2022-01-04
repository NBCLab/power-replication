"""Experiment 1, Analysis Group 3.

Characterizing the relationship between respiration and global BOLD signal with and without
denoising.

RPV correlated with SD of mean cortical signal from:
- TE30
- FIT-R2
- MEDN
- MEDN+GODEC
- MEDN+Nuis-Reg
- MEDN+RVT-Reg
- MEDN+RV-Reg
- MEDN+aCompCor
- MEDN+dGSR
- MEDN+MIR
- MEDN+GSR

Plot respiration time series for deep breaths against mean signal from:
- OC
- MEDN
- MEDN+GODEC
- TE30
- FIT-R2
- MEDN+Nuis-Reg
- MEDN+RVT-Reg
- MEDN+RV-Reg
- MEDN+aCompCor
- MEDN+dGSR
- MEDN+MIR
- MEDN+GSR
"""


def correlate_rpv_with_cortical_sd(
    participants_file, target_file_patterns, mask_pattern
):
    """Perform analysis 1.

    Correlate RPV with standard deviation of mean cortical signal from each of the derivatives,
    across participants.
    Perform one-sided test of significance on correlation coefficient to determine if RPV is
    significantly, positively correlated with SD of mean cortical signal after each denoising
    approach.
    """
    ...


def plot_deep_breath_cortical_signal(
    participants_file,
    deep_breath_indices,
    target_file_patterns,
    dseg_pattern,
):
    """Generate plots for analysis 2.

    Use visually-identified indices of deep breaths from the respiratory trace to extract
    time series from 30 seconds before to 40 seconds after the breath from each of the derivatives,
    then plot the mean signals.
    """
    ...
