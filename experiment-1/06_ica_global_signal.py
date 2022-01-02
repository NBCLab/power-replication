"""Experiment 1, Analysis Group 6.

Does TEDANA retain global BOLD signal in BOLD ICA components?

Carpet plots generated for ICA components and OC data, along with line plots for physiological
traces.

ICA components correlated with mean cortical signal of OC dataset.
-   Record the percentage and number of BOLD-like and non-BOLD-like components correlated with the
    cortical signal at r > 0.5 and r > 0.3 across participants.
-   Mean correlation coefficient for BOLD and non-BOLD components with mean cortical signal is
    calculated for each participant, and distributions of correlation coefficients were compared
    to zero and each other with t-tests.


Mean cortical signal from MEDN is correlated with mean cortical signal from OC for each
participant, and distribution of coefficients is compared to zero with one-sampled t-test.
"""


def plot_components_and_physio():
    """Generate plots for analysis 1."""
    ...


def correlate_ica_with_cortical_signal():
    """Perform analysis 2.

    Correlate each ICA component's time series with the mean cortical signal of the OC dataset.
    Divide the components into BOLD and non-BOLD, then record the percentage and count of each
    type with r > 0.5 and r > 0.3.
    Also record the mean correlation coefficient (after z-transform) for each type.
    Compare the z-transformed coefficients to zero with t-tests.
    """
    ...


def correlate_medn_with_oc():
    """Perform analysis 3.

    Correlate mean cortical signal from MEDN with OC equivalent for each participant,
    convert coefficients to z-values, and perform a t-test against zero on the distribution.
    """
    ...
