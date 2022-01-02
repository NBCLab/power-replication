"""Experiment 1, Analysis Group 5.

Characterizing the relationship between respiration and functional connectivity with and without
denoising.

QC:RSFC and High-Low Motion DDMRA analyses performed on:
- OC
- MEDN Noise
- MEDN
- MEDN+GODEC
- MEDN+Nuis-Reg
- MEDN+RVT-Reg
- MEDN+RV-Reg
- MEDN+aCompCor
- MEDN+dGSR
- MEDN+MIR
- MEDN+GSR

with each of the following used as the quality measure:
- RPV
- mean RV
- mean RVT
"""


def run_ddmra_of_rpv():
    """Run QC:RSFC and high-low analyses on derivatives against RPV."""
    ...


def run_ddmra_of_mean_rv():
    """Run QC:RSFC and high-low analyses on derivatives against mean RV."""
    ...


def run_ddmra_of_mean_rvt():
    """Run QC:RSFC and high-low analyses on derivatives against mean RVT."""
    ...
