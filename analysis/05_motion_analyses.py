"""
Generate distance-dependent motion-related artifact plots
The rank for the intercept (smoothing curve at 35mm) indexes general dependence
on motion (i.e., a mix of global and focal effects), while the rank for the
slope (difference in smoothing curve at 100mm and 35mm) indexes distance
dependence (i.e., focal effects).

QC:RSFC analysis (QC=FD) for OC, ME-DN, ME-DN S0, ME-DN+GODEC, and ME-DN+GSR
data
- Figure 4, Figure 5
- To expand with ME-DN+dGSR and ME-DN+MIR.

High-low motion analysis for OC, ME-DN, ME-DN S0, ME-DN+GODEC, and ME-DN+GSR
data
- Figure 4
- To expand with ME-DN+dGSR and ME-DN+MIR.

Scrubbing analysis for OC, ME-DN, ME-DN S0, ME-DN+GODEC, and ME-DN+GSR
data
- Figure 4
- To expand with ME-DN+dGSR and ME-DN+MIR.

QC:RSFC analysis (QC=RPV) for OC, ME-DN, ME-DN S0, ME-DN+GODEC, and ME-DN+GSR
data
- Figure 5
- To expand with ME-DN+dGSR and ME-DN+MIR.

QC:RSFC analysis (QC=FD) with censored (FD thresh = 0.2) timeseries for OC,
ME-DN, ME-DN+GODEC, ME-DN+GSR, ME-DN+RPCA, and ME-DN+CompCor data
- Figure S10 (ME-DN, ME-DN+GODEC, ME-DN+GSR)
- Figure S13 (OC, ME-DN, ME-DN+GODEC, ME-DN+GSR, ME-DN+RPCA, ME-DN+CompCor)
- To expand with ME-DN+dGSR and ME-DN+MIR.

High-low motion analysis with censored (FD thresh = 0.2) timeseries for ME-DN,
ME-DN+GODEC, and ME-DN+GSR data
- Figure S10
- To expand with ME-DN+dGSR and ME-DN+MIR.
"""
