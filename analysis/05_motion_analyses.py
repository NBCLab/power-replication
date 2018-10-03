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
import sys
import os.path as op

import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
from bids.grabbids import BIDSLayout

# Distance-dependent motion-related artifact analysis code
sys.path.append('/Users/tsalo/Documents/tsalo/ddmra/')
import ddmra

sns.set_style('white')


def run_analyses(dset, pathstr, method,
                 in_dir='/scratch/tsalo006/power-replication/'):
    """

    """
    # Constants
    fd_thresh = 0.2
    n_iters = 10000

    dset_dir = op.join(in_dir, dset)
    layout = BIDSLayout(dset_dir)
    subjects = layout.get_subjects()

    power_dir = op.join(dset_dir, 'derivatives/power')
    fp_dir = op.join(dset_dir, 'derivatives/fmriprep')
    out_dir = op.abspath('../figures/{m}/'.format(m=method))

    imgs = []
    fd_all = []
    for subj in subjects:
        fp_subj_dir = op.join(fp_dir, subj)
        power_subj_dir = op.join(power_dir, subj)
        preproc_dir = op.join(power_subj_dir, 'preprocessed')

        # Get confounds
        conf_file = op.join(fp_subj_dir, 'func',
                            'sub-{0}_task-rest_run-01_bold_confounds.tsv'.format(subj))
        conf_df = pd.read_csv(conf_file, sep='\t')
        fd = conf_df['FramewiseDisplacement'].values
        fd_all.append(fd)

        # Functional data
        func_file = op.join(preproc_dir, pathstr.format(subj=subj))
        imgs.append(nib.load(func_file))

    res_file = op.join(out_dir, 'results.txt')
    v1, v2 = 35, 100  # distances to evaluate
    n_lines = min((n_iters, 50))

    # Run analyses
    ddmra.run(imgs, fd_all, out_dir=out_dir, n_iters=n_iters, qc_thresh=fd_thresh)
    smc_sorted_dists = np.loadtxt(op.join(out_dir, 'smc_sorted_distances.txt'))
    all_sorted_dists = np.loadtxt(op.join(out_dir, 'all_sorted_distances.txt'))

    # QC:RSFC analysis
    # Assess significance
    qcrsfc_rs = np.loadtxt(op.join(out_dir, 'qcrsfc_analysis_values.txt'))
    qcrsfc_smc = np.loadtxt(op.join(out_dir, 'qcrsfc_analysis_smoothing_curve.txt'))
    perm_qcrsfc_smc = np.loadtxt(op.join(out_dir, 'qcrsfc_analysis_null_smoothing_curves.txt'))
    intercept = ddmra.get_val(smc_sorted_dists, qcrsfc_smc, v1)
    slope = (ddmra.get_val(smc_sorted_dists, qcrsfc_smc, v1) -
             ddmra.get_val(smc_sorted_dists, qcrsfc_smc, v2))
    perm_intercepts = ddmra.get_val(smc_sorted_dists, perm_qcrsfc_smc, v1)
    perm_slopes = (ddmra.get_val(smc_sorted_dists, perm_qcrsfc_smc, v1) -
                   ddmra.get_val(smc_sorted_dists, perm_qcrsfc_smc, v2))

    p_inter = ddmra.rank_p(intercept, perm_intercepts, tail='upper')
    p_slope = ddmra.rank_p(slope, perm_slopes, tail='upper')
    with open(res_file, 'w') as fo:
        fo.write('QCRSFC analysis results:\n')
        fo.write('\tIntercept = {0:.04f}, p = {1:.04f}\n'.format(intercept, p_inter))
        fo.write('\tSlope = {0:.04f}, p = {1:.04f}\n'.format(-1*slope, p_slope))

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.regplot(all_sorted_dists, qcrsfc_rs,
                ax=ax, scatter=True, fit_reg=False,
                scatter_kws={'color': 'red', 's': 2., 'alpha': 1})
    ax.axhline(0, xmin=0, xmax=200, color='black', linewidth=3)
    for i in range(n_lines):
        ax.plot(smc_sorted_dists, perm_qcrsfc_smc[i, :], color='black')
    ax.plot(smc_sorted_dists, qcrsfc_smc, color='white')
    ax.set_xlabel('Distance (mm)', fontsize=32)
    ax.set_ylabel('QC:RSFC r\n(QC = mean FD)', fontsize=32, labelpad=-30)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([-0.5, 0.5])
    ax.set_yticklabels([-0.5, 0.5], fontsize=32)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xticklabels([])
    ax.set_xlim(0, 160)
    ax.annotate('35 mm: {0:.04f}\n35-100 mm: {1:.04f}'.format(p_inter, p_slope),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom', fontsize=32)
    fig.tight_layout()
    fig.savefig(op.join(out_dir, 'qcrsfc_analysis.png'), dpi=400)
    del qcrsfc_rs, qcrsfc_smc, perm_qcrsfc_smc

    # High-low motion analysis
    # Assess significance
    hl_corr_diff = np.loadtxt(op.join(out_dir, 'highlow_analysis_values.txt'))
    hl_smc = np.loadtxt(op.join(out_dir, 'highlow_analysis_smoothing_curve.txt'))
    perm_hl_smc = np.loadtxt(op.join(out_dir, 'highlow_analysis_null_smoothing_curves.txt'))
    intercept = ddmra.get_val(smc_sorted_dists, hl_smc, v1)
    slope = (ddmra.get_val(smc_sorted_dists, hl_smc, v1) -
             ddmra.get_val(smc_sorted_dists, hl_smc, v2))
    perm_intercepts = ddmra.get_val(smc_sorted_dists, perm_hl_smc, v1)
    perm_slopes = (ddmra.get_val(smc_sorted_dists, perm_hl_smc, v1) -
                   ddmra.get_val(smc_sorted_dists, perm_hl_smc, v2))

    p_inter = ddmra.rank_p(intercept, perm_intercepts, tail='upper')
    p_slope = ddmra.rank_p(slope, perm_slopes, tail='upper')
    with open(res_file, 'a') as fo:
        fo.write('High-low motion analysis results:\n')
        fo.write('\tIntercept = {0:.04f}, p = {1:.04f}\n'.format(intercept, p_inter))
        fo.write('\tSlope = {0:.04f}, p = {1:.04f}\n'.format(-1*slope, p_slope))

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.regplot(all_sorted_dists, hl_corr_diff,
                ax=ax, scatter=True, fit_reg=False,
                scatter_kws={'color': 'red', 's': 2, 'alpha': 1})
    ax.axhline(0, xmin=0, xmax=200, color='black', linewidth=3)
    for i in range(n_lines):
        ax.plot(smc_sorted_dists, perm_hl_smc[i, :], color='black')
    ax.plot(smc_sorted_dists, hl_smc, color='white')
    ax.set_xlabel('Distance (mm)', fontsize=32)
    ax.set_ylabel(r'High-low motion $\Delta$r', fontsize=32)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([-0.5, 0.5])
    ax.set_yticklabels([-0.5, 0.5], fontsize=32)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xticklabels([])
    ax.set_xlim(0, 160)
    ax.annotate('35 mm: {0:.04f}\n35-100 mm: {1:.04f}'.format(p_inter, p_slope),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom', fontsize=32)
    fig.tight_layout()
    fig.savefig(op.join(out_dir, 'highlow_analysis.png'), dpi=400)
    del hl_corr_diff, hl_smc, perm_hl_smc

    # Scrubbing analysis
    mean_delta_r = np.loadtxt(op.join(out_dir, 'scrubbing_analysis_values.txt'))
    scrub_smc = np.loadtxt(op.join(out_dir, 'scrubbing_analysis_smoothing_curve.txt'))
    perm_scrub_smc = np.loadtxt(op.join(out_dir, 'scrubbing_analysis_null_smoothing_curves.txt'))

    # Assess significance
    intercept = ddmra.get_val(smc_sorted_dists, scrub_smc, v1)
    slope = (ddmra.get_val(smc_sorted_dists, scrub_smc, v1) -
             ddmra.get_val(smc_sorted_dists, scrub_smc, v2))

    perm_intercepts = ddmra.get_val(smc_sorted_dists, perm_scrub_smc, v1)
    perm_slopes = (ddmra.get_val(smc_sorted_dists, perm_scrub_smc, v1) -
                   ddmra.get_val(smc_sorted_dists, perm_scrub_smc, v2))
    p_inter = ddmra.rank_p(intercept, perm_intercepts, tail='upper')
    p_slope = ddmra.rank_p(slope, perm_slopes, tail='upper')
    with open(res_file, 'a') as fo:
        fo.write('Scrubbing analysis results:\n')
        fo.write('\tIntercept = {0:.04f}, p = {1:.04f}\n'.format(intercept, p_inter))
        fo.write('\tSlope = {0:.04f}, p = {1:.04f}\n'.format(-1*slope, p_slope))

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.regplot(all_sorted_dists, mean_delta_r,
                ax=ax, scatter=True, fit_reg=False,
                scatter_kws={'color': 'red', 's': 2, 'alpha': 1})
    ax.axhline(0, xmin=0, xmax=200, color='black', linewidth=3)
    for i in range(n_lines):
        ax.plot(smc_sorted_dists, perm_scrub_smc[i, :],
                color='black')
    ax.plot(smc_sorted_dists, scrub_smc,
            color='white')
    ax.set_xlabel('Distance (mm)', fontsize=32)
    ax.set_ylabel(r'Scrubbing $\Delta$r', fontsize=32)
    ax.set_ylim(-0.05, 0.05)
    ax.set_yticks([-0.05, 0.05])
    ax.set_yticklabels([-0.05, 0.05], fontsize=32)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xticklabels([])
    ax.set_xlim(0, 160)
    ax.annotate('35 mm: {0:.04f}\n35-100 mm: {1:.04f}'.format(p_inter, p_slope),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom', fontsize=32)
    fig.tight_layout()
    fig.savefig(op.join(out_dir, 'scrubbing_analysis.png'), dpi=400)
    del mean_delta_r, scrub_smc, perm_scrub_smc


def run(in_dir='/scratch/tsalo006/power-replication/'):
    dsets = ['ds0', 'ds1']
    methods = ['fit_denoised', 'v3_2_no_gsr', 'v3_2_wavelet', 'v3_2_gsr',
               'v2_5_no_gsr', 'v2_5_wavelet', 'v2_5_gsr']
    pathstrs = ['denoised/fit_denoised/sub-{subj}_task-rest_run-01_t2smap.nii.gz',
                'denoised/v3_2_no_gsr/sub-{subj}_task-rest_run-01_dn_ts_OC.nii.gz',
                'denoised/v3_2_wavelet/sub-{subj}_task-rest_run-01_dn_ts_OC.nii.gz',
                'denoised/v3_2_gsr/sub-{subj}_task-rest_run-01_dn_ts_OC_T1c.nii.gz',
                'denoised/v2_5_no_gsr/sub-{subj}_task-rest_run-01_dn_ts_OC.nii.gz',
                'denoised/v2_5_wavelet/sub-{subj}_task-rest_run-01_dn_ts_OC.nii.gz',
                'denoised/v2_5_gsr/sub-{subj}_task-rest_run-01_dn_ts_OC_T1c.nii.gz',]
    for dset in dsets:
        for j, method in enumerate(methods):
            run_analyses(dset, pathstrs[j], method, in_dir=in_dir)
