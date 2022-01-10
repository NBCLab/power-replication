"""
Generate carpet plots with associated line plots for visual inspection of
individual subjects' processed data. Vertical bands in the carpet plot
(especially ones that line up with motion or respiration shown in the line
plot) indicate changes in global signal, probably caused by noise.

Example subjects' (2) carpet plot of OC, ME-DN S0, and ME-DN data
- Line plot: motion parameters and FD
- Figure 1

Example subjects' (4) carpet plot of FIT R2 data
- Line plots: (1) motion parameters and FD and (2) respiratory belt and heart
rate
-  Figure 2

Example subjects' (1) carpet plot of ME-DN, ME-DN+GODEC low-rank (removed),
ME-DN+GODEC sparse (retained), ME-DN+GSR betas (removed), and ME-DN+GSR
residuals (retained) data
- Line plots: motion parameters and FD
- Figure 3
- Figure S9: ME-DN, ME-DN+GODEC low-rank, and ME-DN+GODEC sparse for two
additional subjects

Example subjects' (2) carpet plot of OC data and ICA component timeseries
- Line plots: motion parameters and FD
- Figure S2

Example subjects' (1) carpet plot of OC, ME-DN S0, ME-DN R2, TE2, FIT S0, and
FIT R2 data
- Line plots: motion parameters and FD
- Figure S3

Example subjects' (2) carpet plot of FIT R2 data
- Line plots: (1) motion parameters and FD and (2) respiratory belt and heart
rate
- Deep breaths are identified with vertical arrows through line plots and
carpet plots
- Figure S4

Example subjects' (2) carpet plot of ME-DN, ME-DN+RVT, ME-DN+RV data
- Line plots: (1) motion parameters and FD and (2) respiratory belt and heart
rate
- Figure S5

Example subjects' (4) carpet plot of FIT R2 and FIT R2+Nuis data
- Line plots: (1) motion parameters and FD and (2) respiratory belt and heart
rate
- Figure S6

Example subjects' (4) carpet plot of ME-DN and ME-DN+Nuis data
- Line plots: motion parameters and FD
- Figure S7

Example subjects' (1) carpet plot of ME-DN, ME-DN+RPCA low-rank, ME-DN+RPCA
sparse, ME-DN+GODEC low-rank, ME-DN+GODEC sparse, ME-DN+GSR betas, ME-DN+GSR
residuals, ME-DN+CompCor betas, and ME-DN+CompCor residuals data
- No line plots
- Figure S12

Example subjects' (1) carpet plot of ME-DN, ME-DN+RPCA low-rank, and
ME-DN+RPCA sparse data for ascending ranks
- No line plots
- Not to be done
- Figure S14
"""
import os.path as op
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from bids.grabbids import BIDSLayout
from nilearn.masking import apply_mask, unmask

sys.path.append("..")
from niworkflows_plots import fMRIPlot


def make_plots(dset, pathstr, method, in_dir="/scratch/tsalo006/power-replication/"):
    """
    Generate carpet/line plots.

    Parameters
    ----------
    dset : str
        Name of dataset
    pathstr : str
        Subfolder and filename to be formatted.
        E.g., ('derivatives/power/{subj}/denoised/t2smap/'
               'sub-{subj}_task-rest_run-01_t2smap.nii.gz')
    method : str
        Method associated with pathstr (e.g., 't2smap')
    """
    dset_dir = op.join(in_dir, dset)
    layout = BIDSLayout(dset_dir)
    subjects = layout.get_subjects()

    power_dir = op.join(dset_dir, "derivatives/power")
    fp_dir = op.join(dset_dir, "derivatives/fmriprep")

    for subj in subjects:
        fp_subj_dir = op.join(fp_dir, subj)
        power_subj_dir = op.join(power_dir, subj)

        preproc_dir = op.join(power_subj_dir, "preprocessed")
        anat_dir = op.join(preproc_dir, "anat")

        # Get TR
        func_file = layout.get(subjects=[subj], task="rest", run="01", suffix=".nii.gz")[0]
        metadata = layout.get_metadata(func_file.filename)
        tr = metadata["RepetitionTime"]

        # Get confounds
        conf_file = op.join(
            fp_subj_dir, subj, "func", "sub-{0}_task-rest_run-01_bold_confounds.tsv"
        )
        conf_df = pd.read_csv(conf_file, sep="\t")
        conf_df = conf_df[["X", "Y", "Z", "RotX", "RotY", "RotZ", "framewise_displacement"]]
        conf_df = conf_df.rename(
            columns={
                "X": "X",
                "Y": "Y",
                "Z": "Z",
                "RotX": "P",
                "RotY": "R",
                "RotZ": "Ya",
                "framewise_displacement": "FD",
            }
        )
        conf_df = conf_df[["X", "P", "Y", "R", "Z", "Ya", "FD"]]

        # Functional data
        func_file = op.join(dset_dir, pathstr.format(subj=subj))
        mask_file = op.join(anat_dir, "total_mask_no_csf.nii.gz")
        seg_file = op.join(anat_dir, "total_mask_segmentation_no_csf.nii.gz")
        func_img = nib.load(func_file)
        mask_img = nib.load(mask_file)
        seg_img = nib.load(seg_file)
        plot = fMRIPlot(func_img, mask_img, tr=tr, conf_df=conf_df, seg_nii=seg_img)

        fig = plt.figure(figsize=(16, 8))
        fig = plot.plot(figure=fig)
        fig.tight_layout()
        fig.savefig("../figures/{m}/{s}_carpet.png".format(s=subj, m=method), dpi=400)


def run(in_dir="/scratch/tsalo006/power-replication/"):
    dsets = ["ds0", "ds1"]
    methods = [
        "fit_denoised",
        "v3_2_no_gsr",
        "v3_2_wavelet",
        "v3_2_gsr",
        "v2_5_no_gsr",
        "v2_5_wavelet",
        "v2_5_gsr",
    ]
    pathstrs = [
        "denoised/fit_denoised/sub-{subj}_task-rest_run-01_t2smap.nii.gz",
        "denoised/v3_2_no_gsr/sub-{subj}_task-rest_run-01_dn_ts_OC.nii.gz",
        "denoised/v3_2_wavelet/sub-{subj}_task-rest_run-01_dn_ts_OC.nii.gz",
        "denoised/v3_2_gsr/sub-{subj}_task-rest_run-01_dn_ts_OC_T1c.nii.gz",
        "denoised/v2_5_no_gsr/sub-{subj}_task-rest_run-01_dn_ts_OC.nii.gz",
        "denoised/v2_5_wavelet/sub-{subj}_task-rest_run-01_dn_ts_OC.nii.gz",
        "denoised/v2_5_gsr/sub-{subj}_task-rest_run-01_dn_ts_OC_T1c.nii.gz",
    ]
    for dset in dsets:
        for j, method in enumerate(methods):
            make_plots(dset, pathstrs[j], method, in_dir=in_dir)
