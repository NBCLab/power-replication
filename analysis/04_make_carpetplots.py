"""

"""
import sys
import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from bids.grabbids import BIDSLayout
from nilearn.masking import apply_mask, unmask

sys.path.append('..')
from niworkflows_plots import fMRIPlot


def make_plots(dset, pathstr, method,
               in_dir='/scratch/tsalo006/power-replication/'):
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

    power_dir = op.join(dset_dir, 'derivatives/power')
    fp_dir = op.join(dset_dir, 'derivatives/fmriprep')

    for subj in subjects:
        fp_subj_dir = op.join(fp_dir, subj)
        power_subj_dir = op.join(power_dir, subj)

        preproc_dir = op.join(power_subj_dir, 'preprocessed')
        anat_dir = op.join(preproc_dir, 'anat')

        # Get TR
        func_file = layout.get(subjects=[subj], task='rest', run='01', suffix='.nii.gz')[0]
        metadata = layout.get_metadata(func_file.filename)
        tr = metadata['RepetitionTime']

        # Get confounds
        conf_file = op.join(fp_subj_dir, subj, 'func',
                            'sub-{0}_task-rest_run-01_bold_confounds.tsv')
        conf_df = pd.read_csv(conf_file, sep='\t')
        conf_df = conf_df[['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ',
                           'FramewiseDisplacement']]
        conf_df = conf_df.rename(columns={'X': 'X',
                                          'Y': 'Y',
                                          'Z': 'Z',
                                          'RotX': 'P',
                                          'RotY': 'R',
                                          'RotZ': 'Ya',
                                          'FramewiseDisplacement': 'FD'})
        conf_df = conf_df[['X', 'P', 'Y', 'R', 'Z', 'Ya', 'FD']]

        # Functional data
        func_file = op.join(dset_dir, pathstr.format(subj=subj))
        mask_file = op.join(anat_dir, 'total_mask_no_csf.nii.gz')
        seg_file = op.join(anat_dir, 'total_mask_segmentation_no_csf.nii.gz')
        func_img = nib.load(func_file)
        mask_img = nib.load(mask_file)
        seg_img = nib.load(seg_file)
        plot = fMRIPlot(func_img, mask_img, tr=tr,
                        conf_df=conf_df, seg_nii=seg_img)

        fig = plt.figure(figsize=(16, 8))
        fig = plot.plot(figure=fig)
        fig.tight_layout()
        fig.savefig('carpet_{s}_{m}.png'.format(s=subj, m=method),
                    dpi=400)
