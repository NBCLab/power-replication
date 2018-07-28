"""
Additional, post-fMRIPrep preprocessing.
"""
import os
import os.path as op

import numpy as np
import nibabel as nib
from bids.grabbids import BIDSLayout
from nilearn.image import resample_to_img
from scipy.ndimage.morphology import binary_erosion


def preprocess(dset, in_dir='/scratch/tsalo006/power-replication/'):
    """
    Perform additional, post-fMRIPrep preprocessing of structural and
    functional MRI data.
    1) Create GM, WM, and CSF masks and resample to 3mm (functional) resolution
    2) Remove first four volumes from each fMRI image
    """
    # LUT values from
    # https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    cort_labels = [3, 17, 18, 19, 20, 42, 53, 54, 55, 56]
    subcort_labels = [9, 10, 11, 12, 13, 26, 48, 49, 50, 52, 58]
    wm_labels = [2, 7, 41, 46, 192]
    csf_labels = [4, 5, 14, 15, 24, 43, 44, 72]
    cereb_labels = [8, 47]

    dset_dir = op.join(in_dir, dset)
    layout = BIDSLayout(dset_dir)
    subjects = layout.get_subjects()

    fp_dir = op.join(dset_dir, 'derivatives/fmriprep')
    fs_dir = op.join(dset_dir, 'derivatives/freesurfer')
    out_dir = op.join(dset_dir, 'derivatives/power')
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    for subject in subjects:
        fp_subj_dir = op.join(fp_dir, subject)
        fs_subj_dir = op.join(fs_dir, subject)
        preproc_dir = op.join(out_dir, subject, 'preprocessed')
        if not op.isdir(preproc_dir):
            os.mkdir(preproc_dir)

        anat_dir = op.join(preproc_dir, 'anat')
        if not op.isdir(anat_dir):
            os.mkdir(anat_dir)

        func_dir = op.join(preproc_dir, 'anat')
        if not op.isdir(func_dir):
            os.mkdir(func_dir)

        # Get echo times in ms as numpy array or list
        echos = layout.get_echoes(subject=subject, modality='func',
                                  type='bold', task='rest', run=1,
                                  extensions=['nii', 'nii.gz'])
        echos = sorted(echos)  # just to be extra safe
        n_echos = len(echos)

        # Create GM, WM, and CSF masks
        # Use preprocessed data from first echo for func resolution
        func_file = op.join(fp_subj_dir, 'func',
                            ('sub-{0}_task-rest_run-01_echo-1_bold_'
                             'space-MNI152NLin2009cAsym_preproc'
                             '.nii.gz').format(subject))
        warp_file = op.join(fp_subj_dir, 'anat',
                            'sub-{0}_run-01_T1w_target-'
                            'MNI152NLin2009cAsym_warp.h5'.format(subject))
        aparc_file = op.join(fs_subj_dir, 'mri/aparc+aseg.mgz')

        # warp T1w-space aparc file to MNI-space
        # how??
        res_aparc_file = op.join(anat_dir, 'res_aparc.nii.gz')

        # select labels for each compartment
        aparc_img = nib.load(res_aparc_file)
        aparc_dat = aparc_img.get_data()
        cort_mask = np.isin(aparc_dat, cort_labels).astype(int)
        subcort_mask = np.isin(aparc_dat, subcort_labels).astype(int)
        wm_mask = np.isin(aparc_dat, wm_labels).astype(int)
        csf_mask = np.isin(aparc_dat, csf_labels).astype(int)
        cereb_mask = np.isin(aparc_dat, cereb_labels).astype(int)

        cort_img = nib.Nifti1Image(cort_mask, aparc_img.affine)
        subcort_img = nib.Nifti1Image(subcort_mask, aparc_img.affine)
        wm_img = nib.Nifti1Image(wm_mask, aparc_img.affine)
        csf_img = nib.Nifti1Image(csf_mask, aparc_img.affine)
        cereb_img = nib.Nifti1Image(cereb_mask, aparc_img.affine)

        func_img = nib.load(func_file)
        aff = wm_img.affine

        # Resample cortical mask to 3mm (functional) resolution with NN interp
        # NOTE: Used for most analyses of "global signal"
        res_cort_img = resample_to_img(cort_img, func_img,
                                       interpolation='nearest')
        res_cort_img.to_filename(op.join(anat_dir, 'cortical_mask.nii.gz'))

        # Resample cortical mask to 3mm (functional) resolution with NN interp
        res_subcort_img = resample_to_img(subcort_img, func_img,
                                          interpolation='nearest')
        res_subcort_img.to_filename(op.join(anat_dir,
                                            'subcortical_mask.nii.gz'))

        # Resample cortical mask to 3mm (functional) resolution with NN interp
        res_cereb_img = resample_to_img(cereb_img, func_img,
                                        interpolation='nearest')
        res_cereb_img.to_filename(op.join(anat_dir, 'cerebellum_mask.nii.gz'))

        # Erode WM mask
        wm_ero0 = wm_img.get_data()
        wm_ero2 = binary_erosion(wm_ero0, iterations=2)
        wm_ero4 = binary_erosion(wm_ero0, iterations=4)

        # Subtract WM mask
        wm_ero02 = wm_ero0 - wm_ero2
        wm_ero24 = wm_ero2 - wm_ero4
        wm_ero02 = nib.Nifti1Image(wm_ero02, aff)  # aka Superficial WM
        wm_ero24 = nib.Nifti1Image(wm_ero24, aff)  # aka Deeper WM
        wm_ero4 = nib.Nifti1Image(wm_ero4, aff)  # aka Deepest WM

        # Resample WM masks to 3mm (functional) resolution with NN interp
        res_wm_ero02 = resample_to_img(wm_ero02, func_img,
                                       interpolation='nearest')
        res_wm_ero24 = resample_to_img(wm_ero24, func_img,
                                       interpolation='nearest')
        res_wm_ero4 = resample_to_img(wm_ero4, func_img,
                                      interpolation='nearest')
        res_wm_ero02.to_filename(op.join(anat_dir, 'wm_ero02.nii.gz'))
        res_wm_ero24.to_filename(op.join(anat_dir, 'wm_ero24.nii.gz'))
        res_wm_ero4.to_filename(op.join(anat_dir, 'wm_ero4.nii.gz'))

        # Erode CSF masks
        csf_ero0 = csf_img.get_data()
        csf_ero2 = binary_erosion(csf_ero0, iterations=2)

        # Subtract CSF masks
        csf_ero02 = csf_ero0 - csf_ero2
        csf_ero02 = nib.Nifti1Image(csf_ero02, aff)  # aka Superficial CSF
        csf_ero2 = nib.Nifti1Image(csf_ero2, aff)  # aka Deeper CSF

        # Resample CSF masks to 3mm (functional) resolution with NN interp
        res_csf_ero02 = resample_to_img(csf_ero02, func_img,
                                        interpolation='nearest')
        res_csf_ero2 = resample_to_img(csf_ero2, func_img,
                                       interpolation='nearest')
        res_csf_ero0.to_filename(op.join(anat_dir, 'csf_ero0.nii.gz'))
        res_csf_ero02.to_filename(op.join(anat_dir, 'csf_ero02.nii.gz'))

        # Remove first four volumes from fMRI volumes
        for i_echo in range(1, n_echos+1):
            echo_file = op.join(fp_subj_dir, 'func',
                                ('sub-{0}_task-rest_run-01_echo-{1}_bold_'
                                 'space-MNI152NLin2009cAsym_preproc'
                                 '.nii.gz').format(subject, i_echo))
            echo_img = nib.load(echo_file)
            echo_data = echo_img.get_data()
            echo_data = echo_data[:, :, :, 4:]
            echo_img = nib.Nifti1Image(echo_data, echo_img.affine)
            echo_img.to_filename(op.join(func_dir,
                                         'sub-{0}_task-rest_run-01_echo-{1}'
                                         '_bold_space-MNI152NLin2009cAsym_'
                                         'powerpreproc'
                                         '.nii.gz'.format(subject, i_echo)))
