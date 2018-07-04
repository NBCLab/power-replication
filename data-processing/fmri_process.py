"""
Additional, post-fMRIPrep preprocessing.
"""
import os
import os.path as op

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
        out_subj_dir = op.join(out_dir, subject)
        if not op.isdir(out_subj_dir):
            os.mkdir(out_subj_dir)

        # Create GM, WM, and CSF masks
        func_file = op.join(fp_subj_dir, 'func/MNI152_blahblah.nii.gz')
        gm_mask_files = []
        wm_mask_files = []
        csf_mask_files = []
        func_img = nib.load(func_file)
        gm_mask = nib.load(gm_mask_files)
        wm_mask = nib.load(wm_mask_files)
        csf_mask = nib.load(csf_mask_files)
        aff = wm_mask.affine

        # Resample GM mask to 3mm (functional) resolution with NN interp
        res_gm_mask = resample_to_img(gm_mask, func_img,
                                      interpolation='nearest')
        # res_gm_mask.to_filename(op.join(out_subj_dir, 'gm_mask.nii.gz'))

        # Erode WM mask
        wm_ero0 = wm_mask.get_data()
        wm_ero2 = binary_erosion(wm_ero0, iterations=2)
        wm_ero4 = binary_erosion(wm_ero0, iterations=4)

        # Subtract WM mask
        wm_ero02 = wm_ero0 - wm_ero2
        wm_ero24 = wm_ero2 - wm_ero4
        wm_ero0 = wm_mask  # just for readability
        wm_ero02 = nib.Nifti1Image(wm_ero02, aff)
        wm_ero24 = nib.Nifti1Image(wm_ero24, aff)

        # Resample WM masks to 3mm (functional) resolution with NN interp
        res_wm_ero0 = resample_to_img(wm_ero0, func_img,
                                      interpolation='nearest')
        res_wm_ero02 = resample_to_img(wm_ero02, func_img,
                                       interpolation='nearest')
        res_wm_ero24 = resample_to_img(wm_ero24, func_img,
                                       interpolation='nearest')
        # res_wm_ero0.to_filename(op.join(out_subj_dir, 'wm_ero0.nii.gz'))
        # res_wm_ero02.to_filename(op.join(out_subj_dir, 'wm_ero02.nii.gz'))
        # res_wm_ero24.to_filename(op.join(out_subj_dir, 'wm_ero24.nii.gz'))

        # Erode CSF masks
        csf_ero0 = csf_mask.get_data()
        csf_ero2 = binary_erosion(csf_ero0, iterations=2)

        # Subtract CSF masks
        csf_ero02 = csf_ero0 - csf_ero2
        csf_ero0 = csf_mask  # just for readability
        csf_ero02 = nib.Nifti1Image(csf_ero02, aff)

        # Resample CSF masks to 3mm (functional) resolution with NN interp
        res_csf_ero0 = resample_to_img(csf_ero0, func_img,
                                       interpolation='nearest')
        res_csf_ero02 = resample_to_img(csf_ero02, func_img,
                                        interpolation='nearest')
        # res_csf_ero0.to_filename(op.join(out_subj_dir, 'csf_ero0.nii.gz'))
        # res_csf_ero02.to_filename(op.join(out_subj_dir, 'csf_ero02.nii.gz'))

        # Remove first four volumes from fMRI volumes
    return None
