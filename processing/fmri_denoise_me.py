"""
Perform multi-echo denoising strategies with tedana.

Also included in this (because it is implemented in tedana) is wavelet
denoising.
"""
import os
import os.path as op
import tedana
from bids.grabbids import BIDSLayout


def run_tedana(dset, in_dir='/scratch/tsalo006/power-replication/'):
    """
    Run tedana workflows.
    - Basic (without GSR, GODEC, or wavelet denoising)
    - With GSR
    - With GODEC
    - With wavelet denoising
    """
    dset_dir = op.join(in_dir, dset)
    layout = BIDSLayout(dset_dir)
    subjects = layout.get_subjects()

    fp_dir = op.join(dset_dir, 'derivatives/fmriprep')
    out_dir = op.join(dset_dir, 'derivatives/power')
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    for subject in subjects:
        fp_subj_dir = op.join(fp_dir, subject)
        out_subj_dir = op.join(out_dir, subject)
        if not op.isdir(out_subj_dir):
            os.mkdir(out_subj_dir)

        # Get echo times in ms as numpy array or list
        echos = layout.get_echoes(subject=subject, modality='func',
                                  type='bold', task='rest', run=1,
                                  extensions=['nii', 'nii.gz'])
        echos = sorted(echos)  # just to be extra safe

        # Get 4D preprocessed file associated with each echo in list
        in_files = []
        echo_times = []
        for echo in echos:
            # Get echo time in ms
            orig_file = layout.get(subject=subject, modality='func',
                                   type='bold', task='rest', run=1,
                                   extensions=['nii', 'nii.gz'], echo=echo)
            metadata = layout.get_metadata(orig_file[0].filename)
            echo_time = metadata['EchoTime'] * 1000
            echo_times.append(echo_time)

            # Get preprocessed file associated with echo
            func_file = op.join(fp_subj_dir, 'func',
                                ('sub-{0}_task-rest_run-01_echo-{1}_bold_'
                                 'space-MNI152NLin2009cAsym_preproc'
                                 '.nii.gz').format(subject, echo))
            in_files.append(func_file)

        # FIT denoised
        tedana.workflows.t2smap(in_files, echo_times, fitmode='ts')

        # TEDANA v3.2 without GSR
        tedana.workflows.tedana(in_files, echo_times, gscontrol=False,
                                label='test', wvpca=False)

        # TEDANA v3.2 with wavelet denoising
        tedana.workflows.tedana(in_files, echo_times, gscontrol=False,
                                label='test', wvpca=True)

        # TEDANA v3.2 with GSR
        tedana.workflows.tedana(in_files, echo_times, gscontrol=True,
                                label='test', wvpca=False)

        # TEDANA v2.5 without GSR
        # TEDANA v2.5 with wavelet denoising
        # TEDANA v2.5 with GSR --> Done in fmri_denoise.py
