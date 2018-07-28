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
        power_subj_dir = op.join(out_dir, subject)
        preproc_dir = op.join(power_subj_dir, 'preprocessed')
        denoise_dir = op.join(power_subj_dir, 'denoised')

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
            func_file = op.join(preproc_dir, 'func',
                                ('sub-{0}_task-rest_run-01_echo-{1}'
                                 '_bold_space-MNI152NLin2009cAsym_'
                                 'powerpreproc'
                                 '.nii.gz').format(subject, echo))
            in_files.append(func_file)

        # FIT denoised
        tedana.workflows.t2smap(in_files, echo_times, fitmode='ts',
                                combmode='t2s', label='fit_denoised')

        # TEDANA v3.2 without GSR
        tedana.workflows.tedana(in_files, echo_times, gscontrol=False,
                                wvpca=False, out_dir=denoise_dir,
                                label='v3_2_no_gsr',
                                selection='kundu_v3_2')

        # TEDANA v3.2 with wavelet denoising
        tedana.workflows.tedana(in_files, echo_times, gscontrol=False,
                                wvpca=True, out_dir=denoise_dir,
                                label='v3_2_wavelet',
                                selection='kundu_v3_2')

        # TEDANA v3.2 with GSR
        tedana.workflows.tedana(in_files, echo_times, gscontrol=True,
                                wvpca=False, out_dir=denoise_dir,
                                label='v3_2_gsr',
                                selection='kundu_v3_2')

        # TEDANA v2.5 without GSR
        tedana.workflows.tedana(in_files, echo_times, gscontrol=False,
                                wvpca=False, out_dir=denoise_dir,
                                label='v2_5_no_gsr',
                                selection='kundu_v2_5')

        # TEDANA v2.5 with wavelet denoising
        tedana.workflows.tedana(in_files, echo_times, gscontrol=False,
                                wvpca=True, out_dir=denoise_dir,
                                label='v2_5_wavelet',
                                selection='kundu_v2_5')

        # TEDANA v2.5 with GSR --> Done in fmri_denoise.py
        # Uses the v2_5_no_gsr results and does more processing on them.
