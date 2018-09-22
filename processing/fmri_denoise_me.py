"""
Perform multi-echo denoising strategies with tedana.

Also included in this (because it is implemented in tedana) is wavelet
denoising.
"""
import os
import os.path as op
import tedana
from bids.grabbids import BIDSLayout


def run_tedana(dset, task, in_dir='/scratch/tsalo006/power-replication/'):
    """
    Run tedana workflows.
    - Basic (without GSR or GODEC)
    - With GSR
    - With GODEC

    Parameters
    ----------
    dset : {'ds000210', 'ds000254', 'ds000258'}
    task : {'rest', 'fingertapping'}
    in_dir : str
    """
    dset_dir = op.join(in_dir, dset)
    layout = BIDSLayout(dset_dir)
    subjects = layout.get_subjects()

    out_dir = op.join(dset_dir, 'derivatives/power')
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    for subject in subjects:
        power_subj_dir = op.join(out_dir, subject)
        preproc_dir = op.join(power_subj_dir, 'preprocessed')
        denoise_dir = op.join(power_subj_dir, 'denoised')

        # Get echo times in ms as numpy array or list
        echos = layout.get_echoes(subject=subject, modality='func',
                                  type='bold', task=task, run=1,
                                  extensions=['nii', 'nii.gz'])
        echos = sorted(echos)  # just to be extra safe

        # Get 4D preprocessed file associated with each echo in list
        in_files = []
        echo_times = []
        for echo in echos:
            # Get echo time in ms
            orig_file = layout.get(subject=subject, modality='func',
                                   type='bold', task=task, run=1,
                                   extensions=['nii', 'nii.gz'], echo=echo)
            metadata = layout.get_metadata(orig_file[0].filename)
            echo_time = metadata['EchoTime'] * 1000
            echo_times.append(echo_time)

            # Get preprocessed file associated with echo
            func_file = op.join(preproc_dir, 'func',
                                ('sub-{0}_task-{1}_run-01_echo-{2}'
                                 '_bold_space-MNI152NLin2009cAsym_'
                                 'powerpreproc'
                                 '.nii.gz').format(subject, task, echo))
            in_files.append(func_file)

        # FIT denoised
        # We retain t2s and s0 timeseries from this method, but do not use
        # optcom or any MEICA derivatives.
        tedana.workflows.t2smap_workflow(in_files, echo_times, fitmode='ts',
                                         out_dir=denoise_dir,
                                         combmode='t2s', label='fit')

        # TEDANA v2.5 with GODEC
        # We use MEDN+GODEC and MEHK+GODEC for GODEC
        # We use MEDN, reconstructed MEDN-noise, MEHK,
        # reconstructed MEHK-noise, optcom, mmix (component timeseries), and
        # comptable (classifications of components) without GODEC
        tedana.workflows.tedana_workflow(in_files, echo_times, gscontrol=False,
                                         ws_denoise='godec',
                                         wvpca=False, out_dir=denoise_dir,
                                         label='meica_v2_5',
                                         selection='kundu_v2_5')
