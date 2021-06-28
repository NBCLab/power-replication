"""Utility functions for processing data."""
import os
import subprocess

from nilearn import image, input_data


def run_command(command, env=None):
    """Run a given command with certain environment variables set."""
    merged_env = os.environ
    if env:
        merged_env.update(env)
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True,
                               env=merged_env)
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() is not None:
            break

    if process.returncode != 0:
        raise Exception("Non zero return code: {0}\n"
                        "{1}\n\n{2}".format(process.returncode, command,
                                            process.stdout.read()))


def _generic_regression(medn_file, mask_file, nuisance_regressors, t_r):
    # Calculate mean-centered version of MEDN data
    mean_img = image.mean_img(medn_file)
    medn_mean_centered_img = image.math_img(
        "img - avg_img",
        img=medn_file,
        avg_img=mean_img,
    )

    # Regress confounds out of MEDN data
    # Note that confounds should be mean-centered and detrended, which the masker will do.
    # See "fMRI data: nuisance regressions" section of Power appendix (page 3).
    regression_masker = input_data.NiftiMasker(
        mask_file,
        smoothing_fwhm=None,
        standardize=False,
        standardize_confounds=True,  # will mean-center them too
        high_variance_confounds=False,
        low_pass=None,
        high_pass=None,
        detrend=True,  # linearly detrends both confounds and data
        t_r=t_r,
        reports=False,
    )
    regression_masker.fit(medn_mean_centered_img)
    # Mask + remove confounds
    denoised_data = regression_masker.transform(
        medn_mean_centered_img,
        confounds=nuisance_regressors,
    )
    # Mask without removing confounds
    raw_data = regression_masker.transform(
        medn_mean_centered_img,
        confounds=None,
    )
    # Calculate residuals (both raw and denoised should have same scale)
    noise_data = raw_data - denoised_data
    denoised_img = regression_masker.inverse_transform(denoised_data)
    noise_img = regression_masker.inverse_transform(noise_data)
    return denoised_img, noise_img
