"""
Generate distance-dependent motion-related artifact plots
The rank for the intercept (smoothing curve at 35mm) indexes general dependence
on motion (i.e., a mix of global and focal effects), while the rank for the
slope (difference in smoothing curve at 100mm and 35mm) indexes distance
dependence (i.e., focal effects).
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import nibabel as nib
import pandas as pd
from nilearn import datasets

# Distance-dependent motion-related artifact analysis code
import ddmra

sns.set_style('whitegrid')


def get_fd(motion):
    # assuming rotations in degrees
    motion[:, :3] = motion[:, :3] * (np.pi/180.) * 50
    motion = np.vstack((np.array([[0, 0, 0, 0, 0, 0]]),
                        np.diff(motion, axis=0)))
    fd = np.sum(np.abs(motion), axis=1)
    return fd


def main():
    # Constants
    n_subjects = 31
    fd_thresh = 0.1
    window = 1000
    v1, v2 = 35, 100  # distances to evaluate
    data = datasets.fetch_adhd(n_subjects=n_subjects)
    n_iters = 10000
    n_lines = min((n_iters, 50))

    # Prepare data
    imgs = []
    fd_all = []

    for i in range(n_subjects):
        func = data.func[i]
        imgs.append(nib.load(func))
        conf = data.confounds[i]
        df = pd.read_csv(conf, sep='\t')
        motion = df[['motion-pitch', 'motion-roll', 'motion-yaw',
                     'motion-x', 'motion-y', 'motion-z']].values
        fd_all.append(get_fd(motion))

    # Run analyses
    results = ddmra.run(imgs, fd_all, n_iters=n_iters, qc_thresh=fd_thresh)

    # QC:RSFC analysis
    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.regplot(results['sorted_dists'], results['qcrsfc_y'],
                ax=ax, scatter=True, fit_reg=False,
                line_kws={'color': 'black', 'linewidth': 3},
                scatter_kws={'color': 'red', 's': 1., 'alpha': 1})
    for i in range(n_lines):
        ax.plot(results['sorted_dists'], results['qcrsfc_null'][i, :], color='black')
    ax.plot(results['sorted_dists'], results['qcrsfc_smc'], color='white')
    ax.set_xlabel('Distance (mm)', fontsize=20)
    ax.set_ylabel('QC:RSFC r\n(QC = mean FD)', fontsize=20)
    fig.savefig('sandbox/qcrsfc_analysis.png', dpi=400)

    # Assess significance
    intercept = ddmra.get_val(results['sorted_dists'], results['qcrsfc_smc'], v1)
    slope = ((ddmra.get_val(results['sorted_dists'], results['qcrsfc_smc'], v1) -
              ddmra.get_val(results['sorted_dists'], results['qcrsfc_smc'], v2)) / (v1 - v2))
    perm_intercepts = ddmra.get_val(results['sorted_dists'], results['qcrsfc_null'], v1)
    perm_slopes = ((ddmra.get_val(results['sorted_dists'], results['qcrsfc_null'], v1) -
                    ddmra.get_val(results['sorted_dists'], results['qcrsfc_null'], v2)) / (v1 - v2))

    p_inter = ddmra.rank_p(intercept, perm_intercepts, tail='upper')
    p_slope = ddmra.rank_p(slope, perm_slopes, tail='upper')
    print('QCRSFC analysis results:')
    print('\tIntercept = {0:.04f}, p = {1:.04f}'.format(intercept, p_inter))
    print('\tSlope = {0:.04f}, p = {1:.04f}'.format(slope, p_slope))

    # High-low motion analysis
    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.regplot(results['sorted_dists'], results['hl_y'],
                ax=ax, scatter=True, fit_reg=False,
                line_kws={'color': 'black', 'linewidth': 3},
                scatter_kws={'color': 'red', 's': 1, 'alpha': 1})
    for i in range(n_lines):
        ax.plot(results['sorted_dists'], results['hl_null'][i, :], color='black')
    ax.plot(results['sorted_dists'], results['hl_smc'], color='white')
    ax.set_xlabel('Distance (mm)', fontsize=20)
    ax.set_ylabel(r'High-low motion $\Delta$r', fontsize=20)
    fig.savefig('sandbox/hilow_analysis.png', dpi=400)

    # Assess significance
    intercept = ddmra.get_val(results['sorted_dists'], results['hl_smc'], v1)
    slope = ((ddmra.get_val(results['sorted_dists'], results['hl_smc'], v1) -
              ddmra.get_val(results['sorted_dists'], results['hl_smc'], v2)) / (v1 - v2))
    perm_intercepts = ddmra.get_val(results['sorted_dists'], results['hl_null'], v1)
    perm_slopes = ((ddmra.get_val(results['sorted_dists'], results['hl_null'], v1) -
                    ddmra.get_val(results['sorted_dists'], results['hl_null'], v2)) / (v1 - v2))

    p_inter = ddmra.rank_p(intercept, perm_intercepts, tail='upper')
    p_slope = ddmra.rank_p(slope, perm_slopes, tail='upper')
    print('High-low motion analysis results:')
    print('\tIntercept = {0:.04f}, p = {1:.04f}'.format(intercept, p_inter))
    print('\tSlope = {0:.04f}, p = {1:.04f}'.format(slope, p_slope))

    # Scrubbing analysis
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.regplot(results['sorted_dists'], results['scrub_y'],
                ax=ax, scatter=True, fit_reg=False,
                line_kws={'color': 'black', 'linewidth': 3},
                scatter_kws={'color': 'red', 's': 1, 'alpha': 1})
    for i in range(n_lines):
        ax.plot(results['sorted_dists'], results['scrub_null'][i, :],
                color='black')
    ax.plot(results['sorted_dists'], results['scrub_smc'],
            color='white')
    ax.set_xlabel('Distance (mm)', fontsize=20)
    ax.set_ylabel(r'Scrubbing $\Delta$r', fontsize=20)
    fig.savefig('sandbox/scrubbing_analysis.png', dpi=400)

    # Assess significance
    intercept = ddmra.get_val(results['sorted_dists'], results['scrub_smc'], v1)
    slope = ((ddmra.get_val(results['sorted_dists'], results['scrub_smc'], v1) -
              ddmra.get_val(results['sorted_dists'], results['scrub_smc'], v2)) / (v1 - v2))

    perm_intercepts = ddmra.get_val(results['sorted_dists'], results['scrub_null'], v1)
    perm_slopes = ((ddmra.get_val(results['sorted_dists'], results['scrub_null'], v1) -
                    ddmra.get_val(results['sorted_dists'], results['scrub_null'], v2)) / (v1 - v2))
    p_inter = ddmra.rank_p(intercept, perm_intercepts, tail='upper')
    p_slope = ddmra.rank_p(slope, perm_slopes, tail='upper')
    print('Scrubbing analysis results:')
    print('\tIntercept = {0:.04f}, p = {1:.04f}'.format(intercept, p_inter))
    print('\tSlope = {0:.04f}, p = {1:.04f}'.format(slope, p_slope))


if __name__ == '__main__':
    main()
