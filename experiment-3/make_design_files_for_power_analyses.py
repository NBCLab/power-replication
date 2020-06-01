"""
"""
from glob import glob
import os.path as op
from os import mkdir
from shutil import copyfile


def make_image_file():
    design_file = 'design.fsf'

    # Each file
    gp_mem = '# Group membership for input {0}\nset fmri(groupmem.{0}) 1\n'
    hi_thing = '# Higher-level EV value for EV 1 and input {0}\nset fmri(evg{0}.1) 1\n'
    f_thing = '# 4D AVW data or FEAT directory ({n})\nset feat_files({n}) "{f}"\n'

    # Once
    n_fls = '# Number of first-level analyses\nset fmri(multiple) {0}\n'
    out = '# Output directory\nset fmri(outputdir) "{0}"\n'
    n_vols = '# Total volumes\nset fmri(npts) {0}\n'

    # Get files
    in_dir = '/home/data/hcp/'
    subdir = 'MNINonLinear/Results/tfMRI_WM/tfMRI_WM_hp200_s4_level2.feat'

    subjects = glob(op.join(in_dir, '*'))
    subjects = [op.basename(s) for s in subjects]
    subjects = sorted([s for s in subjects if s.isdigit()])
    feat_dirs = []
    for s in subjects:
        feat_dir = op.join(in_dir, s, subdir)
        if op.isdir(feat_dir):
            feat_dirs.append(feat_dir)
    n = len(feat_dirs)

    # 0back - fixation
    cope_files = [op.join(fd, 'cope10.feat/stats/cope1.nii.gz') for fd in feat_dirs]

    with open(design_file, 'r') as fo:
        data = fo.read()

    out_dir = '/home/tsalo006/Desktop/hcp-visual/back0.gfeat/'
    if not op.isdir(out_dir):
        mkdir(out_dir)

    data += n_fls.format(n)
    data += '\n'
    data += out.format(out_dir)
    data += '\n'
    data += n_vols.format(n)
    data += '\n'
    for i, f in enumerate(cope_files):
        data += gp_mem.format(i+1)
        data += '\n'
        data += hi_thing.format(i+1)
        data += '\n'
        data += f_thing.format(f=f, n=i+1)
        data += '\n'

    with open(op.join(out_dir, 'visual_power_analysis_design.fsf'), 'w') as fo:
        fo.write(data)

    copyfile(op.join(out_dir, 'visual_power_analysis_design.fsf'),
             'visual_power_analysis_design.fsf')


def make_fingertapping_files():
    design_file = 'design.fsf'

    # Each file
    gp_mem = '# Group membership for input {0}\nset fmri(groupmem.{0}) 1\n'
    hi_thing = '# Higher-level EV value for EV 1 and input {0}\nset fmri(evg{0}.1) 1\n'
    f_thing = '# 4D AVW data or FEAT directory ({n})\nset feat_files({n}) "{f}"\n'

    # Once
    n_fls = '# Number of first-level analyses\nset fmri(multiple) {0}\n'
    out = '# Output directory\nset fmri(outputdir) "{0}"\n'
    n_vols = '# Total volumes\nset fmri(npts) {0}\n'

    # Get files
    in_dir = '/home/data/hcp/'
    subdir = 'MNINonLinear/Results/tfMRI_MOTOR/tfMRI_MOTOR_hp200_s4_level2vol.feat'
    subjects = glob(op.join(in_dir, '*'))
    subjects = [op.basename(s) for s in subjects]
    subjects = sorted([s for s in subjects if s.isdigit()])

    # Contrast 10 is LH-AVG
    # Contrast 12 is RH-AVG
    feat_dirs = []
    for s in subjects:
        feat_dir = op.join(in_dir, s, subdir)
        if op.isdir(feat_dir):
            feat_dirs.append(feat_dir)
    n = len(feat_dirs)

    # Left hand
    cope_files = [op.join(fd, 'cope10.feat/stats/cope1.nii.gz') for fd in feat_dirs]

    with open(design_file, 'r') as fo:
        data = fo.read()

    out_dir = '/home/tsalo006/Desktop/hcp-motor/lh-avg.gfeat/'
    if not op.isdir(out_dir):
        mkdir(out_dir)

    data += n_fls.format(n)
    data += '\n'
    data += out.format(out_dir)
    data += '\n'
    data += n_vols.format(n)
    data += '\n'
    for i, f in enumerate(cope_files):
        data += gp_mem.format(i+1)
        data += '\n'
        data += hi_thing.format(i+1)
        data += '\n'
        data += f_thing.format(f=f, n=i+1)
        data += '\n'

    with open(op.join(out_dir, 'motor_lh_power_analysis_design.fsf'), 'w') as fo:
        fo.write(data)

    copyfile(op.join(out_dir, 'motor_lh_power_analysis_design.fsf'),
             'motor_lh_power_analysis_design.fsf')

    # Right hand
    cope_files = [op.join(fd, 'cope12.feat/stats/cope1.nii.gz') for fd in feat_dirs]

    with open(design_file, 'r') as fo:
        data = fo.read()

    out_dir = '/home/tsalo006/Desktop/hcp-motor/rh-avg.gfeat/'
    if not op.isdir(out_dir):
        mkdir(out_dir)

    data += n_fls.format(n)
    data += '\n'
    data += out.format(out_dir)
    data += '\n'
    data += n_vols.format(n)
    data += '\n'
    for i, f in enumerate(cope_files):
        data += gp_mem.format(i+1)
        data += '\n'
        data += hi_thing.format(i+1)
        data += '\n'
        data += f_thing.format(f=f, n=i+1)
        data += '\n'

    with open(op.join(out_dir, 'motor_rh_power_analysis_design.fsf'), 'w') as fo:
        fo.write(data)

    copyfile(op.join(out_dir, 'motor_rh_power_analysis_design.fsf'),
             'motor_rh_power_analysis_design.fsf')
