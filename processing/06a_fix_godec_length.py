"""Remove last volume from problematic GODEC files.

GODEC adds a single volume to data over a certain size, but the last volume is noise and should be
removed.
See https://github.com/ME-ICA/godec/issues/4 for more info.
"""
import os.path as op
import sys
from glob import glob

import nibabel as nib

sys.path.append("..")

from utils import get_prefixes  # noqa: E402

project_dir = "/home/data/nbc/misc-projects/Salo_PowerReplication"

prefixes = get_prefixes()

dset_dirs = sorted(glob(op.join(project_dir, "dset-*")))
for dset_dir in dset_dirs:
    dset = op.basename(dset_dir)
    print(dset, flush=True)
    dset_prefix = prefixes[dset]
    subject_dirs = sorted(glob(op.join(dset_dir, "derivatives/godec/sub-*")))
    for subject_dir in subject_dirs:
        subject = op.basename(subject_dir)
        print(f"\t{subject}", flush=True)
        subj_prefix = dset_prefix.format(participant_id=subject)
        medn_file = op.join(
            dset_dir,
            "derivatives/tedana",
            subject,
            "func",
            f"{subj_prefix}_desc-optcom_bold.nii.gz",
        )
        assert op.isfile(medn_file), f"File DNE: {medn_file}"
        medn_img = nib.load(medn_file)
        n_vols = medn_img.shape[3]

        godec_files = sorted(glob(op.join(subject_dir, "func", "*.nii.gz")))
        assert len(godec_files) >= 1, op.join(subject_dir, "func", "*.nii.gz")
        for godec_file in godec_files:
            godec_img = nib.load(godec_file)

            if n_vols != godec_img.shape[3]:
                print(f"\t\t{op.basename(godec_file)}", flush=True)
                godec_data = godec_img.get_fdata()
                godec_data = godec_data[:, :, :, :n_vols]
                godec_img = nib.Nifti1Image(godec_data, godec_img.affine, godec_img.header)
                godec_img.to_filename(godec_file)
