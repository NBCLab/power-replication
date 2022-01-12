"""Check that all warped files have been written out."""
import os.path as op
import sys
from pprint import pprint

import nibabel as nib
import pandas as pd

sys.path.append("..")

from utils import get_prefixes, get_prefixes_mni, get_target_files  # noqa: E402

if __name__ == "__main__":
    PROJECT_DIR = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    participants_file = op.join(PROJECT_DIR, "participants.tsv")

    target_files = get_target_files()
    dset_prefixes = get_prefixes()
    dset_mni_prefixes = get_prefixes_mni()

    bad_subs = []
    df = pd.read_table(participants_file)
    for _, row in df.iterrows():
        dataset = row["dset"]
        sub = row["participant_id"]
        dset_mni_subj_prefix = dset_mni_prefixes[dataset].format(participant_id=sub)
        dset_subj_prefix = dset_prefixes[dataset].format(participant_id=sub)

        deriv_dir = op.join(PROJECT_DIR, dataset, "derivatives")

        medn_mni_file = op.join(
            deriv_dir,
            target_files["MEDN"].format(
                prefix=dset_mni_subj_prefix,
                participant_id=sub,
            ),
        )
        if not op.isfile(medn_mni_file):
            print(f"Dataset {dataset} subject {sub} missing MEDN file.")
            print(medn_mni_file)
            bad_subs.append((dataset, sub))
            continue

        for filetype, target_file in target_files.items():
            if ("MEDN+RV" in filetype) and (dataset != "dset-dupre"):
                # Skip physio derivatives for datasets without physio data
                continue

            full_target_file = op.join(
                deriv_dir,
                target_file.format(participant_id=sub, prefix=dset_mni_subj_prefix),
            )
            if not op.isfile(full_target_file):
                print(f"Dataset {dataset} subject {sub} missing {full_target_file}")
                bad_subs.append((dataset, sub))
            else:
                img = nib.load(full_target_file)
                if len(img.shape) != 4:
                    print(f"Dataset {dataset} subject {sub} failed {full_target_file}")
                    bad_subs.append((dataset, sub))

    bad_subs = sorted(list(set(bad_subs)))
    pprint(bad_subs)
