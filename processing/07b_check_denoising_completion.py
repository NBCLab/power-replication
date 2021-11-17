"""Combine individual datasets' participants.tsv files into one."""
import os.path as op
from glob import glob

import pandas as pd

TARGET_FILES = [
    # TE30
    "power/{sub}/func/{prefix}_desc-TE30_bold.nii.gz",
    # OC
    "tedana/{sub}/func/{prefix}_desc-optcom_bold.nii.gz",
    # MEDN
    "tedana/{sub}/func/{prefix}_desc-optcomDenoised_bold.nii.gz",
    # MEDN Noise
    "tedana/{sub}/func/{prefix}_desc-optcomDenoised_errorts.nii.gz",
    # MEDN+MIR
    "tedana/{sub}/func/{prefix}_desc-optcomMIRDenoised_bold.nii.gz",
    # MEDN+MIR Noise
    "tedana/{sub}/func/{prefix}_desc-optcomMIRDenoised_errorts.nii.gz",
    # FIT-R2
    "t2smap/{sub}/func/{prefix}_T2starmap.nii.gz",
    # FIT-S0
    "t2smap/{sub}/func/{prefix}_S0map.nii.gz",
    # MEDN+GODEC (sparse)
    "godec/{sub}/func/{prefix}_desc-GODEC_rank-4_bold.nii.gz",
    # MEDN+GODEC Noise (lowrank)
    "godec/{sub}/func/{prefix}_desc-GODEC_rank-4_lowrankts.nii.gz",
    # MEDN+dGSR
    "rapidtide/{sub}/func/{prefix}_desc-lfofilterCleaned_bold.nii.gz",
    # MEDN+dGSR Noise
    "rapidtide/{sub}/func/{prefix}_desc-lfofilterCleaned_errorts.nii.gz",
    # MEDN+aCompCor
    "nuisance-regressions/{sub}/func/{prefix}_desc-aCompCor_bold.nii.gz",
    # MEDN+aCompCor Noise
    "nuisance-regressions/{sub}/func/{prefix}_desc-aCompCor_errorts.nii.gz",
    # MEDN+GSR
    "nuisance-regressions/{sub}/func/{prefix}_desc-GSR_bold.nii.gz",
    # MEDN+GSR Noise
    "nuisance-regressions/{sub}/func/{prefix}_desc-GSR_errorts.nii.gz",
    # MEDN+Nuis-Reg
    "nuisance-regressions/{sub}/func/{prefix}_desc-NuisReg_bold.nii.gz",
]
PHYSIO_TARGET_FILES = [
    # MEDN+RV-Reg
    "nuisance-regressions/{sub}/func/{prefix}_desc-RVReg_bold.nii.gz",
    # MEDN+RVT-Reg
    "nuisance-regressions/{sub}/func/{prefix}_desc-RVTReg_bold.nii.gz",
]

if __name__ == "__main__":
    PROJECT_DIR = "/home/data/nbc/misc-projects/Salo_PowerReplication/"
    participants_file = op.join(PROJECT_DIR, "participants.tsv")
    bad_subs = []
    df = pd.read_table(participants_file)
    for _, row in df.iterrows():
        dataset = row["dset"]
        sub = row["participant_id"]
        deriv_dir = op.join(PROJECT_DIR, dataset, "derivatives")

        medn_files = sorted(
            glob(
                op.join(
                    deriv_dir,
                    "tedana",
                    sub,
                    "func",
                    "*_desc-optcomDenoised_bold.nii.gz",
                )
            )
        )
        if len(medn_files) != 1:
            bad_subs.append((dataset, sub))
            continue

        medn_file = medn_files[0]
        medn_name = op.basename(medn_file)
        prefix = medn_name.split("desc-")[0].rstrip("_")

        for target_file in TARGET_FILES:
            full_file_pattern = op.join(
                deriv_dir, target_file.format(sub=sub, prefix=prefix)
            )
            matching_files = sorted(glob(full_file_pattern))
            if len(matching_files) == 0:
                print(f"Dataset {dataset} subject {sub} missing {full_file_pattern}")
                bad_subs.append((dataset, sub))

        if dataset == "dset-dupre":
            for target_file in PHYSIO_TARGET_FILES:
                full_file_pattern = op.join(
                    deriv_dir, target_file.format(sub=sub, prefix=prefix)
                )
                matching_files = sorted(glob(full_file_pattern))
                if len(matching_files) == 0:
                    print(
                        f"Dataset {dataset} subject {sub} missing {full_file_pattern}"
                    )
                    bad_subs.append((dataset, sub))

    bad_subs = sorted(list(set(bad_subs)))
    print(bad_subs)
