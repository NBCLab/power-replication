"""Combine individual datasets' participants.tsv files into one."""
import os.path as op
from glob import glob

import pandas as pd

TARGET_FILES = [
    "power/{sub}/func/*desc-TE30_bold.nii.gz",  # TE30
    "tedana/{sub}/func/*desc-optcom_bold.nii.gz",  # OC
    "tedana/{sub}/func/*desc-optcomDenoised_bold.nii.gz",  # MEDN
    "",  # MEDN Noise
    "tedana/{sub}/func/*desc-optcomMIRDenoised_bold.nii.gz",  # MEDN+MIR
    "",  # MEDN+MIR Noise
    "t2smap/{sub}/func/*T2starmap.nii.gz",  # FIT-R2
    "t2smap/{sub}/func/*S0map.nii.gz",  # FIT-S0
    "godec/{sub}/func/*desc-GODEC_rank-4_bold.nii.gz",  # MEDN+GODEC (sparse)
    "godec/{sub}/func/*desc-GODEC_rank-4_lowrankts.nii.gz",  # MEDN+GODEC Noise (lowrank)
    "rapidtide/{sub}/func/*desc-lfofilterCleaned_bold.nii.gz",  # MEDN+dGSR
    "rapidtide/{sub}/func/*desc-noise_bold.nii.gz",  # MEDN+dGSR Noise
    "nuisance-regressions/{sub}/func/*desc-aCompCor_bold.nii.gz",  # MEDN+aCompCor
    "nuisance-regressions/{sub}/func/*desc-aCompCorNoise_bold.nii.gz",  # MEDN+aCompCor Noise
    "nuisance-regressions/{sub}/func/*desc-GSR_bold.nii.gz",  # MEDN+GSR
    "nuisance-regressions/{sub}/func/*desc-GSRNoise_bold.nii.gz",  # MEDN+GSR Noise
]
PHYSIO_TARGET_FILES = [
    "nuisance-regressions/{sub}/func/*desc-RVReg_bold.nii.gz",  # MEDN+RV-Reg
    "nuisance-regressions/{sub}/func/*desc-RVTReg_bold.nii.gz",  # MEDN+RVT-Reg
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
        for target_file in TARGET_FILES:
            full_file_pattern = op.join(deriv_dir, target_file.format(sub=sub))
            matching_files = sorted(glob(full_file_pattern))
            if len(matching_files) == 0:
                print(f"Dataset {dataset} subject {sub} missing {full_file_pattern}")
                bad_subs.append((dataset, sub))

        if dataset == "dset-dupre":
            for target_file in PHYSIO_TARGET_FILES:
                full_file_pattern = op.join(deriv_dir, target_file.format(sub=sub))
                matching_files = sorted(glob(full_file_pattern))
                if len(matching_files) == 0:
                    print(f"Dataset {dataset} subject {sub} missing {full_file_pattern}")
                    bad_subs.append((dataset, sub))

    bad_subs = sorted(list(set(bad_subs)))
    print(bad_subs)
