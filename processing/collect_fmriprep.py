"""
Collect native-space preprocessed data from fMRIPrep working directory and
copy into fMRIPrep derivatives directory, in BIDS format.
"""
import argparse
import os.path as op
from glob import glob
from shutil import copyfile


def collect_fmriprep(deriv_dir, work_dir, subs):
    """
    Collect native-space preprocessed data from fMRIPrep working directory and
    copy into fMRIPrep derivatives directory, in BIDS format.

    Parameters
    ----------
    deriv_dir : str
    work_dir : str
    subs : list
        Cannot include sub- prefix.
    """
    for sub in subs:
        print(f"Wrangling subject {sub}", flush=True)
        sub_in_dir = op.join(work_dir, f"single_subject_{sub}_wf")
        task_dirs = glob(op.join(sub_in_dir, "func_preproc*_task_*_wf"))
        for task_dir in task_dirs:
            bb_wf_dir = op.join(task_dir, "bold_bold_trans_wf")
            bf_dirs = sorted(glob(op.join(bb_wf_dir, "_bold_file_*")))
            for bf_dir in bf_dirs:
                # Collect partially preprocessed data
                bf_dir_list = bf_dir.split("..")
                idx = bf_dir_list.index("sub-{0}".format(sub))
                sub_deriv_dir = op.join(
                    deriv_dir,
                    op.dirname("/".join(bf_dir_list[idx:])),
                )
                bf_filename = bf_dir_list[-1]

                in_file = op.join(
                    bf_dir,
                    "merge",
                    "vol0000_xform-00000_merged.nii.gz",
                )

                # Conform output name
                orig_fn_list = bf_filename.split("_")
                fn_list = orig_fn_list[:]
                fn_list.insert(-1, "space-scanner")
                fn_list.insert(-1, "desc-partialPreproc")
                out_file = op.join(sub_deriv_dir, "_".join(fn_list))
                print(f"Writing {out_file}")
                copyfile(in_file, out_file)


def _get_parser():
    parser = argparse.ArgumentParser(
        description="Collecting partially preprocesed data from fmriprep"
    )
    parser.add_argument(
        "-i",
        "--deriv_dir",
        dest="deriv_dir",
        required=True,
        help="Path to fmriprep dir",
    )
    parser.add_argument(
        "-o",
        "--work_dir",
        dest="work_dir",
        required=True,
        help="Path to work dir",
    )
    parser.add_argument(
        "-s",
        "--subs",
        dest="subs",
        required=True,
        nargs="+",
        help="List with subject IDs",
    )
    return parser


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    collect_fmriprep(**kwargs)


if __name__ == "__main__":
    _main()
