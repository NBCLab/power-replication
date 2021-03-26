#!/usr/bin/env python3
"""Submit jobs for multiple participants."""
import os
import os.path as op
import shutil
import subprocess
import sys
import time
from datetime import datetime
from glob import glob

import argparse
import getpass
import pandas as pd


def is_valid_file(parser, arg):
    """Check if argument is existing file."""
    if not op.isfile(arg) and arg is not None:
        parser.error("The file {0} does not exist!".format(arg))

    return arg


def is_valid_path(parser, arg):
    """Check if argument is existing folder."""
    if not op.isdir(arg) and arg is not None:
        parser.error("The folder {0} does not exist!".format(arg))

    return arg


def copy_dset_files(in_dir, out_dir, sub):
    """Copy top-level and subject files from a BIDS dataset to a path."""
    top_level_files = sorted(glob(op.join(in_dir, "*")))
    top_level_files = [f for f in top_level_files if op.isfile(f)]
    sub_folder = op.join(in_dir, sub)

    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    for top_level_file in top_level_files:
        out_file = op.join(out_dir, op.basename(top_level_file))
        shutil.copyfile(top_level_file, out_file)

    out_sub_folder = op.join(out_dir, sub)
    if not op.isdir(out_sub_folder):
        shutil.copytree(sub_folder, out_sub_folder)


def run(command, env={}):
    """Run a command and allow for specification of environment information.

    Parameters
    ----------
    command: command to be sent to system
    env: parameters to be added to environment
    """
    merged_env = os.environ
    merged_env.update(env)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        env=merged_env,
    )
    while True:
        line = process.stdout.readline()
        line = line.decode("utf-8")
        sys.stdout.write(line)
        sys.stdout.flush()
        if line == "" and process.poll() is not None:
            break

    if process.returncode != 0:
        raise Exception(
            "Non zero return code: {0}\n"
            "{1}\n\n{2}".format(
                process.returncode,
                command,
                process.stdout.read()
            )
        )


def _get_parser():
    """Set up argument parser for scripts."""
    parser = argparse.ArgumentParser(
        description="Submit subject-specific SLURM jobs based on a template"
    )
    parser.add_argument(
        "-t",
        "--template",
        required=True,
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        dest="template_job_file",
        help=(
            "Template job file. Filename should have 'template' in it, which "
            "will be replaced with the participant ID to write out a "
            "participant-specific job file to be submitted."
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        metavar="PATH",
        type=lambda x: is_valid_path(parser, x),
        dest="dset_dir",
        help=(
            "Path to the BIDS dataset."
        ),
    )
    parser.add_argument(
        "-w",
        "--work-dir",
        "--work_dir",
        required=True,
        metavar="PATH",
        type=str,
        dest="work_dir",
        help=(
            "Working directory."
        ),
    )

    subgrp = parser.add_mutually_exclusive_group(required=True)
    subgrp.add_argument(
        "--participant_label",
        "--participant-label",
        action="store",
        nargs="+",
        help=(
            "A space delimited list of participant identifiers or a single "
            "identifier (the sub- prefix can be removed)"
        ),
    )
    subgrp.add_argument(
        "--tsv_file",
        "--tsv-file",
        action="store",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "A tsv file with participant IDs. Matches format of BIDS "
            "participants.tsv file (i.e., IDs are in 'participant_id' column)"
        ),
    )
    subgrp.add_argument(
        "--txt_file",
        "--txt-file",
        action="store",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "A single-column txt file with participant IDs. "
            "Every row is expected to be one participant ID."
        ),
    )

    parser.add_argument(
        "--job_limit",
        "--job-limit",
        action="store",
        type=int,
        help=(
            "Maximum number of jobs to run at once. "
            "If set, you should submit this script via a job, "
            "or else it will keep running on the node "
            "you run it from. Requires argument 'username'."
        ),
        default=None,
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        dest="copy_files",
        help=(
            "Copy relevant files from the BIDS dataset to the working "
            "directory."
        ),
        default=False,
    )
    return parser


def main(
    template_job_file,
    dset_dir,
    work_dir,
    participant_label=None,
    tsv_file=None,
    txt_file=None,
    job_limit=None,
    copy_files=False,
):
    """Submit jobs for multiple participants.

    Optimal for running preprocessing BIDS Apps in parallel instead of
    sequentially.

    Parameters
    ----------
    template_job_file : str
        Template job file. Filename should have "template" in it, which
        will be replaced with the participant ID to write out a
        participant-specific job file to be submitted.
    dset_dir : str
        BIDS dataset directory.
    work_dir : str
        Working directory.
    participant_label : str or list or None, optional
        A list of participant identifiers or a single identifier (the sub-
        prefix can be removed).
        Mutually exclusive with "tsv_file" and "txt_file".
        One of "participant_label", "tsv_file", and "txt_file" must be
        provided.
    tsv_file : str or None, optional
        A tab-delimited file with participant IDs. Matches format of BIDS
        participants.tsv file (i.e., IDs are in "participant_id" column).
        Mutually exclusive with "participant_label" and "txt_file".
        One of "participant_label", "tsv_file", and "txt_file" must be
        provided.
    txt_file : str or None, optional
        A single-column txt file with participant IDs. Every row is expected to
        be one participant ID.
        Mutually exclusive with "participant_label" and "tsv_file".
        One of "participant_label", "tsv_file", and "txt_file" must be
        provided.
    job_limit : int or None, optional
        Maximum number of jobs to run at once. If set, you should submit
        this script via a job, or else it will keep running on the node
        you run it from.
    copy_files : bool, optional
        If True, copy subject and top-level files to the working directory.
        If False, don't.
    """
    if (participant_label is not None) and isinstance(participant_label, list):
        subjects = participant_label[:]
    elif participant_label is not None:
        subjects = [participant_label]
    elif tsv_file is not None:
        participants_df = pd.read_table(tsv_file)
        assert "participant_id" in participants_df.columns
        subjects = participants_df["participant_id"].tolist()
        subjects = [s for s in subjects if s]  # drop vals for empty rows
    elif txt_file is not None:
        with open(txt_file, "r") as fo:
            subjects = fo.readlines()
            subjects = [row.rstrip() for row in subjects]
    else:
        raise Exception(
            'One of "participant_label", "tsv_file", and '
            '"txt_file" must be provided.'
        )

    out_dir = op.join(dset_dir, "derivatives")

    work_dir = op.abspath(work_dir)
    if not op.isdir(work_dir):
        os.mkdir(work_dir)

    jobs_dir = op.join(op.dirname(template_job_file), "jobs")
    if not op.isdir(jobs_dir):
        os.mkdir(jobs_dir)

    if job_limit is not None:
        job_limit = int(job_limit)
        timestamp = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
        job_tracker_file = op.join(
            jobs_dir,
            "queue_count_{}.txt".format(timestamp),
        )
        username = getpass.getuser()

    with open(template_job_file, "r") as fo:
        template_job = fo.read()

    for subject in subjects:
        subject_job_file = template_job_file.replace("template", subject)
        subject_job_file = op.join(jobs_dir, op.basename(subject_job_file))
        # A hack to skip complete subjects
        test_file = op.join(out_dir, "mriqc", "{}_task-movie_echo-5_bold.html".format(subject))
        if op.isfile(test_file):
            print("Subject {} already run. Skipping".format(subject), flush=True)
            continue

        if copy_files:
            sub_temp_dir = op.join(work_dir, subject)
            if not op.isdir(sub_temp_dir):
                os.mkdir(sub_temp_dir)
            sub_dset_dir = op.join(sub_temp_dir, "dset")
            sub_work_dir = op.join(sub_temp_dir, "work")
            copy_dset_files(dset_dir, sub_dset_dir, subject)
        else:
            sub_dset_dir = op.abspath(dset_dir)
            sub_work_dir = op.abspath(work_dir)

        subject_job = template_job.format(
            subject=subject,
            sid=subject.replace("sub-", ""),
            dset_dir=sub_dset_dir,
            work_dir=sub_work_dir,
            out_dir=out_dir,
        )

        with open(subject_job_file, "w") as fo:
            fo.write(subject_job)

        submission_cmd = "sbatch {}".format(subject_job_file)

        # Manage number of concurrent jobs if need be
        if job_limit is None:
            run(submission_cmd)
        else:
            while True:
                # Get current job count
                squeue_cmd = "squeue -u {} > {}".format(
                    username, job_tracker_file
                )
                run(squeue_cmd)

                with open(job_tracker_file, "r") as fo:
                    curr_jobs = fo.readlines()

                n_current_jobs = len(curr_jobs) - 1
                if n_current_jobs >= job_limit:
                    time.sleep(1800)  # 30 minutes
                    continue
                else:
                    run(submission_cmd)
                    break


def _main(argv=None):
    """Entry point."""
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
