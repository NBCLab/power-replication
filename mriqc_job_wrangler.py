#!/usr/bin/env python3
"""Submit jobs for multiple participants."""
import os
import os.path as op
import subprocess
import sys
import time
from datetime import datetime

import argparse
import getpass
import pandas as pd


def is_valid_file(parser, arg):
    """Check if argument is existing file."""
    if not op.isfile(arg) and arg is not None:
        parser.error("The file {0} does not exist!".format(arg))

    return arg


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
        help='Template job file. Filename should have "template" in it, which '
        "will be replaced with the participant ID to write out a "
        "participant-specific job file to be submitted.",
    )

    subgrp = parser.add_mutually_exclusive_group(required=True)
    subgrp.add_argument(
        "--participant_label",
        "--participant-label",
        action="store",
        nargs="+",
        help="A space delimited list of participant identifiers or a single "
        "identifier (the sub- prefix can be removed)",
    )
    subgrp.add_argument(
        "--tsv_file",
        "--tsv-file",
        action="store",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help="A tsv file with participant IDs. Matches format of BIDS "
        'participants.tsv file (i.e., IDs are in "participant_id" column)',
    )
    subgrp.add_argument(
        "--txt_file",
        "--txt-file",
        action="store",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help="A single-column txt file with participant IDs. Every row is "
        "expected to be one participant ID.",
    )

    parser.add_argument(
        "--job_limit",
        "--job-limit",
        action="store",
        type=int,
        help="Maximum number of jobs to run at once. "
        "If set, you should submit "
        "this script via a job, or else it will keep running on the node "
        'you run it from. Requires argument "username".',
        default=None,
    )
    return parser


def _main(argv=None):
    """Entry point."""
    options = _get_parser().parse_args(argv)
    main(**vars(options))


def main(
    template_job_file,
    participant_label=None,
    tsv_file=None,
    txt_file=None,
    job_limit=None,
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

    if job_limit is not None:
        job_limit = int(job_limit)
        timestamp = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
        job_tracker_file = "queue_count_{}.txt".format(timestamp)
        username = getpass.getuser()

    with open(template_job_file, "r") as fo:
        template_job = fo.read()

    for subject in subjects:
        subject_job = template_job.format(subject=subject)
        subject_job_file = template_job_file.replace("template", subject)

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


if __name__ == "__main__":
    _main()
