#!/bin/bash
#SBATCH --job-name=denoising
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=investor
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/denoising-%j-%A-%a.out
#SBATCH --error=/home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/denoising-%j-%A-%a.err
# ------------------------------------------
# USAGE information:
# Use the -array= option.
# Start with 0.
# sbatch 07_run_subject_denoising.sbatch --array=0-778%10
pwd; hostname; date
set -e

#==============Shell script==============#
# Load the software needed
module load singularity-3.5.3

PROJECT="Salo_PowerReplication"
DATA_DIR="/home/data/nbc/misc-projects"

CODE_DIR=${DATA_DIR}/${PROJECT}/code

# Activate the project's miniconda environment
source ${CODE_DIR}/activate_environment

# I have created a combined participants.tsv file only containing the included subjects
# and two columns: participant_id and dataset
tsv_file=${DATA_DIR}/${PROJECT}/participants.tsv

# Parse the participants.tsv file and extract one subject ID from the line corresponding
# to this SLURM task.
dset_name="$(python ${CODE_DIR}/processing/get_tsv_cell.py --column dataset --row ${SLURM_ARRAY_TASK_ID} ${tsv_file})"
subject="$(python ${CODE_DIR}/processing/get_tsv_cell.py --column participant_id --row ${SLURM_ARRAY_TASK_ID} ${tsv_file})"

# Run python script inside fmriprep environment
cmd="python ${CODE_DIR}/processing/04_run_multi-echo_denoising.py \
      --dset ${dset_name} --subject $subject"
# Setup done, run the command
echo Commandline: $cmd
eval $cmd

# Run python script inside fmriprep environment
cmd="python ${CODE_DIR}/processing/05_generate_single-echo_confounds.py \
      --dset ${dset_name} --subject $subject"
# Setup done, run the command
echo Commandline: $cmd
eval $cmd

# Run python script inside fmriprep environment
cmd="python ${CODE_DIR}/processing/06_run_single-echo_denoising.py \
      --dset ${dset_name} --subject $subject"
# Setup done, run the command
echo Commandline: $cmd
eval $cmd

# Output results to a table
echo "$dset_name  sub-$subject      ${SLURM_ARRAY_TASK_ID}  $exitcode" \
      >> ${DATA_DIR}/${PROJECT}/code/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo "Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}"
exit $exitcode

date
