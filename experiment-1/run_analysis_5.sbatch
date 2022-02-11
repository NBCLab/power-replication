#!/bin/bash
#SBATCH --job-name=E1A5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_16C_96G
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-1/experiment01_analysis05_%a.out
#SBATCH --error=/home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-1/experiment01_analysis05_%a.err
# ------------------------------------------
# USAGE information:
# Use the -array= option.
# Start with 0.
# sbatch --array=0-10%10 run_analysis_5.sbatch
pwd; hostname; date
set -e

#==============Shell script==============#
PROJECT="Salo_PowerReplication"
DATA_DIR="/home/data/nbc/misc-projects"

CODE_DIR=${DATA_DIR}/${PROJECT}/code

# Activate the project's miniconda environment
source ${CODE_DIR}/activate_environment

outputs=("MEDN+Nuis-Reg" "OC" "MEDN" "MEDN_Noise" "MEDN+GODEC_sparse" "MEDN+RVT-Reg" "MEDN+RV-Reg" "MEDN+aCompCor" "MEDN+dGSR" "MEDN+MIR" "MEDN+GSR")

# Parse the participants.tsv file and extract one subject ID from the line corresponding
# to this SLURM task.
output_name="${outputs[$SLURM_ARRAY_TASK_ID]}"

# Run python script inside fmriprep environment
cmd="python ${CODE_DIR}/experiment-1/05_resp_and_fc.py ${output_name}"
# Setup done, run the command
echo Commandline: $cmd
eval $cmd

# Output results to a table
echo "$dset_name  sub-$subject      ${SLURM_ARRAY_TASK_ID}  $exitcode" \
      >> ${DATA_DIR}/${PROJECT}/code/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo "Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}"
exit $exitcode

date
