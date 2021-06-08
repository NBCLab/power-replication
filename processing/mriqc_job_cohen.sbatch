#!/bin/bash
#SBATCH --job-name=cohen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=investor
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/mriqc-cohen-%j-%A-%a.out
#SBATCH --error=/home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/mriqc-cohen-%j-%A-%a.err
# ------------------------------------------
# USAGE information:
# Use the -array= option.
# Start with 1.
pwd; hostname; date
set -e

#==============Shell script==============#
# Load the software needed
module load singularity-3.5.3

PROJECT="Salo_PowerReplication"
DSET_NAME="dset-cohen"
IMG_DIR="/home/data/cis/singularity-images"
DATA_DIR="/home/data/nbc/misc-projects"
BIDS_DIR="${DATA_DIR}/${PROJECT}/${DSET_NAME}"
SCRATCH_DIR="/scratch/nbc/tsalo006/${PROJECT}/${DSET_NAME}/mriqc-0.16.1"
DERIVS_DIR="${DATA_DIR}/${PROJECT}/${DSET_NAME}/derivatives"
mkdir -p ${SCRATCH_DIR}
mkdir -p ${DERIVS_DIR}

# Designate a templateflow bind-mount point
SINGULARITYENV_NO_ET=1
export SINGULARITYENV_TEMPLATEFLOW_HOME="/opt/templateflow"

SINGULARITY_CMD="singularity run --home $HOME --cleanenv \
      -B $BIDS_DIR:/data \
      -B ${DERIVS_DIR}:/out \
      -B /home/tsalo006/.cache/templateflow:$SINGULARITYENV_TEMPLATEFLOW_HOME \
      -B $SCRATCH_DIR:/work \
      $IMG_DIR/poldracklab_mriqc_0.16.1.sif"

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

# Compose the command line
cmd="${SINGULARITY_CMD} \
      /data \
      /out/mriqc \
      participant \
      --participant-label $subject \
      -w /work/ \
      --no-sub \
      --nprocs 1"

# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${DATA_DIR}/${PROJECT}/code/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo "Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}"
exit $exitcode

date
