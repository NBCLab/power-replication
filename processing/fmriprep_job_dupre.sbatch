#!/bin/bash
#SBATCH --job-name=fmriprep
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_40C_512G
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/fmriprep-%j-%A-%a.out
#SBATCH --error=/home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/fmriprep-%j-%A-%a.err
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=tsalo006@fiu.edu
# ------------------------------------------

pwd; hostname; date
set -e

#==============Shell script==============#
# Load the software needed
module load singularity-3.5.3

PROJECT="Salo_PowerReplication"
DSET_NAME="dset-dupre"
IMG_DIR="/home/data/cis/singularity-images"
DATA_DIR="/home/data/nbc/misc-projects"
BIDS_DIR="${DATA_DIR}/${PROJECT}/${DSET_NAME}"
SCRATCH_DIR="/scratch/nbc/tsalo006/${PROJECT}/${DSET_NAME}/fmriprep-20.2.1"
DERIVS_DIR="${DATA_DIR}/${PROJECT}/${DSET_NAME}/derivatives"
mkdir -p ${SCRATCH_DIR}
mkdir -p ${DERIVS_DIR}

# Prepare some writeable bind-mount points.
TEMPLATEFLOW_HOST_HOME=$HOME/.cache/templateflow
FMRIPREP_HOST_CACHE=$HOME/.cache/fmriprep
mkdir -p ${FMRIPREP_HOST_CACHE}

# Make sure FS_LICENSE is defined in the container.
FS_LICENSE=$HOME/freesurfer_license.txt

# Designate a templateflow bind-mount point
export SINGULARITYENV_TEMPLATEFLOW_HOME="$HOME/.cache/templateflow"

SINGULARITY_CMD="singularity run --cleanenv \
      -B $BIDS_DIR:/data \
      -B ${DERIVS_DIR}:/out \
      -B ${TEMPLATEFLOW_HOST_HOME}:${SINGULARITYENV_TEMPLATEFLOW_HOME} \
      -B $SCRATCH_DIR:/work \
      $IMG_DIR/poldracklab-fmriprep_20.2.1.sif"

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

# Compose the command line
cmd="${SINGULARITY_CMD} /data \
      /out \
      participant \
      --participant-label $subject \
      -w /work/ \
      --omp-nthreads 8 \
      --nprocs 8 \
      --mem_mb 32000 \
      --output-spaces MNI152NLin6Asym:res-native anat:res-native boldref:res-native \
      --notrack \
      --fs-no-reconall \
      --fs-license-file $FS_LICENSE"

# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?

# Collect partially preprocessed images:
echo
echo "Collecting partially preprocessed images"
SHELL_CMD="singularity exec --cleanenv \
      -B ${BIDS_DIR}:/data \
      -B $DERIVS_DIR/fmriprep:/deriv_dir \
      -B $SCRATCH_DIR/fmriprep_wf:/work_dir \
      $IMG_DIR/poldracklab-fmriprep_20.2.1.sif"

# Run python script inside fmriprep environment
cmd="${SHELL_CMD} python /data/code/processing/collect_fmriprep.py \
      --deriv_dir /deriv_dir \
      --work_dir /work_dir \
      --subs $subject"
# Setup done, run the command
echo Commandline: $cmd
eval $cmd

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${DATA_DIR}/${PROJECT}/code/log/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo "Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}"
exit $exitcode

date
