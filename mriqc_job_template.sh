#!/bin/bash
#---Number of cores
#SBATCH -c 1

#---Job's name in SLURM system
#SBATCH -J {subject}_mriqc

#---Error file
#SBATCH -e /home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/mriqc_{subject}_err

#---Output file
#SBATCH -o /home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/mriqc_{subject}_out

#---Queue name
#SBATCH --account iacc_nbc

#---Partition
#SBATCH -p default-partition
########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=1
. $MODULESHOME/../global/profile.modules
module load singularity-3.5.3

DSET_DIR="/home/data/nbc/misc-projects/Salo_PowerReplication/dset-cambridge/"
WORK_DIR="/scratch/nbc/tsalo006/dset-cambridge-mriqc/{subject}"

SINGULARITYENV_NO_ET=1
SINGULARITYENV_TEMPLATEFLOW_HOME="/opt/templateflow"

# Run MRIQC
singularity run --home $HOME --cleanenv \
    -B /home/tsalo006/.cache/templateflow:$SINGULARITYENV_TEMPLATEFLOW_HOME \
    /home/data/cis/singularity-images/poldracklab_mriqc_0.16.1.sif \
    $DSET_DIR $DSET_DIR/derivatives/mriqc participant \
    --participant-label {sid} \
    -w $WORK_DIR --no-sub \
    --nprocs 1
