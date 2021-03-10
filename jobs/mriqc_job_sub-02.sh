#!/bin/bash
#---Number of cores
#SBATCH -c 1

#---Job's name in SLURM system
#SBATCH -J sub-02_mriqc

#---Error file
#SBATCH -e /home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/mriqc_sub-02_err

#---Output file
#SBATCH -o /home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/mriqc_sub-02_out

#---Queue name
#SBATCH --account iacc_nbc

#---Partition
#SBATCH -p default-partition
########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=1
. $MODULESHOME/../global/profile.modules
module load singularity-3.5.3

DSET_DIR="/home/data/nbc/misc-projects/Salo_PowerReplication/dset-dupre/"
WORK_DIR="/scratch/nbc/tsalo006/dset-dupre-mriqc/"

# Run MRIQC
singularity run --cleanenv \
    /home/data/cis/singularity-images/poldracklab_mriqc_0.15.1.sif \
    $DSET_DIR $DSET_DIR/derivatives/ participant \
    --participant-label 02 \
    -w $DSET_DIR --no-sub \
    --nprocs 1
