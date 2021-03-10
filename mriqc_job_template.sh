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

DSET_DIR="/home/data/nbc/misc-projects/Salo_PowerReplication/dset-dupre/"
WORK_DIR="/scratch/nbc/tsalo006/dset-dupre-mriqc/"

# Run MRIQC
singularity exec --cleanenv \
    -B /home/tsal006/.cache/templateflow:$HOME/.cache/templateflow \
    poldracklab_mriqc.img \
    $DSET_DIR $DSET_DIR/derivatives/ participant \
    --participant-label {subject} \
    -w $DSET_DIR --no-sub \
    --nprocs 1
