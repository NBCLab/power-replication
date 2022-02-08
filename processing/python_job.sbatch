#!/bin/bash
#SBATCH --job-name=power-replication
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=investor

# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/python_stdout
#SBATCH --error=/home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/python_stderr
########################################################
# A generic Python job for the SLURM scheduler
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=4
. $MODULESHOME/../global/profile.modules
source /home/data/nbc/misc-projects/Salo_PowerReplication/code/activate_environment
module load freesurfer-7.1

python /home/data/nbc/misc-projects/Salo_PowerReplication/code/processing/03_derive_masks_and_censor_data.py
