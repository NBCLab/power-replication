#!/bin/bash
#SBATCH --job-name=godec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=investor

# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/godec.out
#SBATCH --error=/home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/godec.err
########################################################
# A generic Python job for the SLURM scheduler
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=4
. $MODULESHOME/../global/profile.modules
source /home/data/nbc/misc-projects/Salo_PowerReplication/code/activate_environment

python /home/data/nbc/misc-projects/Salo_PowerReplication/code/processing/06a_fix_godec_length.py
