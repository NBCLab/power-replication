#!/bin/bash
#SBATCH --job-name=power-replication
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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

#python /home/data/nbc/misc-projects/Salo_PowerReplication/code/processing/03_derive_masks_and_censor_data.py
#python /home/data/nbc/misc-projects/Salo_PowerReplication/code/processing/04_run_multi-echo_denoising.py
#python /home/data/nbc/misc-projects/Salo_PowerReplication/code/processing/05_generate_single-echo_confounds.py
python /home/data/nbc/misc-projects/Salo_PowerReplication/code/processing/06_run_single-echo_denoising.py
