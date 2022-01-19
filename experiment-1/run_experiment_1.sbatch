#!/bin/bash
#SBATCH --job-name=power-exp-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=investor

# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/experiment_1.out
#SBATCH --error=/home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/experiment_1.err
########################################################
# A generic Python job for the SLURM scheduler
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=4
. $MODULESHOME/../global/profile.modules
source /home/data/nbc/misc-projects/Salo_PowerReplication/code/activate_environment

python /home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-1/01_rpv_validation.py
python /home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-1/02_resp_and_motion.py
python /home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-1/03_resp_and_gm.py
python /home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-1/04_hr_and_gm.py
python /home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-1/05_resp_and_fc.py
python /home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-1/06_ica_global_signal.py
