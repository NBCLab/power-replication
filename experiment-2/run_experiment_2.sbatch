#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_16C_96G

# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-2/experiment_2.out
#SBATCH --error=/home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-2/experiment_2.err
########################################################
# A generic Python job for the SLURM scheduler
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=4
. $MODULESHOME/../global/profile.modules
source /home/data/nbc/misc-projects/Salo_PowerReplication/code/activate_environment

# Analysis Group 1 is already run. Takes like a day.
# python /home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-2/01_t2s_global_signal.py
# python /home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-2/02_gm_correlations_across_masks.py
# python /home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-2/03_gm_correlations_after_denoising.py
python /home/data/nbc/misc-projects/Salo_PowerReplication/code/experiment-2/04_methods_for_brain_signals.py
# Analysis Group 5 uses its own job.
