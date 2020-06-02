#!/bin/bash
#---Number of cores
#SBATCH -c 4

#---Job's name in SLURM system
#SBATCH -J hcp_pa

#---Error file
#SBATCH -e power_analysis_glms_err

#---Output file
#SBATCH -o power_analysis_glms_out

#---Queue name
#SBATCH --account iacc_nbc

#---Partition
#SBATCH -p IB_40C_1.5T
########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=$NPROCS
. $MODULESHOME/../global/profile.modules

module load fsl/5.0.10

feat /scratch/tsalo006/motor_lh_power_analysis_design.fsf

feat /scratch/tsalo006/motor_rh_power_analysis_design.fsf

feat /scratch/tsalo006/visual_power_analysis_design.fsf
