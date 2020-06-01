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
#SBATCH -p centos7
########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=$NPROCS
. $MODULESHOME/../global/profile.modules

/home/applications/fsl/6.0.3/bin/feat /home/tsalo006/Desktop/hcp-motor/lh-avg.gfeat/design.fsf

/home/applications/fsl/6.0.3/bin/feat /home/tsalo006/Desktop/hcp-motor/rh-avg.gfeat/design.fsf

/home/applications/fsl/6.0.3/bin/feat /home/tsalo006/Desktop/hcp-visual/back0.gfeat/design.fsf