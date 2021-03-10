#!/bin/bash
#---Number of cores
#SBATCH -c 1

#---Job's name in SLURM system
#SBATCH -J mriqc_wrangler

#---Error file
#SBATCH -e /home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/mriqc_wrangler_err

#---Output file
#SBATCH -o /home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/mriqc_wrangler_out

#---Queue name
#SBATCH --account iacc_nbc

#---Partition
#SBATCH -p default-partition
########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=1
. $MODULESHOME/../global/profile.modules
source /home/data/nbc/misc-projects/Salo_PowerReplication/code/activate_environment

python /home/data/nbc/misc-projects/Salo_PowerReplication/code/mriqc_job_wrangler.py \
    -t /home/data/nbc/misc-projects/Salo_PowerReplication/code/mriqc_job_template.sh \
    --tsv_file /home/data/nbc/misc-projects/Salo_PowerReplication/dset-dupre/participants.tsv \
    --job_limit 5
