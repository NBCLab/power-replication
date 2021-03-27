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
#SBATCH -p investor
########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=1
. $MODULESHOME/../global/profile.modules
source /home/data/nbc/misc-projects/Salo_PowerReplication/code/activate_environment

DSET_NAME="dset-cohen"

python /home/data/nbc/misc-projects/Salo_PowerReplication/code/processing/bids_job_wrangler.py \
    -d /home/data/nbc/misc-projects/Salo_PowerReplication/${DSET_NAME} \
    -w /scratch/nbc/tsalo006/${DSET_NAME} \
    -t /home/data/nbc/misc-projects/Salo_PowerReplication/code/processing/mriqc_job_template.sh \
    --tsv_file /home/data/nbc/misc-projects/Salo_PowerReplication/${DSET_NAME}/participants.tsv \
    --job_limit 20 \
    --copy
