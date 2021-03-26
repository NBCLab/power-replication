#!/bin/bash
#---Number of cores
#SBATCH -c 1

#---Job's name in SLURM system
#SBATCH -J {sid}_fmriprep

#---Error file
#SBATCH -e /home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/fmriprep_{sid}_err

#---Output file
#SBATCH -o /home/data/nbc/misc-projects/Salo_PowerReplication/code/jobs/fmriprep_{sid}_out

#---Queue name
#SBATCH --account iacc_nbc

#---Partition
#SBATCH -p investor
########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=1
. $MODULESHOME/../global/profile.modules
module load singularity-3.5.3

SINGULARITYENV_NO_ET=1
SINGULARITYENV_TEMPLATEFLOW_HOME="/opt/templateflow"

# Run MRIQC
singularity run --home $HOME --cleanenv \
    -B /home/tsalo006/.cache/templateflow:$SINGULARITYENV_TEMPLATEFLOW_HOME \
    /home/data/cis/singularity-images/nipreps_fmriprep_20.2.1.sif \
    {dset_dir} \
    {out_dir} \
    participant \
    --participant-label {sid} \
    -w {work_dir} \
    --nprocs 1 \
    --output-spaces MNI152NLin6Asym:res-native anat:res-native boldref:res-native \
    --fs-license-file /home/tsalo006/freesurfer_license.txt \
    --fs-no-reconall
