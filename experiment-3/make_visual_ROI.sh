# Create V1 masks for experiment 3
module load freesurfer-7.1
module load fsl-6.0.1

export SUBJECTS_DIR="/home/data/nbc/misc-projects/Salo_PowerReplication/dset-dalenberg/derivatives/freesurfer"
export OUTPUT_DIR="/home/data/nbc/misc-projects/Salo_PowerReplication/dset-dalenberg/derivatives/power"

declare -a StringArray=( "sub-01" "sub-03" "sub-04" "sub-05" "sub-06" "sub-07" "sub-08" "sub-09" "sub-10" "sub-11" "sub-12" "sub-13" "sub-14" "sub-15" "sub-16" "sub-17" "sub-18" "sub-19" "sub-20" "sub-21" )

for sub_id in ${StringArray[@]}; do
    echo $sub_id
    mri_label2vol \
        --proj frac 0 1 0.01 \
        --label ${SUBJECTS_DIR}/${sub_id}/label/lh.V1_exvivo.thresh.label \
        --subject ${sub_id} \
        --hemi lh \
        --identity \
        --temp ${SUBJECTS_DIR}/${sub_id}/mri/T1.mgz \
        --o ${OUTPUT_DIR}/${sub_id}/anat/${sub_id}_hemi-L_space-T1w_res-T1w_label-V1_mask.nii.gz

    mri_label2vol \
        --proj frac 0 1 0.01 \
        --label ${SUBJECTS_DIR}/${sub_id}/label/rh.V1_exvivo.thresh.label \
        --subject ${sub_id} \
        --hemi rh \
        --identity \
        --temp ${SUBJECTS_DIR}/${sub_id}/mri/T1.mgz \
        --o ${OUTPUT_DIR}/${sub_id}/anat/${sub_id}_hemi-R_space-T1w_res-T1w_label-V1_mask.nii.gz

    fslmaths \
        ${OUTPUT_DIR}/${sub_id}/anat/${sub_id}_hemi-R_space-T1w_res-T1w_label-V1_mask.nii.gz \
        -max ${OUTPUT_DIR}/${sub_id}/anat/${sub_id}_hemi-L_space-T1w_res-T1w_label-V1_mask.nii.gz \
        ${OUTPUT_DIR}/${sub_id}/anat/${sub_id}_space-T1w_label-V1_mask.nii.gz
done
