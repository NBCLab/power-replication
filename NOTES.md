# Dataset/processing notes

## dset-camcan

MRIQC on sub-CC410040 fails on T1w. Re-running.

Subject sub-CC223115, sub-CC310400, sub-CC320379, sub-CC321053, sub-CC410032, sub-CC420720, and sub-CC510354
lack some echoes. I removed them from the dataset and from the participants.tsv file.

## dset-cambridge

sub-04620 cannot be processed/analyzed, because the nifti file for echo-2 is corrupted.
Someone posted on OpenNeuro about this issue >1 year ago, but received no response.
I just removed the folder (and corresponding entry in participants.tsv) completely.

## dset-cohen

Echos 2 and 3 seem to consistently be impacted by a slice-wise artifact.
I assume the artifact is related to the ASL sequence.
I am not removing these echos though.

## dset-dalenberg

I removed all FlavorRun files.

## dset-dupre

I removed all cuedSGT files from the dataset because I figured that it would be easier
than filtering the dataset in all of my jobs.
