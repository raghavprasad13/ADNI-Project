#!/bin/bash
# Usage ./preprocessing.sh path/to/dataset
# 
# # PET data preprocessing pipeline
# ### Software used:
# * Freesurfer by MGH
# * FMRIB's FSL
#
# ### Preprocessing steps:
# 1. Registration to MNI space
# 2. Application of a smoothing filter. Here, according to the scanner being used, we use FWHM = 4.6mm 
# 
# ### Requirements:
# * gnu-time
# * GNU parallel (different from the parallel implemented in the moreutils package)
# 
# Last updated: 23 April 2020
# Author: Raghav Prasad

source $FREESURFER_HOME/SetUpFreeSurfer.sh

if [[ $# -ne 1 ]]
then
	echo "Usage: "$0" path/to/dataset"
	exit 1
fi

if [[ -z "$FREESURFER_HOME" ]]; then
	echo "ERROR: FreeSurfer not found"
	echo "Perhaps you have not installed FreeSurfer"
	echo -e "Download and install it from here: \033[4mhttps://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall\033[0m"
fi

# problematic_dirs=()

# verify_correctness() {
# 	NUM_FILES=$(ls $1| wc -l)
# 	if [[ $NUM_FILES -lt 5 ]]; then
# 		problematic_dirs+=($1);
# 	fi
# }

apply_smoothing_filter() {
	if [[ `$FSLDIR/bin/fslval $1/mni.nii.gz dim4` -lt 10 ]]; then
		$FREESURFER_HOME/bin/mri_fwhm --smooth-only --i $1/mni.nii.gz --fwhm 4.6 --o $1/out.nii.gz >/dev/null
	else
		$FREESURFER_HOME/bin/mri_fwhm --i $1/mni.nii.gz --fwhm 4.6 --o $1/out.nii.gz >/dev/null
	fi
	# verify_correctness $1
}

registration() {
	if [[ "$1" != "Metadata" ]]; then
		if [[ -e $1/combined.nii.gz ]]; then
			$FREESURFER_HOME/bin/mri_vol2vol --mni152reg --mov $1/combined.nii.gz --o $1/mni.nii.gz >/dev/null
			apply_smoothing_filter $1
		fi
	fi
}

export -f registration
export -f apply_smoothing_filter
# export -f verify_correctness

if [[ $(echo -n $1|tail -c1) = "/" ]];
then
	DATASET_DIR=$(echo -n $1|head -c `expr ${#1} - 1`)
else
	DATASET_DIR=$1
fi

ls -d "$DATASET_DIR"/* | grep -v "Metadata" | parallel -j+0 --progress --eta registration {}

# SCAN_PATHS=$(ls -d "$DATASET_DIR"/*/*/pet1)

# for SCAN_PATH in $SCAN_PATHS; do
# 	verify_correctness $SCAN_PATH
# done

# echo "${problematic_dirs[@]}"

# if [[ ${#problematic_dirs} -gt 0 ]]; then
# 	echo "The following subjects have issues"
# 	for dir in "${problematic_dirs[@]}"; do
# 		subject_name=$(echo "$dir")
# 		echo $subject_name 
# 	done
# 	exit 2
# fi