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
# Last updated: 21 December 2020
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
		$FREESURFER_HOME/bin/mri_fwhm --smooth-only --i $1/mni.nii.gz --fwhm 4.6 --o $1/out.nii.gz >/dev/null 2>&1
	else
		$FREESURFER_HOME/bin/mri_fwhm --i $1/mni.nii.gz --fwhm 4.6 --o $1/out.nii.gz >/dev/null 2>&1
	fi
	# verify_correctness $1
}

registration_and_spatial_realignment() {
	if [[ "$1" != "Metadata" ]]; then
		if [[ -e $1/combined_suv.nii.gz ]]; then
			$FSLDIR/bin/mcflirt -in $1/combined_suv.nii.gz -meanvol -out $1/realigned >/dev/null 2>&1
			$FREESURFER_HOME/bin/mri_vol2vol --mni152reg --mov $1/realigned.nii.gz --o $1/out.nii.gz >/dev/null 2>&1
			# $FSLDIR/bin/mcflirt -in $1/mni.nii.gz -meanvol -out $1/out >/dev/null 2>&1
		fi
	fi
}

export -f registration_and_spatial_realignment
# export -f apply_smoothing_filter
# export -f spatial_realignment
# export -f verify_correctness

if [[ $(echo -n $1|tail -c1) = "/" ]];
then
	DATASET_DIR=$(echo -n $1|head -c `expr ${#1} - 1`)
else
	DATASET_DIR=$1
fi

ls -d "$DATASET_DIR"/* | grep -v "Metadata" | parallel -j+0 --progress --eta registration_and_spatial_realignment {}

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