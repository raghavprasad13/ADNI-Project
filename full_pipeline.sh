#!/bin/bash
#
# Usage: ./full_pipeline.sh path/to/dataset
#
# Author: Raghav Prasad
# Last modified: 29 June 2020

if [[ $# -ne 1 ]]; then
	echo "Usage: "$0" path/to/dataset"
	exit 1
fi

# echo '#########################################################################'
# echo 'STAGE 1: Preprocessing dataset'
# echo '#########################################################################'

# ./preprocessor_1.py -d $1
# ./preprocessor_2.sh $1

# echo '#########################################################################'
# echo 'STAGE 2: Constructing networks'
# echo '#########################################################################'

# ./pet_to_network.r -d $1

# echo '#########################################################################'
# echo 'STAGE 3: Thresholding networks'
# echo '#########################################################################'

# ./thresholding_module_dyconn.py -d $1

echo '#########################################################################'
echo 'STAGE 4: Analyzing networks'
echo '#########################################################################'

./pet_graph_analysis.py -d $1

echo '#########################################################################'
echo 'STAGE 5: Identifying influential nodes'
echo '#########################################################################'

./pet_compute_CI.r -d $1
