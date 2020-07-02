#!/usr/local/bin/Rscript --vanilla
# Usage pet_to_network.r -d path/to/dataset
# 
# Requirements:
# * optparse
# * ppcor
# * reticulate
# * RcppCNPy
# * progress
# 
# Last updated: 3 May 2020
# Author: Raghav Prasad


to_install <- c("optparse", "ppcor", "reticulate", "RcppCNPy", "progress")

install.packages(setdiff(to_install, rownames(installed.packages())), repos = "https://mirrors.ustc.edu.cn/CRAN/")

library("optparse")
library("ppcor")
library('reticulate')
library('RcppCNPy')
library('progress')

 
option_list = list(
  make_option(c("-d", "--dataset"), default=NULL, type="character",
              help="path to dataset directory", metavar="character")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$dataset)){
  print_help(opt_parser);
  stop("At least one argument must be supplied (input file).n", call.=FALSE);
}

FSLDIR <- Sys.getenv('FSLDIR');
if(FSLDIR == '')
	stop("Seems like you don't have FSL installed", call.=FALSE);

dataset_path <- opt$dataset
# print(dataset_path)
parcel_path <- paste(FSLDIR, "/data/atlases/Talairach/Talairach-labels-2mm.nii.gz", sep='')
scan_paths <- Sys.glob(file.path(opt$dataset, "*"))
scan_paths <- scan_paths[- grep("Metadata", scan_paths, value=FALSE)]

for(i in 1:length(scan_paths)) {
    if(!file.exists(file.path(scan_paths[i], 'out.nii.gz')))
        scan_paths <- scan_paths[-i]
}

total <- length(scan_paths)

source_python("pet_helper_funcs.py")

pb <- progress_bar$new(
	format = "Networks constructed (:current/:total) [:bar] (:percent) in :elapsed, eta: :eta",
	total=total, clear=FALSE)

pb$tick(0)
for (scan_path in scan_paths) {
	if(!file.exists(file.path(scan_path, "out.nii.gz")))
		next
	matrices <- time_series_to_matrix(file.path(scan_path, "out.nii.gz"), parcel_path)
	pre_adj_transpose <- t(matrices$pre_adj)
	adj_mat <- pcor(pre_adj_transpose, "pearson")

	adj_mat_path <- file.path(scan_path, "adj_mat.npy")
	percolation_path <- file.path(scan_path, "percolation.npy")
	npySave(adj_mat_path, adj_mat$estimate)
	npySave(percolation_path, matrices$percolation)
	pb$tick()
}




