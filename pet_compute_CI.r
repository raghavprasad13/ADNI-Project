#!/usr/local/bin/Rscript --vanilla
# Usage pet_to_network.r -d path/to/dataset
# 
# Requirements:
# * optparse
# * reticulate
# * stringr
# * progress
# 
# Last updated: 15 August 2020
# Authors: Pranav Mahajan, Raghav Prasad

source("collective_influence_algorithm.R")

to_install <- c("optparse", "reticulate", "stringr", "progress")
install.packages(setdiff(to_install, rownames(installed.packages())), repos = "https://mirrors.ustc.edu.cn/CRAN/")

library(reticulate)
library(stringr)
library(optparse)
library(progress)

np <-import("numpy")

sink(file = "/dev/null", append = FALSE, type = c("output", "message"), split = FALSE)	# prevents verbose output
 
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

datasetPath <- opt$dataset


getInfluentialNodes <- function(scan_path) {
    adj_mat <- np$load(file.path(scan_path, "adj_mat_thresh.npy"))
    adj_mat = (adj_mat != 0)
    pet_graph <- graph_from_adjacency_matrix( adj_mat )

    pet_graph <- set_vertex_attr(pet_graph, "name", value=paste("", as.vector(V(pet_graph)), sep=""))

    # find influencers, remove them, and plot the results after each removal
    g <- getInfluencers(pet_graph, d=2)
    return(g$influencers)
}

juelichAtlasDf <- read.csv('Juelich_MMSE_NPIQ.csv')
juelichAtlasDf <- juelichAtlasDf[, c(1, 5)]
colnames(juelichAtlasDf)[1] <- 'node_num'
colnames(juelichAtlasDf)[2] <- 'roi'
juelichAtlasDf$node_num <- juelichAtlasDf$node_num + 1

mmseDf <- read.csv(file.path(datasetPath, 'stats', 'output_mmse.csv'))
npiqDf <- read.csv(file.path(datasetPath, 'stats', 'output_npiq.csv'))

petIds <- c()
influentialNodeValuesMmse <- c()
influentialNodeValuesNpiq <- c()

pb <- progress_bar$new(
	format = "Networks scanned (:current/:total) [:bar] (:percent) in :elapsed, eta: :eta",
	total=length(npiqDf$PET_ID), clear=FALSE)

pb$tick(0)
for (scan in npiqDf$PET_ID) {
    petIds <- append(petIds, scan)
    influentialNodes <- getInfluentialNodes(file.path(datasetPath, scan))

    entryMmse <- ""
    entryNpiq <- ""

    for(influentialNode in influentialNodes) {
        num <- strtoi(influentialNode)
        roi_name <- subset(juelichAtlasDf, node_num == num)$roi
        roi_name <- str_replace_all(roi_name, " ", ".")

        mmsePcValue <- subset(mmseDf, PET_ID == scan)[[roi_name]]
        npiqPcValue <- subset(npiqDf, PET_ID == scan)[[roi_name]]

        if(!is.null(mmsePcValue)) {
            entryM <- paste(roi_name, ": ", mmsePcValue)
            entryMmse <- paste(entryMmse, ", ", entryM)
        }

        if(!is.null(npiqPcValue)) {
            entryN <- paste(roi_name, ":", npiqPcValue)
            entryNpiq <- paste(entryNpiq, ", ", entryN)
        }
    }

    influentialNodeValuesMmse <- append(influentialNodeValuesMmse, entryMmse)
    influentialNodeValuesNpiq <- append(influentialNodeValuesNpiq, entryNpiq)
    pb$tick()
}

influentialDataMmseDf <- data.frame(petIds, influentialNodeValuesMmse)
colnames(influentialDataMmseDf)[1] <- 'PET_ID'
colnames(influentialDataMmseDf)[2] <- 'Influential node values'

influentialDataNpiqDf <- data.frame(petIds, influentialNodeValuesNpiq)
colnames(influentialDataNpiqDf)[1] <- 'PET_ID'
colnames(influentialDataNpiqDf)[2] <- 'Influential node values'

write.csv(influentialDataMmseDf, file.path(datasetPath, "stats", "influential_mmse.csv"))
write.csv(influentialDataNpiqDf, file.path(datasetPath, "stats", "influential_npiq.csv"))