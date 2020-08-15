#!/usr/bin/env python3
# coding: utf-8
#
# This script thresholds the networks formed in the
# previous stage of the pipeline using a Data-Driven
# thresholding scheme based on Orthogonal Minimal Spanning Trees
#
# Author: Raghav Prasad
# Last modified: 14 August 2020

from dyconnmap.graphs.threshold import threshold_omst_global_cost_efficiency as threshold

import networkx as nx

import numpy as np

from os.path import join, exists
import argparse
from multiprocessing import Pool
from tqdm import tqdm
from glob import glob
import warnings

warnings.filterwarnings('ignore')


def get_thresholded(scan_path):
    adj_mat = np.load(join(scan_path, 'adj_mat.npy'))
    np.nan_to_num(adj_mat, copy=False, posinf=0)

    neg_edges = np.argwhere(adj_mat < 0)
    neg_edges = neg_edges.T
    adj_mat_abs = np.absolute(adj_mat)

    thresholded = threshold(adj_mat_abs)
    thresholded_net = thresholded[1]
    np.fill_diagonal(thresholded_net, 0)
    thresholded_net[neg_edges[0], neg_edges[1]] *= -1

    return (scan_path, thresholded_net)


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to dataset directory")
args = vars(ap.parse_args())

if args['dataset'][-1] == '/':
    args['dataset'] = args['dataset'][:len(args['dataset'])-1]

dataset_path = args['dataset']
scan_paths = glob(join(dataset_path, '*'))

scan_paths = list(filter(lambda path: 'Metadata' not in path, scan_paths))
scan_paths = list(filter(lambda path: 'stats' not in path, scan_paths))
scan_paths = list(filter(lambda path: exists(join(path, 'adj_mat.npy')),
                         scan_paths))


if __name__ == '__main__':
    with Pool() as p:
        with tqdm(total=len(scan_paths), desc='Networks thresholded') as pbar:
            for scan_path, thresholded_net in p.imap_unordered(get_thresholded,
                                                               scan_paths):
                func_net_after = nx.from_numpy_matrix(thresholded_net)
                if func_net_after.number_of_edges() > 0:
                    np.save(join(scan_path, 'adj_mat_thresh.npy'),
                            thresholded_net)
                pbar.update()
