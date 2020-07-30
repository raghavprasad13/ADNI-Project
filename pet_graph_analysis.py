#!/usr/bin/env python3
# coding: utf-8
#
# Author: Raghav Prasad
# Last modified: 30 June 2020

import numpy as np
import pandas as pd
import networkx as nx
import argparse
from networkx.classes.function import selfloop_edges
from tqdm import tqdm
from glob import glob
import xml.etree.ElementTree as ET
from os.path import join, exists
from os import mkdir, environ
from multiprocessing import Pool

FSLDIR = environ["FSLDIR"]

tree = ET.parse(FSLDIR+"/data/atlases/Talairach.xml")
root = tree.getroot()

labels = root.findall('data')[0].findall('label')[1:]

list_of_attribs = sorted(list(labels[0].attrib.keys()))[1:]
list_of_attribs.append('Area')

psych_tests = ['LOGIMEM', 'DIGIF', 'DIGIFLEN', 'DIGIB', 'DIGIBLEN', 'ANIMALS',
               'VEG', 'TRAILA', 'TRAILB', 'WAIS', 'BOSTON']
atlas_df = pd.read_csv('Talairach_Atlas.csv', usecols=['Node #', 'Psych Test',
                                                       'Psych Test 2',
                                                       'Psych Test 3',
                                                       'Psych Test 4',
                                                       'Psych Test 5'])
atlas_df.rename(columns={'Psych Test': 'Psych Test 1'}, inplace=True)
# atlas_df.dropna(subset=['Psych Test '+str(i) for i in range(1, 6)],
#                   how='all', inplace=True)
psych_test_nodes = {psych_test: [atlas_df.iloc[ind]['Node #']
                                 for ind in atlas_df.index
                                 if psych_test in
                                 [atlas_df.iloc[ind]['Psych Test '+str(i)]
                                 for i in range(1, 6)]]
                    for psych_test in psych_tests}


def calculate_percolation(matrix, percolation_mat_path,
                          psych_test=None, threshold=0.4):
    """
    Returns the global mean percolation centrality of a graph
    created from an adjacency matrix and nodal percolation
    values
    Parameters
    ----------
    matrix: array-like, shape (n, n)
            Adjacency matrix where each entry is a number
            between 0 and 1 and represents edge weights

    percolation_mat_path: array-like, shape (n,)
                          Array containing percolation values of
                          all the n nodes

    threshold: float, optional[default=0.6]
               A thresholding constant which will cause
               all edge weights < threshold to be reassigned to 0
    Returns
    -------
    mean_perc_centrality: float
                          Global mean percolation centrality
    """
    global labels, list_of_attribs, psych_test_nodes

    if psych_test is not None:
        threshold = 0

    matrix_copy = matrix.copy()
    matrix_copy[abs(matrix_copy) < threshold] = 0

    percolation = np.load(percolation_mat_path)

    node_attribs = {int(label.attrib['index'])-1: {key: int(label.attrib[key])
                    if key != 'Area' else label.text
                    for key in list_of_attribs} for label in labels}

    for i in range(matrix.shape[0]):
        node_attribs[i].update({'percolation': percolation[i]})

    func_net = nx.from_numpy_array(matrix_copy)

    func_net.remove_edges_from(selfloop_edges(func_net))
    nx.set_node_attributes(func_net, node_attribs)

    if psych_test is not None:
        func_net = func_net.subgraph(psych_test_nodes[psych_test])

    perc_centrality = nx.percolation_centrality(func_net, weight='weight')
    mean_perc_centrality = np.mean(np.array(list(perc_centrality.values())))

    return mean_perc_centrality


def obtain_stats(scan_path):
    global psych_tests
    matrix = np.load(join(scan_path, 'adj_mat.npy'))
    percolation_mat_path = join(scan_path, 'percolation.npy')
    scan_id = scan_path.split('/')[-2]
    # print('Scan ID: ', scan_id)
    perc_list = calculate_percolation(matrix, percolation_mat_path)
    # perc_list.extend([calculate_percolation(matrix, percolation_mat_path,
    #                                         psych_test=psych_test)
    #                   for psych_test in psych_tests])
    return (scan_id, perc_list)


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to dataset directory")
args = vars(ap.parse_args())

if args['dataset'][-1] == '/':
    args['dataset'] = args['dataset'][:len(args['dataset'])-1]

dataset_path = args['dataset']
scan_paths = glob(dataset_path+'/*/')

scan_paths = list(filter(lambda path: 'Metadata' not in path, scan_paths))
scan_paths = list(filter(lambda path: exists(join(path, 'adj_mat.npy')),
                         scan_paths))

result = {}
# with tqdm(total=len(scan_paths)) as pbar:
#     for scan_path in scan_paths:
#         key, value = obtain_stats(scan_path)
#         result.update({key: value})
#         pbar.update()
# exit()
if __name__ == '__main__':
    with Pool() as p:
        with tqdm(total=len(scan_paths)) as pbar:
            for key, value in p.imap_unordered(obtain_stats, scan_paths):
                result.update({key: value})
                pbar.update()

# for scan_path in tqdm(scan_paths, desc='Networks processed'):
#     matrix = np.load(scan_path+'adj_mat.npy')
#     percolation_mat_path = scan_path+'percolation.npy'
#     scan_id = scan_path.split('/')[-3]
#     result.update({scan_id:calculate_percolation(matrix,
#                                                   percolation_mat_path)})


# result = dict(sorted(result.items(), key=lambda t: t[0].split('_')[1]))
cols_to_use = ['PET_ID', 'Subject', 'Gender', 'Hand', 'Age']
cols_to_use.extend(psych_tests)

out_header = ['Mean percolation centrality', ]
# out_header.extend([psych_test+'_PC' for psych_test in psych_tests])

out_df = pd.DataFrame.from_dict(result, orient='index', columns=out_header)
out_df.reset_index(inplace=True)
out_df.rename(columns={'index': 'PET_ID'}, inplace=True)

# cn_df = pd.read_csv('cognitively_normal.csv', usecols=cols_to_use)
# out_cn_df = out_df.merge(cn_df, how='right', on='PET_ID')

# ud_df = pd.read_csv('uncertain_dementia.csv', usecols=cols_to_use)
# out_ud_df = out_df.merge(ud_df, how='right', on='PET_ID')

# ad_df = pd.read_csv('ad_dementia.csv', usecols=cols_to_use)
# out_ad_df = out_df.merge(ad_df, how='right', on='PET_ID')

# print(out_ad_df.head())
# print(out_ad_df.columns)

store_dir_path = join(dataset_path, "stats")
if(not exists(store_dir_path)):
    mkdir(store_dir_path)

out_df.to_csv(join(store_dir_path, 'output.csv'), index=False)
# out_cn_df.to_csv(os.path.join(store_dir_path,
#                               'cognitively_normal_percolation_stats.csv'),
#                  index=False)
# out_ud_df.to_csv(os.path.join(store_dir_path,
#                               'uncertain_dementia_percolation_stats.csv'),
#                  index=False)
# out_ad_df.to_csv(os.path.join(store_dir_path,
#                               'ad_dementia_percolation_stats.csv'),
#                  index=False)
