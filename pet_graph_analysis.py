#!/usr/bin/env python3
# coding: utf-8
#
# This script analyses brain networks wrt certain psychometric tests
# and computes global mean percolation centrality and node-wise
# percolation centrality values for all the nodes associated with
# psychometric tests, namely: MMSE and NPI-Q and outputs the same as
# a CSV. For now, this does not compute NPI-Q PC values.
#
# Author: Raghav Prasad
# Last modified: 14 August 2020

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

try:
    FSLDIR = environ["FSLDIR"]  # make sure you have FSL installed
except KeyError:
    print('ERROR: FSL not installed')

"""
Here we parse the Talairach atlas XML file that comes with FSL to
determine the mapping between the nodes and the anatomical regions
they represent. This mapping will be attached to each of the network
instances created in this script
"""
tree = ET.parse(FSLDIR+"/data/atlases/Juelich.xml")
root = tree.getroot()

labels = root.findall('data')[0].findall('label')

list_of_attribs = sorted(list(labels[0].attrib.keys()))
list_of_attribs.append('Area')

psych_tests = ['MMSE', 'NPI-Q']
atlas_df = pd.read_csv('Juelich_MMSE_NPIQ.csv',
                       usecols=['Node #', 'Anatomical Region', 'Test'])

psych_test_nodes = {psych_test: [atlas_df.iloc[ind]['Node #']
                                 for ind in atlas_df.index
                                 if atlas_df.iloc[ind]['Test'] in [psych_test,
                                                                   'Both']]
                    for psych_test in psych_tests}


def export_csv(result_dict, psych_test='MMSE'):
    global atlas_df, dataset_path

    out_header = ['Mean percolation centrality', ]
    out_header.extend([atlas_df.iloc[ind]['Anatomical Region']
                       for ind in atlas_df.index
                       if atlas_df.iloc[ind]['Test'] in [psych_test, 'Both']])

    out_df = pd.DataFrame.from_dict(result_dict, orient='index',
                                    columns=out_header)
    out_df.reset_index(inplace=True)
    out_df.rename(columns={'index': 'PET_ID'}, inplace=True)

    store_dir_path = join(dataset_path, "stats")
    if not exists(store_dir_path):
        mkdir(store_dir_path)

    export_csv_name = 'output_mmse.csv'
    if psych_test == 'NPI-Q':
        export_csv_name = 'output_npiq.csv'

    out_df.to_csv(join(store_dir_path, export_csv_name), index=False)


def calculate_percolation(matrix, percolation_mat_path, psych_test=None):
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

    Returns
    -------
    perc_centrality_list: list
                          A list of percolation centralities corresponding
                          to the nodes associated with psych_test
                          Returned when psych_test is not None
    or
    mean_perc_centrality: float
                          Global mean percolation centrality
                          Returned when psych_test is None
    """
    global labels, list_of_attribs, psych_test_nodes

    matrix_copy = matrix.copy()

    percolation = np.load(percolation_mat_path)
    # Here we create a dictionary mapping nodes to their attributes
    # such as name of anatomical regions, percolation states, etc.
    node_attribs = {int(label.attrib['index']): {key: int(label.attrib[key])
                    if key != 'Area' else label.text
                    for key in list_of_attribs} for label in labels}

    for i in range(matrix.shape[0]):
        node_attribs[i].update({'percolation': percolation[i]})

    func_net = nx.from_numpy_array(matrix_copy)

    del matrix_copy

    func_net.remove_edges_from(selfloop_edges(func_net))
    nx.set_node_attributes(func_net, node_attribs)

    if psych_test is not None:
        func_net = func_net.subgraph(psych_test_nodes[psych_test])

    perc_centrality = nx.percolation_centrality(func_net, weight='weight')

    if psych_test is not None:
        perc_centrality_list = [perc_centrality[node_index]
                                for node_index in perc_centrality]
        return perc_centrality_list

    mean_perc_centrality = np.mean(np.array(list(perc_centrality.values())))

    return mean_perc_centrality


def obtain_stats(scan_path):
    """
    Returns a 2-tuple of the PET scan ID and a list of all the
    PC values calculated for that scan
    Parameters
    ----------
    scan_path: string
               Path to the scan whose stats are to be calculated
    Returns
    -------
    (scan_id, node_perc_list): 2-tuple
                               PET scan ID and corresponding
                               calculated PC values
    """
    global psych_tests
    scan_id = scan_path.split('/')[-1]

    adj_mat_path = join(scan_path, 'adj_mat_thresh.npy')
    if not exists(adj_mat_path):
        return (scan_id, None)

    matrix = np.load(adj_mat_path)
    percolation_mat_path = join(scan_path, 'percolation.npy')

    mean_pc = calculate_percolation(matrix, percolation_mat_path)

    node_perc_list_mmse = [mean_pc, ]
    node_perc_list_mmse.extend(calculate_percolation(matrix,
                                                     percolation_mat_path,
                                                     psych_test='MMSE'))

    node_perc_list_npiq = [mean_pc, ]
    node_perc_list_npiq.extend(calculate_percolation(matrix,
                                                     percolation_mat_path,
                                                     psych_test='NPI-Q'))

    return (scan_id, [node_perc_list_mmse, node_perc_list_npiq])


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to dataset directory")
args = vars(ap.parse_args())

if args['dataset'][-1] == '/':
    args['dataset'] = args['dataset'][:len(args['dataset'])-1]

dataset_path = args['dataset']
scan_paths = glob(join(dataset_path, '*'))

scan_paths = list(filter(lambda path: 'Metadata' not in path, scan_paths))
scan_paths = list(filter(lambda path: exists(join(path, 'adj_mat.npy')),
                         scan_paths))

result_mmse = {}
result_npiq = {}

if __name__ == '__main__':
    with Pool() as p:
        with tqdm(total=len(scan_paths)) as pbar:
            for key, value in p.imap_unordered(obtain_stats, scan_paths):
                if value is not None:
                    result_mmse.update({key: value[0]})
                    result_npiq.update({key: value[1]})
                pbar.update()

    export_csv(result_mmse)
    export_csv(result_npiq, psych_test='NPI-Q')
