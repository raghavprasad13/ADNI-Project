from scipy.stats import kendalltau as kt
import pandas as pd

from os.path import join
from glob import glob
from itertools import combinations


dataset_path = join('..', 'New_Data')

diagnoses_paths = glob(join(dataset_path, '*'))
radioisotopes = ['AV45', 'FDG', 'PiB']
ranklists = dict()

for diagnosis_path in diagnoses_paths:
    diagnosis = diagnosis_path.split('/')[-1]
    for radioisotope in radioisotopes:
        ranklists[diagnosis+'_'+radioisotope] = pd.read_csv(join(diagnosis_path, radioisotope, 'stats', 'roi_ranking.csv'))['ROI'].tolist()

li = [(key, ranklists[key]) for key in ranklists]

li2 = list(combinations(li, 2))

for tup in li2:
    tup1 = tup[0]
    tup2 = tup[1]

    tup1_name = tup1[0]
    tup2_name = tup2[0]

    tup1_list = tup1[1]
    tup2_list = tup2[1]

    print(tup1_name, ' ', tup2_name, end=': ')
    tau, p = kt(tup1_list, tup2_list)
    print(tau, ', ', p)
