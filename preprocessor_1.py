#!/usr/bin/env python3
# coding: utf-8
#
# This script aids in the preprocessing of the ADNI
# dataset by extracting and creating the appropriate
# directory structure
#
# Author: Raghav Prasad
# Last modified: 1 July 2020

from subprocess import check_call, CalledProcessError
import zipfile
import argparse
from os import remove, mkdir
from os.path import join, exists
from glob import glob
import shutil


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to dataset directory")
args = vars(ap.parse_args())

if args['dataset'][-1] == '/':
    args['dataset'] = args['dataset'][:len(args['dataset'])-1]

dataset_path = args['dataset']

for zfile in glob(join(dataset_path, '*.zip')):
    with zipfile.ZipFile(zfile, 'r') as zf:
        try:
            if zfile.split('/')[-1].split('.')[0].split('_')[-1] != 'metadata':
                zf.extractall(join(dataset_path,
                                   zfile.split('/')[-1].split('.')[0]))
            else:
                zf.extractall(join(dataset_path, 'Metadata'))
        except zipfile.BadZipFile:
            print('WARNING: Bad zip file,', zfile.split('/')[-1])
        except FileNotFoundError:
            print('ERROR: File ', zfile, 'not found')
        finally:
            if exists(zfile):
                remove(zfile)

print('Merging frames...')
for directory in glob(join(dataset_path, '*')):
    if directory.split('/')[-1] == 'Metadata':
        continue
    for subject in glob(join(directory, '*', '*')):
        subject_dir = subject.split('/')[-1]
        for date in glob(join(subject, '*', '*')):
            date_dir = date.split('/')[-1]
            for identifier in glob(join(date, '*')):
                identifier_dir = identifier.split('/')[-1]
                frames = ' '.join(glob(join(identifier, '*.nii')))
                new_dir_name = '~'.join([subject_dir, date_dir,
                                         identifier_dir])
                mkdir(join(dataset_path, new_dir_name))
                try:
                    check_call('fslmerge -t '+join(dataset_path, new_dir_name,
                                                   'combined')+" "+frames,
                               shell=True)
                except CalledProcessError:
                    print('ERROR: fslmerge error')
    shutil.rmtree(directory)
print('DONE')
