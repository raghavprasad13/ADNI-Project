#!/usr/bin/env python3
# coding: utf-8
#
# Author: Raghav Prasad
# Last modified: 21 December 2020

from subprocess import check_call, CalledProcessError
import zipfile
from zipfile import zlib
import xml.etree.ElementTree as ET
import argparse
from os import remove, mkdir
from os.path import join, exists
from glob import glob
import shutil
from tqdm import tqdm


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to dataset directory")
args = vars(ap.parse_args())

if args['dataset'][-1] == '/':
    args['dataset'] = args['dataset'][:len(args['dataset'])-1]

dataset_path = args['dataset']

tracer = dataset_path.split('/')[-1]
print('Tracer: ', tracer)

injection_dose = {'PiB': 555, 'FDG': 185, 'AV45': 370}


def find_xml(xml_file_list, subject, identifier):
    for xml_file_path in xml_file_list:
        xml_file = xml_file_path.split('/')[-1]
        if subject in xml_file and identifier in xml_file:
            return xml_file_path

    return None


print('Unzipping files')
zipfiles = glob(join(dataset_path, '*.zip'))

with tqdm(total=len(zipfiles), desc='Files unzipped') as pbar:
    for zfile in zipfiles:
        with zipfile.ZipFile(zfile, 'r') as zf:
            try:
                if zfile.split('/')[-1].split('.')[0].split('_')[-1] != 'metadata':
                    # key = zfile.split('/')[-1].split('.')[0].split('&')[0].split('=')[-1]
                    # last = zfile.split('/')[-1].split('.')[0].split('=')[-1]
                    zf.extractall(join(dataset_path,
                                  zfile.split('/')[-1].split('.')[0]))
                    # zf.extractall(join(dataset_path, key+'_'+last))
                else:
                    zf.extractall(join(dataset_path, 'Metadata'))
            except zipfile.BadZipFile:
                print('WARNING: Bad zip file,', zfile.split('/')[-1])
            except FileNotFoundError:
                print('ERROR: File ', zfile, 'not found')
            except zlib.error:
                pass
            finally:
                if exists(zfile):
                    remove(zfile)
        pbar.update()

print('Merging frames...')
dirs = glob(join(dataset_path, '*'))
dirs = list(filter(lambda path: 'Metadata' not in path, dirs))

metadata_dir = join(dataset_path, 'Metadata', 'ADNI')
metadata_files = glob(join(metadata_dir, '*.xml'))

with tqdm(total=len(dirs), desc='Scans merged') as pbar:
    for directory in dirs:
        # if directory.split('/')[-1] == 'Metadata':
        #     continue
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

                    metadata_xml = find_xml(metadata_files, subject_dir,
                                            identifier_dir)
                    root = ET.parse(metadata_xml).getroot()
                    subject_weight_g = float(root.findall(".//weightKg")[0].text) * 1000

                    try:
                        c_inj = injection_dose[tracer] / subject_weight_g
                    except ZeroDivisionError:
                        print('Zero division error! Check ' + metadata_xml)
                        continue

                    # print(subject_dir, '\t', identifier_dir, '\t', c_inj)

                    try:
                        check_call('fslmerge -t '+join(dataset_path,
                                                       new_dir_name,
                                                       'combined')+" "+frames,
                                   shell=True)
                    except CalledProcessError:
                        print('ERROR: fslmerge error')

                    try:
                        check_call('fslmaths '+join(dataset_path,
                                                    new_dir_name,
                                                    'combined.nii.gz')+" -div "+str(c_inj)+" "+join(dataset_path,
                                                                                                    new_dir_name,
                                                                                                    'combined_suv'),
                                   shell=True)
                    except CalledProcessError:
                        print('ERROR: fslmaths error')
        shutil.rmtree(directory)
        pbar.update()
