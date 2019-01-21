"""
According given annotations create a consensus annotations
and scale it into particular scales used in dataset

The expected structure of annotations is as follows
ANNOTATIONS/<tissue>/<user>_scale-<number>pc/<csv-file>
The expected structure of dataset is
DATASET/<tissue>/scale-<number>pc/<image-file>

EXAMPLE
-------
>> python run_generate_landmarks.py -a annotations -d dataset

>> python handlers/run_generate_landmarks.py \
    -a /datagrid/Medical/dataset_ANHIR/landmarks_annot \
    -d /datagrid/Medical/dataset_ANHIR/landmarks_all \
    --scales 2 5 10 15 20 25 50 100

Copyright (C) 2014-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import sys
import glob
import logging
import argparse
from functools import partial

import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from handlers.utilities import SCALES, TEMPLATE_FOLDER_SCALE, NB_THREADS
from handlers.utilities import (
    parse_args, create_folder_path, wrap_execute_parallel, list_sub_folders,
    create_consensus_landmarks
)


def create_arg_parser():
    """ argument parser from cmd

    SEE: https://docs.python.org/3/library/argparse.html
    :return {str: ...}:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--path_annots', type=str, required=False,
                        help='path to folder with annotations',
                        default='annotations')
    parser.add_argument('-d', '--path_dataset', type=str, required=False,
                        help='path to the output directory - dataset',
                        default='dataset')
    parser.add_argument('--scales', type=int, required=False, nargs='*',
                        help='generated scales for the dataset', default=SCALES)
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes in parallel',
                        default=NB_THREADS)
    return parser


def generate_consensus_landmarks(path_set, path_dataset):
    """ generate consensus landmarks for a particular image/landmark set

    :param str path_set: path to the set with annotations
    :param str path_dataset: output dataset path (root)
    :return {str: int}:
    """
    path_annots = list_sub_folders(path_set, '*_scale-*pc')
    logging.debug('>> found annotations: %i', len(path_annots))

    dict_lnds, dict_lens = create_consensus_landmarks(path_annots)

    path_scale = os.path.join(path_dataset, os.path.basename(path_set),
                              TEMPLATE_FOLDER_SCALE % 100)
    create_folder_path(path_scale)
    for name in dict_lnds:
        dict_lnds[name].to_csv(os.path.join(path_scale, name))

    return {os.path.basename(path_set): dict_lens}


def dataset_generate_landmarks(path_annots, path_dataset,
                               nb_jobs=NB_THREADS):
    """ generate consensus landmarks in full scale (100%)

    :param str path_annots: path to folder with annotations
    :param str path_dataset: output dataset path
    :param nb_jobs: run parallel jobs
    :return [int]:
    """
    list_sets = list_sub_folders(path_annots)
    logging.info('Found sets: %i', len(list_sets))

    _wrap_lnds = partial(generate_consensus_landmarks, path_dataset=path_dataset)
    counts = list(wrap_execute_parallel(
        _wrap_lnds, sorted(list_sets), nb_jobs=nb_jobs, desc='consensus landmarks'))
    return counts


def scale_set_landmarks(path_set, scales=SCALES):
    """ scale given set with landmarks to particular scales

    :param str path_set: path to image/landmark set
    :param [int] scales: selected output scales
    :return {str: int}: collection of lengths
    """
    logging.debug('> processing: %s', path_set)
    path_scale100 = os.path.join(path_set, TEMPLATE_FOLDER_SCALE % 100)
    if not os.path.isdir(path_scale100):
        logging.error('missing base scale 100pc in "%s"', path_scale100)
        return
    list_csv = glob.glob(os.path.join(path_scale100, '*.csv'))
    logging.debug('>> found landmarks: %i', len(list_csv))
    dict_lnds = {os.path.basename(p): pd.read_csv(p, index_col=0)
                 for p in list_csv}
    set_scales = {}
    for sc in (sc for sc in scales if sc not in [100]):  # drop the base scale
        path_scale = os.path.join(path_set, TEMPLATE_FOLDER_SCALE % sc)
        create_folder_path(path_scale)
        for name in dict_lnds:
            df_scale = dict_lnds[name] * (sc / 100.)
            df_scale.to_csv(os.path.join(path_scale, name))
        set_scales[sc] = len(dict_lnds)
    dict_lens = {os.path.basename(path_set): set_scales}
    return dict_lens


def dataset_scale_landmarks(path_dataset, scales=SCALES, nb_jobs=NB_THREADS):
    """" scale whole dataset

    :param str path_dataset:
    :param [int] scales: selected output scales
    :param nb_jobs: run parallel jobs
    :return [int]:
    """
    list_sets = list_sub_folders(path_dataset)
    logging.info('Found sets: %i', len(list_sets))

    _wrap_scale = partial(scale_set_landmarks, scales=scales)
    counts = list(wrap_execute_parallel(
        _wrap_scale, sorted(list_sets), nb_jobs=nb_jobs, desc='scaling sets'))
    return counts


def main(path_annots, path_dataset, scales, nb_jobs=NB_THREADS):
    count_gene = dataset_generate_landmarks(path_annots, path_dataset,
                                            nb_jobs=nb_jobs)
    count_scale = dataset_scale_landmarks(path_dataset, scales=scales,
                                          nb_jobs=nb_jobs)
    return count_gene, count_scale


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = parse_args(create_arg_parser())
    main(**params)

    logging.info('DONE')
