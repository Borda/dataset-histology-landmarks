"""
According given annotations create a consensus annotations
and scale it into particular scales used in dataset

>> python run_generate_landmarks.py -a annotations -d dataset

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import sys
import glob
import logging
import argparse
import multiprocessing as mproc
from functools import partial

import numpy as np
import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from handlers import utils

NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))


def arg_parse_params():
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
                        help='scales generated for the dataset',
                        default=utils.SCALES)
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes in parallel',
                        default=NB_THREADS)
    args = vars(parser.parse_args())
    logging.info('ARG PARAMETERS: \n %s', repr(args))
    for k in (k for k in args if 'path' in k):
        args[k] = utils.update_path(args[k])
        assert os.path.exists(args[k]), 'missing: (%s) "%s"' % (k, args[k])
    return args


def generate_consensus_landmarks(path_set, path_dataset):
    path_annots = [p for p in glob.glob(os.path.join(path_set, '*'))
                   if os.path.isdir(p)]
    logging.debug('>> found annotations: %i', len(path_annots))

    dict_lnds, dict_lens = utils.create_consensus_landmarks(path_annots)

    path_set = utils.create_folder(path_dataset, os.path.basename(path_set))
    path_scale = utils.create_folder(path_set, utils.TEMPLATE_FOLDER_SCALE % 100)
    for name in dict_lnds:
        dict_lnds[name].to_csv(os.path.join(path_scale, name))

    return {os.path.basename(path_set): dict_lens}


def dataset_generate_landmarks(path_annots, path_dataset,
                               nb_jobs=NB_THREADS):
    list_sets = [p for p in glob.glob(os.path.join(path_annots, '*'))
                 if os.path.isdir(p)]
    logging.info('Found sets: %i', len(list_sets))

    counts = list(utils.wrap_execute_parallel(partial(generate_consensus_landmarks,
                                                      path_dataset=path_dataset),
                                              sorted(list_sets),
                                              desc='consensus lnds',
                                              nb_jobs=nb_jobs))
    return counts


def scale_set_landmarks(path_set, scales=utils.SCALES):
    path_scale100 = os.path.join(path_set, utils.TEMPLATE_FOLDER_SCALE % 100)
    if not os.path.isdir(path_scale100):
        logging.error('missing base scale 100% in "%s"', path_scale100)
        return
    list_csv = glob.glob(os.path.join(path_scale100, '*.csv'))
    logging.debug('>> found landmarks: %i', len(list_csv))
    dict_lnds = {os.path.basename(p): pd.read_csv(p, index_col=0)
                 for p in list_csv}
    if 100 in scales:
        scales.remove(100)  # drop the base scale
    set_scales = {}
    for sc in scales:
        path_scale = utils.create_folder(path_set,
                                         utils.TEMPLATE_FOLDER_SCALE % sc)
        for name in dict_lnds:
            df_scale = dict_lnds[name] * (sc / 100.)
            df_scale.to_csv(os.path.join(path_scale, name))
        set_scales[sc] = len(dict_lnds)
    dict_lens = {os.path.basename(path_set): set_scales}
    return dict_lens


def dataset_scale_landmarks(path_dataset, scales=utils.SCALES,
                            nb_jobs=NB_THREADS):
    list_sets = [p for p in glob.glob(os.path.join(path_dataset, '*'))
                 if os.path.isdir(p)]
    logging.info('Found sets: %i', len(list_sets))

    counts = list(utils.wrap_execute_parallel(partial(scale_set_landmarks,
                                                      scales=scales),
                                              sorted(list_sets),
                                              desc='scaling sets',
                                              nb_jobs=nb_jobs))
    return counts


def main(params):
    count_gene = dataset_generate_landmarks(params['path_annots'],
                                            params['path_dataset'],
                                            nb_jobs=params['nb_jobs'])
    count_scale = dataset_scale_landmarks(params['path_dataset'],
                                          scales=params['scales'],
                                          nb_jobs=params['nb_jobs'])
    return count_gene, count_scale


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = arg_parse_params()
    main(params)

    logging.info('DONE')
