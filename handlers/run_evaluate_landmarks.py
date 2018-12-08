"""
Validate landmarks with a consensus for a particular annotation

The expected structure of annotations is as follows
ANNOTATIONS/<tissue>/<user>_scale-<number>pc/<csv-file>

EXAMPLE
-------
>> python run_evaluate_landmarks.py -a annotations -o output

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import sys
import logging
import argparse
from functools import partial

import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from handlers.utilities import NB_THREADS
from handlers.utilities import (
    assert_paths, find_image_full_size, collect_triple_dir, list_sub_folders,
    wrap_execute_parallel, create_consensus_landmarks, compute_landmarks_statistic,
    parse_path_user_scale
)


def arg_parse_params():
    """ argument parser from cmd

    SEE: https://docs.python.org/3/library/argparse.html
    :return {str: ...}:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--path_annots', type=str, required=False,
                        help='path to folder with annotations',
                        default='annotations')
    parser.add_argument('-i', '--path_dataset', type=str, required=False,
                        help='path to folder with dataset (images)',
                        default='dataset')
    parser.add_argument('-o', '--path_output', type=str, required=False,
                        help='path to the output directory - visualisation',
                        default='output')
    # parser.add_argument('--visual', required=False, action='store_true',
    #                     help='export debug visualisations', default=False)
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes in parallel',
                        default=NB_THREADS)
    args = vars(parser.parse_args())
    logging.info('ARG PARAMETERS: \n %s', repr(args))
    args = assert_paths(args)
    return args


def compute_statistic(path_user, path_refs, path_dataset=None):
    lnds_user, _ = create_consensus_landmarks([path_user])
    lnds_refs, _ = create_consensus_landmarks(path_refs)

    list_stats = []
    set_name, user_name = path_user.split(os.sep)[-2:]
    for csv_name in lnds_user:
        if csv_name not in lnds_refs:
            continue
        im_size = find_image_full_size(path_dataset, set_name,
                                       os.path.splitext(csv_name)[0])
        d_stat = compute_landmarks_statistic(lnds_refs[csv_name],
                                             lnds_user[csv_name],
                                             use_affine=False, im_size=im_size)
        d_stat.update({'name_image_set': set_name,
                       'name_user': user_name,
                       'landmarks': csv_name})
        list_stats.append(d_stat)
    return list_stats


def evaluate_user(user_name, path_annots, path_out, path_dataset=None):
    tissue_sets = list_sub_folders(path_annots)
    stats = []
    for p_set in tissue_sets:
        paths = list_sub_folders(p_set, '*_scale-*pc')
        user_names = [parse_path_user_scale(p)[0].lower()
                      for p in paths]
        paths_lnds_user = [p for p, u in zip(paths, user_names)
                           if u == user_name.lower()]
        paths_lnds_refs = [p for p, u in zip(paths, user_names)
                           if u != user_name.lower()]
        for path_user in paths_lnds_user:
            stats += compute_statistic(path_user, paths_lnds_refs, path_dataset)
    if not stats:
        logging.warning('no statistic collected')
        return 0

    df_stats = pd.DataFrame(stats)
    df_stats.set_index(['name_image_set', 'name_user', 'landmarks'],
                       inplace=True)
    df_stats.to_csv(os.path.join(path_out, 'STATISTIC_%s.csv' % user_name))
    df_stat_short = df_stats.describe().T[['count', 'mean', 'std', 'max']]
    logging.info('USER: %s \n%s \n %s' % (user_name, '=' * 10, df_stat_short))
    return len(df_stats)


def main(path_annots, path_dataset, path_output, nb_jobs=NB_THREADS):
    coll_dirs, _ = collect_triple_dir([path_annots], '', '', with_user=True)
    logging.info('Collected sub-folder: %i', len(coll_dirs))
    user_names = sorted({parse_path_user_scale(d['landmarks'])[0]
                         for d in coll_dirs})
    logging.info('Found users: %s', repr(user_names))
    if len(user_names) < 2:
        logging.info('Not enough user annotations.')

    _evaluate_user = partial(evaluate_user, path_annots=path_annots,
                             path_dataset=path_dataset, path_out=path_output)
    counts = list(wrap_execute_parallel(
        _evaluate_user, user_names, nb_jobs=nb_jobs,
        desc='evaluate @%i-threads' % nb_jobs))
    logging.info('Created %i statistics.', sum(counts))
    return counts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = arg_parse_params()
    main(**params)

    logging.info('DONE')
