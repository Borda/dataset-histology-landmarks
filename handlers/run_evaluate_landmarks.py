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
import glob
import logging
import argparse
import multiprocessing as mproc
from functools import partial

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
    for k in (k for k in args if 'path' in k):
        args[k] = utils.update_path(args[k])
        assert os.path.exists(args[k]), 'missing: (%s) "%s"' % (k, args[k])
    return args


def compute_statistic(path_user, path_refs):
    d_lnds_user, _ = utils.create_consensus_landmarks([path_user])
    d_lnds_refs, _ = utils.create_consensus_landmarks(path_refs)

    list_stats = []
    set_name, user_name = path_user.split(os.sep)[-2:]
    for name in d_lnds_user:
        if not name in d_lnds_refs:
            continue
        d_stat = utils.compute_landmarks_statistic(d_lnds_refs[name],
                                                   d_lnds_user[name],
                                                   use_affine=False)
        d_stat.update({'name_image_set': set_name,
                       'name_user': user_name,
                       'landmarks': name})
        list_stats.append(d_stat)
    return list_stats


def evaluate_user(user_name, path_annots, path_out):
    list_sets = sorted([p for p in glob.glob(os.path.join(path_annots, '*'))
                        if os.path.isdir(p)])
    list_stats = []
    for p_set in list_sets:
        list_paths = sorted([p for p in glob.glob(os.path.join(p_set, '*'))
                              if os.path.isdir(p)])
        user_names = [utils.parse_path_user_scale(p)[0].lower()
                      for p in list_paths]
        path_users = [p for p, u in zip(list_paths, user_names)
                      if u == user_name.lower()]
        path_refs = [p for p, u in zip(list_paths, user_names)
                     if u != user_name.lower()]
        for path_user in path_users:
            list_stats += compute_statistic(path_user, path_refs)
    df_stats = pd.DataFrame(list_stats)
    df_stats.set_index(['name_image_set', 'name_user', 'landmarks'],
                       inplace=True)
    df_stats.to_csv(os.path.join(path_out, 'STATISTIC_%s.csv' % user_name))
    df_stat_short = df_stats.describe().T[['count', 'mean', 'std', 'max']]
    logging.info('USER: %s \n%s \n %s' % (user_name, '=' * 10, df_stat_short))
    return len(df_stats)


def main(params):
    coll_dirs, _ = utils.collect_triple_dir([params['path_annots']], '', '')
    logging.info('Collected sub-folder: %i', len(coll_dirs))
    user_names = sorted({utils.parse_path_user_scale(d['landmarks'])[0]
                         for d in coll_dirs})
    logging.info('Found users: %s', repr(user_names))

    _evaluate_user = partial(evaluate_user, path_annots=params['path_annots'],
                             path_out=params['path_output'])
    counts = list(utils.wrap_execute_parallel(_evaluate_user, user_names,
                                              desc='evaluate',
                                              nb_jobs=params['nb_jobs']))
    return counts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = arg_parse_params()
    main(params)

    logging.info('DONE')
