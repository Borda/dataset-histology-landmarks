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
from functools import partial

import pandas as pd

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from handlers.utilities import TEMPLATE_FOLDER_SCALE, NB_THREADS
from handlers.utilities import (update_path, load_image, find_images,
                                collect_triple_dir, wrap_execute_parallel,
                                create_consensus_landmarks,
                                compute_landmarks_statistic,
                                parse_path_scale, parse_path_user_scale)


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
    for k in (k for k in args if 'path' in k):
        args[k] = update_path(args[k])
        assert os.path.exists(args[k]), 'missing: (%s) "%s"' % (k, args[k])
    return args


def compute_statistic(path_user, path_refs, path_dataset=None):
    lnds_user, _ = create_consensus_landmarks([path_user])
    lnds_refs, _ = create_consensus_landmarks(path_refs)

    list_stats = []
    set_name, user_name = path_user.split(os.sep)[-2:]
    scale = parse_path_scale(path_user)
    for csv_name in lnds_user:
        if csv_name not in lnds_refs:
            continue
        im_size = None  # default - without images
        if path_dataset is not None:
            # TODO: load only the smallest image scale and multiply by ...
            path_img_dir = os.path.join(path_dataset, set_name,
                                        TEMPLATE_FOLDER_SCALE % scale)
            if os.path.isdir(path_img_dir):
                path_imgs = find_images(path_img_dir, os.path.splitext(csv_name)[0])
                im_size = load_image(path_imgs[0]).shape[:2] if path_imgs else None
        d_stat = compute_landmarks_statistic(lnds_refs[csv_name],
                                             lnds_user[csv_name],
                                             use_affine=False, im_size=im_size)
        d_stat.update({'name_image_set': set_name,
                       'name_user': user_name,
                       'landmarks': csv_name})
        list_stats.append(d_stat)
    return list_stats


def evaluate_user(user_name, path_annots, path_out, path_dataset=None):
    list_sets = sorted([p for p in glob.glob(os.path.join(path_annots, '*'))
                        if os.path.isdir(p)])
    list_stats = []
    for p_set in list_sets:
        paths = sorted([p for p in glob.glob(os.path.join(p_set, '*'))
                        if os.path.isdir(p)])
        user_names = [parse_path_user_scale(p)[0].lower()
                      for p in paths]
        paths_lnds_user = [p for p, u in zip(paths, user_names)
                           if u == user_name.lower()]
        paths_lnds_refs = [p for p, u in zip(paths, user_names)
                           if u != user_name.lower()]
        for path_user in paths_lnds_user:
            list_stats += compute_statistic(path_user, paths_lnds_refs,
                                            path_dataset)
    df_stats = pd.DataFrame(list_stats)
    df_stats.set_index(['name_image_set', 'name_user', 'landmarks'],
                       inplace=True)
    df_stats.to_csv(os.path.join(path_out, 'STATISTIC_%s.csv' % user_name))
    df_stat_short = df_stats.describe().T[['count', 'mean', 'std', 'max']]
    logging.info('USER: %s \n%s \n %s' % (user_name, '=' * 10, df_stat_short))
    return len(df_stats)


def main(path_annots, path_dataset, path_output, nb_jobs=NB_THREADS):
    coll_dirs, _ = collect_triple_dir([path_annots], '', '')
    logging.info('Collected sub-folder: %i', len(coll_dirs))
    user_names = sorted({parse_path_user_scale(d['landmarks'])[0]
                         for d in coll_dirs})
    logging.info('Found users: %s', repr(user_names))

    _evaluate_user = partial(evaluate_user, path_annots=path_annots,
                             path_dataset=path_dataset, path_out=path_output)
    counts = list(wrap_execute_parallel(
        _evaluate_user, user_names, nb_jobs=nb_jobs,
        desc='evaluate @%i-threads' % nb_jobs))
    return counts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = arg_parse_params()
    main(**params)

    logging.info('DONE')
