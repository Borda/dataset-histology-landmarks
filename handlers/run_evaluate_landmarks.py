"""
Validate landmarks with a consensus for a particular annotation

The expected structure of annotations is as follows
ANNOTATIONS/<tissue>/<user>_scale-<number>pc/<csv-file>

EXAMPLE
-------
>> python run_evaluate_landmarks.py -a annotations -o output
>> python handlers/run_evaluate_landmarks.py \
    -a /datagrid/Medical/dataset_ANHIR/landmarks_annot \
    -i /datagrid/Medical/dataset_ANHIR/images_private \
    -o /local/borovec/Data/dataset-ANHIR-visu --nb_jobs 2 --visual

Copyright (C) 2014-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import argparse
import logging
import os
import sys
from functools import partial

import matplotlib

if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from birl.utilities.experiments import iterate_mproc_map

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from handlers.utilities import NB_THREADS, TEMPLATE_FOLDER_SCALE
from handlers.utilities import (
    parse_args, find_image_full_size, collect_triple_dir, list_sub_folders,
    create_consensus_landmarks, compute_landmarks_statistic, parse_path_user_scale,
    find_images, load_image, figure_image_landmarks, create_folder_path
)
NAME_FIGURE_COANNOT = 'CO-ANNOTATION___%s.pdf'


def create_arg_parser():
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
    parser.add_argument('--consensus', type=str, required=False,
                        help='method for consensus landmarks',
                        choices=['mean', 'median'], default='mean')
    parser.add_argument('--visual', required=False, action='store_true',
                        help='export co-annotation visualisation', default=False)
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes in parallel',
                        default=NB_THREADS)
    return parser


def visual_coannotation(lnds_user, lnds_refs, path_dataset, path_user,
                        img_name, path_visu):
    """ visualise the co-annotation

    show consensus annotation and use annotation

    :param DF lnds_user: loaded DataFrame
    :param DF lnds_refs: loaded DataFrame
    :param path_dataset: path to the image dataset (root)
    :param name_set: name of the image set
    :param name_user_scale: annotation folder containing user name and used scale
    :param img_name: particular image/annotation/stain name
    :param str|None path_visu: path to output
    :return str: figure path
    """
    name_set, name_user_scale = path_user.split(os.sep)[-2:]
    user, scale = parse_path_user_scale(name_user_scale)
    folder_scale = TEMPLATE_FOLDER_SCALE % scale
    image = None
    if path_dataset is not None and os.path.isdir(path_dataset):
        path_dir = os.path.join(path_dataset, name_set, folder_scale)
        paths_image = find_images(path_dir, img_name)
        image = load_image(paths_image[0]) if paths_image else None

    lnds_user = lnds_user * (scale / 100.)
    lnds_refs = lnds_refs * (scale / 100.)
    fig = figure_image_landmarks(lnds_refs, image, lnds_user, lnds2_name=user)

    fig_name = NAME_FIGURE_COANNOT % img_name
    path_dir = os.path.join(path_visu, name_set, name_user_scale)
    create_folder_path(path_dir)
    path_fig = os.path.join(path_dir, fig_name)
    fig.savefig(path_fig)
    plt.close(fig)
    return path_fig


def compute_statistic(path_user, path_refs, tp_consensus='mean', path_dataset=None,
                      path_visu=None):
    """ aggregate statistics over all his annotations

    :param str path_user: path to user annotation
    :param [str] path_refs: path to annotation of other users
    :param str tp_consensus: type of consensus landmarks
    :param str path_dataset: path to image dataset
    :param str path_visu: path for visualisation (root)
    :return [{}]: list of stat. dictionaries
    """
    assert path_user and path_refs, 'missing user or reference annotation'
    lnds_user, _ = create_consensus_landmarks([path_user])
    lnds_refs, _ = create_consensus_landmarks(path_refs, method=tp_consensus)

    list_stats = []
    name_set, name_user_scale = path_user.split(os.sep)[-2:]
    user, scale = parse_path_user_scale(name_user_scale)
    for csv_name in lnds_user:
        if csv_name not in lnds_refs:
            continue
        im_size = find_image_full_size(path_dataset, name_set,
                                       os.path.splitext(csv_name)[0])
        d_stat = compute_landmarks_statistic(lnds_refs[csv_name],
                                             lnds_user[csv_name],
                                             use_affine=False, im_size=im_size)
        d_stat.update({'image_set': name_set,
                       'user': user,
                       'scale': scale,
                       'landmarks': csv_name})
        list_stats.append(d_stat)
        if path_visu is not None and os.path.isdir(path_visu):
            img_name = os.path.splitext(csv_name)[0]
            visual_coannotation(lnds_user[csv_name], lnds_refs[csv_name], path_dataset,
                                path_user, img_name, path_visu)
    return list_stats


def evaluate_user(user_name, path_annots, path_out, path_dataset=None,
                  tp_consensus='mean', visual=False):
    """ evaluate single user statistic against consensus

    :param str user_name: annotator name
    :param str path_annots: path to the root of annotations
    :param str tp_consensus: type of consensus landmarks
    :param str path_out: path for statistic and visualisation
    :param str path_dataset: path to image dataset
    :param bool visual: visualise also outputs
    :return int: processed items
    """
    tissue_sets = list_sub_folders(path_annots)
    path_visu = path_out if visual else None
    stats = []
    for p_set in tissue_sets:
        paths = list_sub_folders(p_set, '*_scale-*pc')
        user_names = [parse_path_user_scale(p)[0].lower()
                      for p in paths]
        paths_lnds_user = [p for p, u in zip(paths, user_names) if u == user_name.lower()]
        paths_lnds_refs = [p for p, u in zip(paths, user_names) if u != user_name.lower()]
        if not paths_lnds_user or not paths_lnds_refs:
            continue
        for path_user in paths_lnds_user:
            stats += compute_statistic(path_user, paths_lnds_refs, tp_consensus,
                                       path_dataset, path_visu)
    if not stats:
        logging.warning('no statistic collected')
        return 0

    df_stats = pd.DataFrame(stats)
    return df_stats


def main(path_annots, path_dataset, path_output, consensus='mean', visual=False,
         nb_jobs=NB_THREADS):
    coll_dirs, _ = collect_triple_dir([path_annots], '', '', with_user=True)
    logging.info('Collected sub-folder: %i', len(coll_dirs))
    user_names = sorted({parse_path_user_scale(d['landmarks'])[0]
                         for d in coll_dirs})
    logging.info('Found users: %r', user_names)
    if len(user_names) < 2:
        logging.info('Not enough user annotations.')

    _evaluate_user = partial(evaluate_user, path_annots=path_annots,
                             path_dataset=path_dataset, path_out=path_output,
                             tp_consensus=consensus, visual=visual)
    dfs = list(iterate_mproc_map(
        _evaluate_user, user_names, nb_workers=nb_jobs, desc='evaluate'))

    # aggregate results
    df_all = pd.concat(dfs, sort=False)
    df_all.to_csv(os.path.join(path_output, 'STATISTIC__partial.csv'))
    df_short = pd.DataFrame()
    for user, dfg in df_all.groupby('user'):
        stat = dict(dfg['rTRE median'].describe().T[['mean', 'std', 'max']])
        stat = {'%s [median rTRE]' % k: stat[k] for k in stat if k != 'count'}
        stat.update({
            'user': user,
            'count': len(dfg),
        })
        df_short = df_short.append(stat, ignore_index=True)
    df_short.set_index('user', inplace=True)
    logging.info('OVERALL \n%s \n %s' % ('=' * 10, df_short))
    df_short.to_csv(os.path.join(path_output, 'STATISTIC__overview.csv'))

    logging.info('Created %i statistics.', len(df_all))
    return len(df_all)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = parse_args(create_arg_parser())
    main(**params)

    logging.info('DONE')
