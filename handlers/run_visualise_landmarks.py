"""
Visualise landmarks on images for a particular set/scale or whole dataset

The expected structure for dataset is as follows
DATASET/<tissue>/<scale>/<image> and DATASET/<tissue>/<scale>/<csv-file>

EXAMPLE
-------
>> python run_visualise_landmarks.py -l dataset -i dataset -o output

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import sys
import glob
import logging
import argparse
import multiprocessing as mproc

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from handlers import utils

NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))


def arg_parse_params():
    """ argument parser from cmd

    SEE: https://docs.python.org/3/library/argparse.html
    :return {str: ...}:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--path_landmarks', type=str, required=False,
                        help='path to folder with landmarks (particular scale)',
                        default='dataset')
    parser.add_argument('-i', '--path_dataset', type=str, required=False,
                        help='path to folder with dataset (images)',
                        default='dataset')
    parser.add_argument('-o', '--path_output', type=str, required=False,
                        help='path to the output directory - visualisation',
                        default='output')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes in parallel',
                        default=NB_THREADS)
    args = vars(parser.parse_args())
    logging.info('ARG PARAMETERS: \n %s', repr(args))
    for k in (k for k in args if 'path' in k):
        args[k] = utils.update_path(args[k])
        assert os.path.exists(args[k]), 'missing: (%s) "%s"' % (k, args[k])
    return args


def export_visual_set_scale(d_paths):
    list_lnds = sorted(glob.glob(os.path.join(d_paths['landmarks'], '*.csv')))
    list_lnds_imgs = []
    # fined relevant images to the given landmarks
    for p_lnds in list_lnds:
        name_ = os.path.splitext(os.path.basename(p_lnds))[0]
        list_name_like = glob.glob(os.path.join(d_paths['images'], name_ + '.*'))
        p_imgs = [p for p in list_name_like
                  if os.path.splitext(os.path.basename(p))[-1] in utils.IMAGE_EXT]
        if len(p_imgs) > 0:
            list_lnds_imgs.append((p_lnds, sorted(p_imgs)[0]))
    # if there are no images or landmarks, skip it...
    if len(list_lnds_imgs) == 0:
        logging.debug('no image-landmarks to show...')
        return 0
    # create the output folder
    if not os.path.isdir(d_paths['output']):
        os.makedirs(d_paths['output'])
    # draw and export image-landmarks
    for p_lnds, p_img in list_lnds_imgs:
        name_ = os.path.splitext(os.path.basename(p_img))[0]
        fig = utils.figure_image_landmarks(pd.read_csv(p_lnds),
                                           utils.load_image(p_img))
        fig.savefig(os.path.join(d_paths['output'], name_ + '.pdf'))
        plt.close(fig)
    # draw and export PAIRS of image-landmarks
    for i, (p_lnds, p_img0) in enumerate(list_lnds_imgs):
        name0 = os.path.splitext(os.path.basename(p_img0))[0]
        lnd0 = pd.read_csv(p_lnds)
        img0 = utils.load_image(p_img)
        for p_lnds, p_img in list_lnds_imgs[i+1:]:
            name_ = os.path.splitext(os.path.basename(p_img))[0]
            lnd_ = pd.read_csv(p_lnds)
            img_ = utils.load_image(p_img)
            fig = utils.figure_pair_images_landmarks((lnd0, lnd_), (img0, img_),
                                                     names=(name0, name_))
            name = 'PAIR___%s___AND___%s.pdf' % (name0, name_)
            fig.savefig(os.path.join(d_paths['output'], name))
            plt.close(fig)
    return len(list_lnds_imgs)


def main(params):
    assert params['path_landmarks'] != params['path_output'], \
        'this folder "%s" cannot be used as output' % params['path_output']
    assert params['path_dataset'] != params['path_output'], \
        'this folder "%s" cannot be used as output' % params['path_output']

    coll_dirs, _ = utils.collect_triple_dir([params['path_landmarks']],
                                             params['path_dataset'],
                                             params['path_output'])
    logging.info('Collected sub-folder: %i', len(coll_dirs))

    counts = list(utils.wrap_execute_parallel(export_visual_set_scale, coll_dirs,
                                              desc='visualise',
                                              nb_jobs=params['nb_jobs']))
    return counts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = arg_parse_params()
    main(params)

    logging.info('DONE')
