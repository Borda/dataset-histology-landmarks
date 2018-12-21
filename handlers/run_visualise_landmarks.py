"""
Visualise landmarks on images for a particular set/scale or whole dataset

The expected structure for dataset is as follows
 * DATASET/<tissue>/scale-<number>pc/<image>
 * DATASET/<tissue>/scale-<number>pc/<csv-file>

EXAMPLE
-------
>> python run_visualise_landmarks.py -l dataset -i dataset -o output
>> python handlers/run_visualise_landmarks.py \
    -l /datagrid/Medical/dataset_ANHIR/landmarks_annot \
    -i /datagrid/Medical/dataset_ANHIR/images_private \
    -o /local/borovec/Data/dataset-ANHIR-visu --nb_jobs 2

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import logging
import argparse

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import cv2 as cv
    OPENCV = True
except ImportError:
    logging.exception('Missing OpenCV, no image warping will be performed.')
    OPENCV = False

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from handlers.utilities import NB_THREADS, SCALES
from handlers.utilities import (
    assert_paths, load_image, find_images, collect_triple_dir,
    wrap_execute_parallel, estimate_affine_transform,
    figure_pair_images_landmarks, figure_image_landmarks
)

NAME_FIGURE_PAIR = 'PAIR___%s___AND___%s.pdf'
NAME_FIGURE_PAIR_WARPED = 'PAIR___%s___AND___%s___WARPED.pdf'


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
    parser.add_argument('--scales', type=int, required=False, nargs='*',
                        help='select scales for visualization', default=SCALES)
    parser.add_argument('--nb_jobs', type=int, required=False, default=NB_THREADS,
                        help='number of processes in parallel')
    args = vars(parser.parse_args())
    logging.info('ARG PARAMETERS: \n %s', repr(args))
    args = assert_paths(args)
    return args


def load_image_landmarks(lnds_img_pair):
    p_lnds, p_img = lnds_img_pair
    name = os.path.splitext(os.path.basename(p_img))[0]
    lnd = pd.read_csv(p_lnds)
    img = load_image(p_img)
    return name, lnd, img


def warp_affine(img1, img2, lnd1, lnd2):
    nb = min(len(lnd1), len(lnd2))
    pts1, pts2 = lnd1[['X', 'Y']].values[:nb], lnd2[['X', 'Y']].values[:nb]
    _, matrix_inv, _, pts2_warp = estimate_affine_transform(pts1, pts2)
    lnd2_warp = pd.DataFrame(pts2_warp, columns=['X', 'Y'])
    matrix_inv = matrix_inv[:2, :3].astype(np.float64)
    img2_warp = cv.warpAffine(img2, matrix_inv, img1.shape[:2][::-1])
    return img2_warp, lnd2_warp


def export_visual_pairs(lnds_img_pair1, lnds_img_pair2, path_out):
    name1, lnd1, img1 = load_image_landmarks(lnds_img_pair1)
    name2, lnd2, img2 = load_image_landmarks(lnds_img_pair2)
    if img1 is None or img2 is None:
        logging.warning('Fail to load one of required images.')
        return

    fig = figure_pair_images_landmarks((lnd1, lnd2), (img1, img2),
                                       names=(name1, name2))
    fig.savefig(os.path.join(path_out, NAME_FIGURE_PAIR % (name1, name2)))
    plt.close(fig)

    if OPENCV:
        img2_warp, lnd2_warp = warp_affine(img1, img2, lnd1, lnd2)
        del img2, lnd2
        fig = figure_pair_images_landmarks((lnd1, lnd2_warp), (img1, img2_warp),
                                           names=(name1, name2 + ' [WARPED AFFINE]'))
        fig.savefig(os.path.join(path_out, NAME_FIGURE_PAIR_WARPED % (name1, name2)))
        plt.close(fig)


def export_visual_set_scale(d_paths):
    list_lnds = sorted(glob.glob(os.path.join(d_paths['landmarks'], '*.csv')))
    list_lnds_imgs = []
    # fined relevant images to the given landmarks
    for p_lnds in list_lnds:
        name_ = os.path.splitext(os.path.basename(p_lnds))[0]
        p_imgs = find_images(d_paths['images'], name_)
        if p_imgs:
            list_lnds_imgs.append((p_lnds, sorted(p_imgs)[0]))
    # if there are no images or landmarks, skip it...
    if not list_lnds_imgs:
        logging.debug('no image-landmarks to show...')
        return 0
    # create the output folder
    if not os.path.isdir(d_paths['output']):
        os.makedirs(d_paths['output'])
    # draw and export image-landmarks
    for p_lnds, p_img in list_lnds_imgs:
        name_ = os.path.splitext(os.path.basename(p_img))[0]
        img = load_image(p_img)
        if img is None:
            continue
        fig = figure_image_landmarks(pd.read_csv(p_lnds), img)
        fig.savefig(os.path.join(d_paths['output'], name_ + '.pdf'))
        plt.close(fig)
    # draw and export PAIRS of image-landmarks
    path_pairs = [(p1, p2) for i, p1 in enumerate(list_lnds_imgs)
                  for p2 in list_lnds_imgs[i + 1:]]
    for p1, p2 in path_pairs:
        export_visual_pairs(p1, p2, d_paths['output'])
    return len(list_lnds_imgs)


def main(path_landmarks, path_dataset, path_output, scales, nb_jobs=NB_THREADS):
    assert path_landmarks != path_output, \
        'this folder "%s" cannot be used as output' % path_output
    assert path_dataset != path_output, \
        'this folder "%s" cannot be used as output' % path_output

    coll_dirs, _ = collect_triple_dir([path_landmarks], path_dataset,
                                      path_output, scales=scales)
    # filter existing
    coll_dirs = [d for d in coll_dirs
                 if os.path.isdir(d['images']) and os.path.isdir(d['landmarks'])]
    if not coll_dirs:
        logging.info('No sub-folders collected.')
        return 0
    logging.info('Collected sub-folder: %i', len(coll_dirs))

    counts = list(wrap_execute_parallel(
        export_visual_set_scale, coll_dirs, nb_jobs=nb_jobs,
        desc='visualise @%i-threads' % nb_jobs))
    logging.info('Performed %i visualisations', sum(counts))
    return counts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = arg_parse_params()
    main(**params)

    logging.info('DONE')
