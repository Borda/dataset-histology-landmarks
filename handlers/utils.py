"""
General utils used for this collection of scripts

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import re
import glob
import logging
import multiprocessing as mproc

import tqdm
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt

NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))
SCALES = [5, 10, 25, 50, 100]
# template nema for scale folder
TEMPLATE_FOLDER_SCALE = 'scale-%dpc'
# regular expresion patters for determining scale and user
REEXP_FOLDER_ANNOT = 'user-(.\S+)_scale-(\d+)pc'
REEXP_FOLDER_SCALE = '\S*scale-(\d+)pc'
# default figure size for visualisations
FIGURE_SIZE = 18
# expected image extensions
IMAGE_EXT = ['.png', '.jpg', '.jpeg']
COLORS = 'grbm'


def update_path(path, max_depth=5):
    """ bobble up to find a particular path

    :param str path:
    :param int max_depth:
    :return str:

    >>> os.path.isdir(update_path('handlers'))
    True
    >>> os.path.isdir(update_path('no-handlers'))
    False
    """
    path_in = path
    if path.startswith('/'):
        return path
    for _ in range(max_depth):
        if os.path.exists(path):
            break
        path = os.path.join('..', path)

    path = os.path.abspath(path) if os.path.exists(path) else path_in
    return path


def wrap_execute_parallel(wrap_func, iterate_vals,
                          nb_jobs=NB_THREADS, desc=''):
    """ wrapper for execution parallel of single thread as for...

    :param wrap_func: function which will be excited in the iterations
    :param [] iterate_vals: list or iterator which will ide in iterations
    :param int nb_jobs: number og jobs running in parallel
    :param str desc: description for the bar

    >>> [o for o in wrap_execute_parallel(lambda x: x ** 2, range(5), nb_jobs=1)]
    [0, 1, 4, 9, 16]
    >>> [o for o in wrap_execute_parallel(sum, [[0, 1]] * 5, nb_jobs=2)]
    [1, 1, 1, 1, 1]
    """
    iterate_vals = list(iterate_vals)

    tqdm_bar = tqdm.tqdm(total=len(iterate_vals), desc=desc)

    if nb_jobs > 1:
        logging.debug('perform sequential in %i threads', nb_jobs)
        pool = mproc.Pool(nb_jobs)

        for out in pool.imap_unordered(wrap_func, iterate_vals):
            yield out
            if tqdm_bar is not None:
                tqdm_bar.update()
        pool.close()
        pool.join()
    else:
        for out in map(wrap_func, iterate_vals):
            yield out
            if tqdm_bar is not None:
                tqdm_bar.update()


def create_folder(path_base, folder):
    path_folder = os.path.join(path_base, folder)
    if not os.path.isdir(path_folder):
        os.mkdir(path_folder)
    return path_folder


def parse_path_user_scale(path):
    """ from given path with annotation parse user name and scale

    :param str path: path to the user folder
    :return (str, int):

    >>> parse_path_user_scale('user-KO_scale-.5pc')
    ('', nan)
    >>> parse_path_user_scale('user-JB_scale-50pc')
    ('JB', 50)
    >>> parse_path_user_scale('sample/path/user-ck6_scale-25pc')
    ('ck6', 25)
    """
    path = os.path.basename(path)
    obj = re.match(REEXP_FOLDER_ANNOT, path)
    if obj is None:
        return ('', np.nan)
    user, scale = obj.groups()
    scale = int(scale)
    return user, scale


def parse_path_scale(path):
    """ from given path with annotation parse scale

    :param str path: path to the user folder
    :return int:

    >>> parse_path_scale('scale-.1pc')
    nan
    >>> parse_path_scale('user-JB_scale-50pc')
    50
    >>> parse_path_scale('scale-10pc')
    10
    """
    path = os.path.basename(path)
    obj = re.match(REEXP_FOLDER_SCALE, path)
    if obj is None:
        return np.nan
    scale = int(obj.groups()[0])
    return scale


def collect_triple_dir(list_path_lnds, path_dataset, path_out, coll_dirs=None):
    """ collect all subdir up to level of scales

    :param list_path_lnds:
    :param path_dataset:
    :param path_out:
    :param coll_dirs:
    :return:

    >>> coll_dirs, d = collect_triple_dir([update_path('annotations')],
    ...                                   update_path('dataset'), 'output')
    >>> len(coll_dirs) > 0
    True
    >>> 'annotations' in coll_dirs[0]['landmarks'].split(os.sep)
    True
    >>> 'dataset' in coll_dirs[0]['images'].split(os.sep)
    True
    >>> 'output' in coll_dirs[0]['output'].split(os.sep)
    True
    >>> d
    []
    """
    if coll_dirs is None:
        coll_dirs = []
    for path_lnds in list_path_lnds:
        set_name, scale_name = path_lnds.split(os.sep)[-2:]
        scale = parse_path_scale(scale_name)
        if np.isnan(scale):
            sub_dirs = sorted([p for p in glob.glob(os.path.join(path_lnds, '*'))
                               if os.path.isdir(p)])
            coll_dirs, sub_dirs = collect_triple_dir(sub_dirs, path_dataset,
                                                     path_out, coll_dirs)
            continue
        coll_dirs.append({
            'landmarks': path_lnds,
            'images': os.path.join(path_dataset, set_name,
                                   TEMPLATE_FOLDER_SCALE % scale),
            'output': os.path.join(path_out, set_name, scale_name)
        })
    return coll_dirs, []


def create_figure(im_size, max_fig_size=FIGURE_SIZE):
    norm_size = np.array(im_size) / float(np.max(im_size))
    # reverse dimensions and scale by fig size
    fig_size = norm_size[::-1] * max_fig_size
    fig, ax = plt.subplots(figsize=fig_size)
    return fig, ax


def format_figure(fig, ax, im_size, lnds):
    ax.set_xlim([min(0, np.min(lnds[1])), max(im_size[1], np.max(lnds[1]))])
    ax.set_ylim([max(im_size[0], np.max(lnds[0])), min(0, np.min(lnds[0]))])
    fig.tight_layout()
    return fig


def figure_image_landmarks(landmarks, image, max_fig_size=FIGURE_SIZE):
    """ create a figure with images and landmarks

    :param ndarray landmarks: landmark coordinates
    :param ndarray image: 2D image
    :param int max_fig_size:
    :return Figure:

    >>> np.random.seed(0)
    >>> lnds = np.random.randint(-10, 25, (10, 2))
    >>> img = np.random.random((20, 30))
    >>> fig = figure_image_landmarks(lnds, img)
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    >>> df_lnds = pd.DataFrame(lnds, columns=['X', 'Y'])
    >>> fig = figure_image_landmarks(df_lnds, None)
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    """
    if isinstance(landmarks, pd.DataFrame):
        landmarks = landmarks[['X', 'Y']].values
    if image is None:
        image = np.zeros(np.max(landmarks, axis=0) + 25)

    fig, ax = create_figure(image.shape[:2], max_fig_size)

    ax.imshow(image)
    ax.plot(landmarks[:, 0], landmarks[:, 1], 'go')
    ax.plot(landmarks[:, 0], landmarks[:, 1], 'r.')

    for i, lnd in enumerate(landmarks):
        ax.text(lnd[0] + 5, lnd[1] + 5, str(i + 1), fontsize=11, color='black')

    fig = format_figure(fig, ax, image.shape[:2], landmarks)

    return fig


def figure_pair_images_landmarks(pair_landmarks, pair_images, names=None,
                                 max_fig_size=FIGURE_SIZE):
    """ create a figure with image pair and connect related landmarks by line

    :param (ndarray) pair_landmarks: set of landmark coordinates
    :param (ndarray) pair_images: set of 2D image
    :param int max_fig_size:
    :return Figure:

    >>> np.random.seed(0)
    >>> lnds = np.random.randint(-10, 25, (10, 2))
    >>> img = np.random.random((20, 30))
    >>> fig = figure_pair_images_landmarks((lnds, lnds + 5), (img, img))
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    >>> df_lnds = pd.DataFrame(lnds, columns=['X', 'Y'])
    >>> fig = figure_pair_images_landmarks((df_lnds, df_lnds), (img, None))
    >>> isinstance(fig, matplotlib.figure.Figure)
    True
    """
    assert len(pair_landmarks) == len(pair_images), \
        'not equal counts for images (%i) and landmarks (%i)' \
        % (len(pair_landmarks), len(pair_images))
    pair_landmarks = list(pair_landmarks)
    pair_images = list(pair_images)
    for i, landmarks in enumerate(pair_landmarks):
        if isinstance(landmarks, pd.DataFrame):
            pair_landmarks[i] = landmarks[['X', 'Y']].values
    for i, image in enumerate(pair_images):
        if image is None:
            pair_images[i] = np.zeros(np.max(pair_landmarks[i], axis=0) + 25)

    im_size = np.max([img.shape[:2] for img in pair_images], axis=0)
    fig, ax = create_figure(im_size, max_fig_size)

    # draw semi transparent images
    for image in pair_images:
        ax.imshow(image, alpha=(1. / len(pair_images)))

    # draw lined between landmarks
    for i, lnds1 in enumerate(pair_landmarks[1:]):
        lnds0 = pair_landmarks[i]
        for (x0, y0), (x1, y1) in zip(lnds0, lnds1):
            ax.plot([x0, x1], [y0, y1], '-.', color=COLORS[i % len(COLORS)])

    if names is None:
        names = ['image %i' % i for i in range(len(pair_landmarks))]
    # draw all landmarks
    for i, landmarks in enumerate(pair_landmarks):
        ax.plot(landmarks[:, 0], landmarks[:, 1], 'o',
                color=COLORS[i % len(COLORS)], label=names[i])

    assert len(pair_landmarks) > 0, 'missing any landmarks'
    for i, lnd in enumerate(pair_landmarks[0]):
        ax.text(lnd[0] + 5, lnd[1] + 5, str(i + 1), fontsize=11, color='black')

    ax.legend()

    fig = format_figure(fig, ax, im_size, ([lnds[0] for lnds in pair_landmarks],
                                           [lnds[1] for lnds in pair_landmarks]))

    return fig
