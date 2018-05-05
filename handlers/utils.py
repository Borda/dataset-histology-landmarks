"""


Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import re
import logging
import multiprocessing as mproc

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import tqdm
import numpy as np

NB_THREADS = max(1, int(mproc.cpu_count() * 0.9))
SCALES = [5, 10, 25, 50, 100]
FOLDER_SCALE = 'scale-%dpc'
FOLDER_ANNOT = 'user-(.\S+)_scale-(\d+)pc'


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
    if path.startswith('/'):
        return path
    for _ in range(max_depth):
        if os.path.exists(path):
            break
        path = os.path.join('..', path)
    return os.path.abspath(path)


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


def parse_user_scale(path):
    """ from given path with annotation parse user name and scale

    :param str path: path to the user folder
    :return (str, int):

    >>> parse_user_scale('user-KO_scale-.5pc')
    ('', nan)
    >>> parse_user_scale('user-JB_scale-50pc')
    ('JB', 50)
    >>> parse_user_scale('sample/path/user-ck6_scale-25pc')
    ('ck6', 25)
    """
    path = os.path.basename(path)
    obj = re.match(FOLDER_ANNOT, path)
    if obj is None:
        return ('', np.nan)
    user, scale = obj.groups()
    scale = int(scale)
    return user, scale
