"""
Testing most of the executables which do not depend on external data (images)

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import multiprocessing as mproc

import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root

import handlers.utilities as utils
import handlers.run_evaluate_landmarks as r_eval
import handlers.run_generate_landmarks as r_generate
import handlers.run_visualise_landmarks as r_visual

NB_THREADS = max(1, int(mproc.cpu_count() * 0.7))
PATH_ANNOTATIONS = utils.update_path('annotations')
assert os.path.isdir(PATH_ANNOTATIONS), 'missing annot: %s' % PATH_ANNOTATIONS
PATH_DATASET = utils.update_path('dataset')
assert os.path.isdir(PATH_DATASET), 'missing dataset: %s' % PATH_DATASET
PATH_OUTPUT = utils.update_path('output')
if not os.path.isdir(PATH_OUTPUT):
    os.mkdir(PATH_OUTPUT)


def test_00_evaluate_landmarks():
    params = {'path_annots': PATH_ANNOTATIONS,
              'path_dataset': None,
              'path_output': PATH_OUTPUT,
              'nb_jobs': 1}  # coverage is not able to track in parallelism
    counts = r_eval.main(**params)
    assert np.sum(counts) > 0, 'nothing evaluated'


def test_01_generate_landmarks():
    params = {'path_annots': PATH_ANNOTATIONS,
              'path_dataset': PATH_DATASET,
              'scales': utils.SCALES,
              'nb_jobs': 1}  # coverage is not able to track in parallelism
    c_gene, c_scale = r_generate.main(**params)
    assert len(c_gene) > 0, 'nothing generated'
    assert len(c_scale) > 0, 'nothing scaled'


def test_02_visualise_landmarks():
    # NOTE, requite first run the generate script
    params = {'path_landmarks': PATH_DATASET,
              'path_dataset': PATH_DATASET,
              'path_output': PATH_OUTPUT,
              'nb_jobs': 1}  # coverage is not able to track in parallelism
    counts = r_visual.main(**params)
    assert len([n for n in counts if n > 0]) > 0, 'nothing visualised'


if __name__ == '__main__':
    test_00_evaluate_landmarks()
    test_01_generate_landmarks()
    test_02_visualise_landmarks()
