"""
testing most of the executables which do not depend on external data (images)

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import multiprocessing as mproc

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root

import handlers.utils as utils
import handlers.run_generate_landmarks as r_generate

NB_THREADS = max(1, int(mproc.cpu_count() * 0.7))
PATH_ANNOTATIONS = utils.update_path('annotations')
assert os.path.isdir(PATH_ANNOTATIONS), 'missing annot: %s' % PATH_ANNOTATIONS
PATH_LANDMARKS = utils.update_path('landmarks')
if not os.path.isdir(PATH_LANDMARKS):
    os.mkdir(PATH_LANDMARKS)


def test_generate_landmarks():
    params = {'path_annots': PATH_ANNOTATIONS,
              'path_dataset': PATH_LANDMARKS,
              'scales': utils.SCALES,
              'nb_jobs': 1}  # coverage is not able to track in parallelism
    c_gene, c_scale = r_generate.main(params)
    assert len(c_gene) > 0, 'nothing generated'
    assert len(c_scale) > 0, 'nothing scaled'


if __name__ == '__main__':
    test_generate_landmarks()