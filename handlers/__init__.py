import os
import logging

import matplotlib
if os.environ.get('DISPLAY','') == '' \
        and matplotlib.rcParams['backend'] != 'agg':
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')
