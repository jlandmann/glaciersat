"""
Configuration file and options

A number of globals are defined here to be available everywhere.

The configuration idea is basically picked up from OGGM.
"""

import logging
import os
import sys
from collections import OrderedDict
from configobj import ConfigObj, ConfigObjError

# Defaults
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# Local logger
log = logging.getLogger(__name__)

CONFIG_MODIFIED = False


class ResettingOrderedDict(OrderedDict):
    """OrderedDict wrapper that resets our multiprocessing on set.

    Copied from: https://github.com/OGGM/oggm/blob/master/oggm/cfg.py
    """

    def __setitem__(self, key, value):
        global CONFIG_MODIFIED
        OrderedDict.__setitem__(self, key, value)
        CONFIG_MODIFIED = True


class PathOrderedDict(ResettingOrderedDict):
    """Quick "magic" to be sure that paths are expanded correctly.

    Copied from: https://github.com/OGGM/oggm/blob/master/oggm/cfg.py
    """

    def __setitem__(self, key, value):
        # Overrides the original dic to expand the path
        try:
            value = os.path.expanduser(value)
        except AttributeError:
            raise ValueError('The value you are trying to set does not seem to'
                             ' be a valid path: {}'.format(value))

        ResettingOrderedDict.__setitem__(self, key, value)


class DocumentedDict(dict):
    """Quick "magic" to document the BASENAMES entries.

    Copied from: https://github.com/OGGM/oggm/blob/master/oggm/cfg.py
    """

    def __init__(self):
        self._doc = dict()

    def _set_key(self, key, value, docstr=''):
        if key in self:
            raise ValueError('Cannot overwrite a key.')
        dict.__setitem__(self, key, value)
        self._doc[key] = docstr

    def __setitem__(self, key, value):
        # Overrides the original dic to separate value and documentation
        global CONFIG_MODIFIED
        try:
            self._set_key(key, value[0], docstr=value[1])
            CONFIG_MODIFIED = True
        except BaseException:
            raise ValueError('DocumentedDict accepts only tuple of len 2')

    def info_str(self, key):
        """Info string for the documentation."""
        return '    {}'.format(self[key]) + '\n' + '        ' + self._doc[key]

    def doc_str(self, key):
        """Info string for the documentation."""
        return '        {}'.format(self[key]) + '\n' + '            ' + \
               self._doc[key]


# Globals
IS_INITIALIZED = False
CONTINUE_ON_ERROR = False
PARAMS = OrderedDict()
NAMES = OrderedDict()
PATHS = PathOrderedDict()
BASENAMES = DocumentedDict()


_doc = 'Satellite images and derived variables like binary snow maps or snow' \
       ' line altitude.'
BASENAMES['sat_images'] = ('sat_images.nc', _doc)


def initialize(file=None):
    """Read the configuration file containing the run's parameters."""

    global IS_INITIALIZED
    global BASENAMES
    global PARAMS
    global PATHS
    global NAMES
    global CONTINUE_ON_ERROR

    if file is None:
        file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'params.cfg')

    log.info('Parameter file: %s', file)

    try:
        cp = ConfigObj(file, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Config file could not be parsed (%s): %s', file, e)
        sys.exit()

    k = 'landsat8_tiles_ch'
    PARAMS[k] = [str(vk) for vk in cp.as_list(k)]
    k = 'sentinel2_tiles_ch'
    PARAMS[k] = [str(vk) for vk in cp.as_list(k)]
    k = 'cloudcover_range'
    PARAMS[k] = [float(vk) for vk in cp.as_list(k)]
    k = 'cloud_heights_range'
    PARAMS[k] = [float(vk) for vk in cp.as_list(k)]
    k = 'erode_n_pixels'
    PARAMS[k] = cp.as_int(k)
    k = 'dilate_n_pixels'
    PARAMS[k] = cp.as_int(k)

    PATHS['sentinel_download_path'] = cp['sentinel_download_path']

    # Delete non-floats
    ltr = ['sentinel_download_path', 'landsat8_tiles_ch', 'sentinel2_tiles_ch',
           'cloudcover_range', 'cloud_heights_range', 'dilate_n_pixels',
           'erode_n_pixels']
    for k in ltr:
        cp.pop(k, None)

    # Other params are floats
    for k in cp:
        PARAMS[k] = cp.as_float(k)

    IS_INITIALIZED = True
