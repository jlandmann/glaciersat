from configobj import ConfigObj, ConfigObjError
import os
import sys
import numpy as np
import xarray as xr
from typing import Union, Iterable, Tuple

import logging
# Logging options
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
# Module logger
log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def normalized_difference(a: Union[int, float, np.array, xr.DataArray],
                          b: Union[int, float, np.array, xr.DataArray]) \
        -> Union[int, float, np.array, xr.DataArray]:
    """
    Calculate the normalized difference between `a` and `b`.

    .. math:: \frac{a - b}{a + b}

    Parameters
    ----------
    a : int, float, np.array, xr.DataArray
        Minuend.
    b : int, float, np.array, xr.DataArray
        Subtrahend.

    Returns
    -------
    norm_diff: same as input
        Normalized difference between a and b.
    """
    norm_diff = (a - b) / (a + b)
    return norm_diff


def rescale(data: Union[float, np.array, xr.DataArray],
            thresholds: Tuple[float, float]) -> \
        Union[float, np.array, xr.DataArray]:
    """
    Rescale some data to given thresholds, so that data ranges now from 0 to 1.

    Parameters
    ----------
    data : float or np.array or xr.DataArray
        The data to rescale.
    thresholds : tuple
        Tuple containing the min/max value to which to rescale the data.

    Returns
    -------
    same as `data`
        The same data structure as `data`but the values rescaled to range from
        zero to one between the `thresholds`.
    """
    return (data - (thresholds[0])) / (thresholds[1] - thresholds[0])

def get_credentials(credfile: str = None) -> ConfigObj:
    """
    Get credentials to use SentinelHub login.

    Parameters
    ----------
    credfile : str or None
        Credentials file in configObj style. Default: None (take the one under
        the top level directory with the name ".credentials".

    Returns
    -------
    cr: configobj.ConfigObj
        Configuration object with the credentials.
    """
    if credfile is None:
        credfile = os.path.join(os.path.abspath(os.path.dirname(
            os.path.dirname(__file__))), '.credentials')

    try:
        cr = ConfigObj(credfile, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Credentials file could not be parsed (%s): %s',
                     credfile, e)
        sys.exit()

    return cr

