import glaciersat.cfg as cfg
from sentinelsat import SentinelAPI
from configobj import ConfigObj, ConfigObjError
import zipfile
import shapely
import salem
import pandas as pd
import os
import sys
import numpy as np
from scipy import ndimage
import xarray as xr
from xml.etree import ElementTree
from functools import wraps
from collections import OrderedDict
import time
from typing import Union, Iterable, Tuple, List

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
            thresholds: Union[List[float], Tuple[float, float]]) -> \
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


def declutter(data: np.ndarray, n_erode: int, n_dilate: int) -> np.ndarray:
    """
    Remove clutter from data.

    The data should be structured such that all values equal to one. If the
    value for `n_erode` or `n_dilate` is 1, then nothign is changed.

    Parameters
    ----------
    data : np.ndarray
         Array with data containing zeros and ones.
    n_erode : int
         How many pixels shall be eroded, i.e. which is the minimum square side
         length of features that shall survive?
    n_dilate : int
         After erosion, how much shall the remaining features be dilated again?

    Returns
    -------
    decluttered: np.ndarray
        Input array, but with small features removed.
    """

    decluttered = ndimage.binary_dilation(
        ndimage.binary_erosion(data, np.ones((n_erode, n_erode))),
        np.ones((n_dilate, n_dilate))
                                     )
    return decluttered


def retry(exceptions: Union[str, Iterable, tuple, Exception], tries=100,
          delay=60, backoff=1, log_to=None):
    """
    Retry decorator calling the decorated function with an exponential backoff.

    Amended from Python wiki [1]_ and calazan.com [2]_.

    Parameters
    ----------
    exceptions: str or tuple or Exception
        The exception to check. May be a tuple of exceptions to check. If just
        `Exception` is provided, it will retry after any Exception.
    tries: int
        Number of times to try (not retry) before giving up. Default: 100.
    delay: int or float
        Initial delay between retries in seconds. Default: 60.
    backoff: int or float
        Backoff multiplier (e.g. value of 2 will double the delay
        each retry). Default: 1 (no increase).
    log_to: logging.logger
        Logger to use. If None, print.

    References
    -------
    .. [1] https://bit.ly/2NMpF2j
    .. [2] https://bit.ly/3e6FIT9
    """
    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    msg = '{}, Retrying in {} seconds...'.format(e, mdelay)
                    if log_to:
                        log_to.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry


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


def read_xml(filename: str):
    """
    Read a *.XML file at the basic level.

    This is useful e.g. for the Sentinel-2 metadata, which are stored as an
    XML file.

    Parameters
    ----------
    filename : str
        Path to the *.XML file.

    Returns
    -------
    xml.etree.ElementTree:
        An ElementTree object containing the parsed xml data.
    """
    return ElementTree.parse(filename)
