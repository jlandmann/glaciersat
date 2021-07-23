import glaciersat.cfg as cfg
from sentinelsat import SentinelAPI
from configobj import ConfigObj, ConfigObjError
import zipfile
import shapely
import salem
import pandas as pd
import os
import glob
import sys
import numpy as np
from scipy import ndimage
import xarray as xr
from xml.etree import ElementTree
from functools import wraps
from collections import OrderedDict
import time
from typing import Union, Iterable, Tuple, List, Optional

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


def setup_sentinel_api():
    """
    Set up a Sentinel API to interface with Sentinel data on Copernicus Hub.

    Returns
    -------
    api: sentinelsat.SentinelAPI
        An interface.
    """
    cr = get_credentials(os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        '.credentials'))

    api = SentinelAPI(cr['sentinel']['user'], cr['sentinel']['password'],
        api_url="https://apihub.copernicus.eu/apihub/")

    return api


def download_sentinel_tiles(
        date_begin: pd.Timestamp, date_end: pd.Timestamp, tiles: list,
        platform: str = 'Sentinel-2', producttype: str = 'auto',
        cloudcov_min_pct: Optional[float] = None,
        cloudcov_max_pct: Optional[float] = None,
        download_base_dir: Optional[str] = None) -> tuple[
    OrderedDict, dict, dict, dict]:
    """
    Download some Sentinel tiles as specified by tiles and a time span.

    Parameters
    ----------
    date_begin : pd.Timestamp
        Begin date of the time span for which the tiles shall be downloaded.
    date_end : pd.Timestamp
        End date of the time span for which the tiles shall be downloaded.
    tiles : list
        List with tiles names as strings, e.g. ['32TMS', '32TLR'].
    platform : str
        Which platform shall be downloaded. Default: 'Sentinel-2' (optical
        imagery).
    producttype : str
        Which product type shall be downloaded, i.e. which sensor and
        processing level. Allowed are 'S2MSI1C', 'S2MSI2A' and 'auto'. If
        'auto', then the download choice is made based on the availability.
        After 2018-03-31 level 2A data (bottom fo atmosphere reflectance) will
        be downloaded, while before level 1C data (top of atmosphere
        reflectance) will be downloaded. Default: 'auto'.
    cloudcov_min_pct : float
        Minimum allowed cloud cover percentage. Default: None (no limit;
        recommended).
    cloudcov_max_pct : float
        Maximum allowed cloud cover percentage. Default: None (no limit).
    download_base_dir : str
        Top-level download directory for the data.

    Returns
    -------
    products, downloaded, triggered, failed: OrderedDict, list, list, list
    """

    if cloudcov_min_pct is None:
        cloudcov_min_pct = cfg.PARAMS['cloudcover_range'][0]
    if cloudcov_max_pct is None:
        cloudcov_max_pct = cfg.PARAMS['cloudcover_range'][1]
    if download_base_dir is None:
        download_base_dir = cfg.PATHS['sentinel_download_path']

    # first thing to do
    api = setup_sentinel_api()

    # L2A starts only later - we try to get the best possible
    if producttype == 'auto':
        # date when ESA started supplying L2A
        l2a_begin_date = pd.Timestamp('2018-04-01')
        if (date_begin < l2a_begin_date) and (date_end < l2a_begin_date):
            producttype = 'S2MSI1C'
        elif (date_begin >= l2a_begin_date) and (date_end >= l2a_begin_date):
            producttype = 'S2MSI2A'
        elif (date_begin < l2a_begin_date) and (date_end >= l2a_begin_date):
            # call two times
            download_sentinel_tiles(
                date_begin, l2a_begin_date - pd.Timedelta(days=1), tiles=tiles,
                platform=platform, producttype='S2MSI1C')
            download_sentinel_tiles(
                date_begin, l2a_begin_date, tiles=tiles, platform=platform,
                producttype='S2MSI2A')

    query_kwargs = {'platformname': platform, 'producttype': producttype,
                    'date': (date_begin, date_end),
                    'cloudcoverpercentage': "[{} TO {}]".format(
                                cloudcov_min_pct,
                                cloudcov_max_pct)
                                }

    products = OrderedDict()
    for tile in tiles:
        kw = query_kwargs.copy()
        #kw['tileid'] = tile  # products after 2017-03-31
        # search by tiles for S2A needs this line as well:
        kw['filename'] = f'*_T{tile}_*'
        pp = api.query(**kw)
        products.update(pp)
    log.info('{} products found.'.format(len(products)))

    downloaded = {}
    triggered = {}
    failed = {}
    if len(products) == 0:
        return products, downloaded, triggered, failed

    @retry(Exception, tries=100, delay=3600)
    def dl_all(*args, **kwargs):
        log.info('Trying to download {} products...'.format(products))
        d, t, f = api.download_all(*args, **kwargs)
        log.info('{} downloaded, {} triggered, {} failed.'.format(
            len(d), len(t), len(f)))
        return d, t, f

    # create download path if it does not yet exist
    platform_short = (platform[0] + platform[-1]).lower()
    dl_dir = os.path.join(download_base_dir, platform_short,
        producttype[-2:].lower())
    if not os.path.exists(dl_dir):
        os.makedirs(dl_dir)

    downloaded, triggered, failed = dl_all(
        products, directory_path=dl_dir)
    downloaded.update(downloaded)
    failed.update(failed)

    return products, downloaded, triggered, failed



def unzip_sentinel(path: Optional[str] = None, remove_zip: bool = True):
    """
    Unzip some downloaded Sentinel zipped files.

    This function either takes the path to a file or to an entire directory.
    Raises warnings, when "bad zip files" are encountered, which was once a
    server side issue.
    
    Parameters
    ----------
    path : str, optional
        File path or directory that shall be unzipped. If a directory is given,
        the entire path with all (!) subdirectories are searched for zip file
        and all of them are unzipped. Default: None (take
        cfg.PATHS['sentinel_download_path'].
    remove_zip : bool
        Whether or not the zipped file shall be removed after extraction.
        Default: True.

    Returns
    -------
    None
    """
    if path is None:
        path = cfg.PATHS['sentinel_download_path']

    if os.path.isdir(path):
        zippaths = glob.glob(os.path.join(path, '**', '*.zip'), recursive=True)
    elif os.path.isfile(path):
        zippaths = [path]
    else:
        raise ValueError('`Path` has to be either a directory containing zip '
                         'files or a zip file itself.')

    bad_zips = []
    for zp in zippaths:
        dirname = os.path.dirname(zp)
        try:
            with zipfile.ZipFile(zp) as zip_file:
                log.info('Extracting {} ...'.format(os.path.basename(zp)))
                zip_file.extractall(dirname)
            if remove_zip is True:
                os.remove(zp)
        except zipfile.BadZipfile:
            log.warning('Bad zip File: {}'.format(os.path.basename(zp)))
            bad_zips.append(zp)
            continue

    if len(bad_zips) > 0.:
        # should be resolved, but who knows:
        # https://scihub.copernicus.eu/news/News00893
        log.warning('Found bad zip files: {}'.format(bad_zips))
