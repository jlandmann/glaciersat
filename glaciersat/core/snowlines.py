import xarray as xr
from skimage import filters
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
# Module logger
log = logging.getLogger(__name__)


def map_snow_asmag(ds: xr.Dataset, date: pd.Timestamp or None = None,
                   nir_bandname: str = 'B08',
                   roi_shp: str or gpd.GeoDataFrame or None = None) -> \
        xr.DataArray or None:
    """
    Map the snow on a satellite image according to Rastner et al. (2019) [1]_.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing an optical satellite image. Needs to have a `band`
        coordinate which has a value `nir_bandname`.
    date: pd.Timestamp
        Date for which to map the snow. Default: None.
    nir_bandname: str
        Band name for near infrared reflectance. Default: 'B08' (as for
        Sentinel-2).
    roi_shp: str or gpd.GeoDataFrame or None
        Path to a shapefile or geopandas.GeodataFrame that defines the region
        of interest on the dataset. Default: None (use all values in the
        dataset to get the histogram for).


    Returns
    -------
    snow_da: xr.DataArray or None
        Boolean DataArray indicating whether there is snow or not on the
        glacier (1=snow, 0=ice). If no NIR channel is available, None is
        returned.

    References
    ----------
    .. [1] Rastner, P.; Prinz, R.; Notarnicola, C.; Nicholson, L.; Sailer, R.;
        Schwaizer, G. & Paul, F.: On the Automated Mapping of Snow Cover on
        Glaciers and Calculation of Snow Line Altitudes from Multi-Temporal
        Landsat Data. Remote Sensing, Multidisciplinary Digital Publishing
        Institute, 2019, 11, 1410.
    """

    if date is not None:
        try:
            ds = ds.sel(time=date)
        except (ValueError, KeyError):  # dimension or date not present
            pass

    if roi_shp is not None:
        ds = ds.salem.roi(shape=roi_shp)
    n_tot_pix = np.sum(~np.isnan(ds.bands)) / len(ds.bands)

    # todo: is it possible to make this a probabilistic snow map?
    try:
        nir = ds.bands.sel(band=nir_bandname).values / 10000.
    except KeyError:  # no sat image at dates
        log.error('NIR channel not available.')
        return None

    if 'cmask' in ds.data_vars:
        cmask = ds.cmask.values.copy()
        # todo: replace threshold 0. with cfg.PARAMS['cloud_prob_thresh']
        cmask[cmask > 0.] = np.nan
        cmask[cmask <= 0.] = 1.
        assert cmask.ndim <= 2
        nir *= cmask
    else:
        log.warning('No cloud mask information given. Still proceeding and '
                    'pretending a cloud-free scene...')

    # make sure we only work on 1 time step, otherwise next cond. doesn't work
    assert nir.ndim <= 2

    # if too much cloud cover, don't analyze at all
    cloud_cov_ratio = 1 - (np.sum(~np.isnan(nir)) / n_tot_pix)
    # todo: if cloud_cov_ratio <= cfg.PARAMS['max_cloud_cover_ratio']:
    print(cloud_cov_ratio)
    if cloud_cov_ratio <= 0.2:
        val = filters.threshold_otsu(nir[~np.isnan(nir)])
        snow = nir > val
        snow = snow * 1.
        snow[np.isnan(nir)] = np.nan
    else:
        log.error('Masked pixed ratio of the glacier is too high to analyze '
                  'snow cover.')
        snow = np.full_like(nir, np.nan)

    snow_da = xr.DataArray(data=snow, coords={'y': ds.coords['y'],
                                              'x': ds.coords['x']},
                           dims=['y', 'x'])

    return snow_da


