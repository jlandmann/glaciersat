import glaciersat.cfg as cfg
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
        cprob_thresh = cfg.PARAMS['cloud_prob_thresh']
        cmask[cmask > cprob_thresh] = np.nan
        cmask[cmask <= cprob_thresh] = 1.
        assert cmask.ndim <= 2
        nir *= cmask
    else:
        log.warning('No cloud mask information given. Still proceeding and '
                    'pretending a cloud-free scene...')

    # make sure we only work on 1 time step, otherwise next cond. doesn't work
    assert nir.ndim <= 2

    # if too much cloud cover, don't analyze at all
    cloud_cov_ratio = 1 - (np.sum(~np.isnan(nir)) / n_tot_pix)
    if cloud_cov_ratio <= cfg.PARAMS['max_cloud_cover_ratio']:
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


def map_snow_naegeli(ds: xr.Dataset, dem: str or xr.Dataset,
                     date: pd.Timestamp or None = None,
                     roi_shp: str or gpd.GeoDataFrame or None = None,
                     r_crit: float or None = None,) -> \
        xr.DataArray:
    """
    Map snow using the algorithm described in Naegeli et al. (2019) [1]_.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing an optical satellite image. Needs to have an
        `albedo` variable.
    dem: str or xr.Dataset
        Dataset or path to dataset containing a digital elevation model. Must
        be on the same grid as `ds` and have a variable that is called
        `height`.
    date: pd.Timestamp
        Date for which to map the snow. Default: None.
    roi_shp: str or gpd.GeoDataFrame or None
        Path to a shapefile or geopandas.GeodataFrame that defines the region
        of interest on the dataset. Default: None (use all values in the
        dataset to get the histogram for).
    r_crit : float or None
        Critical radius as defined in [1]_. Default: None (retrieve from
        params.cfg).

    Returns
    -------
    snow_da: xr.DataArray
        Boolean DataArray indicating whether there is snow or not on the
        glacier (1=snow, 0=ice).

    References
    ----------
    .. [1] Naegeli, K.; Huss, M. & Hoelzle, M.: Change detection of bare-ice
        albedo in the Swiss Alps. The Cryosphere, 2019, 13, 397-412.
    """
    if isinstance(dem, str):
        dem = xr.open_dataset(dem)

    if date is not None:
        dem = dem.sel(time=date, method="nearest").height.values
    else:
        # take the latest DEM
        dem = dem.isel(time=-1).height.values

    # todo: select roi here

    albedo = ds.albedo
    out_ds = albedo.copy(deep=True)
    out_ds = out_ds.rename('snow')

    # primary surface type evaluation: 1=snow, 0=ice, 0.5=ambiguous
    # nsat = cfg.PARAMS['naegeli_snow_alpha_thresh']
    # niat = cfg.PARAMS['naegeli_ice_alpha_thresh']
    # todo: replace with cfg values
    out_ds = out_ds.where(out_ds <= 0.55, 1.)  # snow
    out_ds = out_ds.where(out_ds >= 0.2, 0.)  # ice
    out_ds = out_ds.where((out_ds < 0.2) | (out_ds > 0.55), 0.5)  # ambiguous

    # only proceed if ambiguous area contains any True values
    if (out_ds == 0.5).any():

        # construct Dataset wil DEM heights filled where area is ambiguous
        dem_amb = out_ds.copy(deep=True)
        dem_amb = dem_amb.where(dem_amb == 0.5)
        dem_amb = dem_amb.where(dem_amb != 0.5, dem)

        albedo_amb = out_ds.where(out_ds == 0.5)
        albedo_amb = albedo_amb.where(albedo_amb != 0.5, albedo)

        # find location with maximum albedo slope
    # todo: sla cannot yet take many values
        alpha_crit, sla = _find_max_albedo_slope_naegeli(albedo_amb, dem_amb)

        # assign surface type based on critical albedo
        out_ds = out_ds.where(albedo < alpha_crit, 1.)
        out_ds = out_ds.where(albedo > alpha_crit, 0.)

        # assign surface type based on critical radius ("outlier suppression")
        # NaN in DEM is excluded (can be at Swiss border, where DEM ends)
        out_ds = out_ds.where(
            (out_ds != 0.5) & (dem < (sla + r_crit)) | np.isnan(dem), 1.)
        out_ds = out_ds.where(
            (out_ds != 0.5) & (dem > (sla - r_crit)) | np.isnan(dem), 0.)

    return out_ds


def _find_max_albedo_slope_naegeli(alpha_amb: xr.Dataset, dem_amb: xr.Dataset,
                                   bin_width: float or None = None,
                                   dim='method') -> tuple:
    """
    Find the maximum albedo elevation gradient in an ambiguous area.

    Parameters
    ----------
    alpha_amb : xr.Dataset
        Dataset containing ambiguous albedo values.
    dem_amb : xr.Dataset
        Dataset containing elevation values in the ambiguous albedo range.
    bin_width : float or None
        Elevation bin width (m) used for calculating the slope. Default: None
        (retrieve from params.cfg).

    Returns
    -------
    alb_max_slope, height_max_slope: tuple of xr.Dataset
        Albedo value at the maximum albedo slope, height at the maximum slope.
    """

    if bin_width is None:
        bin_width = cfg.PARAMS['bin_width']

    # 1) reclassify elevation into bins
    # todo: now we take the overall mean, it should probably done by method
    # dem_min = dem_amb.min(dim=['x', 'y']).astype(int)
    # dem_max = dem_amb.max(dim=['x', 'y']).astype(int)
    dem_min = min(dem_amb.min(dim=['x', 'y']).astype(int))
    dem_max = max(dem_amb.max(dim=['x', 'y']).astype(int))
    bins = np.arange(dem_min, dem_max, bin_width)
    alpha_amb.name = 'alpha'
    dem_amb.name = 'height'
    merged = xr.merge([alpha_amb, dem_amb])
    alpha_amb_binned = merged.groupby_bins('height', bins).mean()

    # todo: Kathrins algorithm doesn't work properly: think of this as improvement:
    #  alpha_amb_binned.alpha.rolling(height_bins=10, center=True).mean()

    # 2) get albedo gradient based on bins
    alpha_amb_binned = alpha_amb_binned.set_coords('height')

    # 3) find maximum gradient index
    max_ix = alpha_amb_binned.differentiate('height').argmax().alpha.item()

    # 4) index albedo and heights with maximum gradient index
    sla = alpha_amb_binned.height[max_ix].item()
    alpha_crit = alpha_amb_binned.alpha[max_ix].item()

    # todo: take care of glaciers with less than 20m elevation range
    # todo: take care of fully snow covered or fully snowfree glacier!?

    return alpha_crit, sla
