from typing import Optional, Union, Iterable, Sized
from glaciersat import utils, cfg
from glaciersat.core import imagery
import xarray as xr
from skimage import filters
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.optimize import least_squares
from functools import partial

# Module logger
log = logging.getLogger(__name__)


def map_snow_linear_unmixing(
        ds: xr.Dataset,
        endmembers: xr.Dataset,
        date: Optional[pd.Timestamp] = None,
        roi_shp: Optional[Union[str, gpd.GeoDataFrame]] = None,
        cloud_mask: Optional[Union[xr.DataArray, np.ndarray]] = None) \
        -> xr.Dataset:
    """
    Map snow on the glacier by linear spectral unmixing.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing an optical satellite image. The image should be
        given as reflectance, i.e. values between zero and one.
    endmembers: xr.Dataset
        Dataset containing the spectral signatures of the endmembers that shall
        be evaluated. E.g. a Dataset could contain two variables `snow`and
        `ice` along the 13 Sentinel-2 channels. It must be reflectances, i.e.
        values (mainly) between zero and one.
    date: pd.Timestamp or None
        Date for which to map the snow. Default: None.
    roi_shp: str or gpd.GeoDataFrame or None
        Path to a shapefile or geopandas.GeoDataFrame that defines the region
        of interest on the dataset. Default: None (use all values in the
        dataset to get the histogram for).
    cloud_mask: xr.DataArray or np.ndarray, optional
        Cloud maks for the scene (saves some calculation time, because cloudy
        areas don't need an evaluation).

    Returns
    -------
    member_probs_ds: xr.Dataset
        Dataset with same spatial dimensions as input `ds`, but variables with
        the names of the `endmembers`, indicating for each pixels the
        probability (0-1) to belong to the endmembers.
    """

    if date is not None:
        ds = ds.sel(time=date)

    if isinstance(roi_shp, str):
        roi_shp = salem.read_shapefile(roi_shp)

    bands = ds.bands.values

    # band 10 is missing in L2A - then linalg fails
    present_bands = \
    np.where([~np.isnan(ds.bands[b]).all() for b in range(len(ds.band))])[0]
    bands = bands[present_bands, ...]

    # make a band-aware choice
    endmember_labels = np.array([k for k, v in endmembers.sel(
        band=ds.band[present_bands].values).data_vars.items()])
    endmembers = np.vstack([v.values for k, v in endmembers.sel(
        band=ds.band[present_bands].values).data_vars.items()])

    # create final array - empty
    member_probs = np.full((endmembers.shape[0], len(ds.y), len(ds.x)), np.nan)

    # see if we have a cloud mask
    if cloud_mask is not None:
        if isinstance(cloud_mask, xr.DataArray):
            cm_roi = csmask_ds.salem.roi(shape=ol).cmask.values
        else:
            cm_roi = cloud_mask.copy()
    else:
        # try to get cloud mask from dataset with image
        if 'cmask' in ds.variables:
            cm_roi = ds.salem.roi(shape=ol).cmask.values
        else:
            log.info('No cloud mask found for spectral unmixing. Assuming no '
                     'clouds...')
            cm_roi = np.zeros_like(ds.bands.isel(band=0))  # random band

    # save calculation cost when iterating
    ds = ds.salem.roi(shape=roi_shp)
    valid = np.where((~np.isnan(ds.bands[0])) & (cm_roi != 1.))

    # optimize in log space to obtain only positive coefficients
    end_const = np.log(endmembers.T)
    print('{} pixels to analyze.'.format(len(valid[0])))
    for pix in range(len(valid[0])):
        pi, pj = valid[0][pix], valid[1][pix]
        ref_const = np.log(bands[:, pi, pj])
        r1 = np.linalg.lstsq(end_const, ref_const, rcond=-1)[0]

        member_probs[:, pi, pj] = r1

    member_probs = np.exp(member_probs)
    assert (member_probs[~np.isnan(member_probs)] >= 0.).all()

    # normalize
    member_probs /= np.sum(member_probs, axis=0)

    member_probs_ds = xr.Dataset(dict(
        [(k, (['y', 'x', 'map_method'], v[..., None])) for k, v in
         zip(endmember_labels, member_probs)]),
                                 coords={'y': ds.y.values, 'x': ds.x.values,
                                         'map_method': ['lsu']})

    return member_probs_ds


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
        except ValueError:  # dimension not present
            pass
    else:  # if only one time, we can safely select it
        if ('time' in ds.coords) and (len(ds.coords['time']) == 1):
            ds = ds.isel(time=0)
        elif ('time' in ds.coords) and (len(ds.coords['time']) != 1):
            raise ValueError('A valid time step must be selected when working '
                             'with multitemporal data.')
        else:  # time not in coords
            pass

    if roi_shp is not None:
        ds = ds.salem.roi(shape=roi_shp)

    # todo: is it possible to make this a probabilistic snow map?
    try:
        # todo: check if really a value 0-1 is needed, or if it doesn't matter
        nir = ds.bands.sel(band=nir_bandname).values
    except KeyError:  # no sat image at dates
        log.error('NIR channel not available.')
        return None

    if 'cmask' in ds.data_vars:
        cmask = ds.cmask.values.copy()
        # cmask already masked to glacier ROI
        cloud_cov_ratio = np.nansum(ds.cmask.values) / np.sum(
            ~np.isnan(ds.cmask.values))
        print('CLOUDCOV_RATIO: ', cloud_cov_ratio)
        cprob_thresh = cfg.PARAMS['cloud_prob_thresh']
        cmask[cmask > cprob_thresh] = np.nan
        cmask[cmask <= cprob_thresh] = 1.
        assert cmask.ndim <= 2
        nir *= cmask
    else:
        cloud_cov_ratio = 0.
        log.warning('No cloud mask information given. Still proceeding and '
                    'pretending a cloud-free scene...')

    # make sure we only work on 1 time step, otherwise next cond. doesn't work
    assert nir.ndim <= 2

    n_valid_pix = np.sum(~np.isnan(nir))

    # if too much cloud cover, don't analyze at all
    if (cloud_cov_ratio > cfg.PARAMS['max_cloud_cover_ratio']) or (
            n_valid_pix == 0.):
        log.error('Masked pixel ratio {:.2f} is higher than the chosen '
                  'threshold max_cloud_cover_ratio or glacier contains only '
                  'NaN.'.format(cloud_cov_ratio))
        snow = np.full_like(nir, np.nan)
    else:
        val = filters.threshold_otsu(nir[~np.isnan(nir)])
        snow = nir > val
        snow = snow * 1.
        snow[np.isnan(nir)] = np.nan

    snow_da = xr.DataArray(data=snow,
                           coords={'y': ds.coords['y'], 'x': ds.coords['x']},
                           dims=['y', 'x'])

    if date is not None:
        snow_da = snow_da.expand_dims(dim='time')
        snow_da = snow_da.assign_coords(time=(['time'], [date]))

    snow_da = snow_da.expand_dims(dim='map_method')
    snow_da = snow_da.assign_coords(map_method=(['map_method'], ['asmag']))

    return snow_da


def get_snow_line_altitude_asmag(ds, date, dem, bin_height=20., n_bins=5):
    """
    Get snow line using the ASMAG method from Rastner et al. (2019) [1]_.

    The method evaluates a blue band histogram and separates it into two
    parts using the Otsu threshold described in [2]_.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset containing a satellite image. It should be given as
        reflectances, i.e. ranging from zero to one.
        todo: Needs a NIR channel named 'B08' at the moment.
    date: pd.Timestamp
        Date for which to process the snow line.
    dem: xr.Dataset
        An elevation error_func to base the snow line search on.
    bin_height: float
        Height of one elevation bin used to look for the snow line altitude
        (m). Default: 20.
    n_bins: int
        Number of bins used to analyze where the snow line altitude is.
        Default: 5.

    Returns
    -------


    References
    ----------
    .. [1] Rastner, P.; Prinz, R.; Notarnicola, C.; Nicholson, L.; Sailer, R.;
        Schwaizer, G. & Paul, F.: On the Automated Mapping of Snow Cover on
        Glaciers and Calculation of Snow Line Altitudes from Multi-Temporal
        Landsat Data. Remote Sensing, Multidisciplinary Digital Publishing
        Institute, 2019, 11, 1410.
    .. [2] Otsu, N.: A threshold selection method from gray-level
        histograms. IEEE transactions on systems, man, and cybernetics, IEEE,
        1979, 9, 62-66.
    """

    # 1) map the snow, if not yet present
    snowmap = map_snow_asmag(ds, date)

    # 2) derive the snow line

    try:
        # Get DEM values
        elevation_grid = dem.height.values
        # Convert DEM to 20 Meter elevation bands:
        cover = []
        for num, height in enumerate(
                np.arange(int(elevation_grid[elevation_grid > 0].min()),
                          int(elevation_grid.max()), bin_height)):
            if num > 0:
                # starting at second iteration:
                while snowmap.shape != elevation_grid.shape:
                    if elevation_grid.shape[0] > snowmap.shape[0] or \
                            elevation_grid.shape[1] > snowmap.shape[
                        1]:  # Shorten elevation grid
                        elevation_grid = elevation_grid[0:snowmap.shape[0],
                                         0:snowmap.shape[1]]
                    if elevation_grid.shape[0] < snowmap.shape[
                        0]:  # Extend elevation grid: append row:
                        try:
                            elevation_grid = np.append(elevation_grid, [
                                elevation_grid[
                                (elevation_grid.shape[0] - snowmap.shape[0]),
                                :]], axis=0)
                        except IndexError:
                            raise  # BUG: very exeptionally, the snow_map is broken -->  # log.error('Snow map is broken - BUG!')  # return
                    if elevation_grid.shape[1] < snowmap.shape[
                        1]:  # append column
                        b = elevation_grid[:,
                            (elevation_grid.shape[1] - snowmap.shape[1]):
                            elevation_grid.shape[1]]
                        elevation_grid = np.hstack((elevation_grid,
                                                    b))  # Expand grid on boundaries to obtain raster in same shape after

                # find all pixels with same elevation between "height" and "height-20":
                while band_height > 0:
                    try:
                        snow_band = snowmap[
                            (elevation_grid > (height - band_height)) & (
                                    elevation_grid < height)]
                    except IndexError:
                        log.error(' Index Error:', elevation_grid.shape,
                                  snowmap.shape)
                    if snow_band.size == 0:
                        band_height -= 1
                    else:
                        break
                # Snow cover on 20 m elevation band:
                if snow_band.size == 0:
                    cover.append(0)
                else:
                    cover.append(
                        snow_band[snow_band == 1].size / snow_band.size)

        num = 0
        if any(loc_cover > 0.5 for loc_cover in cover):
            while num < len(cover):
                # check if there are 5 continuous bands with snow cover > 50%
                if all(bins > 0.5 for bins in cover[num:(num + bands)]):
                    # select lowest band as SLA
                    sla = range(int(elevation_grid[elevation_grid > 0].min()),
                                int(elevation_grid.max()), 20)[num]
                    break  # stop loop
                if num == (len(cover) - bands - 1):
                    # if end of glacier is reached and no SLA found:
                    bands = bands - 1
                    # start search again
                    num = -1
                if len(cover) <= bands:
                    bands = bands - 1
                    num = -1
                num += 1
        else:
            sla = elevation_grid.max()
        dem_ts.close()
        return sla
    except:
        return


def map_snow_naegeli(ds: xr.Dataset, dem: str or xr.Dataset,
                     date: pd.Timestamp or None = None,
                     roi_shp: str or gpd.GeoDataFrame or None = None,
                     alternate: bool = False, r_crit: float or None = None,
                     optimize_r_crit: bool = False) -> xr.DataArray:
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
    date: pd.Timestamp or None, optional
        Date for which to map the snow. Default: None.
    roi_shp: str or gpd.GeoDataFrame or None, optional
        Path to a shapefile or geopandas.GeoDataFrame that defines the region
        of interest on the dataset. Default: None (use all values in the
        dataset to perform the geometrical snow mapping).
    alternate: bool, optional
        If the alternate algorithm to find the critical albedo
        :math:`\alpha_{crit}` shall be applied (described in [2]_). This
        algorithm uses an iterative approach to avoid erroneous detections of
        :math:`\alpha_{crit}` due to noisy albedo data.
    r_crit : float or None, optional
        Critical radius as defined in [1]_. Default: None (retrieve from
        params.cfg).
    optimize_r_crit: bool, optional
        Whether to optimize :math:`r_{crit}` with respect to the conditions on
        the glacier. A Heaviside function is fitted to the elevation profile of
        albedo and its goodness of fit is used to determine the actual
        :math:`r_{crit}` used then:

        .. math:: r_{crit} = \max((1 - R^2) \cdot r_{crit, max}, r_{crit, max})

        where :math:`R^2` is the goodness of fit of the Heaviside function to
        the elevation-albedo profile, and :math:`r_{crit, max}` is the maximum
        possible critical radius. The maximum possible critical radius is
        defined as the maximum distance of the snow line altitude found in the
        process to either bottom or top of the glacier. Default: False (do not
        optimize, but take the value for :math:`r_{crit}` from the
        configuration).

    Returns
    -------
    snow_da: xr.DataArray
        Boolean DataArray indicating whether there is snow or not on the
        glacier (1=snow, 0=ice).

    References
    ----------
    .. [1] Naegeli, K.; Huss, M. & Hoelzle, M.: Change detection of bare-ice
        albedo in the Swiss Alps. The Cryosphere, 2019, 13, 397-412.
    .. [2] Geibel, L.: "SnowIceSen": An Automated Tool to Map Snow and Ice on
        Glaciers with Sentinel-2. Master's Thesis at ETH Zurich, supervised by
        Johannes Landmann and Daniel Farinotti.
    """

    if isinstance(dem, str):
        dem = xr.open_dataset(dem)

    if roi_shp is not None:
        # todo: DEM MUST BE MASKED TO GLACIER!!!! make roi_shape mandatory?
        dem = dem.salem.roi(shape=roi_shp)

    if date is not None:
        try:
            dem = dem.sel(time=date, method="nearest").height.values
        except ValueError:
            dem = dem.height.values
        try:
            ds = ds.sel(time=date)
        except ValueError:  # dimension not present
            pass
    else:
        # take the latest DEM
        try:
            dem = dem.isel(time=-1).height.values
        except ValueError:
            dem = dem.height.values

        if ('time' in ds.coords) and (len(ds.coords['time']) == 1):
            ds = ds.isel(time=0)
        elif ('time' in ds.coords) and (len(ds.coords['time']) != 1):
            raise ValueError('A valid time step must be selected when working '
                             'with multitemporal data.')
        else:  # time not in coords
            pass

    if roi_shp is not None:
        ds = ds.salem.roi(shape=roi_shp)

    albedo = ds.albedo
    if 'cmask' in ds.data_vars:
        cmask = ds.cmask.values.copy()
        # cmask already masked to glacier ROI
        cloud_cov_ratio = np.nansum(ds.cmask.values) / np.sum(
            ~np.isnan(ds.cmask.values))
        print('CLOUDCOV_RATIO: ', cloud_cov_ratio)
        cprob_thresh = cfg.PARAMS['cloud_prob_thresh']
        cmask[cmask > cprob_thresh] = np.nan
        cmask[cmask <= cprob_thresh] = 1.
    else:
        cmask = np.ones_like(albedo.isel(broadband=0))
        cloud_cov_ratio = 0.
        log.warning('No cloud mask information given. Still proceeding and '
                    'pretending a cloud-free scene...')

    albedo *= cmask
    n_valid_pix = np.sum(~np.isnan(albedo))

    out_ds = albedo.copy(deep=True)
    out_ds = out_ds.rename('snow')

    # if too much cloud cover, don't analyze at all
    if (cloud_cov_ratio > cfg.PARAMS['max_cloud_cover_ratio']) or (
            n_valid_pix == 0.):
        log.error('Masked pixel ratio {:.2f} is higher than the chosen '
                  'threshold max_cloud_cover_ratio or glacier contains only '
                  'NaN.'.format(cloud_cov_ratio))
        out_ds[:] = np.nan
        return out_ds

    # primary surface type evaluation: 1=snow, 0=ice, 0.5=ambiguous
    out_ds = primary_surface_type_evaluation(out_ds)

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
        if alternate is True:
            alpha_crit, sla = _find_max_albedo_slope_stepfit(albedo_amb,
                                                             dem_amb)
        else:
            alpha_crit, sla = _find_max_albedo_slope_naegeli(albedo_amb,
                                                             dem_amb)

        # assign surface type based on critical albedo
        out_ds = out_ds.where((albedo < alpha_crit) | np.isnan(albedo), 1.)
        out_ds = out_ds.where((albedo > alpha_crit) | np.isnan(albedo), 0.)

        if (optimize_r_crit is True) and (r_crit is None):
            r_crit_max = np.max(
                np.abs([sla - np.nanmin(dem), np.nanmax(dem) - sla]))

            min_ampl = 0.1  # still arbitrary
            max_ampl = cfg.PARAMS['naegeli_snow_alpha_thresh'] - cfg.PARAMS[
                'naegeli_ice_alpha_thresh']
            min_icpt = cfg.PARAMS['naegeli_ice_alpha_thresh']
            max_icpt = cfg.PARAMS['naegeli_snow_alpha_thresh']

            # stupid, but true: this is double work we wanted to avoid
            merged = xr.merge([albedo_amb, dem_amb])
            bw = cfg.PARAMS['bin_width']
            aab = merged.groupby_bins('height', np.arange(np.nanmin(dem_amb),
                                                          np.nanmax(dem_amb),
                                                          bw)).mean(
                skipna=True)

            # interpolate NaNs linearly (can happen with detached ambig. areas)
            aab = aab.interpolate_na(dim='height_bins', use_coordinate=False,
                fill_value='extrapolate', method='slinear')

            # edge interpolation issues
            aab['alpha'] = aab.alpha.where(aab.alpha >= min_icpt, min_icpt)
            aab['alpha'] = aab.alpha.where(aab.alpha <= max_icpt, max_icpt)

            alpha_in = aab.alpha.values
            height_in = aab.height.values
            r_squared = get_model_fit_r_squared(_root_sum_squared_residuals,
                _step_function, height_in, alpha_in,
                bounds=([min_ampl, min_icpt], [max_ampl, max_icpt]),
                x0=np.array([np.clip(np.nanmax(alpha_in) - np.nanmin(alpha_in),
                                     min_ampl, max_ampl),
                             np.clip(np.nanmin(alpha_in), min_icpt,
                                     max_icpt)]), b=sla)

            # in funny cases, r_squared can be negative
            r_squared = np.clip(r_squared, 0., None)

            # calculate flexible r_crit
            r_crit = np.min([(1 - r_squared) * r_crit_max, r_crit_max])
        elif (optimize_r_crit is False) and (r_crit is None):
            r_crit = cfg.PARAMS['r_crit']
        elif (optimize_r_crit is True) and (r_crit is not None):
            raise ValueError('Conflicting values for r_crit and '
                             'optimize_r_crit: r_crit cannot be optimized when'
                             ' at the same time a fixed values is given.')
        else:
            pass

        # assign surface type based on critical radius ("outlier suppression")
        # NaN in DEM is excluded (can be at Swiss border, where DEM ends)
        out_ds = out_ds.where((dem < (sla + r_crit)) | np.isnan(out_ds), 1.)
        out_ds = out_ds.where((dem > (sla - r_crit)) | np.isnan(out_ds), 0.)

    if date is not None:
        out_ds = out_ds.expand_dims(dim='time')
        out_ds = out_ds.assign_coords(time=(['time'], [date]))

    mmethod_name = 'naegeli_alt' if alternate is True else 'naegeli'
    out_ds = out_ds.expand_dims(dim='map_method')
    out_ds = out_ds.assign_coords(map_method=(['map_method'], [mmethod_name]))

    return out_ds


map_snow_naegeli_alternate = partial(map_snow_naegeli, alternate=True,
                                     optimize_r_crit=True, r_crit=None)


def primary_surface_type_evaluation(
        alpha_ds: xr.DataArray or xr.Dataset) -> xr.DataArray or xr.Dataset:
    """
    Do a primary surface type evaluation after Naegeli et al. (2019) [1]_.

    Based on given albedo thresholds, this algorithm assigns 0 to all pixels
    which are believed to be for certain ice, 1 to those believed to be for
    sure snow, and 0.5 in the so called 'ambiguous range'.

    Parameters
    ----------
    alpha_ds : xr. DataArray or xr.Dataset
        Xarray data structure containing broadband albedo.

    Returns
    -------
    alpha_ds: same as input
        Data structure containing classified albedo.

    References
    ----------
    .. [1] Naegeli, K.; Huss, M. & Hoelzle, M.: Change detection of bare-ice
        albedo in the Swiss Alps. The Cryosphere, 2019, 13, 397-412.
    """

    sat = cfg.PARAMS['naegeli_snow_alpha_thresh']
    iat = cfg.PARAMS['naegeli_ice_alpha_thresh']

    # we also preserve NaNs (they are in the masked region)
    # snow
    alpha_ds = alpha_ds.where((alpha_ds <= sat) | np.isnan(alpha_ds), 1.)
    # ice
    alpha_ds = alpha_ds.where((alpha_ds >= iat) | np.isnan(alpha_ds), 0.)
    # ambiguous
    alpha_ds = alpha_ds.where(
        (alpha_ds < iat) | (alpha_ds > sat) | np.isnan(alpha_ds), 0.5)

    return alpha_ds


def _find_max_albedo_slope_naegeli(alpha_amb: xr.Dataset, dem_amb: xr.Dataset,
                                   bin_width: float or None = None) -> tuple:
    """
    Find the maximum albedo elevation gradient in an ambiguous area.

    Parameters
    ----------
    alpha_amb : xr.Dataset
        Dataset containing ambiguous albedo values.
    dem_amb : xr.Dataset
        Dataset containing elevation values in the ambiguous albedo range.
    bin_width : float or None, optional
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

    # todo: Kathrins algorithm doesn't work properly: think of improvement:
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


def _find_max_albedo_slope_stepfit(alpha_amb: xr.Dataset, dem_amb: xr.Dataset,
                                   bin_width: float or None = None) -> tuple:
    """
    Find the maximum albedo elevation gradient in an ambiguous area.

    The snow line altitude (SLA) and critical albedo :math:`\alpha_{crit}` are
    essential variables in the snow mapping procedure by [1]_. However, it
    might occur that a wrong :math:`\alpha_{crit}` is found, since the highest
    albedo drop in 20m elevations bands is not necessarily always at the
    ice-snow transition.
    To detect to location of the ice-snow transition more safely, we therefore
    employ an iterative method that searches for the maximum slope in the
    albedo profile of the ambiguous elevation range. Initially, we make a guess
    that the highest slope is at the mean elevation of the ambiguous range,
    i.e. splitting up the elevation range into two bins. We then increase the
    number of elevation bins iteratively until we reach a given minimum bin
    width. At each iteration, a step function is fitted to the albedo-
    elevation profile, assuming that a location of the step at the snow
    and ice albedo transition delivers a best goodness of fit. Like this, the
    transition elevation evolves from the centers along neighboring elevation
    bands to its final position.

    Parameters
    ----------
    alpha_amb : xr.Dataset
        Dataset containing ambiguous albedo values.
    dem_amb : xr.Dataset
        Dataset containing elevation values in the ambiguous albedo range.
    bin_width : float or None
        Target elevation bin width (m) used for calculating the slope. Default:
        None (retrieve from params.cfg).

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
    amb_elev_ext = dem_max - dem_min

    # ambiguous range might be smaller than 2 x target bin width => take mean
    if amb_elev_ext <= (2 * bin_width):
        sla = dem_amb.mean(dim=['x', 'y'])
        alpha_crit = alpha_amb.mean(dim=['x', 'y'])
        return alpha_crit, sla

    alpha_amb.name = 'alpha'
    dem_amb.name = 'height'
    merged = xr.merge([alpha_amb, dem_amb])

    # we try to get close to bin_width - rather one time too many
    n_iterations = int(np.ceil((np.log2(amb_elev_ext / bin_width))))

    # first guess SLA and backup alpha_crit
    sla = np.mean([dem_min, dem_max])
    alpha_crit = None
    for i in range(1, n_iterations + 1):
        # 1) bin according to iteration number
        alpha_amb_binned = merged.groupby_bins('height', 2 ** i + 1).mean()

        # 2) get albedo gradient based on bins
        alpha_amb_binned = alpha_amb_binned.set_coords('height')

        # 3) find maximum gradient index
        alpha_slope = alpha_amb_binned.differentiate('height')
        search_ix = np.argmin(np.abs(alpha_amb_binned.height.values - sla))
        search_range = np.unique(
            np.clip(np.arange(search_ix - 2, search_ix + 3), 0,
                    len(alpha_slope.height_bins) - 1))
        try:
            argmax_ix = alpha_slope.isel(
                height_bins=search_range).alpha.argmax().item()
        except ValueError:  # all NaN slice encountered
            continue
        sla = alpha_slope.isel(height_bins=search_range).height.values[
            argmax_ix]
        alpha_crit = \
            alpha_amb_binned.isel(height_bins=search_range).alpha.values[
                argmax_ix]

    return alpha_crit, sla


def _step_function(x: np.array, a: float, c: float, b=0.) -> np.array:
    """
    Create a model for a step function.

    Parameters
    ----------
    x: np.array
        Domain where the step function shall be defined.
    a:  float
        Step amplitude.
    c:  float
        Additive term that determines the intercept, i.e. where the left arm of
        the step intersects the y axis.
    b: float, optional
        X value where the step shall be located. This is made a keyword
        argument in order to able to handle it as a fixed parameter, i.e.
        inserting e.g. a value found in another procedure.
        Default: 0.

    Returns
    -------
    np.array
        Model of step function.
    """
    return (0.5 * (np.sign(x - b) + 1)) * a + c


def _root_sum_squared_residuals(p, model_func, x, y, **kwargs):
    """
    Just a helper residual function as input for scipy.optimize.least_squares.

    This avoids the fact that scipy.optimize.curve_fit cannot take further
    arguments, which are not parameters: see https://bit.ly/3dJnhBI and
    https://bit.ly/2TbBhdZ. In our case, we might want to hand over a fixed
    value for the step position though, when `model_func` is a step function.

    Parameters
    ----------
    p : array_like
        Parameters to optimize.
    model_func: callable
        Function with call signature model_func(x, *params, **kwargs), where
        `x` is the domain, `*params` are the parameters to optimize, and
        `**kwargs` are the model parameters that should not be optimized.
    x : array_like
        Domain where step function shall be defined (sample x values).
    y : array_like
        Sample y values.
    **kwargs: dict, optional
        Optional non-optimized parameters passed on to `model_func`.

    Returns
    -------
    np.array:
        Root sum of squares of residuals.
    """

    model_y = model_func(x, *p, **kwargs)
    return np.sqrt(np.sum(((model_y - y) ** 2)))


def _arctan_function(x, a, b, x0, c):
    """
    Create a model for an arc tangent function.

    # todo: this function can be used to smooth the albedo_elevation profile
       and thus find a robust SLA estimate

    Parameters
    ----------
    x: np.array
        Domain where the step function shall be defined.
    a : float
        Amplitude stretch parameter for arc tangent.
    b : float
        Stretch parameter for arc tangent.
    x0 : float
        X shift parameter for arc tangent (location of saddle).
    c : float
        Y shift parameter for arc tangent.

    Returns
    -------
    np.array
        Model of arc tangent function.
    """
    return a * np.arctan(b * (x - x0)) + c


def get_model_fit_r_squared(error_func: callable, model_func: callable,
                            x: xr.DataArray or np.array,
                            y: xr.DataArray or np.array, bounds: tuple,
                            x0: tuple or None = None, **kwargs) -> float:
    """
    Retrieve :math:`R^2` value from fitting a model to a sample.

    Parameters
    ----------
    error_func: callable
         An error function with call signature error_func(*p, model_func, x, y,
         **kwargs), where `*p` are the model parameters, `model_func` is the
         model used for prediction, `x` are the sample x values, `y` are the
         sample y values, and `**kwargs` are further keyword arguments passed
         on to `model_func`.
    model_func: callable
        Function with call signature model_func(x, *params, **kwargs), where
        `x` is the domain values , `*params` are the parameters to be
        optimized, and `**kwargs` are other parameters of the model that should
        not be optimized.
    x: xr.DataArray or np.array
        X values of the samples to which the model shall be fitted.
    y: xr.DataArray or np.array
        Y values of the samples to which the model shall be fitted.
    bounds: 2-tuple of array_like
        Upper and lower optimization parameter boundaries
        ([a_min, b_min, c_min, ...], [a_max, b_max, c_max, ...]).
    x0: tuple or None, optional
        Initial parameter guesses for the optimization parameters of
        `model_func`.
    **kwargs: dict or None, optional
        Further keyword arguments passed on to `model_func`

    Returns
    -------
    r_squared: float
        :math:`R^2` value (goodness of fit) of fitted step function model to
        elevation-albedo profile.
    """

    res = least_squares(error_func, args=(model_func, x, y), kwargs=kwargs,
                        x0=x0, bounds=bounds)
    residuals = y - model_func(x, *res.x, **kwargs)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


def generate_endmembers_from_otsu(
        ds: xr.Dataset, shape: gpd.GeoDataFrame,
        cloudmask: Optional[xr.Dataset] = None,
        max_cloudcov: Optional[float] = None,
        summer_months_minmax: Optional[tuple] = (7, 9)) -> xr.Dataset:
    """
    Auto-generate snow/ice endmembers on summer scenes with Otsu thresholding.

    # todo: extend to several endmembers
    # todo: rule to better exclude "snow/ice only" glaciers (alpha thresholds?)

    Parameters
    ----------
    ds : xr.Dataset
        Satellite images (with `bands` variable) to work on. Must
        be reflectances, i.e. data value range 0-1.
    shape : gpd.GeoDataFrame
        Outline as region of interest.
    cloudmask:
        Cloud mask to apply: first, to classify the cloud endmembers, but also
        to exclude clouded areas on the region of interest as good as possible.
    max_cloudcov: float or None, optional
        Maximum allowed cloud cover on a scene (ratio between 0 and 1).
        If None, it is parsed from the configuration. Default: None.
    summer_months_minmax: tuple or None, optional
        Which months (number) shall be considered as the 'summer months', i.e.
        where the glacier should have a snow line? These are taken as the base
        for estimating the endmembers. Default: (7,9) (July-September).

    Returns
    -------
    endmembers: xr.Dataset
         Dataset containing the endmembers as variables along the `band`
         coordinate.
    """

    # we are strict: if no cloud mask available, the script will fail
    if cloudmask is None:
        if 'cmask' in ds.variables:
            cloudmask = ds.copy(deep=True)
        else:
            raise ValueError('Cloud mask must be supplied either with the '
                             'imagery or as separate argument (to prevent '
                             'false endmember detection on clouded areas).')
    if max_cloudcov is None:
        max_cloudcov = cfg.PARAMS['max_cloud_cover_ratio']
    summer_range = np.arange(summer_months_minmax[0],
                             summer_months_minmax[1] + 1)

    # select necessary items and convert
    image = ds.bands
    image = image.salem.roi(shape=shape)

    # empty result arrays
    ref_ice = np.full((len(image.band), len(image.time)), np.nan)
    ref_snow = np.full((len(image.band), len(image.time)), np.nan)
    ref_cloud = np.full((len(image.band), len(image.time)), np.nan)

    # we take only all "summer scenes" (they are likely to have a snow line)
    for it, t in enumerate(image.time.values):

        if pd.Timestamp(t).month not in summer_range:
            continue
        cm_roi = cloudmask.sel(time=t).salem.roi(shape=shape).cmask.values

        # Otsu threshold only possible per band - take the overhead
        for j, b in enumerate(image.band.values):
            current_band = image.sel(time=t, band=b).values
            current_band_masked = current_band[
                ~np.isnan(current_band) & (cm_roi == 0.)]
            if np.isnan(current_band_masked).all():
                continue
            current_band_clouds = current_band[
                ~np.isnan(current_band) & (cm_roi != 0.)]
            o_now = filters.threshold_otsu(current_band_masked)
            median_reflect_ice = np.nanmedian(
                current_band_masked[current_band_masked < o_now])
            median_reflect_snow = np.nanmedian(
                current_band_masked[current_band_masked > o_now])

            # snow and ice only when cloud cover is low
            if np.nanmean(cm_roi) < max_cloudcov:
                ref_ice[j, it] = median_reflect_ice
                ref_snow[j, it] = median_reflect_snow
            ref_cloud[j, it] = np.nanmedian(current_band_clouds)

    ref_snow = np.nanmedian(ref_snow, axis=1)
    ref_ice = np.nanmedian(ref_ice, axis=1)
    ref_cloud = np.nanmedian(ref_cloud, axis=1)
    endmembers = xr.Dataset({'snow': (['band'], ref_snow),
                             'ice': (['band'], ref_ice),
                             'clouds': (['band'], ref_cloud)},
                            coords={'band': ds.band.values})
    return endmembers


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import salem
    from matplotlib import colors, cm
    import warnings
    from glaciersat.core import albedo

    cfg.initialize()
    cloud_thresh = cfg.PARAMS['max_cloud_cover_ratio']

    glacier_id = 'RGI50-11.B5616n-1'

    #short_name = 'fin'
    shp_path = 'c:\\users\\johannes\\documents\\modelruns\\CH\\per_glacier\\RGI50-11\\{}\\{}\\outlines.shp'.format(glacier_id[:11], glacier_id)
    dem_path = 'c:\\users\\johannes\\documents\\modelruns\\CH\\per_glacier\\RGI50-11\\{}\\{}\\dem.tif'.format(glacier_id[:11], glacier_id)

    ds = xr.open_dataset(
        #'c:\\users\\johannes\\desktop\\{}_latest.nc'.format(
        #    short_name))
        #'c:\\users\\johannes\\documents\\modelruns\\CH\\per_glacier\\RGI50-11\\{}\\{}\\sat_images.nc'.format(glacier_id[:11], glacier_id))
        'c:\\users\\johannes\\desktop\\fin_latest.nc')
    # convert to reflectance
    ds['bands'] = ds['bands'] / 10000.
    # ds = ds.sel(time='2020-09-11')
    # ds = ds.sel(time='2015-08-29')
    ol = salem.read_shapefile(shp_path)
    ds.attrs['pyproj_srs'] = ol.crs.to_proj4()
    # ds = ds.salem.roi(shape=ol)
    result = []
    if 'albedo' not in ds.data_vars:
        alpha_ens = albedo.get_ensemble_albedo(ds.bands.sel(band='B02'),
                                               ds.bands.sel(band='B03'),
                                               ds.bands.sel(band='B04'),
                                               ds.bands.sel(band='B08'),
                                               ds.bands.sel(band='B11'),
                                               ds.bands.sel(band='B12')).albedo
        alpha_ens.attrs['pyproj_srs'] = ol.crs.to_proj4()
    else:
        alpha_ens = ds.albedo

    # order: 1,2,3,4,5,6,7,8,9,10,11,12,8A
    # 1,2,3,4,5,6,7,8,9,11,12 (from Naegeli), 8A, 10 (read from Naegeli)

    # original: snow, bright ice, dark ice
    # endmembers = np.array([
    #    [0.59, 0.63, 0.67, 0.67, 0.665, 0.66, 0.63, 0.6, 0.5, 0.105, 0.04,
    #     0.04, 0.52],
    #    [0.59, 0.575, 0.55, 0.50, 0.48, 0.46, 0.415, 0.385, 0.29, 0.05, 0.01,
    #     0.01, 0.37],
    #    [0.12, 0.125, 0.135, 0.13, 0.125, 0.12, 0.115, 0.115, 0.085, 0.02,
    #     0.01, 0.01, 0.095]])

    # merged bright and dark ice
    endmembers = np.array([
        [0.59, 0.63, 0.67, 0.67, 0.665, 0.66, 0.63, 0.6, 0.5, 0.105, 0.04,
         0.04, 0.52], np.mean(np.array([[0.59, 0.575, 0.55, 0.50, 0.48,
                                         0.46, 0.415, 0.385, 0.29, 0.05,
                                         0.01, 0.01, 0.37],
                                        [0.12, 0.125, 0.135, 0.13, 0.125,
                                         0.12, 0.115, 0.115, 0.085, 0.02,
                                         0.01, 0.01, 0.095]]), axis=0)])

    # B10 comes from L1C reflectance (no B10 in L2A)
    endmembers_clouds = np.array(
        [0.7657637, 0.81688046, 0.8546996, 0.8671256, 0.8793237, 0.8388719,
         0.80377203, 0.7660221, 0.72458196, 0.14, 0.07805042, 0.07285204,
         0.7467731])

    # from otsu thresholding -B10 is interpolated
    # endmembers = np.array([[0.69586787, 0.71507332, 0.7363538, 0.74294336, 0.75170626,
    #        0.7546415, 0.74678447, 0.7399502, 0.74465635, 0.658109815, 0.57156328,
    #        0.56447413, 0.73288224],
    #       [0.19586787, 0.21507332, 0.2363538, 0.24294336, 0.25170626,
    #        0.2546415, 0.24678447, 0.2399502, 0.24465635, 0.158109815, 0.07156328,
    #        0.06447413, 0.23288224]])
    # from otsu thresholding (only JUN-SEP images) -B10 is interpolated
    # np.array([[0.7096425, 0.72321552, 0.74200752, 0.7473668, 0.75990546,
    #        0.75867552, 0.74852863, 0.74274942, 0.74880117, 0.66, 0.58137598,
    #        0.57796797, 0.7350045],
    #          [0.2096425, 0.22321552, 0.24200752, 0.2473668, 0.25990546,
    #        0.25867552, 0.24852863, 0.24274942, 0.24880117, 0.16, 0.08137598,
    #        0.07796797, 0.2350045]
    #       ])
    # endmembers = np.vstack([endmembers, endmembers_clouds])

    # from Otsu on Rhone  -  B10 is guessed
    # endmembers = np.array([[0.4965, 0.56609999, 0.59500002, 0.63259999, 0.64159998, 0.61344996,
    #        0.5839, 0.58600002, 0.52585005, 0.31, 0.28259999, 0.28934999,
    #        0.55070002],
    #       [0.2157, 0.2335, 0.25400001, 0.2469, 0.25350002, 0.24230001,
    #        0.22669999, 0.21745, 0.2137, 0.0218, 0.03, 0.0212, 0.21029999]#,
    #       #[0.26780001, 0.2793, 0.29859999, 0.29440002, 0.31029998, 0.29585,
    #       # 0.27875001, 0.2652, 0.30360001, 0.14, 0.09220001, 0.09269999, 0.26525]
    #                       ])

    # from Otsu on FIN  -  B10 is guessed
    # endmembers = xr.Dataset({['snow', 'ice']: np.array([[0.71704999, 0.78114998, 0.82624999, 0.838, 0.84514999,
    #   0.79190001, 0.74450001, 0.71200001, 0.70695001, 0.2, 0.15085,
    #   0.13025001, 0.6832],
    #                       [0.33805001, 0.3188, 0.32625, 0.3241, 0.34154999,
    #                        0.321825, 0.30427499, 0.27855, 0.3328, 0.05, 0.01725,
    #                        0.017, 0.27707499]])})
    endmembers = xr.Dataset({'snow': (['band'], np.array(
        [0.71704999, 0.78114998, 0.82624999, 0.838, 0.84514999, 0.79190001,
         0.74450001, 0.71200001, 0.70695001, 0.2, 0.15085, 0.13025001,
         0.6832])), 'ice': (['band'], np.array(
        [0.33805001, 0.3188, 0.32625, 0.3241, 0.34154999, 0.321825, 0.30427499,
         0.27855, 0.3328, 0.05, 0.01725, 0.017, 0.27707499]))}, coords={
        'band': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09',
                 'B10', 'B11', 'B12', 'B8A']})

    #if glacier_id == 'RGI50-11.A51D10':
    #    log.warning('ENDMEMBERS ARE FIXED FOR HÜFIFIRN')
    #    endmembers = xr.Dataset({'snow': (['band'], np.array(
    #        [0.63139999, 0.67644998, 0.73084998, 0.75749999, 0.78525001,
    #   0.74195001, 0.69260001, 0.67124999, 0.63692501, 0.158     ,
    #   0.13635001, 0.6464    ])), 'ice': (['band'], np.array(
    #        [0.36544999, 0.33662501, 0.357125  , 0.361     , 0.39434999,
    #   0.3734    , 0.34899999, 0.31655   , 0.34459999, 0.01685   ,
    #   0.015     , 0.3223    ]))},
    #                            coords={
    #                                'band': ['B01', 'B02', 'B03', 'B04', 'B05',
    #                                         'B06', 'B07', 'B08', 'B09', 'B10',
    #                                         'B11', 'B12', 'B8A']})

    endmembers = generate_endmembers_from_otsu(ds, ol)

    '''
    # based on samples form Findelen
    sshad = ds.bands.values[:, 170:182, 264:278]
    ssun = ds.bands.values[:, 112:124, 277:295]
    isun = ds.bands.values[:, 75:100, 140:165]

    endmembers = np.vstack(
        [np.mean([np.mean(sshad.reshape(13, sshad.shape[1]*sshad.shape[2]), axis=1),
        np.mean(ssun.reshape(13, ssun.shape[1]*ssun.shape[2]), axis=1)], axis=0),
        np.mean(isun.reshape(13, isun.shape[1]*isun.shape[2]), axis=1)])
    '''
    ds['albedo'] = alpha_ens
    dem = xr.open_rasterio(dem_path)
    dem.attrs['pyproj_srs'] = dem.attrs['crs']
    dem = ds.salem.transform(dem.to_dataset(name='height'))
    dem_roi = dem.salem.roi(shape=ol)
    dem = dem.isel(band=0)

    """
    date = pd.Timestamp('2020-08-07')
    cmask = ds.sel(time=date).cmask
    cmask_comb = imagery.geeguide_cloud_mask(ds.sel(time=date).bands)
    cmask_comb = np.clip(cmask_comb + cmask, 0., 1.)
    shadows = imagery.create_cloud_shadow_mask(cmask_comb,
                                               ds.sel(time=date).bands, ds.sel(
            time=date).solar_azimuth_angle.mean(skipna=True).item(), ds.sel(
            time=date).solar_zenith_angle.mean(skipna=True).item())
    csmask = np.clip(cmask_comb + shadows, 0., 1.)
    csmask_ds = cmask.salem.grid.to_dataset()
    csmask_ds['cmask'] = (['y', 'x'], csmask)
    cm_roi = csmask_ds.salem.roi(shape=ol).cmask.values
    um = map_snow_linear_unmixing(ds.sel(time=date), roi_shp=ol,
                             endmembers=endmembers, cloud_mask=cm_roi)
    asm = map_snow_asmag(ds, date=date, roi_shp=ol)
    naeg = map_snow_naegeli(ds, date=date, dem=dem, roi_shp=ol)
    naeg_alt = map_snow_naegeli_alternate(ds, date=date, dem=dem, roi_shp=ol)
    """
    # important! For neaegli method, ds needs to have attribute 'albedo'
    ds['albedo'] = alpha_ens
    for date in ds.time.values:
        if pd.Timestamp(date) != pd.Timestamp('2020-06-29'):
            continue
        cmask = ds.sel(time=date).cmask
        cmask_comb = imagery.geeguide_cloud_mask(ds.sel(time=date).bands)
        cmask_comb = np.clip(cmask_comb + cmask, 0., 1.)
        shadows = imagery.create_cloud_shadow_mask(cmask_comb,
                                           ds.sel(time=date).bands,
                                           ds.sel(
                                               time=date).solar_azimuth_angle.mean(
                                               skipna=True).item(), ds.sel(
                time=date).solar_zenith_angle.mean(skipna=True).item())
        csmask = np.clip(cmask_comb + shadows, 0., 1.)
        csmask_ds = cmask.salem.grid.to_dataset()
        csmask_ds['cmask'] = (['y', 'x'], csmask)
        cm_roi = csmask_ds.salem.roi(shape=ol).cmask.values
        res = map_snow_linear_unmixing(ds.sel(time=date), roi_shp=ol, endmembers=endmembers, cloud_mask=cm_roi)
        asm = map_snow_asmag(ds, date=date, roi_shp=ol)
        naeg = map_snow_naegeli(ds, date=date, dem=dem, roi_shp=ol)
        naeg_alt = map_snow_naegeli_alternate(ds, date=date, dem=dem,
                                           roi_shp=ol)
        #res['snow_asmag'] = asm
        #res['snow_naegeli'] = naeg
        #res['snow_naegeli_alt'] = naeg_alt
        #res['cmask'] = csmask

        data = xr.merge([res,
                         asm.to_dataset(name='snow_asmag'),
                         naeg.to_dataset(name='snow_naeg'),
                         naeg_alt.to_dataset(name='snow_naeg_alt')])
        data['cmask'] = csmask
        result.append(data)

    dem = xr.open_rasterio(dem_path)
    dem.attrs['pyproj_srs'] = dem.attrs['crs']
    dem = ds.salem.transform(dem.to_dataset(name='height'))
    dem_roi = dem.salem.roi(shape=ol)
    dem = dem.isel(band=0)

    # plot snow probability = 50% distribution over elevation
    # time_ix = np.where(ds.time ==pd.Timestamp('2020-09-11'))[0][0]
    # dem['snowprob'] = (('y', 'x'), result[time_ix]['snow'])
    # sl = xr.where((0.48 < dem.snowprob) & (dem.snowprob < 0.52), 1, 0)
    # sl.groupby_bins(dem.height,
    #                bins=np.arange(np.nanmin(dem_roi.height.values),
    #                               np.nanmax(dem_roi.height.values),
    #                               10)).sum().plot()

    # plot overall snow probability over elevation

    warnings.filterwarnings('ignore')


    def median_with_nan_threshold(g, max_nan_ratio=0.1):
        valid_ratio = (g.count().iceprob / g.count().height)
        if valid_ratio < (1 - max_nan_ratio):
            g['iceprob'] = g.iceprob.where(pd.isnull(g.iceprob), np.nan)
            g['snowprob'] = g.snowprob.where(pd.isnull(g.snowprob), np.nan)
        return g.median(skipna=True)


    def std_with_nan_threshold(g, max_nan_ratio=0.1):
        if (g.count().iceprob / g.count().height) < (1 - max_nan_ratio):
            g['iceprob'] = g.iceprob.where(pd.isnull(g.iceprob), np.nan)
            g['snowprob'] = g.snowprob.where(pd.isnull(g.snowprob), np.nan)
        return g.std(skipna=True)

    """
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('hsv', 366)
    fontsize = 16
    plt.rcParams.update({'font.size': fontsize})
    n_processed = 0
    for i, t in enumerate(ds.time.values):
        cmask = ds.sel(time=t).cmask
        cmask_comb = imagery.geeguide_cloud_mask(ds.sel(time=t).bands)
        cmask_comb = np.clip(cmask_comb + cmask, 0., 1.)
        shadows = imagery.create_cloud_shadow_mask(cmask_comb, ds.sel(time=t).bands,
                                           ds.sel
                                               (time=t).solar_azimuth_angle.mean(
                                      skipna=True).item(),
                                           ds.sel
                (time=t).solar_zenith_angle.mean(
                                      skipna=True).item())
        csmask = np.clip(cmask_comb + shadows, 0., 1.)
        csmask_ds = cmask.salem.grid.to_dataset()
        csmask_ds['cmask'] = (['y', 'x'], csmask)
        cm_roi = csmask_ds.salem.roi(shape=ol).cmask.values
        if np.nanmean(cm_roi) > cloud_thresh:
            continue
        n_processed += 1

        time_ix = np.where(ds.time == t)[0][0]
        print(t, time_ix)
        #total_result = result[time_ix]
        #total_result[:, cm_roi == 1.] = np.nan
        total_result = result[time_ix].where(cm_roi == 0., np.nan)
        #snow_result = total_result[0, ...]
        #snow_result = total_result['snow'].values
        # cheap, but works if all snow-related classes have "snow" in the name
        snow_result = np.sum(np.atleast_3d(
            [total_result[k].values for k in total_result.data_vars if 'snow' in k]),
               axis=0)
        # todo: think of non-hardcoding solution
        # if total_result.shape[0] == 3:  # we have two ice classes
        #    ice_result = total_result[1, ...] + total_result[2, ...]
        # if total_result.shape[0] == 2:
        #ice_result = total_result[1, ...]
        #ice_result = total_result['ice'].values
        ice_result = np.sum(np.atleast_3d(
            [total_result[k].values for k in total_result.data_vars if 'ice' in k]),
               axis=0)
        # cloud_result = total_result[2, ...]
        dem_roi['snowprob'] = (('y', 'x'), snow_result)
        dem_roi['iceprob'] = (('y', 'x'), ice_result)
        # dem_roi['cloudprob'] = (('y', 'x'), cloud_result)
        bins = np.arange(np.nanmin(dem_roi.height.values),
                         np.nanmax(dem_roi.height.values), 10)
        dem_roi.groupby_bins(dem_roi.height, bins=bins).mean().snowprob.plot(
            ax=ax, color=cmap(int(pd.Timestamp(t).dayofyear)))
        gb = dem_roi.groupby_bins(dem_roi.height, bins=bins)

        gb_median = gb.map(median_with_nan_threshold,
                         max_nan_ratio=cloud_thresh)
        gb_std = gb.map(std_with_nan_threshold,
                        max_nan_ratio=cloud_thresh)

        fig2, ax2 = plt.subplots(1, 2, figsize=(36, 18))
        # xs = [np.mean([b.left, b.right]) for b, _ in list(gb)]
        # ax2[0].violinplot([g.snowprob.values[~np.isnan(g.snowprob.values)] if (
        #        (g.count().iceprob / g.count().height) > (
        #            1 - cloud_thresh)) else list(
        #    np.full_like(g.snowprob.values, np.nan)) for _, g in list(gb)],
        #                  positions=xs, widths=80., showmedians=True)  # ,
        # ax2[0].violinplot([g.iceprob.values[~np.isnan(g.snowprob.values)] if (
        #        (g.count().iceprob / g.count().height) > (
        #        1 - cloud_thresh)) else list(
        #    np.full_like(g.iceprob.values, np.nan)) for _, g in list(gb)],
        #                  positions=xs, widths=80., showmedians=True)  # ,
        # color=cmap(int(pd.Timestamp(t).dayofyear)))#,
        # yerr = gb_std.snowprob.values
        xs = [np.mean([b.left, b.right]) for b in gb_median.height_bins.values]
        ax2[0].errorbar(xs, gb_median.iceprob.values,
                        color='b',  # cmap(int(pd.Timestamp(t).dayofyear)),
                        yerr=gb_std.iceprob.values, fmt="+-", label='ice prob.', linewidth=2)
        ax2[0].errorbar(xs, gb_median.snowprob.values,
                        color='cyan',  # cmap(int(pd.Timestamp(t).dayofyear)),
                        yerr=gb_std.snowprob.values, fmt="o-", label='snow prob.', linewidth=2)
        # ax2[0].errorbar(xs, gb_median.cloudprob.values,
        #                color=cmap(int(pd.Timestamp(t).dayofyear)),
        #                yerr=gb_std.cloudprob.values, fmt="x-",
        #                label='cloud prob.', linewidth=2)
        ax2[0].set_xlim(np.min(bins), np.max(bins))
        ax2[0].set_ylim(0., 1.)
        ax2[0].legend(fontsize=fontsize)
        ax2[0].set_title('Surface type probability',
                         fontdict={'fontsize': fontsize})
        ax2[0].set_xlabel('Elevation bins', fontdict={'fontsize': fontsize})
        ax2[0].set_ylabel('Probability', fontdict={'fontsize': fontsize})
        # alb = alpha_ens.sel(time=t).mean(dim='broadband').values
        # alb[cm_roi == 1.] = np.nan
        alb = alpha_ens.sel(time=t).mean(dim='broadband')
        alb = alb.salem.roi(shape=ol)
        grey_cmap = plt.get_cmap('Greys_r')
        # grey_cmap.set_bad(color='blue', alpha=0.5)
        # ax2[1].imshow(alb.values, cmap=grey_cmap, aspect='auto', vmin=0., vmax=1.)
        alb.plot.imshow(ax=ax2[1], cmap=grey_cmap, vmin=0., vmax=1.)
        csmask_to_plot = csmask_ds.salem.roi(shape=ol).cmask.where \
            (csmask_ds.salem.roi(shape=ol).cmask == 1., np.nan)
        csmask_to_plot.plot.imshow(ax=ax2[1], cmap='Blues', alpha=0.4)
        # ax2[1].imshow(csmask_to_plot.values, cmap='Blues', alpha=0.4)
        ax2[1].set_title('Broadband albedo (ensemble mean)',
                         fontdict={'fontsize': fontsize})
        ol.plot(ax=ax2[1], facecolor='none', edgecolor='chartreuse',
                linewidth=3, aspect='equal')
        fig2.suptitle(pd.Timestamp(t).strftime('%Y-%m-%d'),
                      fontdict={'fontsize': fontsize})
        fig2.colorbar(
            cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=366),
                              cmap=cmap), ax=ax2[0], label='Day of Year')
        fig2.savefig(
            'c:\\users\\johannes\\documents\\publications\\Paper_Cameras_OptSatellite\\{}_profiles_otsu\\profile_{}.png'.format(
                short_name, pd.Timestamp(t).strftime('%Y-%m-%d')))
        plt.close(fig2)

    fig.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=366), cmap=cmap),
        ax=ax, label='Day of Year')
    fig.savefig(
        'c:\\users\\johannes\\documents\\publications\\Paper_Cameras_OptSatellite\\{}_profiles_otsu\\all_profiles_cloud_thresh_{}.png'.format(
            short_name, cloud_thresh * 100))
    plt.close(fig)
    print('{} snow lines have been processed.'.format(n_processed))

    # eliminate more unclassified clouds with the moisture index
    plt.figure()
    ds_mask = ds.bands.where(ds.cmask == 0)
    test = ds.where(((ds_mask.isel(band=-1) - ds_mask.isel(band=10)) / (
            ds_mask.isel(band=-1) + ds_mask.isel(band=10))).mean(
        dim=['x', 'y'], skipna=True) > 0.8, drop=True)
    (test.cmask.sum(dim=['x', 'y']) / test.cmask.count(dim='time')).plot()
    """