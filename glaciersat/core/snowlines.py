import glaciersat.cfg as cfg
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
    else:  # if only one time, we can safely select it
        if ('time' in ds.coords) and (len(ds.coords['time']) == 1):
            ds = ds.isel(time=0)

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
                     alternate: bool = False, r_crit: float or None = None,
                     optimize_r_crit: bool = False) -> \
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
        dem = dem.sel(time=date, method="nearest").height.values
    else:
        # take the latest DEM
        dem = dem.isel(time=-1).height.values

    albedo = ds.albedo
    if 'cmask' in ds.data_vars:
        cmask = ds.cmask.values.copy()
        cprob_thresh = cfg.PARAMS['cloud_prob_thresh']
        cmask[cmask > cprob_thresh] = np.nan
        cmask[cmask <= cprob_thresh] = 1.
        albedo *= cmask
    else:
        log.warning('No cloud mask information given. Still proceeding and '
                    'pretending a cloud-free scene...')

    if roi_shp is not None:
        albedo = albedo.salem.roi(shape=roi_shp)
    out_ds = albedo.copy(deep=True)
    out_ds = out_ds.rename('snow')

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
            max_ampl = cfg.PARAMS['naegeli_snow_alpha_thresh'] - \
                       cfg.PARAMS['naegeli_ice_alpha_thresh']
            min_icpt = cfg.PARAMS['naegeli_ice_alpha_thresh']
            max_icpt = cfg.PARAMS['naegeli_snow_alpha_thresh']

            # stupid, but true: this is double work we wanted to avoid
            merged = xr.merge([albedo_amb, dem_amb])
            bw = cfg.PARAMS['bin_width']
            aab = merged.groupby_bins('height', np.arange(np.nanmin(dem_amb),
                                                          np.nanmax(dem_amb),
                                                          bw)).mean()
            alpha_in = aab.alpha.values
            height_in = aab.height.values
            r_squared = get_model_fit_r_squared(
                _root_sum_squared_residuals, _step_function, height_in,
                alpha_in, bounds=([min_ampl, min_icpt], [max_ampl, max_icpt]),
                x0=np.array([np.nanmax(alpha_in) - np.nanmin(alpha_in),
                             np.nanmin(alpha_in)]), b=sla)

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
        out_ds = out_ds.where(
            (out_ds != 0.5) & (dem < (sla + r_crit)) | np.isnan(dem), 1.)
        out_ds = out_ds.where(
            (out_ds != 0.5) & (dem > (sla - r_crit)) | np.isnan(dem), 0.)

    return out_ds


map_snow_naegeli_alternate = partial(map_snow_naegeli, alternate=True,
                                     optimize_r_crit=True, r_crit=None)


def primary_surface_type_evaluation(alpha_ds: xr.DataArray or xr.Dataset) -> \
        xr.DataArray or xr.Dataset:
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
    alpha_ds = alpha_ds.where((alpha_ds < iat) | (alpha_ds > sat) |
                              np.isnan(alpha_ds), 0.5)

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
        argmax_ix = alpha_slope.isel(
            height_bins=search_range).alpha.argmax().item()
        sla = alpha_slope.isel(height_bins=search_range).height.values[
            argmax_ix]
        alpha_crit = alpha_amb_binned.isel(
            height_bins=search_range).alpha.values[argmax_ix]

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
