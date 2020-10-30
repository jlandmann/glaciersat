import numpy as np
import xarray as xr
import pandas as pd

xr.set_options(keep_attrs=True)


def get_broadband_albedo_knap(g, nir):
    """
    Get broad band albedo after Knap (1999) ([1]_) as used in [2]_.

    Band designations:
    Landsat 8: g: b3, nir: b5 ([2]_).
    Sentinel 2: g: b3, nir: b8 ([2]_).

    Parameters
    ----------
    g: np.array
        Green band.
    nir: np.array
        Near-infrared band.

    Returns
    -------
    np.array:
        Broad band albedo after [1]_.

    References
    ----------
    .. [1] Knap, W. H.; Reijmer, C. H. & Oerlemans, J.: Narrowband to
        broadband conversion of Landsat TM glacier albedos. International
        Journal of Remote Sensing, Taylor & Francis, 1999, 20, 2091-2110.
    .. [2] Naegeli, K.; Damm, A.; Huss, M.; Wulf, H.; Schaepman, M. & Hoelzle,
        M.: Cross-Comparison of Albedo Products for Glacier Surfaces Derived
        from Airborne and Satellite (Sentinel-2 and Landsat 8) Optical Data
        Remote Sensing, 2017, 9.
    """

    return 0.726 * g - 0.322 * g ** 2 - 0.015 * nir + 0.581 * nir ** 2


def get_broadband_albedo_liang(b: np.ndarray, r: np.ndarray, nir: np.ndarray,
                               swir1: np.ndarray, swir2: np.ndarray) \
        -> np.ndarray:
    """
    Get broad band albedo after Liang (2001) ([1]_) as used in [2]_.

    Band designations:
    Landsat 8: b: b2, r: b4, nir: b8, swir1: b6, swir2: b7
    Sentinel 2: b: b2, r: b4, nir: b8, swir1: b11, swir2: b12

    Parameters
    ----------
    b: np.array
        Blue band.
    r: np.array
        Red band.
    nir: np.array
        Near-infrared band.
    swir1: np.array
        Short wave-infrared band 1.
    swir2: np.array
        Short wave-infrared band 2.

    Returns
    -------
    np.array:
        Broad band albedo after [1]_.

    References
    ----------
    .. [1] Liang, S.: Narrowband to broadband conversions of land surface
        albedo I: Algorithms. Remote Sensing of Environment, 2001, 76,
        213 - 238.
    .. [2] Naegeli, K.; Damm, A.; Huss, M.; Wulf, H.; Schaepman, M. & Hoelzle,
        M.: Cross-Comparison of Albedo Products for Glacier Surfaces Derived
        from Airborne and Satellite (Sentinel-2 and Landsat 8) Optical Data
        Remote Sensing, 2017, 9.
    """
    return 0.356 * b + 0.130 * r + 0.373 * nir + 0.085 * swir1 + 0.072 * \
           swir2 - 0.0018


def get_broadband_albedo_bonafoni(b: np.ndarray, g: np.ndarray, r: np.ndarray,
                                  nir: np.ndarray, swir1: np.ndarray,
                                  swir2: np.ndarray) -> np.ndarray:
    """
    Get broad band albedo after Bonafoni (2020) ([1]_).

    Band designations:
    Sentinel 2: b:b2, r: b4, nir: b8, swir1: b11, swir2: b12
    The papers designs this albedo for Sentinel only.

    Parameters
    ----------
    b: np.array
        Blue band.
    g: np.array
        Green band.
    r: np.array
        Red band.
    nir: np.array
        Near-infrared band.
    swir1: np.array
        Short wave-infrared band 1.
    swir2: np.array
        Short wave-infrared band 2.

    Returns
    -------
    np.array:
        Broad band albedo after [1]_.

    References
    ----------
    .. [1] Bonafoni, S., & Sekertekin, A. (2020). Albedo Retrieval From
        Sentinel-2 by New Narrow-to-Broadband Conversion Coefficients. IEEE
        Geoscience and Remote Sensing Letters.
    """

    return b * 0.2266 + g * 0.1236 + r * 0.1573 + nir * 0.3417 + \
           swir1 * 0.1170 + swir2 * 0.0338


def get_proxy_albedo_mccarthy(r: np.ndarray, g: np.ndarray, b: np.ndarray) \
        -> np.ndarray:
    """
    Get a 'proxy albedo' (sum of RGB) as discussed with D. Rounce in McCarthy.

    The sum is divided by three to mimic a "normal reflectance" band.

    Parameters
    ----------
    r: np.ndarray
        Red band.
    g: np.ndarray
        Green band.
    b: np.ndarray
        Blue band.

    Returns
    -------
    np.ndarray
        Array with proxy for albedo.
    """
    return (r + g + b) / 3.


def get_ensemble_albedo(b, g, r, nir, swir1, swir2):
    """
    Get an ensemble estimate of all available albedo methods.

    At the moment

    Returns
    -------
    np.ndarray:
        The
    """
    a_knap = get_broadband_albedo_knap(g, nir)
    a_liang = get_broadband_albedo_liang(b, r, nir, swir1, swir2)
    a_bonafoni = get_broadband_albedo_bonafoni(b, g, r, nir, swir1, swir2)

    # return the same type as input
    if np.array([isinstance(a, np.ndarray) for a in
                 [a_knap, a_liang, a_bonafoni]]).all():
        return np.vstack([a_knap, a_liang, a_bonafoni])
    elif np.array([isinstance(a, xr.DataArray) for a in
                   [a_knap, a_liang, a_bonafoni]]).all():
        # minimal coords, since there might be "band" left, for example
        a_ens = xr.concat([a_knap, a_liang, a_bonafoni],
                          pd.Index(['knap', 'liang', 'bonafoni'],
                          name='broadband'), coords='minimal',
                          compat='override', combine_attrs='identical')
        a_ens = a_ens.reset_coords('band', drop=True)
        a_ens = a_ens.to_dataset(name='albedo', promote_attrs=True)
        return a_ens
    else:
        raise ValueError('Data types of the albedo ensemble members are: {}, '
                         'but allowed are only `xr.DataArray` or `np.array`.'.
                         format([type(o) for o in [a_knap, a_liang,
                                                   a_bonafoni]]))
