from typing import Union
import xarray as xr
import os
from glob import glob
import pandas as pd
import salem
import numpy as np
import geopandas as gpd
from glaciersat.core import albedo
from glaciersat import utils

import logging

log = logging.getLogger(__name__)


class SatelliteImageMeta:
    """
    Metadata class for a satellite image.

    This class should hold info about the scene (sensor, platform, footprint,
    maybe cloud percentage (mask) etc, but not supply the actual values
    """

    def __init__(self, path):
        if path is not None:
            self.path = path
        else:
            self.path = None

        self.sensor = None
        self.platform = None
        self.scene_footprint = None
        self.cloud_mask = None
        self.cloud_area_percent = None
        self.proc_level = None

    def __repr__(self):

        summary = ['<' + type(self).__module__ + '.' +
                   type(self).__name__ + '>']
        if self.sensor is not None:
            summary += ['  Sensor: ' + self.sensor]
        if self.platform is not None:
            summary += ['  Platform: ' + self.platform]
        if self.scene_footprint is not None:
            summary += ['  Footprint: ' + 'True']
        else:
            summary += ['  Footprint: ' + 'False']
        if self.cloud_mask is not None:
            summary += ['  Cloud Mask: ' + 'True']
        else:
            summary += ['  Cloud Mask: ' + 'False']
        if self.path is not None:
            summary += ['  Origin Path: ' + self.path]

        return '\n'.join(summary) + '\n'

    def load_data(self):
        return SatelliteImage(path=self.path)


class S2ImageMeta(SatelliteImageMeta):
    """
    Metadata class for a Sentinel-2 satellite image.

    This class should hold info about the scene (sensor, platform, footprint,
    maybe cloud percentage (mask) etc, but not supply the actual values.
    """

    def __init__(self, path):
        super().__init__(path=path)
        self.path = path

        # todo: we take B01 as representative for all others: ok?
        sf_path = glob(os.path.join(path, '**', '**', '**',
                                    'MSK_DETFOO_B01.gml'))
        if len(sf_path) == 0:
            log.warning('Scene footprint for {} not available.'.format(
                os.path.basename(path)))
        else:
            self.get_scene_footprint(sf_path[0])

        fname_split = os.path.basename(path).split('_')
        self.date = pd.Timestamp(str(fname_split[6][:8]))
        self.sensor = fname_split[1][:3]
        self.platform = fname_split[0]
        self.proc_level = fname_split[1][3:]

        cm_path = glob(os.path.join(path, '**', '**', '**',
                                    'MSK_CLOUDS_B00.gml'))
        print('CM: ', cm_path)
        if len(cm_path) == 0:
            self.cloud_mask = None
        else:
            # todo: get_cloud_mask_from_gml needs self.grid for rasterizing
            self.cloud_mask = None
            self.cloud_area_percent = None
            #self.cloud_mask = self.get_cloud_mask_from_gml(cm_path[0])
            #self.cloud_area_percent = (np.sum(self.cloud_mask == 1)/ self.cloud_mask.size) * 100

    def load_data(self):
        return S2Image(safe_path=self.path)

    def overlaps_shape(self, shape, percentage=100.):
        """
        Check if the satellite image overlaps a shape by a given percentage.

        Parameters
        ----------
        shape: str od gpd.GeoDataFrame
             Path to shapefile or geopandas.GeoDataFrame for region of
             interest, e.g. a glacier outline.
        percentage : float, optional
             Percentage of `shape` that should intersect the satellite image
             scene footprint. Default: 100. (satellite image has to **contain**
             `shape` fully)

        Returns
        -------
        overlap_bool: bool
            Whether or not the scene footprint contains the given percentage of
            the shape.
        """
        if isinstance(shape, str):
            shape = salem.read_shapefile(shape)

        ratio = percentage / 100.
        if ratio == 1.:
            overlap_bool = self.scene_footprint.contains(
                shape.to_crs(self.scene_footprint.crs)).item()
        elif 0. < ratio < 1.:
            shape_reproj = shape.to_crs(self.scene_footprint.crs)
            shape_area = shape_reproj.area.item()
            intsct_area = self.scene_footprint.intersection(
                shape_reproj).area.item()
            if (intsct_area / shape_area) >= ratio:
                overlap_bool = True
            else:
                overlap_bool = False
        else:
            raise ValueError("Overlap percentage must be between 0 und 100.")

        return overlap_bool

    def get_scene_footprint(self, fp_path: str) -> None:
        """
        Get and unify a Sentinel *.GML scene footprint.

        Parameters
        ----------
        fp_path : str
            Path to a *.GML file containing a Sentinel scene footprint.

        Returns
        -------
        None
        """
        fp = gpd.read_file(fp_path)
        fp_union = fp.unary_union
        fp_gdf = gpd.GeoSeries(fp_union, crs=fp.crs)

        self.scene_footprint = fp_gdf

    def get_cloud_mask_from_gml(self, cmask_path: str) -> np.ndarray:
        """
        Rasterize a Sentinel *.GML cloud mask onto a given grid.

        Parameters
        ----------
        cmask_path : str
            Path to a *.GML file containing a Sentinel cloud mask.

        Returns
        -------
        cmask_raster: np.ndarray
            Cloud mask as a numpy array.
        """
        try:
            cmask = gpd.read_file(cmask_path)
        except ValueError:  # Fiona ValueError: Null layer: '' when empty
            # assume no clouds then (mask of Zeros)
            cmask_raster = self.grid.region_of_interest()
            return cmask_raster

        cmask_u = cmask.unary_union
        cmask_raster = self.grid.region_of_interest(geometry=cmask_u,
                                                    crs=cmask.crs)
        return cmask_raster


class LandsatImageMeta(SatelliteImageMeta):
    """
    Metadata class for a Landsat satellite image.

    This class should hold info about the scene (sensor, platform, footprint,
    maybe cloud percentage (mask) etc, but not supply the actual values.
    """

    def __init__(self, path):
        super().__init__(path=path)
        if path is not None:
            self.path = path
        else:
            self.path = None

        self.sensor = None
        self.platform = None
        self.scene_footprint = None
        self.cloud_mask = None
        self.cloud_area_percent = None
        self.proc_level = None

    def load_data(self):
        return LandsatImage(path=self.path)


class SatelliteImage:
    """
    Base class for satellite images.

    Attributes
    ----------
    data. xr.Dataset
        Dataset holding the image values.
    path: str
        Path where the data originate from.
    sensor: str
        Sensor that has acquired the image.
    platform: str
        Platform on which the recording sensor is mounted.
    scene_footprint: geopandas.GeoSeries
        Footprint geometry of the image. This makes it easier, for example, to
        check whether an object of interest intersects with with the image.

    Parameters
    ----------
    ds: xr.Dataset or None
        The dataset to construct the class from. If `None`, then the class is
        tried to be constructed from `path`. This is why `ds` is mutually
        exclusive with `path`. Default: None.
    path: str or None
        The path to the directory to construct the class from. If `None`, then
        the class is tried to be constructed from `ds`. This is why `path` is
        mutually exclusive with `ds`. Default: None.
    """

    def __init__(self, ds=None, path=None):

        if ds is not None:
            self.data = ds
        else:
            self.data = None

        if path is not None:
            self.path = path
        else:
            self.path = None

        self.sensor = None
        self.platform = None
        self.scene_footprint = None
        self.cloud_mask = None
        self.cloud_area_percent = None
        self.proc_level = None

    def __repr__(self):

        summary = ['<' + type(self).__module__ + '.' + type(self).__name__ +
                   '>']
        if self.sensor is not None:
            summary += ['  Sensor: ' + self.sensor]
        if self.platform is not None:
            summary += ['  Platform: ' + self.platform]
        if self.scene_footprint is not None:
            summary += ['  Footprint: ' + 'True']
        else:
            summary += ['  Footprint: ' + 'False']
        if self.cloud_mask is not None:
            summary += ['  Cloud Mask: ' + 'True']
        else:
            summary += ['  Cloud Mask: ' + 'False']
        if self.data is not None:
            summary += ['  Data: ' + self.data.__repr__()]
        if self.path is not None:
            summary += ['  Origin Path: ' + self.path]

        return '\n'.join(summary) + '\n'

    def get_scene_footprint(self, fp_path):
        """
        Get and unify a scene footprint geometry file.

        Parameters
        ----------
        fp_path : str
            Path to a file containing a scene footprint.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def overlaps_shape(self, shape, percentage=100.):
        """
        Check if the satellite image overlaps a shape by a given percentage.

        Parameters
        ----------
        shape: str od gpd.GeoDataFrame
             Path to shapefile or geopandas.GeoDataFrame for region of
             interest, e.g. a glacier outline.
        percentage : float, optional
             Percentage of `shape` that should intersect the satellite image
             scene footprint. Default: 100. (satellite image has to **contain**
             `shape` fully)

        Returns
        -------
        overlap_bool: bool
            Whether or not the scene footprint contains the given percentage of
            the shape.
        """
        if isinstance(shape, str):
            shape = salem.read_shapefile(shape)

        ratio = percentage / 100.
        if ratio == 1.:
            overlap_bool = self.scene_footprint.contains(
                shape.to_crs(self.scene_footprint.crs)).item()
        elif 0. < ratio < 1.:
            shape_reproj = shape.to_crs(self.scene_footprint.crs)
            shape_area = shape_reproj.area.item()
            intsct_area = self.scene_footprint.intersection(
                shape_reproj).area.item()
            if (intsct_area / shape_area) >= ratio:
                overlap_bool = True
            else:
                overlap_bool = False
        else:
            raise ValueError("Overlap percentage must be between 0 und 100.")

        return overlap_bool


class S2Image(SatelliteImage):

    def __init__(self, ds=None, safe_path=None):
        super().__init__(ds=ds, path=safe_path)

        self.band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                           'B08', 'B09', 'B10', 'B11', 'B12', 'B8A']
        self.cloud_mask_names = ['CMASK']
        self.band_names_short = ['ca', 'b', 'g', 'r', 'vre1', 'vre2', 'vre3',
                                 'nir', 'wv', 'swirc', 'swir1', 'swir2',
                                 'nnir']
        self.band_names_long = ['Coastal Aerosol', 'Blue', 'Green', 'Red',
                                'Vegetation Red Edge 1',
                                'Vegetation Red Edge 2',
                                'Vegetation Red Edge 3', 'Near Infrared',
                                'Water Vapour', 'Short Wave Infrared - Cirrus',
                                'Short Wave Infrared 1',
                                'Short Wave Infrared 2',
                                'Narrow Near Infrared']
        self.bands_60m = ['B01', 'B09', 'B10']
        self.bands_20m = ['B05', 'B06', 'B07', 'B11', 'B12', 'B8A']
        self.bands_10m = ['B02', 'B03', 'B04', 'B08']

        if safe_path is not None:
            self.path = safe_path
            self.proc_level = os.path.basename(safe_path).split('_')[1][3:]
        else:
            self.path = None
            try:
                self.proc_level = ds.attrs['proc_level']
            except KeyError:
                pass  # None is inherited default

        # band 10 not present for processing level 2a
        if self.proc_level.lower() == 'l2a':
            b10_ix = self.band_names.index('B10')
            self.band_names.remove('B10')
            self.bands_60m.remove('B10')
            del self.band_names_short[b10_ix]
            del self.band_names_long[b10_ix]

        # scaling factor (e.g. for albedo computation)
        self.scale_fac = 10000.

        self.grid = None

        if (ds is not None) and (safe_path is None):
            if isinstance(ds, xr.Dataset):
                self.data = ds
            else:
                raise ValueError('`ds` must be an xarray.Dataset.')
        elif (ds is None) and (safe_path is not None):
            self.data = self.from_safe_format(safe_path)
        elif (ds is not None) and (safe_path is not None):
            raise ValueError('Keyword arguments `ds` and `safe_path` are '
                             'mutually exclusive.')
        elif (ds is None) and (safe_path is None):
            raise ValueError('Either of the keywords `ds` and `safe_path` must'
                             ' be given.')
        else:
            raise ValueError('Value for `ds` and/or `safe_path` not accepted.')

    def from_safe_format(self, safe_path: str) -> xr.DataArray:
        """
        Generate class from .SAFE format.

        Parameters
        ----------
        safe_path : str
            Path to a folder in *.SAFE format.

        Returns
        -------
        all_bands: xr.DataArray
            An xarray DataArray with all bands.
        """

        # check if we want to unzip
        if safe_path.endswith('.zip'):
            safe_path = safe_path[:-4]  # clip off '*.zip'
            raise NotImplementedError('Unzipping is not yet supported.')

        bpaths = []
        for b in self.band_names:
            fl = glob(os.path.join(safe_path, '**', '**', '**', '**',
                                   str('*' + b + '*.jp2')), recursive=True)
            # sort and take the one with the highest resolution (level 2A)
            bpaths.append(sorted(fl)[0])
        bands_open = [xr.open_rasterio(p, chunks={'x': 500, 'y': 500})
                      for p in bpaths]

        # interpolate to 10m resolution:
        fine_ix = [self.band_names.index(fb) for fb in self.bands_10m]
        for bi, bname in enumerate(self.band_names):
            if bi not in fine_ix:
                interp = bands_open[bi].isel(band=0).interp_like(
                    bands_open[fine_ix[0]], method='nearest',
                    assume_sorted=False)
                tf = list(interp.transform)
                tf[0] = bands_open[fine_ix[0]].transform[0]
                tf[4] = bands_open[fine_ix[0]].transform[4]
                interp.attrs['transform'] = tuple(tf)
                interp.attrs['res'] = bands_open[fine_ix[0]].res
                bands_open[bi] = interp
            else:
                bands_open[bi] = bands_open[bi].isel(band=0)
            bands_open[bi] = bands_open[bi].reset_coords('band', drop=True)

        # set grid from first open Dataset
        random_ds = bands_open[0].to_dataset(name='var', promote_attrs=True)
        random_ds.attrs['pyproj_srs'] = random_ds.crs.split('=')[1]
        self.grid = salem.grid_from_dataset(random_ds)

        all_bands = xr.concat(bands_open, pd.Index(self.band_names,
                                                   name='band'))

        # pick random file to get a date and expand
        date = pd.Timestamp(
            os.path.basename(bpaths[0]).split('_')[1].split('T')[0])
        all_bands = all_bands.expand_dims(dim='time')
        all_bands = all_bands.assign_coords(time=(['time'], [date]))

        # for salem
        all_bands.attrs['pyproj_srs'] = all_bands.crs.split('=')[1]
        all_bands.name = 'bands'

        # save disk space when writing later
        all_bands.encoding.update(
            {'dtype': 'int16', 'scale_factor': 1e-5, '_FillValue': -9999,
             'zlib': True})

        # process cloud mask
        self.cloud_mask = self.get_cloud_mask()

        # should not happen
        if self.cloud_mask is None:
            all_bands = all_bands.to_dataset(name='bands',
                                             promote_attrs=True)
        else:
            cmask_da = self.cloud_mask.copy(deep=True)
            cmask_da = cmask_da.expand_dims(dim='time')
            cmask_da = cmask_da.assign_coords(time=(['time'], [date]))
            # attrs should be the same anyway, but first has 'pyproj_srs'
            all_bands = xr.merge([all_bands, cmask_da],
                                 combine_attrs='no_conflicts')

        # process scene footprint
        # todo: we take B01 as representative for all others: ok?
        sf_path = glob(
            os.path.join(safe_path, '**', '**', '**', 'MSK_DETFOO_B01.gml'))
        if len(sf_path) == 0:
            log.warning('Scene footprint for {} not available.'.format(
                os.path.basename(safe_path)))
        else:
            self.get_scene_footprint(sf_path[0])

        return all_bands

    def get_cloud_mask_from_gml(self, cmask_path: str) -> xr.DataArray:
        """
        Rasterize a Sentinel *.GML cloud mask onto a given grid.

        Parameters
        ----------
        cmask_path : str
            Path to a *.GML file containing a Sentinel cloud mask.

        Returns
        -------
        cmask_da: xr.DataArray
            Cloud mask as an xr.DataArray.
        """
        try:
            cmask = gpd.read_file(cmask_path)
        except ValueError:  # Fiona ValueError: Null layer: '' when empty
            # assume no clouds then (mask of Zeros)
            cmask_raster = self.grid.region_of_interest()
            return cmask_raster

        cmask_u = cmask.unary_union
        cmask_raster = self.grid.region_of_interest(geometry=cmask_u,
                                                    crs=cmask.crs)
        cmask_da = xr.DataArray(
            cmask_raster,
            coords={'x': self.grid.x_coord, 'y': self.grid.y_coord},
            dims=('y', 'x'),
            name='cmask',
            attrs={'pyproj_srs': self.grid.crs})
        return cmask_da

    def get_cloud_mask_from_scl(self, cmask_path: str) -> xr.DataArray:
        """
        Get cloud information from scene classification (2A processing only).

        This cloud mask should in principle be better the the one from GML.

        Parameters
        ----------
        cmask_path : str
            Path to a *.GML file containing a Sentinel cloud mask.

        Returns
        -------
        cmask_raster: xr.DataArray
            Cloud mask as a DataArray.
        """

        scene_class = xr.open_rasterio(cmask_path).isel(band=0)
        # set the magic attribute
        scene_class.attrs['pyproj_srs'] = scene_class.attrs['crs']

        # select "odd" classes:
        # 2 (dark areas), 3 (cloud shadows), 8 (medium cloud prob.),
        # 9 (high cloud prob.)
        # todo: check 6 (cloud shadows on glaciers sometimes interpreted as water)
        cmask = xr.where((scene_class == 2) | (scene_class == 3) |
                            (scene_class == 8) | (scene_class == 9), 1, 0)
        cmask.attrs['pyproj_srs'] = scene_class.attrs['pyproj_srs']
        cmask = self.grid.to_dataset().salem.transform(
            cmask.to_dataset(name='cmask'))

        return cmask.cmask

    def get_cloud_mask(self, cm_path=None) -> xr.DataArray or None:
        """

        Parameters
        ----------
        cm_path :

        Returns
        -------

        """

        if cm_path is not None:
            if cm_path.endswith('jp2'):
                cmask = self.get_cloud_mask_from_scl(cm_path)
            elif cm_path.endswith('gml'):
                cmask = self.get_cloud_mask_from_gml(cm_path)
            else:
                raise ValueError('Given cloud mask path must be a ".GML" or '
                                 '"*.jp2" file.')
        else:
            # level 2A should have the best cloud mask - get it if possible
            if self.proc_level.lower() == 'l2a':
                cm_path = glob(os.path.join(
                    self.path, '**', '**', '**', 'R20m', '*SCL_20m.jp2'))[0]
                # todo: avoid double code, and try l1c if l2a fails
                if len(cm_path) == 0:
                    log.warning('Cloud mask for {} not available.'.format(
                        os.path.basename(self.path)))
                    return None
                cmask = self.get_cloud_mask_from_scl(cm_path)
            elif self.proc_level.lower() == 'l1c':
                cm_path = glob(os.path.join(self.path, '**', '**', '**',
                                            'MSK_CLOUDS_B00.gml'))[0]
                if len(cm_path) == 0:
                    log.warning('Cloud mask for {} not available.'.format(
                        os.path.basename(self.path)))
                    return None
                cmask = self.get_cloud_mask_from_gml(cm_path)
            else:
                raise ValueError(
                    'If no cloud mask path is given, the "proc_level" '
                    'attribute must be set.')

        # save disk when writing later
        cmask.encoding.update(
            {'dtype': 'int8', 'scale_factor': 0.01, '_FillValue': -99,
             'zlib': True})

        return cmask

    def get_scene_footprint(self, fp_path: str) -> None:
        """
        Get and unify a Sentinel *.GML scene footprint.

        Parameters
        ----------
        fp_path : str
            Path to a *.GML file containing a Sentinel scene footprint.

        Returns
        -------
        None
        """
        fp = gpd.read_file(fp_path)
        fp_union = fp.unary_union
        fp_gdf = gpd.GeoSeries(fp_union, crs=fp.crs)

        self.scene_footprint = fp_gdf

    def get_ensemble_albedo(self, return_ds: bool = False) -> \
            xr.Dataset or SatelliteImage:
        """
        Get an ensemble albedo of the satellite reflectances using three
        methods.

        Parameters
        ----------
        return_ds : bool, optional
            Whether to return the `xarray.Dataset` (True), or a
            `SatelliteImage` instance. The latter also has the footprint as an
            attribute, which is useful for further processing. Default: False
            (return `SatelliteImage`).

        Returns
        -------
        alpha_ens: xr.Dataset or glaciersat.core.imagery.SatelliteImage
            Ensemble albedo object, depending on the value of `return_ds`.
        """
        sf = self.scale_fac
        alpha_ens = albedo.get_ensemble_albedo(
            self.data.bands.sel(band='B02') / sf,
            self.data.bands.sel(band='B03') / sf,
            self.data.bands.sel(band='B04') / sf,
            self.data.bands.sel(band='B08') / sf,
            self.data.bands.sel(band='B11') / sf,
            self.data.bands.sel(band='B12') / sf)
        alpha_ens = xr.merge([alpha_ens, self.data.cmask],
                             combine_attrs='no_conflicts')
        # save disk space when writing later
        alpha_ens.albedo.encoding.update(
            {'dtype': 'int16', 'scale_factor': 0.0001, '_FillValue': -9999,
             'zlib': True})
        if return_ds is True:
            return alpha_ens
        else:
            alpha_ens = SatelliteImage(alpha_ens)
            alpha_ens.scene_footprint = self.scene_footprint.copy(deep=True)
            alpha_ens.cloud_mask = self.cloud_mask.copy()
            alpha_ens.path = self.path
            return alpha_ens

    def get_ndvi(self):
        """
        Normalised difference vegetation index.

        Returns
        -------
        xr.DataArray:
            DataArray containing the NDVI.
        """
        return ndvi(self.data.bands.sel(band='B08'),
                    self.data.bands.sel(band='B04'))

    def get_ndmi(self):
        """
        Normalised difference moisture index.

        Returns
        -------
        xr.DataArray:
            DataArray containing the NDMI.
        """
        return ndmi(self.data.bands.sel(band='B08'),
                    self.data.bands.sel(band='B11'))

    def get_ndsi(self):
        """
        Normalised difference snow index.

        Returns
        -------
        xr.DataArray:
            DataArray containing the NDSI.
        """
        return ndsi(self.data.bands.sel(band='B03'),
                    self.data.bands.sel(band='B11'))

    def get_ndwi(self):
        """
        Normalised difference water index.

        Returns
        -------
        xr.DataArray:
            DataArray containing the NDWI.
        """
        return ndsi(self.data.bands.sel(band='B03'),
                    self.data.bands.sel(band='B08'))

class LandsatImage(SatelliteImage):
    def __init__(self):
        super().__init__()

    def from_download_file(self, download_path):
        pass


def ndvi(nir: Union[xr.DataArray, np.ndarray, float, int],
         red: Union[xr.DataArray, np.ndarray, float, int]) -> \
        Union[xr.DataArray, np.ndarray, float, int]:
    """
    Calculate the normalised difference vegetation index.

    Sentinel-2: B08, B04

    Parameters
    ----------
    nir : xr.DataArray or np.ndarray or float or int
        Near infrared band acquisition.
    red : xr.DataArray or np.ndarray or float or int
        Red band acquisition.

    Returns
    -------
    same as input:
        Normalised difference vegetation index.
    """
    return utils.normalized_difference(nir, red)


def ndwi(g: Union[xr.DataArray, np.ndarray, float, int],
         nir: Union[xr.DataArray, np.ndarray, float, int]) -> \
        Union[xr.DataArray, np.ndarray, float, int]:
    """
    Calculate the normalised difference water index.

    Sentinel-2: B03, B08

    Parameters
    ----------
    g : xr.DataArray or np.ndarray or float or int
        Green band acquisition.
    nir : xr.DataArray or np.ndarray or float or int
        Near infrared band acquisition.

    Returns
    -------
    same as input:
        Normalised difference water index.
    """
    return utils.normalized_difference(g, nir)


def ndsi(g: Union[xr.DataArray, np.ndarray, float, int],
         swir1: Union[xr.DataArray, np.ndarray, float, int]) -> \
        Union[xr.DataArray, np.ndarray, float, int]:
    """
    Calculate the normalised difference snow index.

    Sentinel-2: B03, B11

    Parameters
    ----------
    g : xr.DataArray or np.ndarray or float or int
        Green band acquisition.
    swir1 : xr.DataArray or np.ndarray or float or int
        Short wave infrared band acquisition.

    Returns
    -------
    same as input:
        Normalised difference snow index.
    """
    return utils.normalized_difference(g, swir1)


def ndmi(nir: Union[xr.DataArray, np.ndarray, float, int],
         swir1: Union[xr.DataArray, np.ndarray, float, int]) -> \
        Union[xr.DataArray, np.ndarray, float, int]:
    """
    Normalized difference moisture index.

    Sentinel-2: B8A, B11

    Parameters
    ----------
    nir : xr.DataArray or np.ndarray or float or int
        Near infrared band acquisition.
    swir1 : xr.DataArray or np.ndarray or float or int
        Short wave infrared band acquisition.

    Returns
    -------
    same as input:
        Normalised difference moisture index.
    """
    return utils.normalized_difference(nir, swir1)


def hollstein_fig5_shadow_class(
        image: Union[xr.Dataset, S2Image]) -> xr.DataArray:
    """
    Calculating shadows according to [Hollstein et al. (2016)]_, fig. 5.

    # todo: this currently works for SEN-2 only

    Parameters
    ----------
    image : xr.Dataset or imagery.S2Image
        The Sentinel satellite image (with `bands` variable) to work on. Must
        be reflectances, i.e. data value range 0-1.

    Returns
    -------
    xr.DataArray
        Boolean mask with shadows as True.

    References
    ----------
    .. [Hollstein et al. (2016)]: Hollstein, A., Segl, K., Guanter, L., Brell,
        M., & Enesco, M. (2016). Ready-to-use methods for the detection of
        clouds, cirrus, snow, shadow, water and clear sky pixels in Sentinel-2
        MSI images. Remote Sensing, 8(8), 666.
    """

    if isinstance(image, S2Image):
        image = image.data

    green = image.bands.sel(band='B03')
    nir_comp = image.bands.sel(band='B8A')

    return (green < 0.325) & (nir_comp < 0.166) & (nir_comp > 0.039)


def hollstein_fig6_shadow_class(
        image: Union[xr.Dataset, S2Image]) -> xr.DataArray:
    """
    Calculating shadows according to [Hollstein et al. (2016)]_, fig. 6.

    # todo: this currently works for SEN-2 only

    Parameters
    ----------
    image : xr.Dataset or imagery.S2Image
        The Sentinel satellite image (with `bands` variable) to work on. Must
        be reflectances, i.e. data value range 0-1.

    Returns
    -------
    xr.DataArray
        Boolean mask with shadows as True.

    References
    ----------
    .. [Hollstein et al. (2016)]: Hollstein, A., Segl, K., Guanter, L., Brell,
        M., & Enesco, M. (2016). Ready-to-use methods for the detection of
        clouds, cirrus, snow, shadow, water and clear sky pixels in Sentinel-2
        MSI images. Remote Sensing, 8(8), 666.
    """

    if isinstance(image, S2Image):
        image = image.data

    nir_comp = image.bands.sel(band='B8A')
    vre2 = image.bands.sel(band='B06')
    green = image.bands.sel(band='B03')
    swir2 = image.bands.sel(band='B12')
    wv = image.bands.sel(band='B09')

    path1 = (nir_comp < 0.156) & ((vre2 - green) < -0.025) & (
            (swir2 - wv) < -0.016)
    path2 = (nir_comp < 0.156) & ((vre2 - green) > -0.025) & (
            (swir2 - wv) < 0.084)
    return path1 | path2


def hollstein_fig7_shadow_class(
        image: Union[xr.Dataset, S2Image]) -> xr.DataArray:
    """
    Calculating shadows according to [Hollstein et al. (2016)]_, fig. 7.

    # todo: this currently works for SEN-2 only

    Parameters
    ----------
    image : xr.Dataset or imagery.S2Image
        The Sentinel satellite image (with `bands` variable) to work on. Must
        be reflectances, i.e. data value range 0-1.

    Returns
    -------
    xr.DataArray
        Boolean mask with shadows as True.

    References
    ----------
    .. [Hollstein et al. (2016)]: Hollstein, A., Segl, K., Guanter, L., Brell,
        M., & Enesco, M. (2016). Ready-to-use methods for the detection of
        clouds, cirrus, snow, shadow, water and clear sky pixels in Sentinel-2
        MSI images. Remote Sensing, 8(8), 666.
    """
    if isinstance(image, S2Image):
        image = image.data

    nir_comp = image.bands.sel(band='B8A')
    wv = image.bands.sel(band='B09')
    blue = image.bands.sel(band='B02')
    green = image.bands.sel(band='B03')
    swirc = image.bands.sel(band='B10')
    swir2 = image.bands.sel(band='B12')

    path1 = (nir_comp < 0.181) & (nir_comp < 0.051) & (wv < 0.01) & (
            blue < 0.073)
    path2 = (nir_comp < 0.181) & (nir_comp < 0.051) & (wv > 0.01) & (
            green < 0.074)
    path3 = (nir_comp < 0.181) & (nir_comp > 0.051) & (swir2 < 0.097) & (
            swirc < 0.011)
    path4 = (nir_comp < 0.181) & (nir_comp > 0.051) & (swir2 > 0.097) & (
            swirc > 0.010)
    return path1 | path2 | path3 | path4


def hollstein_fig8_shadow_class(
        image: Union[xr.Dataset, S2Image]) -> xr.DataArray:
    """
    Calculating shadows according to [Hollstein et al. (2016)]_, fig. 8.

    # todo: this currently works for SEN-2 only

    Parameters
    ----------
    image : xr.Dataset or imagery.S2Image
        The Sentinel satellite image (with `bands` variable) to work on. Must
        be reflectances, i.e. data value range 0-1.

    Returns
    -------
    xr.DataArray
        Boolean mask with shadows as True.

    References
    ----------
    .. [Hollstein et al. (2016)]: Hollstein, A., Segl, K., Guanter, L., Brell,
        M., & Enesco, M. (2016). Ready-to-use methods for the detection of
        clouds, cirrus, snow, shadow, water and clear sky pixels in Sentinel-2
        MSI images. Remote Sensing, 8(8), 666.
    """

    if isinstance(image, S2Image):
        image = image.data

    green = image.bands.sel(band='B03')
    nir_comp = image.bands.sel(band='B8A')
    vre3 = image.bands.sel(band='B07')
    wv = image.bands.sel(band='B09')
    swir1 = image.bands.sel(band='B11')
    vre1 = image.bands.sel(band='B05')
    ca = image.bands.sel(band='B01')

    path1 = (green < 0.319) & (nir_comp < 0.166) & ((green - vre3) < 0.027) & (
            (wv - swir1) > -0.097)
    path2 = (green < 0.319) & (nir_comp < 0.166) & ((green - vre3) > 0.027) & (
            (wv - swir1) > 0.021)
    path3 = (green > 0.319) & ((vre1 / swir1) > 4.33) & (green < 0.525) & (
            (ca / vre1) > 1.184)
    return path1 | path2 | path3


def additional_threshold_to_find_more_snow_in_shadow(
        image: Union[xr.Dataset, S2Image]) -> xr.DataArray:
    """
    A try to make the endmembers profiles from [Naegeli et al. (2017)]_
    distinguishable in shadowed regions.

    Parameters
    ----------
    image : xr.Dataset or imagery.S2Image
        The Sentinel satellite image (with `bands` variable) to work on. Must
        be reflectances, i.e. data value range 0-1.

    Returns
    -------
    xr.DataArray
        Boolean mask with shadows as True.

    References
    ----------
    .. [Naegeli et al. (2017)]: Naegeli, K., Damm, A., Huss, M., Wulf, H.,
        Schaepman, M., & Hoelzle, M. (2017). Cross-comparison of albedo
        products for glacier surfaces derived from airborne and satellite
        (Sentinel-2 and Landsat 8) optical data. Remote Sensing, 9(2), 110.
    """

    if isinstance(image, S2Image):
        image = image.data
    # based on comparing histograms
    return (image.bands.sel(band='B01') > 0.331) & (
            image.bands.sel(band='B09') > 0.294) & ((image.bands.sel(
        band='B01') - image.bands.sel(band='B10')) > 0.294)
