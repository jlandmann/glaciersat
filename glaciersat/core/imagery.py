import xarray as xr
import os
from glob import glob
import pandas as pd
import salem
import numpy as np
import geopandas as gpd
from glaciersat.core import albedo

import logging

log = logging.getLogger(__name__)


class SatelliteImage:

    def __init__(self):
        self.sensor = None
        self.platform = None
        self.scene_footprint = None

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
        super().__init__()

        self.band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                           'B08', 'B09', 'B10', 'B11', 'B12', 'B8A']
        self.cloud_mask_names = ['CMASK']
        self.bands_names_short = ['ca', 'b', 'g', 'r', 'vre1', 'vre2', 'vre3',
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
            raise NotImplementedError('Unzipping is not yet supported.')

        bpaths = []
        for b in self.band_names:
            fl = glob(os.path.join(safe_path, '**', '**', '**', '**',
                                   str('*' + b + '.jp2')), recursive=True)
            bpaths.append(fl[0])
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

        # process cloud mask
        cm_path = glob(
            os.path.join(safe_path, '**', '**', '**', 'MSK_CLOUDS_B00.gml'))
        if len(cm_path) == 0:
            log.warning('Cloud mask for {} not available.'.format(
                os.path.basename(safe_path)))
            all_bands = all_bands.to_dataset(name='bands',
                                             promote_attrs=True)
        else:
            self.cloud_mask = self.get_cloud_mask_from_gml(cm_path[0])
            cmask_da = xr.DataArray(self.cloud_mask,
                                    coords=bands_open[0].coords,
                                    dims=bands_open[0].dims, name='cmask',
                                    attrs=bands_open[0].attrs)
            cmask_da = cmask_da.expand_dims(dim='time')
            cmask_da = cmask_da.assign_coords(time=(['time'], [date]))
            cmask_da.attrs['pyproj_srs'] = all_bands.crs.split('=')[1]
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

        # for saving later
        all_bands.encoding['zlib'] = True

        return all_bands

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

    def get_ensemble_albedo(self):
        sf = self.scale_fac
        return albedo.get_ensemble_albedo(self.data.bands.sel(band='B02') / sf,
                                          self.data.bands.sel(band='B03') / sf,
                                          self.data.bands.sel(band='B04') / sf,
                                          self.data.bands.sel(band='B08') / sf,
                                          self.data.bands.sel(band='B11') / sf,
                                          self.data.bands.sel(band='B12') / sf)


class LandsatImage(SatelliteImage):
    def __init__(self):
        super().__init__()

    def from_download_file(self, download_path):
        pass
