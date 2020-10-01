import xarray as xr
import os
from glob import glob
import pandas as pd
from glaciersat.core import albedo


class SatelliteImage:
    def __init__(self):
        pass


class S2Image(SatelliteImage):

    def __init__(self, ds=None, safe_path=None):
        super().__init__()

        self.band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                           'B08', 'B09', 'B10', 'B11', 'B12', 'B8A']
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

    def from_safe_format(self, safe_path):
        """
        Generate class from .SAFE format.

        Parameters
        ----------
        safe_path :

        Returns
        -------

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

        all_bands = xr.concat(bands_open, pd.Index(self.band_names,
                                                   name='band'))
        # for salem
        all_bands.attrs['pyproj_srs'] = all_bands.crs.split('=')[1]
        return all_bands

    def get_ensemble_albedo(self):
        sf = self.scale_fac
        return albedo.get_ensemble_albedo(self.data.sel(band='B02') / sf,
                                          self.data.sel(band='B03') / sf,
                                          self.data.sel(band='B04') / sf,
                                          self.data.sel(band='B08') / sf,
                                          self.data.sel(band='B11') / sf,
                                          self.data.sel(band='B12') / sf)


class LandsatImage(SatelliteImage):
    def __init__(self):
        super().__init__()

    def from_download_file(self, download_path):
        pass
