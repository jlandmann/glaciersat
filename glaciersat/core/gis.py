import xarray as xr
import salem
import os
import pandas as pd
import numpy as np
from glaciersat.core import imagery

import logging
log = logging.getLogger(__file__)


def crop_sat_image_to_glacier(ds: xr.Dataset or imagery.SatelliteImage,
                              gdir_candidates: list = None,
                              out_dirs: list = None, grids: list = None,
                              min_overlap: float = 100.,
                              shapes: list or None = None) -> None:
    """
    Crop a satellite image into glacier domains and append to existing images.

    Parameters
    ----------

    ds : xr.Dataset or glaciersat.core.imagery.SatelliteImage
        Object containing the satellite image.
    gdir_candidates: list of GlacierDirectories, optional
        List with potential GlacierDirectories that might be included in the
        scene. Mutually exclusive with `grids`. Default: None.
    out_dirs: list of str, optional
        List with according output directories for the case that `grids` is
        given.
    grids: list of salem.Grid, optional
        List with salem.Grids defining a glacier region to which the data shall
        be clipped. Mutually exclusive with `gdir_candidates`. Default: None.
    min_overlap: float
        Minimum overlap percentage of satellite image and glacier. Default:
        100. (glacier must be contained fully in satellite image footprint).
    shapes: list or None, optional
        List of paths to shapes of glaciers. Must be in the same order like
        `gdir_candidates` or `grids`, respectively. If `None` and
        `gdir_candidates` is given, shapes will be retrieved from the outlines
        in the glacier directory. Default: None.

    Returns
    -------
    None.
    """

    if (gdir_candidates is not None) and (grids is not None):
        raise ValueError('The keywords "gdir_candidates" and "grids" are '
                         'mutually exclusive.')
    elif (gdir_candidates is None) and (grids is None):
        raise ValueError('Either of the keywords "gdir_candidates" or "grids" '
                         'must be given.')
    elif (gdir_candidates is not None) and (grids is None):
        grids = [salem.Grid.from_json(g.get_filepath('glacier_grid')) for g in
                 gdir_candidates]
    else:
        pass

    if (shapes is None) and (gdir_candidates is not None):
        shapes = [gdir_candidates[i].get_filepath('outlines') for i in
                  range(len(gdir_candidates))]

    # cheap way to pre-assess whether given glacier is in the image at all
    if callable(getattr(ds, 'overlaps_shape', None)) and (shapes is not None):
        cand_in_img = [ds.overlaps_shape(i, percentage=min_overlap) for i in
                       shapes]
        if np.array(cand_in_img).any() is False:  # no intersection at all
            log.info(
                'No intersection of the satellite image and the supplied grids'
                ' at the given level ({}%) at all.'.format(min_overlap))
            return
    else:  # postpone to next step (more expensive)
        cand_in_img = None

    # we need to load for this operation
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        ds = ds.data

    ds.load()

    if not isinstance(ds, xr.Dataset):
        ds = ds.to_dataset(name='albedo', promote_attrs=True)

    for i, grid in enumerate(grids):

        if (cand_in_img is not None) and (cand_in_img[i] is False):
            continue

        grid_ds = grid.to_dataset()
        ds_glacier = grid_ds.salem.transform(ds, interp='linear')

        # can be that grid is outside satellite image
        # todo: what about half coverage (edge) and clouds? check with outline?
        if pd.isnull(ds_glacier.to_array()).all():
            # todo: log something here?
            continue

        if gdir_candidates is not None:
            gi = gdir_candidates[i]
            if gi.has_file('sat_images'):
                with xr.open_dataset(gi.get_filepath('sat_images')) as exist:
                    exist.load()
                    ds_total = xr.merge([ds_glacier, exist],
                                        combine_attrs='no_conflicts',
                                        compat='override')
                ds_total.to_netcdf(gi.get_filepath('sat_images'))
            else:
                ds_glacier.to_netcdf(gi.get_filepath('sat_images'))
        elif out_dirs is not None:
            gi = out_dirs[i]
            fp = os.path.join(gi, 'sat_images.nc')
            if os.path.exists(fp):
                with xr.open_dataset(fp) as exist:
                    exist.load()
                    ds_total = xr.merge([ds_glacier, exist],
                                        combine_attrs='no_conflicts',
                                        compat='override')
                ds_total.to_netcdf(fp)
            else:
                ds_glacier.to_netcdf(fp)