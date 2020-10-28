import xarray as xr
import salem
import os
import pandas as pd


def rasterize_shape(shp, resolution):
    """
    Rasterize a shapefile.

    In this application, this is especially useful e.g. for rasterizing cloud
    masks.

    # todo: take this func from salem

    Parameters
    ----------
    shp :
    resolution : float
        Target resolution in meters (m).

    Returns
    -------
    raster: salem.

    """
    #mask_all_touched = grid.region_of_interest(shape=, all_touched=True)


def crop_sat_image_to_glacier(ds: xr.Dataset, gdir_candidates: list = None,
                              out_dirs: list = None, grids: list = None):
    """
    Crop a satellite image into glacier domains and append to existing images.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the satellite image.
    gdir_candidates: list of GlacierDirectories, optional
        List with potential GlacierDirectories that might be included in the
        scene. Mutually exclusive with `grids`. Default: None.
    out_dirs: list of str, optional
        List with according output directories for the case that `grids` is
        given.
    grids: list of `salem.Grid`s, optional
        List with salem.Grids defining a glacier region to which the data shall
        be clipped. Mutually exclusive with `gdir_candidates`. Default: None.

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

    ds.load()
    for i, grid in enumerate(grids):
        grid_ds = grid.to_dataset()
        if not isinstance(ds, xr.Dataset):
            ds = ds.to_dataset(name='albedo', promote_attrs=True)
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
                    ds_total = xr.merge([exist, ds_glacier],
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
                    ds_total = xr.merge([exist, ds_glacier],
                                        combine_attrs='no_conflicts',
                                        compat='override')
                ds_total.to_netcdf(fp)
            else:
                ds_glacier.to_netcdf(fp)