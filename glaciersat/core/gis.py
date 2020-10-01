import xarray as xr
import salem


def crop_sat_image_to_glacier(ds: xr.Dataset, gdir_candidates: list = None,
                              grids: list = None):
    """
    Crop a satellite image into glacier domains and append to existing images.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the satellite image.
    gdir_candidates: list of GlacierDirectories, optional
        List with potential GlacierDirectories that might be included in the
        scene. Mutually exclusive with `grids`. Default: None.
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

    for i, grid in enumerate(grids):
        ds_glacier = ds.salem.subset(grid=grid)
        ds_glacier.name = 'albedo'
        grid_ds = grid.to_dataset()
        ds_glacier = grid_ds.salem.transform(ds_glacier, interp='linear')

        if gdir_candidates is not None:
            gi = gdir_candidates[i]
            if gi.has_file('sat_images'):
                with xr.open_dataset(gi.get_filepath('sat_images')) as exist:
                    ds_total = xr.merge([exist, ds_glacier])
                ds_total.to_dataset()
            else:
                ds_glacier.to_dataset(gi.get_filepath('sat_images'))
