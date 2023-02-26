from typing import Union, Optional
import os
import logging
import numpy as np
import salem
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.topology import TopologicalError
from glaciersat.core import imagery, gis, snowlines
from glaciersat.core.imagery import SatelliteImage, S2Image, LandsatImage, \
    SatelliteImageMeta
from glaciersat import cfg
import dask

try:
    from oggm import GlacierDirectory
except ImportError:
    pass

try:
    from crampon import GlacierDirectory
except ImportError:
    pass


# todo: good idea here?
from multiprocessing.pool import ThreadPool

dask.config.set(scheduler="threads", pool=ThreadPool(10))

log = logging.getLogger()
log.setLevel(logging.ERROR)


def distribute_scene_to_domains(
    image: Optional[Union[SatelliteImage, S2Image, LandsatImage,
                          SatelliteImageMeta]] = None,
    gdirs: Optional[list] = None,
    grids: Optional[list] = None,
) -> None:
    """
    Distribute a satellite scene to a set of domains.

    This can be useful when e.g. distributing a satellite image to OGGM/CRAMPON
    GlacierDirectories.

    Parameters
    ----------
    image : SatelliteImage, S2Image, LandsatImage, SatelliteImageMeta, optional
        Satellite image to distribute. If it is a Meta object, data will only
        be loaded if the scene intersects with the grids specified.
    gdirs : list, optional
        List of GlacierDirectories. The advantage of passing GlacierDirectories
        directly is that actually the glacier outlines can be used to check for
        intersection with the scene and not the domain (the would cause "false
        positive" intersections and thus unnecessary data rubbish.
        Needs to have either OGGM or CRAMPON accessible. Default: None.
    grids : list, optional
         List with paths to domain grids (JSON) that should be intersected
         with. Default: None.

    Returns
    -------
    None
    """

    # mutually exclusive arguments
    if gdirs is not None and grids is not None:
        raise ValueError(
            'Arguments `gdirs` and `grids` are mutually exclusive.'
        )

    if gdirs is not None:
        grids = [g.grid for g in gdirs]
        outlines = [g.read_shapefile('outlines') for g in gdirs]
    else:
        grids_gdfs = [g.to_geometry(to_crs=image.grid.proj.crs) for g in grids]
        outlines = pd.concat(grids_gdfs)
        outlines['Area'] = [g.dx * np.abs(g.dy) * g.nx * g.ny for g in grids]

    # make a union for a quicker overlap search (less errors)
    outlines["fake_col"] = 0
    ol_union = outlines.dissolve(by="fake_col")
    ol_union = ol_union.buffer(0)
    ol_union.drop("fake_col", axis=1, inplace=True)  # tidy up

    # todo: replace with clever solution
    #base_paths = [
    #    os.path.join("/scratch/landmanj/modelruns/CH/per_glacier/RGI50-11/",
    #        v[:11], v) for v in grids.RGIId.values]

    # Check if any overlap at all (otherwise takes long)
    try:
        if not image.overlaps_shape(ol_union, percentage=(np.min(
            outlines.Area) / np.sum(outlines.Area)) * 100):
                log.info("No glacier on given scene.")
    except TopologicalError:
        pass

    # if more than one, see if we can reduce the amount of processed glaciers
    cand_in_img = [image.overlaps_shape(
        gpd.GeoSeries(i[1]["geometry"], crs=outlines.crs)) for i in
        outlines.iterrows()]

    # if first check has failed due to broken topology
    if not np.array(cand_in_img).any():
        log.info("No glacier on given scene (second attempt).")
    else:
        log.info("{} overlaps found.".format(
            np.count_nonzero(np.array(cand_in_img))))

    if image.is_meta:
        image = image.load_data()  # imagery.S2Image(safe_path=p)
    alpha_ens = image.get_ensemble_albedo()

    # write out files for easier retrieval next time
    image.data.to_netcdf(os.path.join(os.path.dirname(image.path),
        os.path.basename(image.path).split(".")[0] + ".nc", ))
    alpha_ens.data.to_netcdf(os.path.join(os.path.dirname(image.path),
        os.path.basename(image.path).split(".")[0] + "_alpha.nc", ))

    gridpaths = [("glacier_grid.json") for b in base_paths]
    dem_paths = [os.path.join(b, "homo_dem_ts.nc") for b in base_paths]
    roi_paths = [os.path.join(b, "outlines.shp") for b in base_paths]

    # select only candidates that intersect
    gridpaths = list(np.array(gridpaths)[cand_in_img])
    dem_paths = list(np.array(dem_paths)[cand_in_img])
    roi_paths = list(np.array(roi_paths)[cand_in_img])

    grids = [salem.Grid.from_json(g) for g in gridpaths]

    # process albedo
    log.info(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        ": PROCESS ALBEDO...")
    gis.crop_sat_image_to_glacier(alpha_ens, grids=grids,
        out_dirs=[os.path.dirname(g) for g in gridpaths],
        shapes=[r for r in roi_paths], )

    # process bands
    log.info(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        ": PROCESS BANDS...")
    gis.crop_sat_image_to_glacier(image, grids=grids,
        out_dirs=[os.path.dirname(g) for g in gridpaths],
        shapes=[r for r in roi_paths])

    """
    for j, b in enumerate(base_paths):
        print('GLACIER: ', b)

        gridpath = os.path.join(b, 'glacier_grid.json')
        dem_path = os.path.join(b, 'homo_dem_ts.nc')
        roi_path = os.path.join(b, 'outlines.shp')
        grid = salem.Grid.from_json(gridpath)

        # test if dem has pyproj_srs
        try:
            with xr.open_dataset(dem_path) as d1:
                d1t = d1.copy(deep=True)
                d1.close()
                d1t.salem
        except (RuntimeError, AttributeError):
            try:
                ol = salem.read_shapefile(roi_path)
                d1t.attrs['pyproj_srs'] = ol.crs.to_proj4()
                d1t.to_netcdf(dem_path)
            except (RuntimeError, KeyError):  # homo_dem_ts missing:
                print(
                    'Probably homo_dem_ts is missing: continuing with another glacier...')
            continue

        try:
            cropped = xr.open_dataset(os.path.join(b, 'sat_images.nc'))
        except FileNotFoundError:
            print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                  ': Satellite image file not available\n')
            continue

        avail_dates = cropped.time.values.copy()

        for date in avail_dates:

            print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                  ': Processing alternate snow distribution...')
            try:
                tt = snowlines.map_snow_naegeli_alternate(cropped,
                    dem=dem_path, date=date, roi_shp=roi_path)
            except ValueError:
                print('DATE:, ', date)
                cropped.bands.sel(time=date, band='B08').plot()
                raise

            if tt.isel(broadband=0).isnull().all():
                print('We forget this scene...')
                # return
                continue

            print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                  ': Processing Naegeli snow distribution...')
            t = snowlines.map_snow_naegeli(cropped, dem=dem_path, date=date,
                                           roi_shp=roi_path)

            cropped_nir = xr.open_dataset(
                os.path.join(b, 'sat_images.nc'.format(proc_level)))
            print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                  ': Processing ASMAG snow distribution...')

            ttt = snowlines.map_snow_asmag(cropped_nir, date=date,
                                           roi_shp=roi_path)

            snowmaps = xr.merge([t.to_dataset(name='snow', promote_attrs=True),
                                 tt.to_dataset(name='snow',
                                               promote_attrs=True),
                                 ttt.to_dataset(name='snow',
                                                promote_attrs=True)])

            assim_path = os.path.join(b, 'assim_data.nc'.format(proc_level))
            if os.path.exists(assim_path):
                with xr.open_dataset(assim_path) as exist:
                    exist.load()
                    ds_total = snowmaps.combine_first(exist)
                ds_total.to_netcdf(assim_path, encoding={
                    'snow': {'dtype': 'int8', '_FillValue': 99, 'zlib': True}})
            else:
                snowmaps.to_netcdf(assim_path, encoding={
                    'snow': {'dtype': 'int8', '_FillValue': 99, 'zlib': True}})

            success_cnt += 1
            print('SUCCESS NO: ', success_cnt)
    """


    date = image.date
    for g in gdirs:

        snowprob_exist = xr.open_dataset(g.get_filepath('snowprob'))
        ds = xr.open_dataset(g.get_filepath('sat_images'))
        ol = g.read_shapefile('outlines')

        dem = xr.open_rasterio(g.get_filepath('dem'))
        dem.attrs['pyproj_srs'] = dem.attrs['crs']
        dem = ds.salem.transform(dem.to_dataset(name='height'))
        dem_roi = dem.salem.roi(shape=ol)
        dem = dem.isel(band=0)

        # todo: this shouldn't be run every time: Save them somewhere
        endmembers = snowlines.generate_endmembers_from_otsu(ds, ol)

        cmask = ds.sel(time=image.date).cmask
        cmask_comb = imagery.geeguide_cloud_mask(ds.sel(time=date).bands)
        cmask_comb = np.clip(cmask_comb + cmask, 0., 1.)
        shadows = imagery.create_cloud_shadow_mask(cmask_comb,
                                                   ds.sel(time=date).bands,
                                                   ds.sel(
                                                       time=date).solar_azimuth_angle.mean(
                                                       skipna=True).item(),
                                                   ds.sel(
                                                       time=date).solar_zenith_angle.mean(
                                                       skipna=True).item())
        csmask = np.clip(cmask_comb + shadows, 0., 1.)
        csmask_ds = cmask.salem.grid.to_dataset()
        csmask_ds['cmask'] = (['y', 'x'], csmask)
        cm_roi = csmask_ds.salem.roi(shape=ol).cmask.values
        res = snowlines.map_snow_linear_unmixing(ds.sel(time=date), roi_shp=ol,
                                       endmembers=endmembers,
                                       cloud_mask=cm_roi)
        asm = snowlines.map_snow_asmag(ds, date=date, roi_shp=ol)
        naeg = snowlines.map_snow_naegeli(ds, date=date, dem=dem, roi_shp=ol)
        naeg_alt = snowlines.map_snow_naegeli_alternate(ds, date=date, dem=dem,
                                              roi_shp=ol)

        data = xr.merge([res, asm.to_dataset(name='snow_asmag'),
                         naeg.to_dataset(name='snow_naeg'),
                         naeg_alt.to_dataset(name='snow_naeg_alt')])
        data['cmask'] = csmask

        snowprob_new = snowprob_exist.update(data)
        snowprob_exist.close()

        snowprob_new.to_netcdf(g.get_filepath('snowprob'))
