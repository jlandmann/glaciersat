#############
API Reference
#############

.. currentmodule:: glaciersat

This page provides an automatically generated documentation for glacierat's API. Not all functions are listed here, but hopefully all useful elements.


Image Interfaces
================

Classes for different scenes from differents platforms are implemented here. For a more efficient processing, we also allow to load only metadata.

.. currentmodule:: glaciersat.core.imagery
.. autosummary::
    :toctree: generated/
    :nosignatures:

    SatelliteImageMeta
    SatelliteImage
    S2Image

Albedo methods
==============

These methods can calculate albedo from reflectance.

.. currentmodule:: glaciersat.core.albedo
.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_broadband_albedo_knap
    get_broadband_albedo_liang
    get_broadband_albedo_bonafoni
    get_proxy_albedo_mccarthy
    get_ensemble_albedo


Snow line methods
=================

Here we offer a bunch of tools to calculate snow lines in different ways without manual intervention.

.. currentmodule:: glaciersat.core.snowlines
.. autosummary::
    :toctree: generated/
    :nosignatures:

    map_snow_asmag
    map_snow_naegeli
    map_snow_naegeli_alternate

GIS methods
===========

Here are some arbitrary GIS functions.

.. currentmodule:: glaciersat.core.gis
.. autosummary::
    :toctree: generated/
    :nosignatures:

    crop_sat_image_to_glacier