.. glaciersat documentation master file, created by
   sphinx-quickstart on Wed Oct 21 18:21:51 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to glaciersat - monitoring glaciers from space!
=======================================================

Glaciersat is a modular system that is made to make the retrieval and processing of satellite data for glacier monitoring easy.

Glaciersat has been initiated as a sister project of `CRAMPON - Cryospheric Monitoring and Prediction Online <https://crampon.readthedocs.org/>`_.
It mainly serves as supplier for obervational data within CRAMPON, which can then be assimilated in the mass balance models.
To view our latest updates on the state of glaciers in Switzerland , go to our `our website <https://crampon.glamos.ch/>`_.



Satellite Data Acquisition
^^^^^^^^^^^^^^^^^^^^^^^^^^


* :doc:`introduction`


.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Satellite Data Acquisition

    introduction.rst


Preprocessing
^^^^^^^^^^^^^

Cropping etc

* :doc:`preprocessing`


.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Preprocessing

    preprocessing.rst


Optical satellite data
^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`albedo`
* :doc:`snowlines`
* :doc:`opt-uncertainties`


.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Optical satellite data

    albedo.rst
    snowlines.rst
    opt-uncertainties.rst


SAR data
^^^^^^^^



* :doc:`melt-area`
* :doc:`sar-uncertainties`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: SAR data

    melt-area.rst
    sar-uncertainties.rst


Use glaciersat
^^^^^^^^^^^^^^

Glaciersat is primarily designed for application to glaciers.
However, it can in principle also be applied to the monitoring of other kinds of geometric objects.
Here you can find an overview of what glaciersat is capable of.

* :doc:`api`
* :doc:`data-sources`
* :doc:`pitfalls`
* :doc:`whats-new`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Use glaciersat

    api.rst
    data-sources.rst
    pitfalls.rst
    whats-new.rst

About
-----
    Most of glaciersat has been implemented by `Johannes Landmann <https://vaw.ethz.ch/personen/person-detail.html?persid=234293>`_
