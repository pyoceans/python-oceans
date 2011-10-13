===============
Python Seawater
===============

This python package contains a python re-write of two MatlabTM Toolboxes and some
extra fun functions. Typical usage often looks like this::

    >>> import seawater.gibbs as gsw
    >>> gsw.SA_from_SP(SP, p, lon, lat)

    >>> import seawater.csiro as sw
    >>> sw.test()

Main modules
============

gibbs
-----

The `Gibbs Sea Water <http://www.teos-10.org/software.htm>`_ (GSW v2.0).

A oceanographic toolbox of the International Thermodynamic Equation of
Seawater - 2010 or TEOS-10.

Contains the functions for evaluating the thermodynamic properties of pure water
(using IAPWS-09) and seawater (using IAPWS-08 for the saline part).

csiro
-----

The `CSIRO seawater toolbox <http://www.cmar.csiro.au/datacentre/ext_docs/seawater.htm>`_
(SEAWATER-3.3) for calculating the properties of sea water. The package uses the
formulas from Unesco's joint panel on oceanographic tables and standards, UNESCO
1981 and UNESCO 1983 (EOS-80) .

The EOS-80 library is considered now obsolete; it is provided here for
compatibility with old scripts, and to allow a smooth transition to the
new TEOS-10.

extra
-----

Just some extra functions and a wave calculator.

gibbs vs. csiro
^^^^^^^^^^^^^^^

.. role:: raw-math(raw)
    :format: latex html

This table shows some function names in the giibs library and the corresponding function names in the csiro library.

+---------------------------------------+-------------------------------+-------------------------------------------------------+
| **Variable**                          | **SeaWater & ESO‐80**         | **Gibbs‐SeaWater (GSW) & TEOS‐10**                    |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| Absolute Salinity                     |          NA                   | gsw.SA_from_SP(SP,p,long,lat)                         |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| Conservative Temperature              |          NA                   | gsw.CT_from_t(SA,t,p)                                 |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| density (i.e. in situ density)        |  sw.dens(SP,t,p)              | gsw.rho_CT(SA,CT,p), or gsw.rho(SA,t,p), or           |
|                                       |                               | gsw.rho_CT25(SA,CT,p)                                 |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| potential density                     |  sw.pden(SP,t,p,pr)           | gsw.rho_CT(SA,CT,pr), or                              |
|                                       |                               | gsw.rho_CT25(SA,CT,pr)                                |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| potential temperature                 |  sw.ptmp(SP,t,p,pr)           | gsw.pt_from_t(SA,t,p,pr)                              |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| $\sigma_0$, using                     |  sw.dens(SP, $\theta_o$, 0)   | gsw.sigma0_CT(SA,CT)                                  |
|  $\theta_o$ = sw.ptmp(SP,t,p,0)       |  -1000 kg m :sup:`-3`         |                                                       |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| $\sigma_2$, using                     |  sw.dens(SP,$\theta_2$, 2000) | gsw.sigma2_CT(SA,CT)                                  |
|  $\theta_2$ = sw.ptmp(SP,t,p,2000)    |  -1000 kg m :sup:`-3`         |                                                       |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| $\sigma_4$, using                     |  sw.dens(SP,$\theta_4$, 4000) | gsw.sigma2_CT(SA,CT)                                  |
|  $\theta_4$ = sw.ptmp(SP,t,p,2000)    |  -1000 kg m :sup:`-3`         |                                                       |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| specific volume anomaly               |  sw.svan(SP,t,p)              | gsw.specvol_anom_CT(SA,CT,p)  or                      |
|                                       |                               | gsw.specvol_anom_CT25(SA,CT,p)                        |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| dynamic height anomaly                | -sw.gpan(SP,t,p)              | gsw.geo_strf_dyn_height(SA,CT,p,delta_p,interp_style) |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| geostrophic velocity                  |  sw.gvel(ga,lat,long)         | gsw.geostrophic_velocity(geo_str,long,lat,p)          |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| N :sup:`2`                            |  sw.bfrq(SP,t,p,lat)          | gsw.Nsquared_CT25(SA,CT,p,lat)                        |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| pressure from height                  |  sw.pres(-z,lat)              | gsw.p_from_z(z,lat)                                   |
| (SW uses depth, not height)           |                               |                                                       |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| height from pressure                  |  z =  -sw.dpth(p,lat)         | gsw.z_from_p(p,lat)                                   |
| (SW outputs depth, not height)        |                               |                                                       |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| in situ temperature from pt           |  sw.temp(SP,pt,p,pr)          | gsw.pt_from_t(SA,pt,pr,p)                             |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| sound speed                           |  sw.svel(SP,t,p)              | gsw.sound_speed(SA,t,p)                               |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| isobaric heat capacity                |  sw.cp(SP,t,p)                | gsw.cp(SA,t,p)                                        |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| adiabatic lapse rate*                 |  sw.adtg(SP,t,p)              | gsw.adiabatic_lapse_rate(SA,t,p)                      |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| SP from cndr,  (PSS‐78)               |  sw.salt(cndr,t,p)            | gsw.SP_from_cndr(cndr,t,p)                            |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| cndr from SP,  (PSS‐78)               |  sw.cndr(SP,t,p)              | gsw.cndr_from_SP(SP,t,p)                              |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| distance                              |  sw.dist(lat,long,units)      | gsw.distance(long,lat,p)                              |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| gravitational acceleration            |  sw.g(lat,z)                  | gsw.grav(lat,p)                                       |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| Coriolis parameter                    |  sw.f(lat)                    | gsw.f(lat)                                            |
+---------------------------------------+-------------------------------+-------------------------------------------------------+
| testing of all functions              |  sw.test()                    | gsw.test()                                            |
+---------------------------------------+-------------------------------+-------------------------------------------------------+

\* The SW and GSW functions output the adiabatic lapse rate in different units, being  K (dbar) :sup:`-1`  and  K Pa :sup:`-1`  respectively.


Thanks
======

* Bjørn Ådlandsvik - Testing unit and several bug fixes
* Eric Firing - Support for masked arrays, re-write of _delta_SA
* Trevor J. McDougall (and all of SCOR/IAPSO WG127) for making available the Matlab and Fortran versions of this software

Acknowledgments
---------------

* SCOR/IAPSO WG127. Most of the gibbs module is derived from the GSW Oceanographic Toolbox of TEOS-10.

* CSIRO. The csiro.py is a re-write of their seawater matlab toolbox.


More information:
    http://pypi.python.org/pypi/seawater/

Note on versioning:

The MAJOR.MINOR.MICRO will be used to represent:

MAJOR == The matlab version from the TEOS-10 Group

MINOR == Significant changes made in the python version

MICRO == Bug fixes only
