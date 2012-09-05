# -*- coding: utf-8 -*-
def ftopo(x, y, topofile='/home/andre/home_almada/lado/general_data/gebco/gebco15-40s_30-52w_30seg.nc'):
  """
  Usage
  -----
  H,D,Xo,Yo = ftopo(x, y, topofile='/home/andre/home_almada/lado/general_data/gebco/gebco_08_30seg.nc')

  Description
  -----------
  Finds the depth of points of coordinates 'x','y' using GEBCO data set.

  Parameters
  ----------
  x         : 1D array
              Array containing longitudes of the points of unknown depth.

  y         : 1D array
              Array containing latitudes of the points of unknown depth.

  topofile  : string, optional
              String containing path to the GEBCO netCDF file.

  Returns
  -------
  H         : 1D array
              Array containing depths of points closest to the input X,Y coordinates.

  X         : 1D array
              Array of horizontal distance associated with array 'H'.

  D         : 1D array
              Array containing distances (in km) from the input X,Y coordinates to the data points.

  Xo        : 1D array
              Array containing longitudes of the data points.

  Yo        : 1D array
              Array containing latitudes of the data points.

  NOTES
  -------
  This function reads the entire netCDF file before extracting the wanted data.
  Therefore, it does not handle the full GEBCO dataset (1.8 GB) efficiently.

  TODO
  -------
  Make it possible to work with the full gebco dataset, by extracting only the wanted indexes.

  Code History
  ---------------------------------------
  Author of the original Matlab code (ftopo.m, ETOPO2 dataset): Marcelo Andrioni <marceloandrioni@yahoo.com.br>
  December 2008: Modification performed by Cesar Rocha <cesar.rocha@usp.br> to handle ETOPO1 dataset.
  July 2012: Python Translation and modifications performed by André Palóczy Filho <paloczy@gmail.com>
  to handle GEBCO dataset (30 arc seconds resolution).
  ======================================================================================================================
  """
  from numpy import asanyarray,arange,meshgrid,reshape,flipud,sqrt,append,abs,cumsum
  from seawater.csiro import dist
  from netCDF4 import Dataset

  def near(x, x0):
    """Function near(x, x0) Given an 1D array x and a scalar x0,
    returns the index of the element of x closest to x0."""  
    nearest_value_idx = (abs(x-x0)).argmin()
    return nearest_value_idx
  #----------------------------------------------------------------------------------------------#
  (x,y) = map(asanyarray,(x,y))

  ###############################################
  ### Opening netCDF file and extracting data ###
  ###############################################
  grid = Dataset(topofile)
  yyr = grid.variables['y_range'][:]
  xxr = grid.variables['x_range'][:]
  spacing = grid.variables['spacing'][:]
  dx, dy = spacing[0], spacing[1]
  ### Creating lon and lat 1D arrays ###
  xx = arange(xxr[0], xxr[1], dx); xx = xx + dx/2
  yy = arange(yyr[0], yyr[1], dy); yy = yy + dy/2
  h = grid.variables['z'][:]
  grid.close()

  ##########################################################
  ### Retrieving nearest point for each input coordinate ###
  ##########################################################
  A = asanyarray([])
  xx, yy = meshgrid(xx, yy)
  ni, nj = xx.shape[0], yy.shape[1]
  h = reshape(h, (ni,nj))
  h = flipud(h)
  Xo = A.copy()
  Yo = A.copy()
  H = A.copy()
  D = A.copy()
  for I in xrange(x.size):
    ix = near(xx[0,:], x[I])
    iy = near(yy[:,0], y[I])
    H = append(H, h[iy,ix])
    D = append(D, dist([x[I], xx[0,ix]],[y[I], yy[iy,0]],units='km')[0]) # calculating distance between input and GEBCO points
    Xo = append(Xo, xx[0,ix])
    Yo = append(Yo, yy[iy,0])
  X = append(0, cumsum(dist(Xo,Yo,units='km')[0])) # calculating distance axis

  return H,X,D,Xo,Yo
