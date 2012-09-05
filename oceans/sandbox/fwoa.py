# -*- coding: utf-8 -*-
def fwoa(x, y, woafile='/home/andre/lado/general_data/woa2009/woa2009_annual.mat'):
  """
  Usage
  -----
  T,S,X,D,Xo,Yo = fwoa(x, y, woafile='/home/andre/lado/general_data/woa2009/woa2009_annual.mat')

  Description
  -----------
  Gets the TS profiles in the World Ocean Atlas (WOA) 2009 data set whose coordinates
  are closest to the input coordinates 'x','y'.

  Parameters
  ----------
  x         : 1D array
              Array containing longitudes of the points of unknown depth.

  y         : 1D array
              Array containing latitudes of the points of unknown depth.

  woafile   : string, optional
              String containing path to the WOA .mat file.

  Returns
  -------
  T         : 2D array
              Array containing the Temperature profiles closest to the input X,Y coordinates.

  S         : 2D array
              Array containing the Salinity (PSS-78) profiles closest to the input X,Y coordinates.

  X         : 1D array
              Array of horizontal distance associated with the TS profiles recovered.

  D         : 1D array
              Array containing distances (in km) from the input X,Y coordinates to the TS profiles.

  Xo        : 1D array
              Array containing longitudes of the WOA TS profiles.

  Yo        : 1D array
              Array containing latitudes of the WOA TS profiles.

  NOTES
  -------
  This function only reads .mat files, converted from the original netCDF ones.

  TODO
  -------
  Implement netCDF file reading (Original WOA 2009 format)
  Implement option to retrieve linearly interpolated profiles instead of nearest ones
  ======================================================================================================================
  """
  from numpy import asanyarray,arange,append,abs,cumsum,nan,ones
  from scipy.io import loadmat
  from seawater.csiro import dist

  def near(x, x0):
    """Function near(x, x0) Given an 1D array x and a scalar x0,
    returns the index of the element of x closest to x0."""  
    nearest_value_idx = (abs(x-x0)).argmin()
    return nearest_value_idx
  #----------------------------------------------------------------------------------------------#
  x,y = map(asanyarray, (x,y))

  #########################
  ### Reading .mat file ###
  #########################
  d = loadmat(woafile)
  xx = d['lon']
  yy = d['lat']
  TT = d['temp']
  SS = d['salt']

  #############################################################
  ### Retrieving nearest profiles for each input coordinate ###
  #############################################################
  A = asanyarray([])
  B = nan*ones((TT.shape[2],x.size))
  Xo = A.copy()
  Yo = A.copy()
  D = A.copy()
  T = B.copy()
  S = B.copy()
  for I in xrange(x.size):
    ix = near(xx[0,:], x[I])
    iy = near(yy[:,0], y[I])
    T[:,I] = TT[iy,ix,:]
    S[:,I] = SS[iy,ix,:]
    D = append(D, dist([x[I], xx[0,ix]],[y[I], yy[iy,0]],units='km')[0]) # calculating distance between input and nearest WOA points
    Xo = append(Xo, xx[0,ix])
    Yo = append(Yo, yy[iy,0])
  X = append(0, cumsum(dist(Xo,Yo,units='km')[0])) # calculating distance axis

  return T,S,X,D,Xo,Yo