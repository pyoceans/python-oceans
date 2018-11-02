import numpy as np


def in_polygon(xp, yp, polygon, transform=None, radius=0.0):
    """
    Check is points `xp` and `yp` are inside the `polygon`.
    Polygon is a `matplotlib.path.Path` object.

    https://stackoverflow.com/questions/21328854/shapely-and-matplotlib-point-in-polygon-not-accurate-with-geolocation

    Examples
    --------
    >>> from matplotlib.path import Path
    >>> polygon = Path([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    >>> x1, y1 = 0.5, 0.5
    >>> x2, y2 = 1, 1
    >>> x3, y3 = 0, 1.5
    >>> in_polygon([x1, x2, x3], [y1, y2, y3], polygon)
    array([ True, False, False])

    """
    xp, yp = map(np.atleast_1d, (xp, yp))
    points = np.atleast_2d([xp, yp]).T
    return polygon.contains_points(points, transform=None, radius=0.0)


def gamma_G_north_atlantic(SP, pt):
    """
    Polynomials definitions: North Atlantic. VERSION 1: WOCE dataset.

    """
    Fit = np.array([[0., 0., 0.868250629754601], [1., 0., 4.40022403081395],
                    [0., 1., 0.0324341891674178], [2., 0., -6.45929201288070],
                    [1., 1., -9.92256348514822], [0., 2., 1.72145961018658],
                    [3., 0., -19.3531532033683], [2., 1., 66.9856908160296],
                    [1., 2., -12.9562244122766], [0., 3., -3.47469967954487],
                    [4., 0., 66.0796772714637], [3., 1., -125.546334295077],
                    [2., 2., 7.73752363817384], [1., 3., 10.1143932959310],
                    [0., 4., 5.56029166412630], [5., 0., -54.5838313094697],
                    [4., 1., 70.6874394242861], [3., 2., 36.2272244269615],
                    [2., 3., -26.0173602458275], [1., 4., -0.868664167905995],
                    [0., 5., -3.84846537069737], [6., 0., 10.8620520589394],
                    [5., 1., 0.189417034623553], [4., 2., -36.2275575056843],
                    [3., 3., 22.6867313196590], [2., 4., -8.16468531808416],
                    [1., 5., 5.58313794099231], [0., 6., -0.156149127884621]
                    ])

    gamma_NAtl = Fit[0, 2] * np.ones_like(SP)
    for k in range(1, len(Fit)):
        i = Fit[k, 0]
        j = Fit[k, 1]
        gamma_NAtl = gamma_NAtl + Fit[k, 2] * (SP ** i * pt ** j)

    return gamma_NAtl


def gamma_G_south_atlantic(SP, pt):
    """
    Polynomials definitions: South Atlantic. VERSION 1: WOCE dataset.

    """
    Fit = np.array([[0., 0., 0.970176813506429], [1., 0., 0.755382324920216],
                    [0., 1., 0.270391840513646], [2., 0., 10.0570534575124],
                    [1., 1., -3.30869686476731], [0., 2., -0.702511207122356],
                    [3., 0., -29.0124086439839], [2., 1., -3.60728647124795],
                    [1., 2., 10.6725319826530], [0., 3., -0.342569734311159],
                    [4., 0., 22.1708651635369], [3., 1., 61.1208402591733],
                    [2., 2., -61.0511562956348], [1., 3., 14.6648969886981],
                    [0., 4., -3.14312850717262], [5., 0., 13.0718524535924],
                    [4., 1., -106.892619745231], [3., 2., 74.4131690710915],
                    [2., 3., 5.18263256656924], [1., 4., -12.1368518101468],
                    [0., 5., 2.73778893334855], [6., 0., -15.8634717978759],
                    [5., 1., 51.7078062701412], [4., 2., -15.8597461367756],
                    [3., 3., -35.0297276945571], [2., 4., 28.7899447141466],
                    [1., 5., -8.73093192235768], [0., 6., 1.25587481738340]
                    ])

    gamma_SAtl = Fit[0, 2] * np.ones_like(SP)

    for k in range(1, len(Fit)):
        i = Fit[k, 0]
        j = Fit[k, 1]
        gamma_SAtl = gamma_SAtl + Fit[k, 2] * (SP ** i * pt ** j)

    return gamma_SAtl


def gamma_G_pacific(SP, pt):
    """
    Polynomials definitions: Pacific. VERSION 1: WOCE_dataset.

    """
    Fit = np.array([[0., 0., 0.990419160678528], [1., 0., 1.10691302482411],
                    [0., 1., 0.0545075600726227], [2., 0., 5.48298954708578],
                    [1., 1., -1.81027781763969], [0., 2., 0.673362062889351],
                    [3., 0., -9.59966716439147], [2., 1., -11.1211267642241],
                    [1., 2., 6.94431859780735], [0., 3., -3.35534931941803],
                    [4., 0., -15.7911318241728], [3., 1., 86.4094941684553],
                    [2., 2., -63.9113580983532], [1., 3., 23.1248810527697],
                    [0., 4., -1.19356232779481], [5., 0., 48.3336456682489],
                    [4., 1., -145.889251358860], [3., 2., 95.6825154064427],
                    [2., 3., -8.43447476300482], [1., 4., -16.0450914593959],
                    [0., 5., 3.51016478240624], [6., 0., -28.5141488621899],
                    [5., 1., 72.6259160928028], [4., 2., -34.7983038993856],
                    [3., 3., -21.9219942747555], [2., 4., 25.1352444814321],
                    [1., 5., -5.58077135773059], [0., 6., 0.0505878919989799]
                    ])

    gamma_Pac = Fit[0, 2] * np.ones_like(SP)

    for k in range(1, len(Fit)):
        i = Fit[k, 0]
        j = Fit[k, 1]
        gamma_Pac = gamma_Pac + Fit[k, 2] * (SP ** i * pt ** j)

    return gamma_Pac


def gamma_G_indian(SP, pt):
    """
    Polynomials definitions: Indian. VERSION 1: WOCE_dataset.

    """
    Fit = np.array([[0., 0., 0.915127744449523], [1., 0., 2.52567287174508],
                    [0., 1., 0.276709571734987], [2., 0., -0.531583207697361],
                    [1., 1., -5.95006196623071], [0., 2., -1.29591003712053],
                    [3., 0., -6.52652369460365], [2., 1., 23.8940719644002],
                    [1., 2., -0.628267986663373], [0., 3., 3.75322031850245],
                    [4., 0., 1.92080379786486], [3., 1., 0.341647815015304],
                    [2., 2., -39.2270069641610], [1., 3., 14.5023693075710],
                    [0., 4., -5.64931439477443], [5., 0., 20.3803121236886],
                    [4., 1., -64.7046763005989], [3., 2., 88.0985881844501],
                    [2., 3., -30.0525851211887], [1., 4., 4.04000477318118],
                    [0., 5., 0.738499368804742], [6., 0., -16.6137493655149],
                    [5., 1., 46.5646683140094], [4., 2., -43.1528176185231],
                    [3., 3., 0.754772283610568], [2., 4., 13.2992863063285],
                    [1., 5., -6.93690276392252], [0., 6., 1.42081034484842]
                    ])

    gamma_Ind = Fit[0, 2] * np.ones_like(SP)

    for k in range(1, len(Fit)):
        i = Fit[k, 0]
        j = Fit[k, 1]
        gamma_Ind = gamma_Ind + Fit[k, 2] * (SP ** i * pt ** j)

    return gamma_Ind


def gamma_G_southern_ocean(SP, pt, p):
    """
    Polynomials definitions: Southern Ocean. VERSION 1: WOCE_dataset.

    """
    Fit_N = np.array([[0., 0., 0.874520046342081], [1., 0., -1.64820627969497],
                      [0., 1., 2.05462556912973], [2., 0., 28.0996269467290],
                      [1., 1., -8.27848721520081], [0., 2., -9.03290825881587],
                      [3., 0., -91.0872821653811], [2., 1., 34.8904015133508],
                      [1., 2., 0.949958161544143], [0., 3., 21.4780019724540],
                      [4., 0., 133.921771803702], [3., 1., -50.0511970208864],
                      [2., 2., -4.44794543753654], [1., 3., -11.7794732139941],
                      [0., 4., -21.0132492641922], [5., 0., -85.1619212879463],
                      [4., 1., 7.85544471116596], [3., 2., 44.5061015983665],
                      [2., 3., -32.9544488911897], [1., 4., 31.2611766088444],
                      [0., 5., 4.26251346968625], [6., 0., 17.2136374200161],
                      [5., 1., 13.4683704071999], [4., 2., -27.7122792678779],
                      [3., 3., 11.9380310360096], [2., 4., 1.95823443401631],
                      [1., 5., -10.8585153444218], [0., 6., 1.44257249650877]
                      ])

    gamma_SOce_N = Fit_N[0, 2] * np.ones_like(SP)

    for k in range(1, len(Fit_N)):
        i = Fit_N[k, 0]
        j = Fit_N[k, 1]
        gamma_SOce_N = gamma_SOce_N + Fit_N[k, 2] * (SP ** i * pt ** j)

    Fit_S = np.array([[0., 0., 0.209190309846492], [1., 0., -1.92636557096894],
                      [0., 1., -3.06518655463115], [2., 0., 9.06344944916046],
                      [1., 1., 2.96183396117389], [0., 2., 39.0265896421229],
                      [3., 0., -15.3989635056620], [2., 1., 3.87221350781949],
                      [1., 2., -53.6710556192301], [0., 3., -215.306225218700],
                      [4., 0., 8.31163564170743], [3., 1., -3.14460332260582],
                      [2., 2., 3.68258441217306], [1., 3., 264.211505260770],
                      [0., 4., 20.1983279379898]
                      ])

    p_ref, pt_ref, c_pt = 700., 2.5, 0.65

    gamma_A = Fit_S[0, 2] * np.ones_like(SP)

    for k in range(1, len(Fit_S)):
        i = Fit_S[k, 0]
        j = Fit_S[k, 1]
        gamma_A = gamma_A + Fit_S[k, 2] * (SP ** i * pt ** j)

    gamma_SOce_S = (gamma_A * np.exp(-p / p_ref) *
                    (1. / 2. - 1. / 2. * np.tanh((40. * pt - pt_ref) / c_pt)))

    gamma_SOce = gamma_SOce_N + gamma_SOce_S
    return gamma_SOce


def gamma_GP_from_SP_pt(SP, pt, p, lon, lat):
    """
    Global Polynomial of Neutral Density with respect to Practical Salinity
    and potential temperature.

    Calculates the Global Polynomial of Neutral Density gammma_GP using an
    approximate form gamma_poly of Neutral Density on each oceanic basin:
    North Atlantic, South Atlantic, Pacific, Indian and Southern Ocean.  Each
    function is a polynomial of 28 terms, function of Practical Salinity,
    potential temperature.  The function on the Southern Ocean contains another
    function which is a polynomial of 15 terms of Practical Salinity and
    potential temperature times by a pressure and a potential temperature terms
    which is effective close to Antarctica in shallow waters.  The polynomials
    on each ocean basins are combined to form the Global Polynomial using tests
    in latitude and longitude to recognize the basin and weighting functions to
    make the functions zip together where the oceanic basins communicate.

    Parameter
    ---------
    SP : array_like
         Practical salinity [psu]
    pt : array_like
         Potential temperature [ITS-90 deg C]
    p : array_like
        Sea pressure [dbar] (i.e. absolute pressure - 10.1325 dbar)
    lon : number
          Longitude [0-360]
    lat : latitude

    Returns
    -------
    gamma_GP : array
               Global Polynomial of Neutral Density with [kg/m^3] respect to
               Practical Salinity and potential temperature.

    Examples
    --------
    >>> from oceans.sw_extras.gamma_GP_from_SP_pt import gamma_GP_from_SP_pt
    >>> SP = [35.066, 35.086, 35.089, 35.078, 35.025, 34.851, 34.696, 34.572,
    ...      34.531, 34.509, 34.496, 34.452, 34.458, 34.456, 34.488, 34.536,
    ...      34.579, 34.612, 34.642, 34.657, 34.685, 34.707, 34.72, 34.729]
    >>> pt = [12.25, 12.21, 12.09, 11.99, 11.69, 10.54, 9.35, 8.36, 7.86, 7.43,
    ...       6.87, 6.04, 5.5, 4.9, 4.04, 3.29, 2.78, 2.45, 2.211, 2.011,
    ...       1.894, 1.788, 1.554, 1.38]
    >>> p = [1.0, 48.0, 97.0, 145.0, 194.0, 291.0, 388.0, 485.0, 581.0, 678.0,
    ...      775.0, 872.0, 969.0, 1066.0, 1260.0, 1454.0, 1647.0, 1841.0,
    ...      2020.0, 2216.0, 2413.0, 2611.0, 2878.0, 3000.0]
    >>> lon, lat, n = [187.317, -41.6667, 24]
    >>> gamma_GP_from_SP_pt(SP, pt, p, lon, lat)
    array([26.66339976, 26.68613362, 26.71169809, 26.72286813, 26.74102625,
           26.82472769, 26.91707848, 26.9874849 , 27.03564777, 27.08512861,
           27.15880197, 27.24506111, 27.32438575, 27.40418818, 27.54227885,
           27.67691837, 27.77693976, 27.84683646, 27.90297626, 27.9428694 ,
           27.98107846, 28.01323277, 28.05769996, 28.09071215])

    Author
    ------
    Guillaume Serazin, Paul Barker & Trevor McDougall   [help@teos-10.org]

    VERSION NUMBER: 1.0 (27th October, 2011)

    """
    from matplotlib.path import Path

    SP, pt, p, lon, lat = list(map(np.asanyarray, (SP, pt, p, lon, lat)))
    SP, pt, p, lon, lat = np.broadcast_arrays(SP, pt, p, lon, lat)

    # Normalization of the variables.
    SP = SP / 42.
    pt = pt / 40.

    # Computation of the polynomials on each oceanic basin.
    gamma_NAtl = gamma_G_north_atlantic(SP, pt)
    gamma_SAtl = gamma_G_south_atlantic(SP, pt)
    gamma_Pac = gamma_G_pacific(SP, pt)
    gamma_Ind = gamma_G_indian(SP, pt)
    gamma_SOce = gamma_G_southern_ocean(SP, pt, p)
    # gamma_Arc = np.zeros_like(SP) * np.NaN

    # Definition of the Indian part.
    io_lon = np.array([100, 100, 55, 22, 22, 146, 146, 133.9, 126.94, 123.62,
                       120.92, 117.42, 114.11, 107.79, 102.57, 102.57, 98.79,
                       100])

    io_lat = np.array([20, 40, 40, 20, -90, -90, -41, -12.48, -8.58, -8.39,
                       -8.7, -8.82, -8.02, -7.04, -3.784, 2.9, 10, 20])

    # Definition of the Pacific part.
    po_lon = np.array([100, 140, 240, 260, 272.59, 276.5, 278.65, 280.73,
                       295.217, 290, 300, 294, 290, 146, 146, 133.9, 126.94,
                       123.62, 120.92, 117.42, 114.11, 107.79, 102.57, 102.57,
                       98.79, 100.])

    po_lat = np.array([20, 66, 66, 19.55, 13.97, 9.6, 8.1, 9.33, 0, -52.,
                       -64.5, -67.5, -90, -90, -41, -12.48, -8.58, -8.39,
                       -8.7, -8.82, -8.02, -7.04, -3.784, 2.9, 10, 20])

    # Definition of the polygon filters.
    io_polygon = Path(list(zip(io_lon, io_lat)))
    po_polygon = Path(list(zip(po_lon, po_lat)))
    i_inter_indian_pacific = (in_polygon(lon, lat, io_polygon) *
                              in_polygon(lon, lat, po_polygon))

    i_indian = np.logical_xor(in_polygon(lon, lat, io_polygon), i_inter_indian_pacific)
    i_pacific = in_polygon(lon, lat, po_polygon)
    i_atlantic = (1 - i_pacific) * (1 - i_indian)

    # Definition of the Atlantic weighting function.
    charac1_sa = lat < -10.
    charac2_sa = np.logical_and(lat <= 10., lat >= -10.)
    w_sa = charac1_sa + charac2_sa * (0.5 + 0.5 *
                                      np.cos(np.pi * (lat + 10.) / 20.))

    # Definition of the Southern Ocean weighting function.
    charac1_so = lat < -40.
    charac2_so = np.logical_and(lat <= -20., lat >= -40.)
    w_so = charac1_so + charac2_so * (0.5 + 0.5 *
                                      np.cos(np.pi * (lat + 40.) / 20.))

    # Combination of the North and South Atlantic.
    gamma_Atl = (1. - w_sa) * gamma_NAtl + w_sa * gamma_SAtl

    # Combination of the middle parts.
    gamma_middle = (i_pacific * gamma_Pac + i_atlantic * gamma_Atl +
                    i_indian * gamma_Ind)

    # Combination of the Northern and Southern parts.
    gamma_GP = w_so * gamma_SOce + (1. - w_so) * gamma_middle

    # Set NaN in the arctic region.
    gamma_GP[lat > 66.] = np.NaN

    # De-normalization.
    gamma_GP = 20. * gamma_GP - 20

    return gamma_GP
