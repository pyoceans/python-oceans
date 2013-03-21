# -*- coding: utf-8 -*-

# Only used on the test routine.
import sys
from platform import uname
from time import asctime, localtime

from csiro import *


def test(fileout='python-test.txt'):
    r"""Copy of the Matlab test.

    Execute test routines to verify SEAWATER Library routines for your
    platform. Prints output to file.

    Notes
    ------
    This is only to reproduce sw_test.m from the original. A better more
    complete test is performed via doctest.

    Modifications: Phil Morgan
                   03-12-12. Lindsay Pender, Converted to ITS-90.
                   10-01-14. Filipe Fernandes, Python translation.
    """
    f = open(fileout, 'w')

    print >>f, '**********************************************'
    print >>f, '    TEST REPORT    '
    print >>f, ''
    print >>f, ' SEA WATER LIBRARY ', __version__
    print >>f, ''
    print >>f, ''
    # Show some info about this Python
    print >>f, 'python version:', sys.version
    print >>f, ' on ', uname()[0], uname()[-1], ' computer'
    print >>f, ''
    print >>f,  asctime(localtime())
    print >>f, '**********************************************'
    print >>f, ''

    # Test MAIN MODULE  ptmp.
    module = 'ptmp'
    submodules = 'adtg'

    print >>f, '*************************************'
    print >>f, '**  TESTING MODULE: ', module
    print >>f, '**  and SUB-MODULE: ', submodules
    print >>f, '*************************************'

    # test 1 - data from Unesco 1983 p45
    T = np.array([[0,  0,  0,  0,  0,  0],
            [10, 10, 10, 10, 10, 10],
            [20, 20, 20, 20, 20, 20],
            [30, 30, 30, 30, 30, 30],
            [40, 40, 40, 40, 40, 40]])

    T = T / 1.00024

    S = np.array([[25, 25, 25, 35, 35, 35],
            [25, 25, 25, 35, 35, 35],
            [25, 25, 25, 35, 35, 35],
            [25, 25, 25, 35, 35, 35],
            [25, 25, 25, 35, 35, 35]])

    P = np.array([[0, 5000, 10000, 0, 5000, 10000],
            [0, 5000, 10000, 0, 5000, 10000],
            [0, 5000, 10000, 0, 5000, 10000],
            [0, 5000, 10000, 0, 5000, 10000],
            [0, 5000, 10000, 0, 5000, 10000]])

    Pr = np.array([0, 0, 0, 0, 0, 0])

    UN_ptmp = np.array([[0, -0.3061, -0.9667,  0, -0.3856, -1.0974],
                    [10,  9.3531,  8.4684, 10,  9.2906,  8.3643],
                    [20, 19.0438, 17.9426, 20, 18.9985, 17.8654],
                    [30, 28.7512, 27.4353, 30, 28.7231, 27.3851],
                    [40, 38.4607, 36.9254, 40, 38.4498, 36.9023]])

    PT = ptmp(S, T, P, Pr) * 1.00024

    # DISPLAY RESULTS
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, 'Comparison of accepted values from UNESCO 1983 '
    print >>f, ' (Unesco Tech. Paper in Marine Sci. No. 44, p45)'
    print >>f, '********************************************************'

    m, n = S.shape  # TODO: so many loops there must be a better way.
    for icol in range(0, n):
        print >>f, '   Sal  Temp  Press     PTMP       ptmp'
        print >>f, '  (psu)  (C)   (db)     (C)          (C)'
        result = np.vstack((S[:, icol], T[:, icol], P[:, icol],
        UN_ptmp[:, icol], PT[:, icol]))
        for iline in range(0, m):
            print >>f, " %4.0f  %4.0f   %5.0f   %8.4f  %11.5f" % tuple(result[:, iline])

        print >>f, ''

    # test MAIN MODULE  svan
    module     = 'svan'
    submodules = 'dens dens0 smow seck pden ptmp'

    print >>f, '*************************************'
    print >>f, '**  TESTING MODULE: ', module
    print >>f, '**  and SUB-MODULE: ', submodules
    print >>f, '*************************************'

    # test DATA FROM: Unesco Tech. Paper in Marine Sci. No. 44, p22
    s = np.array([0,     0,  0,     0, 35,    35, 35,   35])
    p = np.array([0, 10000,  0, 10000,  0, 10000,  0, 10000])
    t = np.array([0,     0, 30,    30,  0,     0, 30,    30]) / 1.00024

    UN_svan = np.array([2749.54, 2288.61, 3170.58, 3147.85,
                        0.0,    0.00,  607.14,  916.34])

    SVAN    = svan(s, t, p)

    # DISPLAY RESULTS
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, 'Comparison of accepted values from UNESCO 1983'
    print >>f, ' (Unesco Tech. Paper in Marine Sci. No. 44, p22)'
    print >>f, '********************************************************'
    print >>f, ''
    print >>f, '   Sal  Temp  Press        SVAN        svan'
    print >>f, '  (psu)  (C)   (db)    (1e-8*m3/kg)  (1e-8*m3/kg)'
    result = np.vstack([s, t, p, UN_svan, 1e+8*SVAN])
    for iline in range( 0, len(SVAN) ):
        print >>f,  " %4.0f  %4.0f   %5.0f   %11.2f    %11.3f" % tuple(result[:,iline])

    # test MAIN MODULE salt
    module     = 'salt'
    submodules = 'salrt salrp sals'
    print >>f, '*************************************'
    print >>f, '**  TESTING MODULE: ', module
    print >>f, '**  and SUB-MODULE: ', submodules
    print >>f, '*************************************'

    # test 1 - data from Unesco 1983 p9
    R    = np.array([  1,       1.2,       0.65]) # cndr = R
    T    = np.array([ 15,        20,          5]) / 1.00024
    P    = np.array([  0,      2000,       1500])
    #Rt   = np.array([  1, 1.0568875, 0.81705885])
    UN_S = np.array([35, 37.245628,  27.995347])

    S    = salt(R, T, P)

    # DISPLAY RESULTS
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, 'Comparison of accepted values from UNESCO 1983 '
    print >>f, '(Unesco Tech. Paper in Marine Sci. No. 44, p9)'
    print >>f, '********************************************************'
    print >>f, ''
    print >>f, '   Temp    Press       R              S           salt'
    print >>f, '   (C)     (db)    (no units)       (psu)          (psu) '
    table = np.vstack([T, P, R, UN_S, S])
    m,n = table.shape
    for iline in range( 0, n ):
        print >>f, " %4.0f       %4.0f  %8.2f      %11.6f  %14.7f" % tuple(table[:,iline])

    # test MAIN MODULE cndr
    module     = 'cndr'
    submodules = 'salds'
    print >>f, '*************************************'
    print >>f, '**  TESTING MODULE: ', module
    print >>f, '**  and SUB-MODULE: ', submodules
    print >>f, '*************************************'

    # test 1 - data from Unesco 1983 p9
    T    = np.array([  0, 10, 0, 10, 10, 30]) / 1.00024
    P    = np.array([  0,  0, 1000, 1000, 0, 0])
    S    = np.array([ 25, 25, 25, 25, 40, 40])
    UN_R = np.array([ 0.498088, 0.654990, 0.506244, 0.662975, 1.000073, 1.529967])
    R    = cndr(S, T, P)

    # DISPLAY RESULTS
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, 'Comparison of accepted values from UNESCO 1983 '
    print >>f, ' (Unesco Tech. Paper in Marine Sci. No. 44, p14)'
    print >>f, '********************************************************'
    print >>f, ''
    print >>f, '   Temp    Press       S            cndr         cndr'
    print >>f, '   (C)     (db)      (psu)        (no units)    (no units) '
    table = np.vstack([T, P, S, UN_R, R])
    m,n = table.shape
    for iline in range( 0, n ):
        print >>f, " %4.0f       %4.0f   %8.6f   %11.6f  %14.8f" % tuple(table[:,iline])

    # test MAIN MODULE depth
    module     = 'depth'
    print >>f, ''
    print >>f, '*************************************'
    print >>f, '**  TESTING MODULE: ', module
    print >>f, '*************************************'

    # test DATA - matrix "pressure", vector "lat"  Unesco 1983 data p30.
    lat = np.array([0, 30, 45, 90])
    P   = np.array([[  500,   500,   500,  500],
                 [ 5000,  5000,  5000, 5000],
                 [10000, 10000, 10000, 10000]])

    UN_dpth = np.array([[  496.65,  496.00,  495.34,  494.03],
                    [ 4915.04, 4908.56, 4902.08, 4889.13],
                    [ 9725.47, 9712.65, 9699.84, 9674.23]])

    dpth = depth(P, lat)

    # DISPLAY RESULTS
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, 'Comparison of accepted values from Unesco 1983 '
    print >>f, '(Unesco Tech. Paper in Marine Sci. No. 44, p28)'
    print >>f, '********************************************************'

    for irow in range(0, 3):
        print >>f, ''
        print >>f, '    Lat       Press     DPTH      dpth'
        print >>f, '  (degree)    (db)     (meter)    (meter)'
        table = np.vstack( [ lat, P[irow,:], UN_dpth[irow,:], dpth[irow,:] ] )
        m,n   = table.shape
        for iline in range(0, n):
            print >>f, "  %6.3f     %6.0f   %8.2f   %8.3f" % tuple(table[:,iline])

    # test MAIN MODULE fp
    module     = 'fp'
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, '**  TESTING MODULE: ', module
    print >>f, '********************************************************'

    # test 1 - UNESCO DATA p.30
    S     = np.array([ [5, 10, 15, 20, 25, 30, 35, 40],
                    [5, 10, 15, 20, 25, 30, 35, 40] ])

    P     = np.array([ [  0,   0,   0,   0,   0,   0,   0,   0],
                    [500, 500, 500, 500, 500, 500, 500, 500] ])


    UN_fp = np.array([[-0.274, -0.542, -0.812, -1.083, -1.358, -1.638, -1.922, -2.212],
                    [-0.650, -0.919, -1.188, -1.460, -1.735, -2.014, -2.299, -2.589] ])

    FP    = fp(S, P)

    # DISPLAY RESULTS
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, 'Comparison of accepted values from UNESCO 1983 '
    print >>f, ' (Unesco Tech. Paper in Marine Sci. No. 44, p30)'
    print >>f, '********************************************************'

    for irow in range(0, 2):
        print >>f, ''
        print >>f, '   Sal   Press      fp        fp'
        print >>f, '  (psu)   (db)      (C)        (C)'
        table = np.vstack( [ S[irow,:], P[irow,:], UN_fp[irow,:], FP[irow,:] ] )
        m,n   = table.shape
        for iline in range( 0, n ):
            print >>f, " %4.0f   %5.0f   %8.3f  %11.4f" % tuple(table[:,iline])

    # test MAIN MODULE cp
    module     = 'cp'
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, '**  TESTING MODULE: ', module
    print >>f, '********************************************************'

    # test 1 -
    # DATA FROM POND AND PICKARD INTRO. DYNAMICAL OCEANOGRAPHY 2ND ED. 1986
    T     = np.array([[ 0,  0,  0,  0,  0,  0],
                   [10, 10, 10, 10, 10, 10],
                   [20, 20, 20, 20, 20, 20],
                   [30, 30, 30, 30, 30, 30],
                   [40, 40, 40, 40, 40, 40]]) / 1.00024

    S     = np.array([[25, 25, 25, 35, 35, 35],
                   [25, 25, 25, 35, 35, 35],
                   [25, 25, 25, 35, 35, 35],
                   [25, 25, 25, 35, 35, 35],
                   [25, 25, 25, 35, 35, 35]])

    P     = np.array([[0, 5000, 10000, 0, 5000, 10000],
                   [0, 5000, 10000, 0, 5000, 10000],
                   [0, 5000, 10000, 0, 5000, 10000],
                   [0, 5000, 10000, 0, 5000, 10000],
                   [0, 5000, 10000, 0, 5000, 10000]])

    UN_cp = np.array([[4048.4,  3896.3,  3807.7,  3986.5,  3849.3,  3769.1],
                   [4041.8,  3919.6,  3842.3,  3986.3,  3874.7,  3804.4],
                   [4044.8,  3938.6,  3866.7,  3993.9,  3895.0,  3828.3],
                   [4049.1,  3952.0,  3883.0,  4000.7,  3909.2,  3844.3],
                   [4051.2,  3966.1,  3905.9,  4003.5,  3923.9,  3868.3]])

    CP    = cp(S, T, P)

    # DISPLAY RESULTS
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, 'Comparison of accepted values from UNESCO 1983 '
    print >>f, ' (Unesco Tech. Paper in Marine Sci. No. 44, p37)'
    print >>f, '********************************************************'

    m,n = S.shape
    for icol in range(0, n):
        print >>f, ''
        print >>f, '   Sal  Temp  Press      Cp        cp'
        print >>f, '  (psu)  (C)   (db)    (J/kg.C)   (J/kg.C)'
        result = np.vstack( [ S[:,icol], T[:,icol], P[:,icol],
                            UN_cp[:,icol], CP[:,icol] ] )
        for iline in range(0, m):
            print >>f, " %4.0f  %4.0f   %5.0f   %8.1f  %11.2f" % tuple(result[:,iline])

    # test MAIN MODULE svel
    module     = 'svel'
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, '**  TESTING MODULE: ', module
    print >>f, '********************************************************'

    # test 1 -
    # DATA FROM POND AND PICKARD INTRO. DYNAMICAL OCEANOGRAPHY 2ND ED. 1986
    T = np.array([[0,  0,  0,  0,  0,  0],
                  [10, 10, 10, 10, 10, 10],
                  [20, 20, 20, 20, 20, 20],
                  [30, 30, 30, 30, 30, 30],
                  [40, 40, 40, 40, 40, 40]]) / 1.00024

    S = np.array([[25, 25, 25, 35, 35, 35],
                  [25, 25, 25, 35, 35, 35],
                  [25, 25, 25, 35, 35, 35],
                  [25, 25, 25, 35, 35, 35],
                  [25, 25, 25, 35, 35, 35]])

    P = np.array([[0, 5000, 10000, 0, 5000, 10000],
                  [0, 5000, 10000, 0, 5000, 10000],
                  [0, 5000, 10000, 0, 5000, 10000],
                  [0, 5000, 10000, 0, 5000, 10000],
                  [0, 5000, 10000, 0, 5000, 10000]])

    UN_svel = np.array([[ 1435.8, 1520.4, 1610.4, 1449.1, 1534.0, 1623.2],
                     [1477.7, 1561.3, 1647.4, 1489.8, 1573.4, 1659.0],
                     [1510.3, 1593.6, 1676.8, 1521.5, 1604.5, 1687.2],
                     [1535.2, 1619.0, 1700.6, 1545.6, 1629.0, 1710.1],
                     [1553.4, 1638.0, 1719.2, 1563.2, 1647.3, 1727.8]])

    SVEL    = svel(S, T, P)

    # DISPLAY RESULTS
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, 'Comparison of accepted values from UNESCO 1983 '
    print >>f, ' (Unesco Tech. Paper in Marine Sci. No. 44, p50)'
    print >>f, '********************************************************'

    m,n = SVEL.shape
    for icol in range(0, n):
        print >>f, ''
        print >>f, '   Sal  Temp  Press     SVEL       svel'
        print >>f, '  (psu)  (C)   (db)     (m/s)       (m/s)'

        result = np.vstack( [ S[:,icol], T[:,icol], P[:,icol], UN_svel[:,icol], SVEL[:,icol] ] )
        for iline in range(0, m):
            print >>f, " %4.0f  %4.0f   %5.0f   %8.1f  %11.3f" % tuple(result[:,iline])

    # test SUBMODULES alpha beta aonb
    submodules     = 'alpha beta aonb'
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, '**  and SUB-MODULE: ', submodules
    print >>f, '********************************************************'

    # DATA FROM MCDOUOGALL 1987
    s    = 40
    PT   = 10
    p    = 4000
    beta_lit  = 0.72088e-03
    aonb_lit  = 0.34763
    alpha_lit = aonb_lit*beta_lit

    BETA  = beta( s, PT, p, pt=True)
    ALPHA = alpha(s, PT, p, pt=True)
    AONB  = aonb( s, PT, p, pt=True)

    # DISPLAY RESULTS
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, 'Comparison of accepted values from MCDOUGALL 1987 '
    print >>f, '********************************************************'

    print >>f, ''
    print >>f, '   Sal  Temp  Press     BETA       beta'
    print >>f, '  (psu)  (C)   (db)   (psu^-1)     (psu^-1)'
    table = np.hstack( [ s, PT, p, beta_lit, BETA ] )
    print >>f, " %4.0f  %4.0f   %5.0f   %11.4e  %11.5e" % tuple(table)

    print >>f, ''
    print >>f, '   Sal  Temp  Press     AONB       aonb'
    print >>f, '  (psu)  (C)   (db)   (psu C^-1)   (psu C^-1)'
    table = np.hstack( [s, PT, p, aonb_lit, AONB] )
    print >>f, " %4.0f  %4.0f   %5.0f   %8.5f  %11.6f" % tuple(table)

    print >>f, ''
    print >>f, '   Sal  Temp  Press     ALPHA       alpha'
    print >>f, '  (psu)  (C)   (db)    (psu^-1)     (psu^-1)'
    table = np.hstack( [ s, PT, p, alpha_lit, ALPHA ] )
    print >>f, " %4.0f  %4.0f   %5.0f   %11.4e  %11.4e" % tuple(table)

    # test MAIN MODULES  satO2 satN2 satAr
    module     = 'satO2 satN2 satAr'
    print >>f, ''
    print >>f, '********************************************************'
    print >>f, '**  TESTING MODULE: ', module
    print >>f, '********************************************************'

    # Data from Weiss 1970
    T      = np.array([[ -1, -1],
                    [ 10, 10],
                    [ 20, 20],
                    [ 40, 40]]) / 1.00024

    S      = np.array([[ 20, 40],
                    [ 20, 40],
                    [ 20, 40],
                    [ 20, 40]])

    lit_O2 = np.array([[ 9.162, 7.984],
                    [ 6.950, 6.121],
                    [ 5.644, 5.015],
                    [ 4.050, 3.656]])

    lit_N2 =  np.array([[ 16.28, 14.01],
                     [ 12.64, 11.01],
                     [ 10.47,  9.21],
                     [  7.78,  6.95]])

    lit_Ar =  np.array([[ 0.4456, 0.3877],
                     [ 0.3397, 0.2989],
                     [ 0.2766, 0.2457],
                     [ 0.1986, 0.1794]])

    O2     = satO2(S, T)
    N2     = satN2(S, T)
    Ar     = satAr(S, T)

    # DISPLAY RESULTS
    print >>f, ''
    print >>f, '************************************************************'
    print >>f, 'Comparison of accepted values from Weiss, R.F. 1979 '
    print >>f, '"The solubility of nitrogen, oxygen and argon in water'
    print >>f, ' and seawater." Deep-Sea Research., 1970, Vol 17, pp721-735.'
    print >>f, '************************************************************'

    m,n = S.shape
    for icol in range(0, n):
        print >>f, ''
        print >>f, '   Sal  Temp      O2         satO2'
        print >>f, '  (psu)  (C)      (ml/l)     (ml/l)'
        result = np.vstack( [ S[:,icol], T[:,icol],
                            lit_O2[:,icol], O2[:,icol] ] )
        for iline in range(0, m):
            print >>f, " %4.0f  %4.0f    %8.2f   %9.3f" % tuple(result[:,iline])

    for icol in range(0, n):
        print >>f, ''
        print >>f, '   Sal  Temp      N2         satN2'
        print >>f, '  (psu)  (C)      (ml/l)     (ml/l)'
        result = np.vstack( [ S[:,icol], T[:,icol],
                            lit_N2[:,icol], N2[:,icol] ] )
        for iline in range(0, m):
            print >>f, " %4.0f  %4.0f    %8.2f  %9.3f" % tuple(result[:,iline])

    for icol in range(0, n):
        print >>f, ''
        print >>f, '   Sal  Temp      Ar         satAr'
        print >>f, '  (psu)  (C)      (ml/l)     (ml/l)'
        result = np.vstack( [ S[:,icol], T[:,icol],
                              lit_Ar[:,icol], Ar[:,icol] ] )
        for iline in range(0, m):
            print >>f, " %4.0f  %4.0f     %8.4f  %9.4f" % tuple(result[:,iline])

if __name__=='__main__':
    test()
