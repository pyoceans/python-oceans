import os

import numpy as np


_default_path = os.path.join(os.path.dirname(__file__), 'data')


def LineNormals2D(Vertices, Lines):
    """
    This function calculates the normals, of the line points using the
    neighboring points of each contour point, and forward an backward
    differences on the end points.

    N = LineNormals2D(V, L)

    inputs,
      V : List of points/vertices 2 x M
    (optional)
      Lines : A N x 2 list of line pieces, by indices of the vertices
            (if not set assume Lines=[1 2; 2 3 ; ... ; M-1 M])

    outputs,
      N : The normals of the Vertices 2 x M

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.load(os.path.join(_default_path, 'testdata.npz'))
    >>> Lines, Vertices = data['Lines'], data['Vertices']
    >>> N = LineNormals2D(Vertices, Lines)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1)
    >>> _ = ax.plot(np.c_[Vertices[:, 0], Vertices[:,0 ] + 10 * N[:, 0]].T,
    ...            np.c_[Vertices[:, 1], Vertices[:, 1] + 10 * N[:, 1]].T)

    Function based on LineNormals2D.m written by
    D.Kroon University of Twente (August 2011)

    """
    eps = np.spacing(1)

    if isinstance(Lines, np.ndarray):
        pass
    elif not Lines:
        Lines = np.c_[np.arange(1, Vertices.shape[0]),
                      np.arange(2, Vertices.shape[0] + 1)]
    else:
        raise ValueError(f'Expected np.array but got {Lines:!r}.')

    # Calculate tangent vectors.
    DT = Vertices[Lines[:, 0] - 1, :] - Vertices[Lines[:, 1] - 1, :]

    # Make influence of tangent vector 1/Distance (Weighted Central
    # Differences.  Points which are closer give a more accurate estimate of
    # the normal).
    LL = np.sqrt(DT[:, 0] ** 2 + DT[:, 1] ** 2)
    DT[:, 0] = DT[:, 0] / np.maximum(LL ** 2, eps)
    DT[:, 1] = DT[:, 1] / np.maximum(LL ** 2, eps)

    D1 = np.zeros_like(Vertices)
    D2 = np.zeros_like(Vertices)
    D1[Lines[:, 0] - 1, :] = DT
    D2[Lines[:, 1] - 1, :] = DT
    D = D1 + D2

    # Normalize the normal.
    LL = np.sqrt(D[:, 0] ** 2 + D[:, 1] ** 2)
    N = np.zeros_like(D)
    N[:, 0] = -D[:, 1] / LL
    N[:, 1] = D[:, 0] / LL

    return N


def LineCurvature2D(Vertices, Lines=None):
    """
    This function calculates the curvature of a 2D line. It first fits
    polygons to the points. Then calculates the analytical curvature from
    the polygons.

    k = LineCurvature2D(Vertices,Lines)

    inputs,
      Vertices : A M x 2 list of line points.
      (optional)
      Lines : A N x 2 list of line pieces, by indices of the vertices
            (if not set assume Lines=[1 2; 2 3 ; ... ; M-1 M])

    outputs,
      k : M x 1 Curvature values

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.load(os.path.join(_default_path, 'testdata.npz'))
    >>> Lines, Vertices = data['Lines'], data['Vertices']
    >>> k = LineCurvature2D(Vertices, Lines)
    >>> N = LineNormals2D(Vertices, Lines)
    >>> k = k * 100
    >>> fig, ax = plt.subplots(nrows=1, ncols=1)
    >>> _ = ax.plot(np.c_[Vertices[:, 0], Vertices[:, 0] + k * N[:, 0]].T,
    ...             np.c_[Vertices[:, 1], Vertices[:, 1] + k * N[:, 1]].T, 'g')
    >>> _ = ax.plot(np.c_[Vertices[Lines[:, 0] - 1, 0],
    ...                   Vertices[Lines[:, 1] - 1, 0]].T,
    ...             np.c_[Vertices[Lines[:, 0] - 1, 1],
    ...                   Vertices[Lines[:, 1] - 1, 1]].T, 'b')
    >>> _ = ax.plot(Vertices[:, 0], Vertices[:, 1], 'r.')

    Function based on LineCurvature2D.m written by
    D.Kroon University of Twente (August 2011)

    """
    # If no line-indices, assume a x[0] connected with x[1], x[2] with x[3].
    if isinstance(Lines, np.ndarray):
        pass
    elif not Lines:
        Lines = np.c_[np.arange(1, Vertices.shape[0]),
                      np.arange(2, Vertices.shape[0] + 1)]
    else:
        raise ValueError('Cannot recognized {!r}.'.format(Lines))

    # Get left and right neighbor of each points.
    Na = np.zeros(Vertices.shape[0], dtype=np.int)
    Nb = np.zeros_like(Na)
    # As int because we use it to index an array...

    Na[Lines[:, 0] - 1] = Lines[:, 1]
    Nb[Lines[:, 1] - 1] = Lines[:, 0]

    # Check for end of line points, without a left or right neighbor.
    checkNa = Na == 0
    checkNb = Nb == 0

    Naa, Nbb = Na, Nb

    Naa[checkNa] = np.where(checkNa)[0]
    Nbb[checkNb] = np.where(checkNb)[0]

    # If no left neighbor use two right neighbors, and the same for right.
    Na[checkNa] = Nbb[Nbb[checkNa]]
    Nb[checkNb] = Naa[Naa[checkNb]]

    # ... Also, I remove `1` to get python indexing correctly.
    Na -= 1
    Nb -= 1

    # Correct for sampling differences.
    Ta = -np.sqrt(np.sum((Vertices - Vertices[Na, :]) ** 2, axis=1))
    Tb = np.sqrt(np.sum((Vertices - Vertices[Nb, :]) ** 2, axis=1))

    # If no left neighbor use two right neighbors, and the same for right.
    Ta[checkNa] = -Ta[checkNa]
    Tb[checkNb] = -Tb[checkNb]

    x = np.c_[Vertices[Na, 0], Vertices[:, 0], Vertices[Nb, 0]]
    y = np.c_[Vertices[Na, 1], Vertices[:, 1], Vertices[Nb, 1]]
    M = np.c_[np.ones_like(Tb),
              -Ta,
              Ta ** 2,
              np.ones_like(Tb),
              np.zeros_like(Tb),
              np.zeros_like(Tb),
              np.ones_like(Tb),
              -Tb,
              Tb ** 2]

    invM = inverse3(M)
    a = np.zeros_like(x)
    b = np.zeros_like(a)
    a[:, 0] = (invM[:, 0, 0] * x[:, 0] +
               invM[:, 1, 0] * x[:, 1] +
               invM[:, 2, 0] * x[:, 2])

    a[:, 1] = (invM[:, 0, 1] * x[:, 0] +
               invM[:, 1, 1] * x[:, 1] +
               invM[:, 2, 1] * x[:, 2])

    a[:, 2] = (invM[:, 0, 2] * x[:, 0] +
               invM[:, 1, 2] * x[:, 1] +
               invM[:, 2, 2] * x[:, 2])

    b[:, 0] = (invM[:, 0, 0] * y[:, 0] +
               invM[:, 1, 0] * y[:, 1] +
               invM[:, 2, 0] * y[:, 2])

    b[:, 1] = (invM[:, 0, 1] * y[:, 0] +
               invM[:, 1, 1] * y[:, 1] +
               invM[:, 2, 1] * y[:, 2])

    b[:, 2] = (invM[:, 0, 2] * y[:, 0] +
               invM[:, 1, 2] * y[:, 1] +
               invM[:, 2, 2] * y[:, 2])

    # Calculate the curvature from the fitted polygon.
    k = (2 * (a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]) /
         ((a[:, 1] ** 2 + b[:, 1] ** 2) ** (3 / 2)))

    return k


def inverse3(M):
    """
    This function does inv(M), but then for an array of 3x3 matrices.

    """
    adjM = np.zeros((M.shape[0], 3, 3))
    adjM[:, 0, 0] = M[:, 4] * M[:, 8] - M[:, 7] * M[:, 5]
    adjM[:, 0, 1] = -(M[:, 3] * M[:, 8] - M[:, 6] * M[:, 5])
    adjM[:, 0, 2] = M[:, 3] * M[:, 7] - M[:, 6] * M[:, 4]
    adjM[:, 1, 0] = -(M[:, 1] * M[:, 8] - M[:, 7] * M[:, 2])
    adjM[:, 1, 1] = M[:, 0] * M[:, 8] - M[:, 6] * M[:, 2]
    adjM[:, 1, 2] = -(M[:, 0] * M[:, 7] - M[:, 6] * M[:, 1])
    adjM[:, 2, 0] = M[:, 1] * M[:, 5] - M[:, 4] * M[:, 2]
    adjM[:, 2, 1] = -(M[:, 0] * M[:, 5] - M[:, 3] * M[:, 2])
    adjM[:, 2, 2] = M[:, 0] * M[:, 4] - M[:, 3] * M[:, 1]

    detM = (M[:, 0] * M[:, 4] * M[:, 8] - M[:, 0] * M[:, 7] * M[:, 5] -
            M[:, 3] * M[:, 1] * M[:, 8] + M[:, 3] * M[:, 7] * M[:, 2] +
            M[:, 6] * M[:, 1] * M[:, 5] - M[:, 6] * M[:, 4] * M[:, 2])

    return adjM / detM[:, None, None]
