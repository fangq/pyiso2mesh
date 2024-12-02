"""@package docstring
Iso2Mesh for Python - Primitive shape meshing functions

Copyright (c) 2024 Edward Xu <xu.ed at neu.edu>
              2019-2024 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "getexeext",
    "rotatevec3d",
]
##====================================================================================
## dependent libraries
##====================================================================================

import sys
import numpy as np

##====================================================================================
## implementations
##====================================================================================

def getexeext():

    if sys.platform=='linux':
        ext = '.mexa64'
    elif 'win' in sys.platform:
        ext = '.exe'
    elif sys.platform == 'aarch64':
        ext = '.mexmaca64'
    else:
        print('Unable to find extension type')

    return ext

#_________________________________________________________________________________________________________

def rotatevec3d(pt, v1, u1=None, p0=None):
    """
    Rotate 3D points from one Cartesian coordinate system to another.

    Parameters:
    pt : numpy.ndarray
        3D points defined in a standard Cartesian system where a unitary
        z-vector is (0,0,1), 3 columns for x, y and z.
    v1 : numpy.ndarray
        The unitary z-vector for the target coordinate system.
    u1 : numpy.ndarray, optional
        The unitary z-vector for the source coordinate system, if ignored,
        u1=(0,0,1).
    p0 : numpy.ndarray, optional
        Offset of the new coordinate system, if ignored, p0=(0,0,0).

    Returns:
    newpt : numpy.ndarray
        The transformed 3D points.
    """

    if u1 is None:
        u1 = np.array([0, 0, 1])
    if p0 is None:
        p0 = np.array([0, 0, 0])

    u1 = u1 / np.linalg.norm(u1)
    v1 = v1 / np.linalg.norm(v1)

    R, s = rotmat2vec(u1, v1)
    newpt = (R @ pt.T * s).T

    if p0 is not None:
        p0 = p0.flatten()
        newpt += np.tile(p0, (newpt.shape[0], 1))

    return newpt

#_________________________________________________________________________________________________________

def rotmat2vec(u, v):
    """
    [R,s]=rotmat2vec(u,v)

    the rotation matrix from vector u to v, satisfying R*u*s=v

    input:
      u: a 3D vector in the source coordinate system;
      v: a 3D vector in the target coordinate system;

    output:
      R: a rotation matrix to transform normalized u to normalized v
      s: a scaling factor, so that R*u*s=v
    """
    s = np.linalg.norm(v) / np.linalg.norm(u)
    u1 = u / np.linalg.norm(u)
    v1 = v / np.linalg.norm(v)

    k = np.cross(u1, v1)
    if not np.any(k):  # u and v are parallel
        R = np.eye(3)
        return R, s

    # Rodrigues's formula:
    costheta = np.dot(u1, v1)
    R = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = costheta * np.eye(3) + R + np.outer(k, k) * (1 - costheta) / np.sum(k ** 2)

    return R, s

