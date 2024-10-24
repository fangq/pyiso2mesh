"""@package docstring
Iso2Mesh for Python - File I/O module

Copyright (c) 2024 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "affinemap",
    "meshinterp",
    "meshremap",
    "proj2mesh",
    "dist2surf",
    "regpt2surf",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np

##====================================================================================
## implementations
##====================================================================================


def affinemap(pfrom, pto):
    """
    Calculate an affine transform (A matrix and b vector) to map 3D vertices
    from one space to another using least square solutions.

    Parameters:
    pfrom : numpy array (n x 3), points in the original space
    pto : numpy array (n x 3), points in the mapped space

    Returns:
    A : numpy array (3 x 3), the affine transformation matrix
    b : numpy array (3 x 1), the translation vector
    """
    ptnum = pfrom.shape[0]

    if pto.shape[0] != ptnum:
        raise ValueError("Two inputs should have the same size")

    bsubmat = np.eye(3)
    amat = np.zeros((ptnum * 3, 9))

    for i in range(ptnum):
        amat[i * 3 : (i + 1) * 3, :] = np.kron(bsubmat, pfrom[i, :])

    amat = np.hstack([amat, np.tile(bsubmat, (ptnum, 1))])
    bvec = pto.T.flatten()

    x, _, _, _ = np.linalg.lstsq(amat, bvec, rcond=None)
    A = x[:9].reshape(3, 3).T
    b = x[-3:]

    return A, b


def meshinterp(fromval, elemid, elembary, fromelem, initval=None):
    """
    Interpolate nodal values from the source mesh to the target mesh based on a linear interpolation.

    Args:
        fromval: Values defined at the source mesh nodes. The row or column number
                 must be the same as the source mesh node count (matching elemid length).
        elemid: IDs of the source mesh element that encloses the target mesh nodes; a vector of length
                equal to the target mesh node count.
        elembary: Barycentric coordinates of each target mesh node within the source mesh elements.
                  The sum of each row is 1, with 3 or 4 columns.
        fromelem: The element list of the source mesh.
        initval: Optional initial values for the target nodes.

    Returns:
        newval: A 2D array where rows equal the target mesh nodes, and columns equal
                the value numbers defined at each source mesh node.
    """

    # If fromval is a single row, convert it to a column vector
    if fromval.shape[0] == 1:
        fromval = fromval.T

    # Find valid indices (non-NaN element IDs)
    idx = np.where(~np.isnan(elemid))[0]

    # Reshape fromval to match the size of elembary and fromelem
    allval = np.reshape(
        fromval[fromelem[elemid[idx], :], :],
        (len(idx), elembary.shape[1], fromval.shape[1]),
    )

    # Perform linear interpolation using barycentric coordinates
    tmp = np.array(
        [np.sum(elembary[idx, :] * x, axis=1) for x in np.rollaxis(allval, 2)]
    )

    # Initialize newval with initval or NaN
    if initval is not None:
        newval = initval
    else:
        newval = np.full((len(elemid), fromval.shape[1]), np.nan)

    # Assign interpolated values to newval at valid indices
    newval[idx, :] = np.squeeze(np.stack(tmp, axis=2))

    return newval


def meshremap(fromval, elemid, elembary, toelem, nodeto):
    """
    Redistribute nodal values from the source mesh to the target mesh
    so that the sum of each property on each mesh is the same.

    Parameters:
    fromval: Values defined at the source mesh nodes; should be a 1D or 2D array
             with the same number of rows or columns as the source mesh node count.
    elemid: IDs of the target mesh element that encloses the source mesh nodes; a vector.
    elembary: Barycentric coordinates of each source mesh node within the target mesh elements;
              sum of each row is 1, expect 3 or 4 columns (or N-D).
    toelem: Element list of the target mesh.
    nodeto: Total number of target mesh nodes.

    Returns:
    newval: A 2D array with rows equal to the target mesh nodes and columns equal to
            the value numbers defined at each source mesh node.
    """

    # Ensure fromval is a column vector if it is a row vector
    if fromval.shape[0] == 1:
        fromval = fromval.T

    # Ensure fromval's number of columns matches the length of elemid
    if fromval.shape[1] == len(elemid):
        fromval = fromval.T

    # Initialize output array newval
    newval = np.zeros((nodeto, fromval.shape[1]))

    # Ignore NaN elements
    idx = ~np.isnan(elemid)
    fromval = fromval[idx, :]
    elembary = elembary[idx, :]
    idx = elemid[idx]

    # Compute nodal values weighted by barycentric coordinates
    nodeval = np.repeat(
        fromval[:, np.newaxis, :], elembary.shape[1], axis=1
    ) * np.repeat(elembary[:, np.newaxis, :], fromval.shape[1], axis=1).transpose(
        0, 2, 1
    )

    # Accumulate contributions to target mesh nodes
    for i in range(elembary.shape[1]):
        ix, iy = np.meshgrid(toelem[idx, i], np.arange(fromval.shape[1]))
        nval = nodeval[:, :, i].T
        newval += np.add.at(newval, (ix.flatten(), iy.flatten()), nval.flatten())

    return newval


def proj2mesh(v, f, data):
    """
    Projects the scalar data onto the triangular mesh.

    Args:
    v: Vertices of the mesh (nn x 3).
    f: Faces of the mesh (ne x 3).
    data: Scalar values associated with vertices (nn x 1).

    Returns:
    proj_data: Projected data values on the mesh faces.
    """
    # Initialize projected data
    proj_data = np.zeros(f.shape[0])

    for i in range(f.shape[0]):
        # Get vertex indices for the face
        indices = f[i, :]

        # Compute the average of the scalar values at the vertices
        proj_data[i] = np.mean(data[indices])

    return proj_data


def dist2surf(p, v, f):
    """
    Computes the signed distance from points to a triangular surface.

    Args:
    p: Points (nx3 array).
    v: Vertices of the surface (mx3 array).
    f: Faces of the surface (px3 array, indices into vertices).

    Returns:
    dist: Signed distances from points to the surface.
    nearest: Nearest point indices on the surface.
    """
    dist = np.zeros(p.shape[0])
    nearest = np.zeros(p.shape[0], dtype=int)

    for i in range(p.shape[0]):
        point = p[i]
        closest_dist = np.inf

        for j in range(f.shape[0]):
            # Get vertices of the face
            tri = v[f[j]]
            d, _ = point_to_triangle_distance(point, tri)
            if d < closest_dist:
                closest_dist = d
                nearest[i] = j

        dist[i] = closest_dist

    return dist, nearest


def point_to_triangle_distance(point, tri):
    """
    Compute the distance from a point to a triangle.

    Args:
    point: The point (1x3 array).
    tri: The triangle (3x3 array of vertices).

    Returns:
    distance: Distance from the point to the triangle.
    nearest_point: The nearest point on the triangle.
    """
    # Vector calculations to find the nearest point on the triangle
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    v2 = point - tri[0]

    # Compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # Barycentric coordinates
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    # Check if point is inside the triangle
    if (u >= 0) and (v >= 0) and (u + v <= 1):
        # Point is inside the triangle
        nearest_point = tri[0] + u * v0 + v * v1
    else:
        # Point is outside the triangle, compute the distance to edges or vertices
        # (Placeholder for edge/vertex distance computation)
        nearest_point = None  # Replace with actual nearest point calculation
        distance = np.inf  # Placeholder

    distance = (
        np.linalg.norm(point - nearest_point) if nearest_point is not None else np.inf
    )

    return distance, nearest_point


def regpt2surf(pt, v, f):
    """
    Projects a set of points onto a triangular surface.

    Args:
    pt: Points to be projected (n x 3 array).
    v: Vertices of the surface (m x 3 array).
    f: Faces of the surface (p x 3 array, indices into vertices).

    Returns:
    proj_pt: Projected points on the surface.
    nearest: Indices of the nearest face for each point.
    """
    proj_pt = np.zeros_like(pt)
    nearest = np.zeros(pt.shape[0], dtype=int)

    for i in range(pt.shape[0]):
        point = pt[i]
        closest_dist = np.inf

        for j in range(f.shape[0]):
            # Get the vertices of the triangle
            tri = v[f[j]]
            d, nearest_point = point_to_triangle_distance(point, tri)

            if d < closest_dist:
                closest_dist = d
                proj_pt[i] = nearest_point
                nearest[i] = j

    return proj_pt, nearest
