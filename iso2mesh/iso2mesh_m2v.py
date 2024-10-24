"""@package docstring
Iso2Mesh for Python - Mesh-to-volume mesh rasterization

Copyright (c) 2024 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = ["m2v", "mesh2vol", "mesh2mask", "qmeshcut", "barycentricgrid"]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import matplotlib.pyplot as plt

##====================================================================================
## implementations
##====================================================================================


def m2v(*args):
    """
    Shortcut for mesh2vol, rasterizing a tetrahedral mesh to a volume.

    Parameters:
    Same as mesh2vol function.

    Returns:
    Volumetric representation of the mesh.
    """
    return mesh2vol(*args)


def mesh2vol(node, elem, xi, yi=None, zi=None):
    """
    Fast rasterization of a 3D mesh to a volume with tetrahedron index labels.

    Parameters:
    node: Node coordinates (Nx3 array)
    elem: Tetrahedron element list (Nx4 array)
    xi: Grid or number of divisions along x-axis
    yi: (Optional) Grid along y-axis
    zi: (Optional) Grid along z-axis

    Returns:
    mask: 3D volume where voxel values correspond to the tetrahedron index
    weight: (Optional) Barycentric weights for each voxel
    """

    # Check if xi is scalar or list of grid divisions
    if isinstance(xi, (int, float)) and yi is None and zi is None:
        mn = np.min(node, axis=0)
        mx = np.max(node, axis=0)
        df = (mx[:3] - mn[:3]) / xi
    elif len(xi) == 3 and yi is None and zi is None:
        mn = np.min(node, axis=0)
        mx = np.max(node, axis=0)
        df = (mx[:3] - mn[:3]) / xi
    elif yi is not None and zi is not None:
        mx = [np.max(xi), np.max(yi), np.max(zi)]
        mn = [np.min(xi), np.min(yi), np.min(zi)]
        df = [np.min(np.diff(xi)), np.min(np.diff(yi)), np.min(np.diff(zi))]
    else:
        raise ValueError("At least xi input is required")

    xi = np.arange(mn[0], mx[0], df[0])
    yi = np.arange(mn[1], mx[1], df[1])
    zi = np.arange(mn[2], mx[2], df[2])

    if node.shape[1] != 3 or elem.shape[1] <= 3:
        raise ValueError("node must have 3 columns; elem must have 4 columns")

    mask = np.zeros((len(xi) - 1, len(yi) - 1, len(zi) - 1), dtype=int)
    weight = None

    if len(elem.shape) > 1:
        weight = np.zeros((4, len(xi) - 1, len(yi) - 1, len(zi) - 1))

    fig = plt.figure()
    for i in range(len(zi)):
        cutpos, cutvalue, facedata, elemid = qmeshcut(elem, node, zi[i])
        if cutpos is None:
            continue

        maskz, weightz = mesh2mask(cutpos, facedata, xi, yi, fig)
        idx = np.where(~np.isnan(maskz))
        mask[:, :, i] = maskz

        if weight is not None:
            eid = facedata[maskz[idx]]
            maskz[idx] = (
                cutvalue[eid[:, 0]] * weightz[0, idx]
                + cutvalue[eid[:, 1]] * weightz[1, idx]
                + cutvalue[eid[:, 2]] * weightz[2, idx]
                + cutvalue[eid[:, 3]] * weightz[3, idx]
            )
            weight[:, :, :, i] = weightz

    plt.close(fig)
    return mask, weight


def mesh2mask(node, face, xi, yi=None, hf=None):
    """
    Fast rasterization of a 2D mesh to an image with triangle index labels.

    Parameters:
    node: Node coordinates (N by 2 or N by 3 array)
    face: Triangle surface (N by 3 or N by 4 array)
    xi: Grid or number of divisions along x-axis
    yi: (Optional) Grid along y-axis
    hf: (Optional) Handle to a figure for faster rendering

    Returns:
    mask: 2D image where pixel values correspond to the triangle index
    weight: (Optional) Barycentric weights for each triangle
    """

    # Determine grid size from inputs
    if isinstance(xi, (int, float)) and yi is None:
        mn = np.min(node, axis=0)
        mx = np.max(node, axis=0)
        df = (mx[:2] - mn[:2]) / xi
    elif len(xi) == 2 and yi is None:
        mn = np.min(node, axis=0)
        mx = np.max(node, axis=0)
        df = (mx[:2] - mn[:2]) / xi
    elif yi is not None:
        mx = [np.max(xi), np.max(yi)]
        mn = [np.min(xi), np.min(yi)]
        df = [np.min(np.diff(xi)), np.min(np.diff(yi))]
    else:
        raise ValueError("At least xi input is required")

    # Error checking for input sizes
    if node.shape[1] <= 1 or face.shape[1] <= 2:
        raise ValueError(
            "node must have 2 or 3 columns; face must have at least 3 columns"
        )

    # If no figure handle is provided, create one
    if hf is None:
        fig = plt.figure()
    else:
        plt.clf()

    # Rasterize the mesh to an image
    plt.gca().patch.set_visible(False)
    plt.gca().set_position([0, 0, 1, 1])

    cmap = plt.get_cmap("jet", len(face))
    plt.pcolormesh(node[:, 0], node[:, 1], np.arange(len(face)), cmap=cmap)

    # Set axis limits
    plt.xlim([mn[0], mx[0]])
    plt.ylim([mn[1], mx[1]])
    plt.clim([1, len(face)])
    output_size = np.round((mx[:2] - mn[:2]) / df).astype(int)

    # Rendering or saving to image
    mask = np.zeros(output_size, dtype=np.int32)
    if hf is None:
        plt.close(fig)

    # Optional weight calculation (if requested)
    weight = None
    if yi is not None:
        weight = barycentricgrid(node, face, xi, yi, mask)

    return mask, weight


def barycentricgrid(node, face, xi, yi, mask):
    """
    Compute barycentric weights for a grid.

    Parameters:
    node: Node coordinates
    face: Triangle surface
    xi: x-axis grid
    yi: y-axis grid
    mask: Rasterized triangle mask

    Returns:
    weight: Barycentric weights for each triangle
    """
    xx, yy = np.meshgrid(xi, yi)
    idx = ~np.isnan(mask)
    eid = mask[idx]

    t1 = node[face[:, 0], :]
    t2 = node[face[:, 1], :]
    t3 = node[face[:, 2], :]

    # Calculate barycentric coordinates
    tt = (t2[:, 1] - t3[:, 1]) * (t1[:, 0] - t3[:, 0]) + (t3[:, 0] - t2[:, 0]) * (
        t1[:, 1] - t3[:, 1]
    )
    w = np.zeros((len(idx), 3))
    w[:, 0] = (t2[eid, 1] - t3[eid, 1]) * (xx[idx] - t3[eid, 0]) + (
        t3[eid, 0] - t2[eid, 0]
    ) * (yy[idx] - t3[eid, 1])
    w[:, 1] = (t3[eid, 1] - t1[eid, 1]) * (xx[idx] - t3[eid, 0]) + (
        t1[eid, 0] - t3[eid, 0]
    ) * (yy[idx] - t3[eid, 1])
    w[:, 0] /= tt[eid]
    w[:, 1] /= tt[eid]
    w[:, 2] = 1 - w[:, 0] - w[:, 1]

    weight = np.zeros((3, mask.shape[0], mask.shape[1]))
    weight[0, idx] = w[:, 0]
    weight[1, idx] = w[:, 1]
    weight[2, idx] = w[:, 2]

    return weight
