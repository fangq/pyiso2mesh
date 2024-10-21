"""@package docstring
Iso2Mesh for Python - Primitive shape meshing functions

Copyright (c) 2024 Edward Xu <xu.ed at neu.edu>
              2019-2024 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "meshgrid5",
    "meshgrid6",
    "surfedge",
    "volface",
    "surfplane",
    "surfacenorm",
    "nodesurfnorm",
    "plotsurf",
    "plotasurf",
    "meshcentroid",
    "varargin2struct",
    "jsonopt",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import matplotlib.pyplot as plt
import sys
from itertools import permutations


def meshgrid5(*args):
    args = list(args)

    n = len(args)
    if n != 3:
        raise ValueError("only works for 3D case!")

    for i in range(n):
        v = args[i]
        if len(v) % 2 == 0:
            args[i] = np.linspace(v[0], v[-1], len(v) + 1)

    # create a single n-d hypercube
    cube8 = np.array(
        [
            [1, 4, 5, 13],
            [1, 2, 5, 11],
            [1, 10, 11, 13],
            [11, 13, 14, 5],
            [11, 13, 1, 5],
            [2, 3, 5, 11],
            [3, 5, 6, 15],
            [15, 11, 12, 3],
            [15, 11, 14, 5],
            [11, 15, 3, 5],
            [4, 5, 7, 13],
            [5, 7, 8, 17],
            [16, 17, 13, 7],
            [13, 17, 14, 5],
            [5, 7, 17, 13],
            [5, 6, 9, 15],
            [5, 8, 9, 17],
            [17, 18, 15, 9],
            [17, 15, 14, 5],
            [17, 15, 5, 9],
            [10, 13, 11, 19],
            [13, 11, 14, 23],
            [22, 19, 23, 13],
            [19, 23, 20, 11],
            [13, 11, 19, 23],
            [11, 12, 15, 21],
            [11, 15, 14, 23],
            [23, 21, 20, 11],
            [23, 24, 21, 15],
            [23, 21, 11, 15],
            [16, 13, 17, 25],
            [13, 17, 14, 23],
            [25, 26, 23, 17],
            [25, 22, 23, 13],
            [13, 17, 25, 23],
            [17, 18, 15, 27],
            [17, 15, 14, 23],
            [26, 27, 23, 17],
            [27, 23, 24, 15],
            [23, 27, 17, 15],
        ]
    ).T

    # build the complete lattice
    nodecount = [len(arg) for arg in args]

    if any(count < 2 for count in nodecount):
        raise ValueError("Each dimension must be of size 2 or more.")

    node = lattice(*args)
    # print(node)

    ix, iy, iz = np.meshgrid(
        np.arange(1, nodecount[0] - 1, 2),
        np.arange(1, nodecount[1] - 1, 2),
        np.arange(1, nodecount[2] - 1, 2),
        indexing="ij",
    )
    ind = np.ravel_multi_index(
        (ix.flatten() - 1, iy.flatten() - 1, iz.flatten() - 1), nodecount
    )

    nodeshift = np.array(
        [
            0,
            1,
            2,
            nodecount[0],
            nodecount[0] + 1,
            nodecount[0] + 2,
            2 * nodecount[0],
            2 * nodecount[0] + 1,
            2 * nodecount[0] + 2,
        ]
    )
    nodeshift = np.concatenate(
        (
            nodeshift,
            nodeshift + nodecount[0] * nodecount[1],
            nodeshift + 2 * nodecount[0] * nodecount[1],
        )
    )

    nc = len(ind)
    elem = np.zeros((nc * 40, 4), dtype=int)
    for i in range(nc):
        elem[np.arange(0, 40) + (i * 40), :] = (
            np.reshape(nodeshift[cube8.flatten() - 1], (4, 40)).T + ind[i]
        )

    return node, elem


# _________________________________________________________________________________________________________


def meshgrid6(*args):
    # dimension of the lattice
    n = len(args)

    # create a single n-d hypercube     # list of node of the cube itself
    vhc = (
        np.array(list(map(lambda x: list(bin(x)[2:].zfill(n)), range(2**n)))) == "1"
    ).astype(int)

    # permutations of the integers 1:n
    p = list(permutations(range(1, n + 1)))
    p = p[::-1]
    nt = len(p)
    thc = np.zeros((nt, n + 1), dtype=int)

    for i in range(nt):
        thc[i, :] = np.where(
            np.all(np.diff(vhc[:, np.array(p[i]) - 1], axis=1) >= 0, axis=1)
        )[0]

    # build the complete lattice
    nodecount = np.array([len(arg) for arg in args])
    if np.any(nodecount < 2):
        raise ValueError("Each dimension must be of size 2 or more.")
    node = lattice(*args)

    # unrolled index into each hyper-rectangle in the lattice
    ind = [np.arange(nodecount[i] - 1) for i in range(n)]
    ind = np.meshgrid(*ind, indexing="ij")
    ind = np.array(ind).reshape(n, -1).T
    k = np.cumprod([1] + nodecount[:-1].tolist())

    ind = 1 + ind @ k.T  # k[:-1].reshape(-1, 1)
    nind = len(ind)
    offset = vhc @ k.T
    elem = np.zeros((nt * nind, n + 1), dtype=int)
    L = np.arange(1, nind + 1).reshape(-1, 1)

    for i in range(nt):
        elem[L.flatten() - 1, :] = np.tile(ind, (n + 1, 1)).T + np.tile(
            offset[thc[i, :]], (nind, 1)
        )
        L += nind

    elem = elem - 1

    return node, elem


# _________________________________________________________________________________________________________


def lattice(*args):
    # generate a factorial lattice in n variables
    n = len(args)
    sizes = [len(arg) for arg in args]
    c = np.meshgrid(*args, indexing="ij")
    g = np.zeros((np.prod(sizes), n))
    for i in range(n):
        g[:, i] = c[i].flatten()
    return g


# _________________________________________________________________________________________________________


def surfedge(f, *varargin):
    if f.size == 0:
        return np.array([]), None

    findjunc = 0

    if f.shape[1] == 3:
        edges = np.vstack(
            (f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])
        )  # create all the edges
    elif f.shape[1] == 4:
        edges = np.vstack(
            (f[:, [0, 1, 2]], f[:, [1, 0, 3]], f[:, [0, 2, 3]], f[:, [1, 3, 2]])
        )  # create all the edges
    else:
        raise ValueError("surfedge only supports 2D and 3D elements")

    edgesort = np.sort(edges, axis=1)
    _, ix, jx = np.unique(edgesort, axis=0, return_index=True, return_inverse=True)

    # if isoctavemesh:
    #     u = np.unique(jx)
    #     if f.shape[1] == 3 and findjunc:
    #         qx = u[np.histogram(jx, bins=u)[0] > 2]
    #     else:
    #         qx = u[np.histogram(jx, bins=u)[0] == 1]
    # else:
    vec = np.bincount(jx, minlength=max(jx) + 1)
    if f.shape[1] == 3 and findjunc:
        qx = np.where(vec > 2)[0]
    else:
        qx = np.where(vec == 1)[0]

    openedge = edges[ix[qx], :]
    elemid = None
    if len(varargin) >= 2:
        elemid, iy = np.unravel_index(ix[qx], f.shape)

    return openedge, elemid


# _________________________________________________________________________________________________________


def volface(t):
    openedge, elemid = surfedge(t)
    return openedge, elemid


# _________________________________________________________________________________________________________


def surfplane(node, face):
    # plane=surfplane(node,face)
    #
    # plane equation coefficients for each face in a surface
    #
    # input:
    #   node: a list of node coordinates (nn x 3)
    #   face: a surface mesh triangle list (ne x 3)
    #
    # output:
    #   plane: a (ne x 4) array, in each row, it has [a b c d]
    #        to denote the plane equation as "a*x+b*y+c*z+d=0"
    AB = node[face[:, 1], :3] - node[face[:, 0], :3]
    AC = node[face[:, 2], :3] - node[face[:, 0], :3]

    N = np.cross(AB, AC)
    d = -np.dot(N, node[face[:, 0], :3].T)
    plane = np.column_stack((N, d))
    return plane


# _________________________________________________________________________________________________________


def surfacenorm(node, face, *args):
    # Compute the normal vectors for a triangular surface.
    #
    # Parameters:
    #  node : np.ndarray
    #      A list of node coordinates (nn x 3).
    #  face : np.ndarray
    #       A surface mesh triangle list (ne x 3).
    #  args : list
    #      A list of optional parameters, currently surfacenorm supports:
    #      'Normalize': [1|0] if set to 1, the normal vectors will be unitary (default).
    # Returns:
    #  snorm : np.ndarray
    #      Output surface normal vector at each face.
    opt = varargin2struct(*args)

    snorm = surfplane(node, face)
    snorm = snorm[:, :3]

    if jsonopt("Normalize", 1, opt):
        snorm = snorm / np.sqrt(np.sum(snorm**2, axis=1, keepdims=True))

    return snorm


# _________________________________________________________________________________________________________


def nodesurfnorm(node, elem):
    #  nv=nodesurfnorm(node,elem)
    #
    #  calculate a nodal norm for each vertix on a surface mesh (surface
    #   can only be triangular or cubic)
    #
    # parameters:
    #      node: node coordinate of the surface mesh (nn x 3)
    #      elem: element list of the surface mesh (3 columns for
    #            triangular mesh, 4 columns for cubic surface mesh)
    #      pt: points to be projected, 3 columns for x,y and z respectively
    #
    # outputs:
    #      nv: nodal norms (vector) calculated from nodesurfnorm.m
    #          with dimensions of (size(v,1),3)
    nn = node.shape[0]
    ne = elem.shape[0]

    ev = surfacenorm(node, elem)

    nv = np.zeros((nn, 3))
    ev2 = np.tile(ev, (1, 3))

    for i in range(ne):
        nv[elem[i, :], :] += ev2[i, :].reshape(3, 3).T

    nvnorm = np.sqrt(np.sum(nv * nv, axis=1))
    idx = np.where(nvnorm > 0)[0]

    if len(idx) < nn:
        print(
            "Warning: found interior nodes, their norms will be set to zeros; to remove them, please use removeisolatednodes.m from iso2mesh toolbox"
        )

        nv[idx, :] = nv[idx, :] / nvnorm[idx][:, np.newaxis]
    else:
        nv = nv / nvnorm[:, np.newaxis]

    return nv


# _________________________________________________________________________________________________________


def plotsurf(node, face, *args):
    rngstate = np.random.get_state()
    if len(sys.argv) >= 2:
        randseed = int("623F9A9E", 16)  # "U+623F U+9A9E"

        if "ISO2MESH_RANDSEED" in globals():
            randseed = globals()["ISO2MESH_RANDSEED"]
        np.random.seed(randseed)

        if isinstance(face, list):
            sc = np.random.rand(10, 3)
            length = len(face)
            newsurf = [[] for _ in range(10)]
            # reorganizing each labeled surface into a new list
            for i in range(length):
                fc = face[i]
                if isinstance(fc, list) and len(fc) >= 2:
                    if fc[1] + 1 > 10:
                        sc[fc[1] + 1, :] = np.random.rand(1, 3)
                    if fc[1] + 1 >= len(newsurf):
                        newsurf[fc[1] + 1] = []
                    newsurf[fc[1] + 1].append(fc[0])
                else:  # unlabeled facet is tagged by 0
                    if isinstance(fc, list):
                        newsurf[0].append(np.array(fc).flatten())
                    else:
                        newsurf[0].append(fc)

            plt.hold(True)
            h = []
            newlen = len(newsurf)

            for i in range(newlen):
                if not newsurf[i]:
                    continue
                try:
                    subface = np.array(newsurf[i]).T
                    if subface.shape[0] > 1 and subface.ndim == 2:
                        subface = subface.T
                    h.append(
                        plt.Patch(
                            vertices=node, faces=subface, facecolor=sc[i, :], *args
                        )
                    )
                except:
                    for j in range(len(newsurf[i])):
                        h.append(
                            plt.Patch(
                                vertices=node,
                                faces=newsurf[i][j],
                                facecolor=sc[i, :],
                                *args
                            )
                        )
        else:
            if face.shape[1] == 4:
                tag = face[:, 3]
                types = np.unique(tag)
                plt.hold(True)
                h = []
                for i in range(len(types)):
                    if node.shape[1] == 3:
                        h.append(
                            plotasurf(
                                node,
                                face[tag == types[i], 0:3],
                                facecolor=np.random.rand(3, 1),
                                *args
                            )
                        )
                    else:
                        h.append(plotasurf(node, face[tag == types[i], 0:3], *args))
            else:
                h = plotasurf(node, face, *args)

    if h:
        plt.axis("equal")
    #        if np.all(np.array(plt.gca().view) == [0, 90]):
    #            plt.view(3)

    if h and len(args) >= 1:
        return h

    np.random.set_state(rngstate)


# _________________________________________________________________________________________________________


def plotasurf(node, face, *args):
    if face.shape[1] <= 2:
        h = plotedges(node, face, *args)
    else:
        if node.shape[1] == 4:
            h = plt.trisurf(
                face[:, 0:3], node[:, 0], node[:, 1], node[:, 2], node[:, 3], *args
            )
        else:
            fig = plt.figure(figsize=(16, 9))
            h = plt.axes(projection="3d")
            # Creating color map
            my_cmap = plt.get_cmap("hot")
            # Creating plot
            trisurf = h.plot_trisurf(
                node[:, 0], node[:, 1], face, node[:, 2], cmap=my_cmap
            )

    if "h" in locals():
        return h


# _________________________________________________________________________________________________________


def meshcentroid(v, f):
    #
    # centroid=meshcentroid(v,f)
    #
    # compute the centroids of a mesh defined by nodes and elements
    # (surface or tetrahedra) in R^n space
    #
    # input:
    #      v: surface node list, dimension (nn,3)
    #      f: surface face element list, dimension (be,3)
    #
    # output:
    #      centroid: centroid positions, one row for each element
    #
    if not isinstance(f, list):
        ec = v[f[:, :], :]
        print(ec.shape)
        centroid = np.squeeze(np.mean(ec, axis=1))
    else:
        length_f = len(f)
        centroid = np.zeros((length_f, v.shape[1]))
        try:
            for i in range(length_f):
                fc = f[i]
                if fc:  # need to set centroid to NaN if fc is empty?
                    vlist = fc[0]
                    centroid[i, :] = np.mean(
                        v[vlist[~np.isnan(vlist)], :], axis=0
                    )  # Note to Ed check if this is functioning correctly
        except Exception as e:
            raise ValueError("malformed face cell array") from e
    return centroid


# _________________________________________________________________________________________________________


def varargin2struct(*args):
    opt = {}
    length = len(args)
    if length == 0:
        return opt

    i = 0
    while i < length:
        if isinstance(args[i], dict):
            opt = {**opt, **args[i]}  # Merging dictionaries
        elif isinstance(args[i], str) and i < length - 1:
            opt[args[i].lower()] = args[i + 1]
            i += 1
        else:
            raise ValueError(
                "input must be in the form of ...,'name',value,... pairs or structs"
            )
        i += 1

    return opt


# _________________________________________________________________________________________________________


def jsonopt(key, default, *args):
    val = default
    if len(args) <= 0:
        return val
    key0 = key.lower()
    opt = args[0]
    if isinstance(opt, dict):
        if key0 in opt:
            val = opt[key0]
        elif key in opt:
            val = opt[key]
    return val
