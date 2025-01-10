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
    "meshabox",
    "meshacylinder",
    "meshanellip",
    "meshunitsphere",
    "meshasphere",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import matplotlib.pyplot as plt
import sys
import iso2mesh as im
from itertools import permutations, combinations



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
                                *args,
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
                                *args,
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


#_________________________________________________________________________________________________________

def meshabox(p0, p1, opt, nodesize=1):
    """
    Create the surface and tetrahedral mesh of a box geometry.

    Parameters:
    p0: Coordinates (x, y, z) for one end of the box diagonal
    p1: Coordinates (x, y, z) for the other end of the box diagonal
    opt: Maximum volume of the tetrahedral elements
    nodesize: (Optional) Size of the elements near each vertex.
              Can be a scalar or an 8x1 array.

    Returns:
    node: Node coordinates, 3 columns for x, y, and z respectively
    face: Surface mesh faces, each row represents a face element
    elem: Tetrahedral elements, each row represents a tetrahedron
    """
    if nodesize is None:
        nodesize = 1

    # Call to surf2mesh function to generate the surface mesh and volume elements
    node, elem, ff = im.surf2mesh(np.array([]), np.array([]), p0, p1, 1, opt, regions=None, holes=None, dobbx=nodesize)

    # Reorient the mesh elements
    elem,_,_ = im.meshreorient(node, elem[:, :4])

    # Extract the surface faces from the volume elements
    face = volface(elem)[0]

    return node, face, elem

#_________________________________________________________________________________________________________

def meshunitsphere(tsize, **kwargs):

    dim = 60
    esize = tsize * dim
    thresh = dim / 2 - 1

    xi, yi, zi = np.meshgrid(
        np.arange(0, dim + 0.5, 0.5),
        np.arange(0, dim + 0.5, 0.5),
        np.arange(0, dim + 0.5, 0.5),
    )
    dist = thresh - np.sqrt((xi - 30) ** 2 + (yi - 30) ** 2 + (zi - 30) ** 2)
    dist[dist < 0] = 0

    # Call a vol2restrictedtri equivalent in Python here (needs a custom function)
    node, face = im.vol2restrictedtri(dist, 1, (dim, dim, dim), dim**3, 30, esize, esize, 40000)

    node = (node - 0.5) * 0.5
    node, face, _ = im.removeisolatednode(node, face)
    node = (node - 30) / 28
    r0 = np.sqrt(np.sum(node**2, axis=1))
    node = node / r0[:, None]

#    if not 'maxvol' in kwargs:
#        maxvol = tsize**3

    maxvol = kwargs['maxvol'] if 'maxvol' in kwargs else tsize**3

    # Call a surf2mesh equivalent in Python here (needs a custom function)
    node, elem, face = im.surf2mesh(node, face, np.array([-1, -1, -1]) * 1.1, np.array([1, 1, 1]) * 1.1, 1, maxvol)

    return node, face, elem

#_________________________________________________________________________________________________________

def meshasphere(c0, r, tsize, maxvol=None):
    if maxvol is None:
        maxvol = tsize**3

    if maxvol is not None:
        node, face, elem = meshunitsphere(tsize / r, maxvol=maxvol / (r**3))
    else:
        node, face, elem = meshunitsphere(tsize / r)

    node = node * r + np.tile(np.array(c0).reshape(1, -1), (node.shape[0], 1))

    return node, face, elem #if maxvol is not None else (node, face)

#_________________________________________________________________________________________________________

def meshacylinder(c0, c1, r, **kwargs):#tsize=0, maxvol=0, ndiv=20):

    tsize = kwargs['tsize'] if 'tsize' in kwargs else 0
    maxvol = kwargs['maxvol'] if 'maxvol' in kwargs else 0
    ndiv = kwargs['ndiv'] if 'ndiv' in kwargs else 20

    if len(np.array([r])) == 1:
        r = np.array([r, r])

    if any(np.array(r) <= 0) or np.all(c0 == c1):
        raise ValueError("Invalid cylinder parameters")

    c0 = np.array(c0).reshape(-1, 1)
    c1 = np.array(c1).reshape(-1, 1)
    v0 = c1 - c0
    len_axis = np.linalg.norm(v0)

    if tsize == 0:
        tsize = min(np.append(r,len_axis)) / 10

    if maxvol == 0:
        maxvol = tsize**3 / 5

    dt = 2 * np.pi / ndiv
    theta = np.arange(dt, 2 * np.pi + dt, dt)
    cx = np.outer(np.array(r), np.cos(theta))
    cy = np.outer(np.array(r), np.sin(theta))

    p0 = np.column_stack((cx[0,:], cy[0,:], np.zeros(ndiv)))
    p1 = np.column_stack((cx[1,:], cy[1,:], len_axis * np.ones(ndiv)))

    pp = np.vstack((p0, p1))
    no = im.rotatevec3d(pp, v0.T) + np.tile(c0.T, (pp.shape[0], 1))

    #face = np.empty((0,4))
    face = []
    for i in range(ndiv - 1):
        #face = np.vstack((face, np.array([i, i + ndiv, i + ndiv + 1, i + 1])))
        face.append([[[i, i + ndiv, i + ndiv + 1, i + 1]], [1]])

    face.append([[[ndiv - 1, 2 * ndiv - 1, ndiv, 0]], [1]])
    face.append([[list(range(ndiv))],[2]])
    face.append([[list(range(ndiv, 2 * ndiv))], [3]])

    if tsize == 0 and maxvol == 0:
        return no, face

    if not 'tsize' in kwargs:
        tsize = len_axis/10

    if not 'maxvol' in kwargs:
        maxvol = tsize**3

    node, elem, *_ = im.surf2mesh(no, face, np.min(no, axis=0), np.max(no, axis=0), 1, maxvol,regions=np.array([[0,0,1]]),holes=np.array([]))
    face, *_ = volface(elem[:,0:4])

    return node, face, elem

#_________________________________________________________________________________________________________

def meshgrid5(*args):
    args = list(args)

    n = len(args)
    if n != 3:
      raise ValueError('only works for 3D case!')

    for i in range(n):
      v = args[i]
      if len(v) % 2 == 0:
        args[i] = np.linspace(v[0], v[-1], len(v) + 1)

    # create a single n-d hypercube
    cube8 = np.array([
        [1, 4, 5, 13], [1, 2, 5, 11], [1, 10, 11, 13], [11, 13, 14, 5],
        [11, 13, 1, 5], [2, 3, 5, 11], [3, 5, 6, 15], [15, 11, 12, 3],
        [15, 11, 14, 5], [11, 15, 3, 5], [4, 5, 7, 13], [5, 7, 8, 17],
        [16, 17, 13, 7], [13, 17, 14, 5], [5, 7, 17, 13], [5, 6, 9, 15],
        [5, 8, 9, 17], [17, 18, 15, 9], [17, 15, 14, 5], [17, 15, 5, 9],
        [10, 13, 11, 19], [13, 11, 14, 23], [22, 19, 23, 13], [19, 23, 20, 11],
        [13, 11, 19, 23], [11, 12, 15, 21], [11, 15, 14, 23], [23, 21, 20, 11],
        [23, 24, 21, 15], [23, 21, 11, 15], [16, 13, 17, 25], [13, 17, 14, 23],
        [25, 26, 23, 17], [25, 22, 23, 13], [13, 17, 25, 23], [17, 18, 15, 27],
        [17, 15, 14, 23], [26, 27, 23, 17], [27, 23, 24, 15], [23, 27, 17, 15]]).T

    # build the complete lattice
    nodecount = [len(arg) for arg in args]

    if any(count < 2 for count in nodecount):
      raise ValueError('Each dimension must be of size 2 or more.')

    node = lattice(*args)
    #print(node)

    ix, iy, iz = np.meshgrid(np.arange(1, nodecount[0] - 1, 2),
                              np.arange(1, nodecount[1] - 1, 2),
                              np.arange(1, nodecount[2] - 1, 2), indexing='ij')
    ind = np.ravel_multi_index((ix.flatten()-1, iy.flatten()-1, iz.flatten()-1), nodecount)

    nodeshift = np.array([0, 1, 2, nodecount[0], nodecount[0]+1, nodecount[0]+2, 2*nodecount[0], 2*nodecount[0]+1, 2*nodecount[0]+2])
    nodeshift = np.concatenate((nodeshift, nodeshift + nodecount[0] * nodecount[1], nodeshift + 2 * nodecount[0] * nodecount[1]))

    nc = len(ind)
    elem = np.zeros((nc * 40, 4), dtype=int)
    for i in range(nc):
      elem[np.arange(0,40) + (i * 40), :] = np.reshape(nodeshift[cube8.flatten()-1], (4, 40)).T + ind[i]

    return node, elem

#_________________________________________________________________________________________________________

def meshgrid6(*args):
    # dimension of the lattice
    n = len(args)

    # create a single n-d hypercube     # list of node of the cube itself
    vhc = (np.array(list(map(lambda x: list(bin(x)[2:].zfill(n)), range(2**n)))) == '1').astype(int)

    # permutations of the integers 1:n
    p = list(permutations(range(1, n + 1)))
    p = p[::-1]
    nt = len(p)
    thc = np.zeros((nt, n + 1), dtype=int)

    for i in range(nt):
        thc[i, :] = np.where(np.all(np.diff(vhc[:, np.array(p[i])-1], axis=1) >= 0, axis=1))[0]

    # build the complete lattice
    nodecount = np.array([len(arg) for arg in args])
    if np.any(nodecount < 2):
        raise ValueError('Each dimension must be of size 2 or more.')
    node = lattice(*args)

    # unrolled index into each hyper-rectangle in the lattice
    ind = [np.arange(nodecount[i] - 1) for i in range(n)]
    ind = np.meshgrid(*ind, indexing='ij')
    ind = np.array(ind).reshape(n, -1).T
    k = np.cumprod([1] + nodecount[:-1].tolist())

    ind = 1 + ind @ k.T #k[:-1].reshape(-1, 1)
    nind = len(ind)
    offset = vhc @ k.T
    elem = np.zeros((nt * nind, n + 1), dtype=int)
    L = np.arange(1, nind + 1).reshape(-1, 1)

    for i in range(nt):
        elem[L.flatten() - 1, :] = np.tile(ind, (n+1, 1)).T + np.tile(offset[thc[i, :]], (nind, 1))
        L += nind

    elem = elem - 1;

    return node, elem

#_________________________________________________________________________________________________________

def lattice(*args):
    n = len(args)
    sizes = [len(arg) for arg in args]
    grids = np.meshgrid(*args, indexing="ij")
    grid = np.zeros((np.prod(sizes), n))
    for i in range(n):
        grid[:, i] = grids[i].ravel()
    return grid


def latticegrid(x1, x2, x3, *args):
    if len(args) < 1:
        raise ValueError("At least one additional argument is required.")

    mesh_dims = [len(x1), len(x2), len(x3)]
    total_nodes = np.prod(mesh_dims)
    if total_nodes < 8:
        raise ValueError("Insufficient nodes for lattice grid.")

    node = np.zeros((total_nodes, 3))
    node[:, 0] = np.repeat(np.repeat(x1, len(x2)), len(x3))
    node[:, 1] = np.repeat(np.tile(x2, len(x1)), len(x3))
    node[:, 2] = np.tile(x3, len(x1) * len(x2))

    elem = np.zeros(
        (total_nodes * 40, 4), dtype=int
    )  # Assuming a max of 40 elements per node for placeholders
    cnt = 0
    for i in range(mesh_dims[0] - 1):
        for j in range(mesh_dims[1] - 1):
            for k in range(mesh_dims[2] - 1):
                n1 = i * mesh_dims[1] * mesh_dims[2] + j * mesh_dims[2] + k
                n2 = n1 + 1
                n3 = n1 + mesh_dims[2]
                n4 = n2 + mesh_dims[2]
                n5 = n1 + mesh_dims[1] * mesh_dims[2]
                n6 = n2 + mesh_dims[1] * mesh_dims[2]
                n7 = n3 + mesh_dims[1] * mesh_dims[2]
                n8 = n4 + mesh_dims[1] * mesh_dims[2]
                elem[cnt : cnt + 8] = np.array(
                    [
                        [n1, n2, n4, n3],
                        [n1, n3, n7, n5],
                        [n1, n5, n6, n2],
                        [n2, n6, n8, n4],
                        [n3, n4, n8, n7],
                        [n5, n7, n8, n6],
                    ]
                )
                cnt += 8

    return node, elem[:cnt]


def extrudecurve(c0, c1, curve, ndiv):
    if len(c0) != len(c1) or len(c0) != 3:
        raise ValueError("c0 and c1 must be 3D points of the same dimension!")

    if ndiv < 1:
        raise ValueError("ndiv must be at least 1!")

    curve = np.array(curve)
    if curve.shape[1] != 3:
        raise ValueError("curve must be a Nx3 array!")

    ncurve = curve.shape[0]
    nodes = np.zeros((ndiv * ncurve, 3))
    for i in range(ndiv):
        alpha = i / (ndiv - 1)  # linear interpolation factor
        point = (1 - alpha) * c0 + alpha * c1
        nodes[i * ncurve : (i + 1) * ncurve, :] = curve + point

    elem = np.zeros((ncurve * (ndiv - 1) * 2, 4), dtype=int)
    for i in range(ndiv - 1):
        for j in range(ncurve):
            if j < ncurve - 1:
                elem[i * ncurve * 2 + j * 2, :] = [
                    i * ncurve + j,
                    (i + 1) * ncurve + j,
                    (i + 1) * ncurve + (j + 1),
                    i * ncurve + (j + 1),
                ]
                elem[i * ncurve * 2 + j * 2 + 1, :] = [
                    (i + 1) * ncurve + j,
                    (i + 1) * ncurve + (j + 1),
                    i * ncurve + (j + 1),
                    i * ncurve + j,
                ]

    return nodes, elem


def meshcylinders(c0, c1, r, tsize=0, maxvol=0, ndiv=20):
    if np.any(np.array(r) <= 0):
        raise ValueError("Radius must be greater than zero.")

    if np.array(c0).shape != (3,) or np.array(c1).shape != (3,):
        raise ValueError("c0 and c1 must be 3D points.")

    if len(r) == 1:
        r = [r[0], r[0]]

    r = np.array(r).flatten()

    if len(r) == 2:
        r = np.array([r[0], r[0], r[1]])

    len_axis = np.linalg.norm(np.array(c1) - np.array(c0))

    if tsize == 0:
        tsize = min(r) / 10

    if maxvol == 0:
        maxvol = tsize**3 / 5

    node, face, elem = meshacylinder(c0, c1, r, tsize, maxvol, ndiv)

    return node, face, elem


def highordertet(node, elem, order=2, opt=None):
    """
    Generate a higher-order tetrahedral mesh by refining a linear tetrahedral mesh.

    Args:
        node: Nodal coordinates of the linear tetrahedral mesh (n_nodes, 3).
        elem: Element connectivity (n_elements, 4).
        order: Desired order of the output mesh (default is 2 for quadratic mesh).
        opt: Optional dictionary to control mesh refinement options.

    Returns:
        newnode: Nodal coordinates of the higher-order tetrahedral mesh.
        newelem: Element connectivity of the higher-order tetrahedral mesh.
    """

    if order < 2:
        raise ValueError("Order must be greater than or equal to 2")

    if opt is None:
        opt = {}

    # Example: linear to quadratic conversion (order=2)
    if order == 2:
        newnode, newelem = lin_to_quad_tet(node, elem)
    else:
        raise NotImplementedError(
            f"Higher order {order} mesh refinement is not yet implemented"
        )

    return newnode, newelem


def lin_to_quad_tet(node, elem):
    """
    Convert linear tetrahedral elements (4-node) to quadratic tetrahedral elements (10-node).

    Args:
        node: Nodal coordinates (n_nodes, 3).
        elem: Element connectivity (n_elements, 4).

    Returns:
        newnode: Nodal coordinates of the quadratic mesh.
        newelem: Element connectivity of the quadratic mesh.
    """

    n_elem = elem.shape[0]
    n_node = node.shape[0]

    # Initialize new node and element lists
    edge_midpoints = {}
    new_nodes = []
    new_elements = []

    for i in range(n_elem):
        element = elem[i]
        quad_element = list(element)  # Start with linear nodes

        # Loop over each edge of the tetrahedron
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        for e in edges:
            n1, n2 = sorted([element[e[0]], element[e[1]]])
            edge_key = (n1, n2)

            if edge_key not in edge_midpoints:
                # Compute midpoint and add it as a new node
                midpoint = (node[n1] + node[n2]) / 2
                new_nodes.append(midpoint)
                edge_midpoints[edge_key] = n_node + len(new_nodes) - 1

            quad_element.append(edge_midpoints[edge_key])

        new_elements.append(quad_element)

    # Combine old and new nodes
    newnode = np.vstack([node, np.array(new_nodes)])
    newelem = np.array(new_elements)

    return newnode, newelem


def elemfacecenter(node, elem):
    """
    Generate barycentric dual-mesh face center nodes and indices for each tetrahedral element.

    Args:
        node: List of node coordinates.
        elem: List of elements (each row contains the indices of nodes forming each tetrahedral element).

    Returns:
        newnode: Coordinates of new face-center nodes.
        newelem: Indices of the face-center nodes for each original tetrahedral element.
    """

    # Find unique faces from the elements (tetrahedral mesh)
    faces, idx, newelem = uniqfaces(elem[:, :4])

    # Extract the coordinates of the nodes forming these faces
    newnode = node[faces.flatten(), :3]

    # Reshape newnode to group coordinates of nodes in each face
    newnode = newnode.reshape(3, 3, faces.shape[0])

    # Compute the mean of the coordinates to find the face centers
    newnode = np.mean(newnode, axis=1)

    return newnode, newelem


def barydualmesh(node, elem, flag=None):
    """
    Generate barycentric dual-mesh by connecting edge, face, and element centers.

    Parameters:
    node : numpy.ndarray
        List of input mesh nodes.
    elem : numpy.ndarray
        List of input mesh elements (each row contains indices of nodes for each element).
    flag : str, optional
        If 'cell', outputs `newelem` as cell arrays (each with 4 nodes).

    Returns:
    newnode : numpy.ndarray
        All new nodes in the barycentric dual-mesh (made of edge, face, and element centers).
    newelem : numpy.ndarray or list
        Indices of face nodes for each original tet element, optionally in cell array format.
    """

    # Compute edge-centers
    enodes, eidx = highordertet(node, elem)

    # Compute face-centers
    fnodes, fidx = elemfacecenter(node, elem)

    # Compute element centers
    c0 = meshcentroid(node, elem[:, : min(elem.shape[1], 4)])

    # Concatenate new nodes and their indices
    newnode = np.vstack((enodes, fnodes, c0))

    newidx = np.hstack(
        (
            eidx,
            fidx + enodes.shape[0],
            np.arange(1, elem.shape[0] + 1).reshape(-1, 1)
            + enodes.shape[0]
            + fnodes.shape[0],
        )
    )

    # Element connectivity for barycentric dual-mesh (using original indexing)
    newelem = (
        np.array(
            [
                [1, 8, 11, 7],
                [2, 7, 11, 9],
                [3, 9, 11, 8],
                [4, 7, 11, 10],
                [5, 8, 11, 10],
                [6, 9, 11, 10],
            ]
        ).T
        - 1
    )  # Adjust to 0-based indexing for Python

    newelem = newidx[:, newelem.flatten()]

    newelem = newelem.reshape((elem.shape[0], 4, 6))
    newelem = np.transpose(newelem, (0, 2, 1))
    newelem = newelem.reshape((elem.shape[0] * 6, 4))

    # If the 'cell' flag is set, return `newelem` as a list of lists (cells)
    if flag == "cell":
        newelem = [newelem[i, :].tolist() for i in range(newelem.shape[0])]

    return newnode, newelem


def extrudesurf(no, fc, vec):
    """
    Create an enclosed surface mesh by extruding an open surface.

    Parameters:
    no : ndarray
        2D array containing the 3D node coordinates of the original surface.
    fc : ndarray
        2D array representing the triangular faces of the original surface.
        Each row corresponds to a triangle defined by indices of 3 nodes.
    vec : array or scalar
        If an array, defines the extrusion direction. If scalar, the normal vector
        is used and multiplied by this scalar for extrusion.

    Returns:
    node : ndarray
        3D node coordinates for the generated surface mesh.
    face : ndarray
        Triangular face patches of the generated surface mesh.
    """

    nlen = no.shape[0]  # Number of nodes in the original surface

    if len(vec) > 1:  # Extrude using a specified vector
        node = np.vstack([no, no + np.tile(vec, (nlen, 1))])
    else:  # Extrude along the surface normal
        node = np.vstack([no, no + vec * nodesurfnorm(no, fc)])

    face = np.vstack([fc, fc + nlen])  # Create top and bottom faces

    # Find surface edges and create side faces
    edge = surfedge(fc)
    sideface = np.hstack([edge, edge[:, [0]] + nlen])
    sideface = np.vstack([sideface, edge + nlen, edge[:, [1]]])

    face = np.vstack([face, sideface])  # Combine all faces

    # Perform mesh repair (fix degenerate elements, etc.)
    node, face = meshcheckrepair(node, face)

    return node, face


def meshanellip(c0, rr, tsize, maxvol=None):
    """
    Create the surface and tetrahedral mesh of an ellipsoid.

    Parameters:
    c0 : list or ndarray
        Center coordinates [x0, y0, z0] of the ellipsoid.
    rr : list or ndarray
        Radii of the ellipsoid. If rr is:
            - Scalar: a sphere with radius rr.
            - 1x3 or 3x1 vector: specifies the ellipsoid radii [a, b, c].
            - 1x5 or 5x1 vector: specifies [a, b, c, theta, phi], where theta and phi are rotation angles along the z and x axes.
    tsize : float
        Maximum surface triangle size on the ellipsoid.
    maxvol : float, optional
        Maximum volume of the tetrahedral elements.

    Returns:
    node : ndarray
        Node coordinates, 3 columns for x, y, and z respectively.
    face : ndarray
        Surface mesh face elements (each row has 3 vertices).
    elem : ndarray, optional
        Tetrahedral mesh elements (each row has 4 vertices).
    """

    rr = np.asarray(rr).flatten()

    if len(rr) == 1:
        rr = [rr[0], rr[0], rr[0]]  # Sphere case
    elif len(rr) == 3:
        pass  # Already in ellipsoid format
    elif len(rr) != 5:
        raise ValueError("Invalid rr length. See help for details.")

    rmax = min(rr[:3])

    if maxvol is None:
        maxvol = tsize**3  # Set maxvol based on tsize if not provided

    # Call meshunitsphere to generate unit sphere mesh
    if maxvol:
        node, face, elem = meshunitsphere(tsize / rmax, maxvol / (rmax**3))
    else:
        node, face = meshunitsphere(tsize / rmax)

    # Scale the unit sphere to the ellipsoid
    node = node @ np.diag(rr[:3])

    if len(rr) == 5:
        theta = rr[3]
        phi = rr[4]

        # Rotation matrices for theta (z-axis) and phi (x-axis)
        Rz = np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        Rx = np.array(
            [[1, 0, 0], [0, np.cos(phi), np.sin(phi)], [0, -np.sin(phi), np.cos(phi)]]
        )

        # Apply rotation to the node coordinates
        node = (Rz @ (Rx @ node.T)).T

    # Translate the ellipsoid to the center c0
    node += np.array(c0).reshape(1, 3)

    return node, face, elem if maxvol else (node, face)
