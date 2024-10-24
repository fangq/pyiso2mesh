"""@package docstring
Iso2Mesh for Python - Mesh data queries and manipulations

Copyright (c) 2024 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = [
    "finddisconnsurf",
    "surfedge",
    "volface",
    "extractloops",
    "meshconn",
    "meshcentroid",
    "nodevolume",
    "elemvolume",
    "neighborelem",
    "layersurf",
    "faceneighbors",
    "edgeneighbors",
    "maxsurf",
    "flatsegment",
    "surfplane",
    "surfinterior",
    "surfpart",
    "surfseeds",
    "meshquality",
    "meshedge",
    "meshface",
    "surfacenorm",
    "nodesurfnorm",
    "uniqedges",
    "uniqfaces",
    "innersurf",
    "outersurf",
    "surfvolume",
    "insurface",
    "advancefront",
    "meshreorient",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
from itertools import combinations

##====================================================================================
## implementations
##====================================================================================


def finddisconnsurf(f):
    """
    Extract disconnected surfaces from a cluster of surfaces.

    Parameters:
    f : numpy.ndarray
        Faces defined by node indices for all surface triangles.

    Returns:
    facecell : list
        Separated disconnected surface node indices.
    """

    facecell = []  # Initialize output list
    subset = np.array([])  # Initialize an empty subset array

    # Loop until all faces are processed
    while f.size > 0:
        # Find the indices of faces connected to the first face
        idx = np.isin(f, f[0, :]).reshape(f.shape)
        ii = np.where(np.sum(idx, axis=1))[0]

        # Continue until all connected faces are processed
        while ii.size > 0:
            # Append connected faces to the subset
            subset = np.vstack((subset, f[ii, :])) if subset.size else f[ii, :]
            f = np.delete(f, ii, axis=0)  # Remove processed faces
            idx = np.isin(f, subset).reshape(f.shape)  # Update connection indices
            ii = np.where(np.sum(idx, axis=1))[0]  # Find next set of connected faces

        # If the subset is non-empty, append it to the output
        if subset.size > 0:
            facecell.append(subset)
            subset = np.array([])  # Reset subset

    return facecell


def surfedge(f, junction=0):
    """
    Find the edge of an open surface or surface of a volume.

    Parameters:
    f : numpy.ndarray
        Input surface face element list, dimension (n, 3) or (n, 4).
    junction : int, optional
        If set to 1, allows finding junctions (edges with more than two connected triangles).

    Returns:
    openedge : numpy.ndarray
        List of edges of the specified surface.
    elemid : numpy.ndarray, optional
        Corresponding index of the tetrahedron or triangle with an open edge.
    """

    if f.size == 0:
        return np.array([]), None

    # Create all the edges based on the dimensionality of elements (triangles or tetrahedrons)
    if f.shape[1] == 3:
        edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    elif f.shape[1] == 4:
        edges = np.vstack(
            [f[:, [0, 1, 2]], f[:, [1, 0, 3]], f[:, [0, 2, 3]], f[:, [1, 3, 2]]]
        )
    else:
        raise ValueError(
            "surfedge only supports 2D (triangles) and 3D (tetrahedrons) elements"
        )

    # Sort each edge for uniqueness (since edge direction does not matter)
    edgesort = np.sort(edges, axis=1)

    # Find unique edges
    _, ix, jx = np.unique(edgesort, axis=0, return_index=True, return_inverse=True)

    # Histogram to count occurrences of each edge
    vec = np.bincount(jx)

    # Find open edges (edges that appear only once, or more if junctions are needed)
    if f.shape[1] == 3 and junction:
        qx = np.where(vec > 2)[0]
    else:
        qx = np.where(vec == 1)[0]

    openedge = edges[ix[qx], :]

    # If element indices are requested, compute them
    elemid = None
    if f.shape[1] == 3 and openedge.size > 0:
        elemid = ix[qx]

    return openedge, elemid


def volface(t):
    """
    Find the surface patches of a volume.

    Parameters:
    t : numpy.ndarray
        Input volumetric element list (tetrahedrons), dimension (ne, 4).

    Returns:
    openface : numpy.ndarray
        List of faces of the specified volume.
    elemid : numpy.ndarray, optional
        The corresponding index of the tetrahedron with an open edge or triangle.
    """

    # Use surfedge function to find the boundary faces of the volume
    openface, elemid = surfedge(t)

    return openface, elemid


def extractloops(edges):
    """
    Extract individual loops or polyline segments from a collection of edges.

    Parameters:
    edges : numpy.ndarray
        A two-column matrix recording the starting/ending points of all edge segments.

    Returns:
    loops : list
        Output list of polyline or loop segments, with NaN separating each loop/segment.
    """

    loops = []

    # Remove degenerate edges (edges where the start and end points are the same)
    edges = edges[edges[:, 0] != edges[:, 1], :]

    # Initialize the loop with the first edge
    loops.extend(edges[0, :])
    loophead = edges[0, 0]
    loopend = edges[0, 1]
    edges = np.delete(edges, 0, axis=0)

    while edges.size > 0:
        # Find the index of the edge connected to the current loop end
        idx = np.concatenate(
            [np.where(edges[:, 0] == loopend)[0], np.where(edges[:, 1] == loopend)[0]]
        )

        if len(idx) > 1:
            # If multiple connections found, select the first
            idx = idx[0]

        if len(idx) == 0:
            # If no connection found (open-line segment)
            idx_head = np.concatenate(
                [
                    np.where(edges[:, 0] == loophead)[0],
                    np.where(edges[:, 1] == loophead)[0],
                ]
            )
            if len(idx_head) == 0:
                # If both open ends are found, start a new loop
                loops.append(np.nan)
                loops.extend(edges[0, :])
                loophead = edges[0, 0]
                loopend = edges[0, 1]
                edges = np.delete(edges, 0, axis=0)
            else:
                # Flip and trace the other end of the loop
                loophead, loopend = loopend, loophead
                lp = np.flip(loops)
                seg = np.where(np.isnan(lp))[0]
                if len(seg) == 0:
                    loops = lp.tolist()
                else:
                    loops = (loops[: len(loops) - seg[0]] + lp[: seg[0]]).tolist()
            continue

        # Trace along a single line thread
        if len(idx) == 1:
            ed = edges[idx, :].flatten()
            ed = ed[ed != loopend]
            newend = ed[0]
            if newend == loophead:
                # If a loop is completed
                loops.extend([loophead, np.nan])
                edges = np.delete(edges, idx, axis=0)
                if edges.size == 0:
                    break
                loops.extend(edges[0, :])
                loophead = edges[0, 0]
                loopend = edges[0, 1]
                edges = np.delete(edges, 0, axis=0)
                continue
            else:
                loops.append(newend)

            loopend = newend
            edges = np.delete(edges, idx, axis=0)

    return loops


def meshconn(elem, nn):
    """
    Create a node neighbor list from a mesh.

    Parameters:
    elem : numpy.ndarray
        Element table of the mesh, where each row represents an element and its node indices.
    nn : int
        Total number of nodes in the mesh.

    Returns:
    conn : list
        A list of length `nn`, where each element is a list of all neighboring node IDs for each node.
    connnum : numpy.ndarray
        A vector of length `nn`, indicating the number of neighbors for each node.
    count : int
        Total number of neighbors across all nodes.
    """

    # Initialize conn as a list of empty lists
    conn = [[] for _ in range(nn)]
    dim = elem.shape

    # Loop through each element and populate the conn list
    for i in range(dim[0]):
        for j in range(dim[1]):
            conn[elem[i, j] - 1].extend(
                elem[i, :]
            )  # Adjust for 0-based indexing in Python

    count = 0
    connnum = np.zeros(nn, dtype=int)

    # Loop through each node to remove duplicates and self-references
    for i in range(nn):
        if len(conn[i]) == 0:
            continue
        # Remove duplicates and self-references
        neig = np.unique(conn[i])
        neig = neig[neig != i + 1]  # Remove self-reference, adjust for 0-based indexing
        conn[i] = neig.tolist()
        connnum[i] = len(conn[i])
        count += connnum[i]

    return conn, connnum, count


def meshcentroid(v, f):
    """
    Compute the centroids of a mesh defined by nodes and elements (surface or tetrahedra) in R^n space.

    Parameters:
    v : numpy.ndarray
        Surface node list, dimension (nn, 3).
    f : numpy.ndarray or list
        Surface face element list, dimension (be, 3), or a cell array (list in Python) of elements.

    Returns:
    centroid : numpy.ndarray
        Centroid positions, one row for each element.
    """

    if not isinstance(f, list):
        # If f is not a list, treat it as a regular element array
        ec = np.reshape(v[f[:, : f.shape[1]], :], (v.shape[1], f.shape[1], f.shape[0]))
        centroid = np.mean(ec, axis=1).T
    else:
        # If f is a list (similar to a cell array in MATLAB)
        len_f = len(f)
        centroid = np.zeros((len_f, v.shape[1]))

        for i in range(len_f):
            fc = f[i]
            if fc is not None and len(fc) > 0:
                vlist = fc[0]
                valid_vlist = vlist[~np.isnan(vlist)]
                centroid[i, :] = np.mean(v[valid_vlist.astype(int), :], axis=0)
            else:
                # Optionally set centroid to NaN if fc is empty
                centroid[i, :] = np.nan

    return centroid


def nodevolume(node, elem, evol=None):
    """
    Calculate the volumes of the cells in the barycentric dual-mesh.
    This is different from Voronoi cells, which belong to the circumcentric dual mesh.

    Parameters:
    node : numpy.ndarray
        Node coordinates.
    elem : numpy.ndarray
        Element table of a mesh.
    evol : numpy.ndarray, optional
        Element volumes for each element (if not provided, it will be computed).

    Returns:
    nodevol : numpy.ndarray
        Volume values for all nodes.
    """

    # Determine if the mesh is 3D or 4D based on the number of nodes per element
    dim = 4 if elem.shape[1] == 4 else 3

    # If element volumes (evol) are not provided, calculate them
    if evol is None:
        evol = elemvolume(node, elem[:, :dim])

    elemnum = elem.shape[0]
    nodenum = node.shape[0]

    # Initialize node volume array
    nodevol = np.zeros(nodenum)

    # Loop through each element and accumulate the volumes
    for i in range(elemnum):
        nodevol[elem[i, :dim]] += evol[i]

    # Divide by the dimensionality to get the final node volumes
    nodevol /= dim

    return nodevol


def elemvolume(node, elem, option=None):
    """
    Calculate the volume for a list of simplices (elements).

    Parameters:
    node : numpy.ndarray
        Node coordinates.
    elem : numpy.ndarray
        Element table of a mesh.
    option : str, optional
        If 'signed', the volume is the raw determinant; otherwise, absolute values are returned.

    Returns:
    vol : numpy.ndarray
        Volume values for all elements.
    """

    if elem.shape[1] == node.shape[1]:  # For 2D elements (triangles)
        enum = elem.shape[0]
        vol = np.zeros(enum)
        acol = np.ones(3)  # Column of ones for determinant computation

        for i in range(enum):
            e1 = np.linalg.det(np.c_[node[elem[i, :], 1], node[elem[i, :], 2], acol])
            e2 = np.linalg.det(np.c_[node[elem[i, :], 2], node[elem[i, :], 0], acol])
            e3 = np.linalg.det(np.c_[node[elem[i, :], 0], node[elem[i, :], 1], acol])
            vol[i] = np.sqrt(e1 * e1 + e2 * e2 + e3 * e3) * 0.5
        return vol

    dim = elem.shape[1]  # For higher-dimensional elements (3D tetrahedra)
    enum = elem.shape[0]
    vol = np.zeros(enum)

    for i in range(enum):
        detmat = np.vstack([node[elem[i, :], :].T, np.ones(dim)])
        vol[i] = np.linalg.det(detmat)

    if option == "signed":
        vol /= np.prod(np.arange(1, node.shape[1] + 1))
    else:
        vol = np.abs(vol) / np.prod(np.arange(1, node.shape[1] + 1))

    return vol


def neighborelem(elem, nn):
    """
    Create a node neighbor list from a mesh.

    Parameters:
    elem : numpy.ndarray
        Element table of the mesh, where each row represents an element and its node indices.
    nn : int
        Total number of nodes in the mesh.

    Returns:
    conn : list
        A list of length `nn`, where each element is a list of all neighboring element IDs for each node.
    connnum : numpy.ndarray
        A vector of length `nn`, indicating the number of neighboring elements for each node.
    count : int
        Total number of neighboring elements across all nodes.
    """

    # Initialize conn as a list of empty lists
    conn = [[] for _ in range(nn)]
    dim = elem.shape

    # Loop through each element and populate the conn list
    for i in range(dim[0]):
        for j in range(dim[1]):
            conn[elem[i, j] - 1].append(i + 1)  # Adjust for 0-based indexing

    count = 0
    connnum = np.zeros(nn, dtype=int)

    # Loop through each node to sort neighbors and calculate total counts
    for i in range(nn):
        conn[i] = sorted(conn[i])
        connnum[i] = len(conn[i])
        count += connnum[i]

    return conn, connnum, count


def layersurf(elem, **kwargs):
    """
    face, labels = layersurf(elem, opt)
    or
    face, labels = layersurf(elem, option1=value1, option2=value2, ...)

    Process a multi-layered tetrahedral mesh to extract the layer surface meshes.

    Arguments:
    elem : an Nx5 integer array representing the tetrahedral mesh element list.
           The first 4 columns represent the tetrahedral element node indices,
           and the last column represents tissue labels.

    Optional kwargs:
    order : str, default '>=', options ['>=', '=', '<=']
        Determines how to process layers:
        '>=' (default): outmost layer has the lowest label count;
        '<=': innermost is lowest;
        '=': surface of each label is extracted individually.
    innermost : array-like, default [0]
        Labels treated as innermost regions, its boundary extracted using '=='.
        By default, label 0 is assumed to be the innermost (i.e., nothing enclosed inside).
    unique : bool, default False
        If True, removes duplicated triangles. If False, keeps all triangles.
    occurrence : str, default 'first', options ['first', 'last']
        If 'first', unique operator keeps the duplicated triangle with the lowest label number;
        otherwise, keeps the triangle with the highest label number.

    Returns:
    face : Nx4 array
        Extracted surface faces.
    labels : list
        Unique sorted labels in the mesh.
    """
    # Process input options
    opt = kwargs
    outsideislower = opt.get("order", ">=")
    dounique = opt.get("unique", False)
    innermost = opt.get("innermost", [0])
    occurrence = opt.get("occurrence", "first")

    labels = np.sort(np.unique(elem[:, 4]))
    face = []

    # Process each label
    for i in range(len(labels)):
        if outsideislower == ">=" and labels[i] not in innermost:
            newface = volface(elem[elem[:, 4] >= labels[i], :4])
        elif outsideislower == "<=" and labels[i] not in innermost:
            newface = volface(elem[elem[:, 4] <= labels[i], :4])
        else:
            newface = volface(elem[elem[:, 4] == labels[i], :4])

        # Add label to faces
        newface = np.hstack((newface, np.full((newface.shape[0], 1), labels[i])))
        face.append(newface)

    face = np.vstack(face)

    # Remove duplicate triangles if unique option is enabled
    if dounique:
        face[:, :3] = np.sort(face[:, :3], axis=1)
        uniqface, idx = np.unique(face[:, :3], axis=0, return_index=True)
        face = np.hstack((uniqface, face[idx, -1].reshape(-1, 1)))

    return face, labels


def faceneighbors(t, opt=None):
    """
    facenb = faceneighbors(t, opt)

    Find the 4 face-neighboring elements of a tetrahedron.

    Arguments:
    t   : tetrahedron element list, 4 columns of integers.
    opt : if 'surface', return the boundary triangle list
          (same as face output from v2m).
          if 'rowmajor', return boundary triangles in row-major order.

    Output:
    facenb : If opt is 'surface', returns the list of boundary triangles.
             Otherwise, returns element neighbors for each element. Each
             row contains 4 numbers representing the element indices
             sharing triangular faces [1 2 3], [1 2 4], [1 3 4], and
             [2 3 4]. A 0 indicates no neighbor (i.e., boundary face).
    """
    # Generate faces from tetrahedral elements
    faces = np.hstack(
        (t[:, [0, 1, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 3]], t[:, [1, 2, 3]])
    )
    faces = np.sort(faces, axis=1)

    # Find unique faces and their indices
    _, ix, jx = np.unique(faces, axis=0, return_inverse=True)

    # Determine the system being used
    isoctavemesh = False  # Adjust this based on your environment
    if isoctavemesh:
        u = np.unique(jx)
        qx = u[np.histogram(jx, bins=np.arange(max(jx) + 2))[0] == 2]
    else:
        vec = np.histogram(jx, bins=np.arange(max(jx) + 2))[0]
        qx = np.where(vec == 2)[0]

    nn = np.max(t)
    ne = t.shape[0]
    facenb = np.zeros_like(t)

    # Identify duplicate faces and their element pairings
    ujx, ii = np.unique(jx, return_index=True)
    _, ii2 = np.unique(jx, return_index=True)

    # List of element pairs that share a common face
    iddup = np.vstack((ii[qx], ii2[qx])).T
    faceid = np.ceil(iddup / ne).astype(int)
    eid = np.mod(iddup, ne)
    eid[eid == 0] = ne

    # Rearrange the list into an element format
    for i in range(len(qx)):
        facenb[eid[i, 0] - 1, faceid[i, 0] - 1] = eid[i, 1]
        facenb[eid[i, 1] - 1, faceid[i, 1] - 1] = eid[i, 0]

    # 0s in facenb indicate boundary faces with no neighbors
    # Handle special cases based on the second argument
    if opt is not None:
        if opt == "surface":
            facenb = faces[np.where(facenb == 0)[0], :]
        elif opt == "rowmajor":
            index = np.arange(len(faces)).reshape(-1, 4).T.flatten()
            faces = faces[index, :]
            facenb = faces[np.where(facenb.T == 0)[0], :]
        else:
            raise ValueError(f'Unsupported option "{opt}".')

    return facenb


def edgeneighbors(t, opt=None):
    """
    edgenb = edgeneighbors(t, opt)

    Find neighboring triangular elements in a triangular surface.

    Arguments:
    t   : a triangular surface element list, 3 columns of integers.
    opt : (optional) If 'general', return edge neighbors for a general triangular surface.
          Each edge can be shared by more than 2 triangles. If ignored, assumes all
          triangles are shared by no more than 2 triangles.

    Output:
    edgenb : If opt is not supplied, edgenb is a size(t, 1) by 3 array, with each element
             being the triangle ID of the edge neighbor of that triangle. For each row,
             the neighbors are listed in the order of those sharing edges [1, 2], [2, 3],
             and [3, 1] between the triangle nodes.
             If opt = 'general', edgenb is a list of arrays, where each entry lists the edge neighbors.
    """
    # Generate the edges from the triangle elements
    edges = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    edges = np.sort(edges, axis=1)

    # Find unique edges and their indices
    _, ix, jx = np.unique(edges, axis=0, return_index=True, return_inverse=True)

    ne = t.shape[0]  # Number of triangles
    if opt == "general":
        edgenb = [np.where(jx == jx[i])[0] for i in range(len(jx))]
        return [np.setdiff1d(nb, [i]) for i, nb in enumerate(edgenb)]

    # Determine boundary neighbors
    vec = np.bincount(jx)
    qx = np.where(vec == 2)[
        0
    ]  # Get indices where edges are shared by exactly 2 triangles

    edgenb = np.zeros_like(t)
    ujx, first_idx = np.unique(jx, return_index=True)
    _, last_idx = np.unique(jx[::-1], return_index=True)
    last_idx = len(jx) - last_idx - 1

    # Find the element pairs that share an edge
    iddup = np.vstack([first_idx[qx], last_idx[qx]]).T
    faceid = (iddup // ne) + 1
    eid = iddup % ne
    eid[eid == 0] = ne

    # Assign neighboring elements
    for i in range(len(qx)):
        edgenb[eid[i, 0], faceid[i, 0]] = eid[i, 1]
        edgenb[eid[i, 1], faceid[i, 1]] = eid[i, 0]

    # Handle boundary edges (where no neighbor exists)
    return edgenb


def maxsurf(facecell, node=None):
    """
    f, maxsize = maxsurf(facecell, node)

    Return the surface with the maximum number of elements or total area from a cell array of surfaces.

    Arguments:
    facecell : a list of arrays, each element representing a face array.
    node     : optional, node list. If given, the output is the surface with the largest surface area.

    Output:
    f        : the surface data (node indices) for the surface with the most elements (or largest area if node is given).
    maxsize  : if node is not provided, maxsize is the row number of f.
               If node is given, maxsize is the total area of f.
    """
    maxsize = -1
    maxid = -1

    # If node is provided, calculate area for each surface
    if node is not None:
        areas = np.zeros(len(facecell))
        for i in range(len(facecell)):
            areas[i] = np.sum(elemvolume(node[:, :3], facecell[i]))
        maxsize = np.max(areas)
        maxid = np.argmax(areas)
        f = facecell[maxid]
        return f, maxsize
    else:
        # Find the surface with the most elements
        for i in range(len(facecell)):
            if len(facecell[i]) > maxsize:
                maxsize = len(facecell[i])
                maxid = i

        f = []
        if maxid >= 0:
            f = facecell[maxid]

        return f, maxsize


def flatsegment(node, edge):
    """
    mask = flatsegment(node, edge)

    Decompose edge loops into flat segments along arbitrary planes of the bounding box.

    Arguments:
    node : Nx3 array of x, y, z coordinates for each node of the mesh.
    edge : vector separated by NaN, each segment is a closed polygon consisting of node IDs.

    Output:
    mask : list, each element is a closed polygon on the x/y/z plane.

    Author: Qianqian Fang
    Date: 2008/04/08

    Notes:
    This function is fragile: it cannot handle curves with many collinear nodes near corner points.
    """

    idx = edge
    nn = len(idx)
    val = np.zeros(nn)

    # Check for nearly flat tetrahedrons
    for i in range(nn):
        tet = np.mod(np.arange(i, i + 4), nn)
        tet[tet == 0] = nn
        # Calculate determinant to determine flatness
        val[i] = (
            abs(np.linalg.det(np.hstack((node[idx[tet], :], np.ones((4, 1)))))) > 1e-5
        )

    val = np.concatenate((val, val[:2]))
    mask = []
    oldend = 0
    count = 0

    # Decompose into flat segments
    for i in range(nn):
        if val[i] == 1 and val[i + 1] == 1 and val[i + 2] == 0:
            val[i + 2] = 2
            mask.append(idx[oldend : i + 3])
            count += 1
            oldend = i + 2
        else:
            mask.append(np.concatenate((idx[oldend:], mask[0])))
            break

    return mask


def mesheuler(face):
    """
    X, V, E, F = mesheuler(face)

    Compute Euler's characteristic of a mesh.

    Arguments:
    face : a closed surface mesh (Mx3 array where M is the number of faces and each row contains vertex indices)

    Output:
    X : Euler's characteristic (X = 2 - 2 * g, where g is the genus)
    V : number of vertices
    E : number of edges
    F : number of faces

    Author: Qianqian Fang
    This function is part of the iso2mesh toolbox (http://iso2mesh.sf.net)
    """

    # Number of vertices
    V = len(np.unique(face))

    # Construct edges from faces
    E = np.vstack((face[:, [0, 2]], face[:, [0, 1]], face[:, [1, 2]]))
    E = np.unique(
        np.sort(E, axis=1), axis=0
    )  # Sort edge vertex pairs and remove duplicates
    E = len(E)

    # Number of faces
    F = face.shape[0]

    # Euler's characteristic formula: X = V - E + F
    X = V - E + F

    return X, V, E, F


def orderloopedge(edge):
    """
    newedge = orderloopedge(edge)

    Order the node list of a simple loop based on the connection sequence.

    Arguments:
    edge : an Nx2 array where each row is an edge defined by two integers (start/end node index).

    Output:
    newedge : Nx2 array of reordered edge node list.

    Author: Qianqian Fang
    Date: 2008/05

    Notes:
    This function cannot process bifurcations.
    """

    ne = edge.shape[0]
    newedge = np.zeros_like(edge)
    newedge[0, :] = edge[0, :]

    for i in range(1, ne):
        row, col = np.where(edge[i:, :] == newedge[i - 1, 1])
        if len(row) == 1:
            newedge[i, :] = [newedge[i - 1, 1], edge[row[0] + i, 1 - col[0]]]
            edge[[i, row[0] + i], :] = edge[[row[0] + i, i], :]
        elif len(row) >= 2:
            raise ValueError("Bifurcation is found, exiting.")
        elif len(row) == 0:
            raise ValueError(f"Open curve at node {newedge[i - 1, 1]}")

    return newedge


def bbxflatsegment(node, edge):
    """
    mask = bbxflatsegment(node, edge)

    Decompose edge loops into flat segments along arbitrary planes of the bounding box.

    Arguments:
    node : Nx3 array of x, y, z coordinates for each node of the mesh.
    edge : vector separated by NaN, each segment is a closed polygon consisting of node IDs.

    Output:
    mask : list, each element is a closed polygon on the x/y/z plane.

    Author: Qianqian Fang
    Date: 2008/04/08

    Notes:
    This function is fragile: it cannot handle curves with many collinear nodes near corner points.
    """

    idx = edge
    nn = len(idx)
    val = np.zeros(nn)

    # Check for nearly flat tetrahedrons
    for i in range(nn):
        tet = np.mod(np.arange(i, i + 4), nn)
        tet[tet == 0] = nn
        # Calculate determinant to determine flatness
        val[i] = (
            abs(np.linalg.det(np.hstack((node[idx[tet], :], np.ones((4, 1)))))) > 1e-5
        )

    val = np.concatenate((val, val[:2]))
    mask = []
    oldend = 0
    count = 0

    # Decompose into flat segments
    for i in range(nn):
        if val[i] == 1 and val[i + 1] == 1 and val[i + 2] == 0:
            val[i + 2] = 2
            mask.append(idx[oldend : i + 3])
            count += 1
            oldend = i + 2
        else:
            mask.append(np.concatenate((idx[oldend:], mask[0])))
            break

    return mask


def surfplane(node, face):
    """
    plane = surfplane(node, face)

    Calculate plane equation coefficients for each face in a surface.

    Parameters:
    node : numpy array
        A list of node coordinates (nn x 3)
    face : numpy array
        A surface mesh triangle list (ne x 3)

    Returns:
    plane : numpy array
        A (ne x 4) array where each row has [a, b, c, d] to represent
        the plane equation as "a*x + b*y + c*z + d = 0"
    """

    # Compute vectors AB and AC from the triangle vertices
    AB = node[face[:, 1], :] - node[face[:, 0], :]
    AC = node[face[:, 2], :] - node[face[:, 0], :]

    # Compute normal vectors to the triangles using cross product
    N = np.cross(AB, AC)

    # Compute the plane's d coefficient by taking the dot product of normal vectors with vertex positions
    d = -np.einsum("ij,ij->i", N, node[face[:, 0], :])

    # Return the plane coefficients [a, b, c, d]
    plane = np.hstack((N, d[:, np.newaxis]))

    return plane


def surfinterior(node, face):
    """
    pt, p0, v0, t, idx = surfinterior(node, face)

    Identify a point that is enclosed by the (closed) surface.

    Arguments:
    node : a list of node coordinates (nn x 3)
    face : a surface mesh triangle list (ne x 3)

    Output:
    pt  : the interior point coordinates [x, y, z]
    p0  : ray origin used to determine the interior point
    v0  : the vector used to determine the interior point
    t   : ray-tracing intersection distances (with signs) from p0. The intersection coordinates
          can be expressed as p0 + t[i] * v0
    idx : index to the face elements that intersect with the ray, order matches that of t

    Author: Qianqian Fang
    This function is part of the iso2mesh toolbox (http://iso2mesh.sf.net)
    """

    pt = []
    len_faces = face.shape[0]

    for i in range(len_faces):
        p0 = np.mean(
            node[face[i, :3], :], axis=0
        )  # Calculate the centroid of the triangle
        plane = surfplane(node, face[i, :])  # Plane equation for the current triangle
        v0 = plane[:3]  # Use the plane normal vector as the direction of the ray
        t, u, v = raytrace(p0, v0, node, face[:, :3])  # Perform ray-tracing
        idx = np.where((u >= 0) & (v >= 0) & (u + v <= 1.0) & (~np.isinf(t)))[
            0
        ]  # Filter valid intersections

        # Sort and ensure ray intersections are valid
        ts, uidx = np.unique(np.sort(t[idx]), return_index=True)
        if len(ts) > 0 and len(ts) % 2 == 0:
            ts = ts.reshape((2, len(ts) // 2))
            tdiff = ts[1, :] - ts[0, :]
            maxi = np.argmax(tdiff)
            pt = (
                p0 + v0 * (ts[0, maxi] + ts[1, maxi]) * 0.5
            )  # Calculate the midpoint of the longest segment
            idx = idx[uidx]
            t = t[idx]
            break

    return pt, p0, v0, t, idx


def surfpart(f, loopedge):
    """
    elist = surfpart(f, loopedge)

    Partition a triangular surface using a closed loop defined by existing edges.

    Parameters:
    f : numpy array
        Surface face element list, dimension (n, 3) or (n, 4)
    loopedge : numpy array
        A 2-column array specifying a closed loop in counter-clockwise order.

    Returns:
    elist : numpy array
        List of triangles that is enclosed by the loop.
    """
    elist = []

    # Check if input is empty
    if f.size == 0 or loopedge.size == 0:
        return np.array(elist)

    # Handle triangular or quadrilateral elements
    if f.shape[1] == 3:
        # Create edges from triangles
        edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    elif f.shape[1] == 4:
        # Create edges from quadrilaterals
        edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 3]], f[:, [3, 0]]])
    else:
        raise ValueError("surfpart only supports triangular and quadrilateral elements")

    # Advance the front using the edges and loop
    elist, front = advancefront(edges, loopedge)

    # Continue advancing the front until no more elements can be added
    while front.size > 0:
        elist0, front0 = advancefront(edges, front)
        elist = np.unique(np.vstack([elist, elist0]), axis=0)
        front = front0

    return elist


def surfseeds(node, face):
    """
    seeds = surfseeds(node, face)

    Calculate a set of interior points, with each enclosed by a closed
    component of a surface.

    Parameters:
    node : numpy array
        A list of node coordinates (nn x 3).
    face : numpy array
        A surface mesh triangle list (ne x 3).

    Returns:
    seeds : numpy array
        Interior point coordinates for each closed surface component.
    """

    # Find disconnected surface components
    fc = finddisconnsurf(face[:, 0:3])
    len_fc = len(fc)

    # Initialize seed points array
    seeds = np.zeros((len_fc, 3))

    # For each disconnected component, calculate the interior point
    for i in range(len_fc):
        seeds[i, :] = surfinterior(node, fc[i])

    return seeds


def meshquality(node, elem, maxnode=4):
    """
    quality = meshquality(node, elem, maxnode=4)

    Compute the Joe-Liu mesh quality measure of an N-D mesh (N <= 3).

    Parameters:
    node : numpy array
        Node coordinates of the mesh (nn x 3).
    elem : numpy array
        Element table of an N-D mesh (ne x (N+1)).
    maxnode : int, optional
        Maximum number of nodes per element (default is 4 for tetrahedral).

    Returns:
    quality : numpy array
        A vector of the same length as size(elem,1), with each element being
        the Joe-Liu mesh quality metric (0-1) of the corresponding element.
        A value close to 1 represents higher mesh quality (1 means equilateral tetrahedron);
        a value close to 0 means a nearly degenerated element.

    Reference:
    A. Liu, B. Joe, Relationship between tetrahedron shape measures,
    BIT 34 (2) (1994) 268-287.
    """

    if elem.shape[1] > maxnode:
        elem = elem[:, :maxnode]

    enum = elem.shape[0]

    # Compute element volume
    vol = elemvolume(node, elem)

    # Compute edge lengths
    edges = meshedge(elem)
    ed = node[edges[:, 0], :] - node[edges[:, 1], :]
    ed = np.sum(ed**2, axis=1)
    ed = np.sum(ed.reshape((enum, len(ed) // enum)), axis=1)

    dim = elem.shape[1] - 1
    coeff = 10 / 9  # for tetrahedral elements

    if dim == 2:
        coeff = 1

    # Compute quality metric
    quality = (
        coeff
        * dim
        * 2 ** (2 * (1 - 1 / dim))
        * 3 ** ((dim - 1) / 2)
        * vol ** (2 / dim)
        / ed
    )

    # Normalize quality if max quality > 1
    maxquality = np.max(quality)
    if maxquality > 1:
        quality = quality / maxquality

    return quality


def meshedge(elem, opt=None):
    """
    edges = meshedge(elem, opt=None)

    Return all edges in a surface or volumetric mesh.

    Parameters:
    elem : numpy array
        Element table of a mesh (supports N-dimensional space elements).
    opt : dict, optional
        Optional input. If opt is provided as a dictionary, it can have the following field:
        - opt['nodeorder']: If 1, assumes the elem node indices are in CCW orientation;
                            if 0, uses combinations to order edges.

    Returns:
    edges : numpy array
        Edge list; each row represents an edge, specified by the starting and
        ending node indices. The total number of edges is size(elem,1) x comb(size(elem,2),2).
        All edges are ordered by looping through each element.
    """

    # Determine element dimension and the combination of node pairs for edges
    dim = elem.shape
    edgeid = list(combinations(range(dim[1]), 2))
    len_edges = len(edgeid)

    # Initialize edge list
    edges = np.zeros((dim[0] * len_edges, 2), dtype=int)

    # Populate edges by looping through each element
    for i in range(len_edges):
        edges[i * dim[0] : (i + 1) * dim[0], :] = np.vstack(
            [elem[:, edgeid[i][0]], elem[:, edgeid[i][1]]]
        ).T

    return edges


def meshface(elem, opt=None):
    """
    faces = meshface(elem, opt=None)

    Return all faces in a surface or volumetric mesh.

    Parameters:
    elem : numpy array
        Element table of a mesh (supports N-dimensional space elements).
    opt : dict, optional
        Optional input. If provided, `opt` can contain the following field:
        - opt['nodeorder']: If 1, assumes the elem node indices are in CCW orientation;
                            otherwise, uses combinations to order faces.

    Returns:
    faces : numpy array
        Face list; each row represents a face, specified by node indices.
        The total number of faces is size(elem,1) x comb(size(elem,2),3).
        All faces are ordered by looping through each element.
    """

    dim = elem.shape
    faceid = list(combinations(range(dim[1]), 3))
    len_faces = len(faceid)

    # Initialize face list
    faces = np.zeros((dim[0] * len_faces, 3), dtype=int)

    # Populate faces by looping through each element
    for i in range(len_faces):
        faces[i * dim[0] : (i + 1) * dim[0], :] = np.vstack(
            [elem[:, faceid[i][0]], elem[:, faceid[i][1]], elem[:, faceid[i][2]]]
        ).T

    return faces


def surfacenorm(node, face, normalize=True):
    """
    snorm = surfacenorm(node, face, normalize=True)

    Compute the normal vectors for a triangular surface.

    Parameters:
    node : numpy array
        A list of node coordinates (nn x 3).
    face : numpy array
        A surface mesh triangle list (ne x 3).
    normalize : bool, optional
        If set to True, the normal vectors will be unitary (default is True).

    Returns:
    snorm : numpy array
        Output surface normal vector at each face.
    """

    # Compute the normal vectors using surfplane (function must be defined)
    snorm = surfplane(node, face)[:, :3]

    # Normalize the normal vectors if requested
    if normalize:
        snorm = snorm / np.linalg.norm(snorm, axis=1)[:, np.newaxis]

    return snorm


def nodesurfnorm(node, elem):
    """
    nv = nodesurfnorm(node, elem)

    Calculate a nodal normal for each vertex on a surface mesh (the surface
    can only be triangular or cubic).

    Parameters:
    node : numpy array
        Node coordinates of the surface mesh (nn x 3).
    elem : numpy array
        Element list of the surface mesh (3 columns for triangular mesh,
        4 columns for cubic surface mesh).

    Returns:
    nv : numpy array
        Nodal normals calculated for each node (nn x 3).
    """

    nn = node.shape[0]  # Number of nodes
    ne = elem.shape[0]  # Number of elements
    nedim = elem.shape[1]  # Element dimension

    # Compute element normals
    ev = surfacenorm(node, elem)

    # Initialize nodal normals
    nv = np.zeros((nn, 3))

    # Sum element normals for each node
    for i in range(ne):
        nv[elem[i, :], :] += ev[i, :]

    # Normalize nodal normals
    nvnorm = np.linalg.norm(nv, axis=1)
    idx = np.where(nvnorm > 0)[0]

    if len(idx) < nn:
        print("Warning: Found interior nodes, their norms will be set to zeros.")
        nv[idx, :] = nv[idx, :] / nvnorm[idx][:, np.newaxis]
    else:
        nv = nv / nvnorm[:, np.newaxis]

    return nv


def uniqedges(elem):
    """
    edges, idx, edgemap = uniqedges(elem)

    Return the unique edge list from a surface or tetrahedral mesh.

    Parameters:
    elem : numpy array
        A list of elements, where each row is a list of nodes for an element.
        The input `elem` can have 2, 3, or 4 columns.

    Returns:
    edges : numpy array
        Unique edges in the mesh, denoted by pairs of node indices.
    idx : numpy array
        Indices of the unique edges in the raw edge list (returned by meshedge).
    edgemap : numpy array
        Index of the raw edges in the output list (for triangular meshes).
    """

    # Handle cases based on element size
    if elem.shape[1] == 2:
        edges = elem
    elif elem.shape[1] >= 3:
        edges = meshedge(elem)
    else:
        raise ValueError("Invalid input: element size not supported.")

    # Find unique edges and indices
    sorted_edges = np.sort(edges, axis=1)
    uedges, idx, jdx = np.unique(
        sorted_edges, axis=0, return_index=True, return_inverse=True
    )
    edges = edges[idx, :]

    # Compute edgemap if requested
    edgemap = None
    if elem.shape[1] > 2:
        edgemap = jdx.reshape(elem.shape[0], -1)

    return edges, idx, edgemap


def uniqfaces(elem):
    """
    faces, idx, facemap = uniqfaces(elem)

    Return the unique face list from a surface or tetrahedral mesh.

    Parameters:
    elem : numpy array
        A list of elements, where each row contains node indices for an element.
        The input `elem` can have 2, 3, or 4 columns.

    Returns:
    faces : numpy array
        Unique faces in the mesh, denoted by triplets of node indices.
    idx : numpy array
        Indices of the unique faces in the raw face list (returned by meshface).
    facemap : numpy array
        Index of the raw faces in the output list (for triangular meshes).
    """

    # Determine faces based on element size
    if elem.shape[1] == 3:
        faces = elem
    elif elem.shape[1] >= 4:
        faces = meshface(elem)
    else:
        raise ValueError("Invalid input: element size not supported.")

    # Find unique faces and their indices
    sorted_faces = np.sort(faces, axis=1)
    ufaces, idx, jdx = np.unique(
        sorted_faces, axis=0, return_index=True, return_inverse=True
    )
    faces = faces[idx, :]

    # Compute facemap if requested
    facemap = None
    if elem.shape[1] > 3:
        facemap = jdx.reshape(elem.shape[0], -1)

    return faces, idx, facemap


def innersurf(node, face, outface=None):
    """
    Extract the interior triangles (shared by two enclosed compartments)
    of a complex surface.

    Parameters:
    node: Node coordinates
    face: Surface triangle list
    outface: (Optional) the exterior triangle list, if not provided,
             will be computed using outersurf().

    Returns:
    inface: The collection of interior triangles of the surface mesh
    """

    # If outface is not provided, compute it using outersurf
    if outface is None:
        outface = outersurf(node, face)

    # Check membership of sorted faces in sorted outface, row-wise
    tf = ismember_rows(np.sort(face, axis=1), np.sort(outface, axis=1))

    # Select faces not part of the exterior (tf == 0)
    inface = face[tf == 0, :]

    return inface


def ismember_rows(A, B):
    """
    Check if rows of A are present in B.

    Parameters:
    A: Input array A
    B: Input array B

    Returns:
    A boolean array where True indicates the presence of a row of A in B
    """
    dtype = np.dtype((np.void, A.dtype.itemsize * A.shape[1]))
    A_view = np.ascontiguousarray(A).view(dtype)
    B_view = np.ascontiguousarray(B).view(dtype)
    return np.in1d(A_view, B_view)


def outersurf(node, face):
    """
    Extract the outer-most shell of a complex surface mesh.

    Parameters:
    node: Node coordinates
    face: Surface triangle list

    Returns:
    outface: The outer-most shell of the surface mesh
    """

    face = face[:, :3]  # Limit face to first 3 columns
    ed = surfedge(face)  # Find surface edges

    # If surface is open, raise an error
    if ed.size != 0:
        raise ValueError(
            "Open surface detected, close it first. Consider meshcheckrepair() with meshfix option."
        )

    # Fill the surface and extract the volume's outer surface
    no, el = fillsurf(node, face)
    outface = volface(el)

    # Remove isolated nodes
    no, outface = removeisolatednode(no, outface)

    # Check matching of node coordinates
    maxfacenode = np.max(outface)
    I, J = ismember_rows(np.round(no[:maxfacenode, :] * 1e10), np.round(node * 1e10))

    # Map faces to the original node set
    outface = J[outface]

    # Remove faces with unmapped (zero-indexed) nodes
    ii, jj = np.where(outface == 0)
    outface = np.delete(outface, ii, axis=0)

    return outface


def surfvolume(node, face, option=None):
    """
    Calculate the enclosed volume for a closed surface.

    Parameters:
    node: Node coordinates
    face: Surface triangle list
    option: (Optional) additional option, currently unused

    Returns:
    vol: Total volume of the enclosed space
    """

    face = face[:, :3]  # Limit face to first 3 columns
    ed = surfedge(face)  # Detect surface edges

    # If surface is open, raise an error
    if ed.size != 0:
        raise ValueError(
            "Open surface detected, you need to close it first. Consider meshcheckrepair() with the meshfix option."
        )

    # Fill the surface and calculate the volume of enclosed elements
    no, el = fillsurf(node, face)
    vol = elemvolume(no, el)

    # Sum the volume of all elements
    vol = np.sum(vol)

    return vol


def insurface(node, face, points):
    """
    Test if a set of 3D points is located inside a 3D triangular surface.

    Parameters:
    node: Node coordinates (Nx3 array)
    face: Surface triangle list (Mx3 array)
    points: A set of 3D points to test (Px3 array)

    Returns:
    tf: A binary vector of length equal to the number of points.
        1 indicates the point is inside the surface, and 0 indicates outside.
    """

    from scipy.spatial import Delaunay

    # Fill the surface and get nodes and elements
    no, el = fillsurf(node, face)

    # Check if points are inside the surface using Delaunay triangulation
    tri = Delaunay(no)
    tf = tri.find_simplex(points) >= 0

    # Set points inside the surface to 1, and outside to 0
    tf = tf.astype(int)

    return tf


def advancefront(edges, loop, elen=3):
    """
    advance an edge-front on an oriented surface to the next separated by
    one-element width

    Author: Qianqian Fang
    Date: 2012/02/09

    Input:
    edges: edge list of an oriented surface mesh, must be in CCW order
    loop: a 2-column array, specifying a closed loop in CCW order
    elen: node number inside each element, if ignored, elen is set to 3

    Output:
    elist: list of triangles enclosed between the two edge-fronts
    nextfront: a new edge loop list representing the next edge-front
    """

    # Initialize output variables
    elist = []
    nextfront = []

    # Check if elen is provided, if not, set to 3
    if elen is None:
        elen = 3

    # Find edges that are part of the loop
    hasedge, loc = ismember(loop, edges)

    # If any edges in loop are not in the mesh, raise an error
    if not np.all(hasedge):
        raise ValueError("Loop edge is not defined in the mesh")

    # Calculate number of nodes in the mesh
    nodenum = len(edges) // elen

    # Find unique elements in the loop
    elist = np.unique((loc - 1) % nodenum) + 1

    # Get the corresponding edges for elist
    nextfront = edges[elist, :]

    # Loop through remaining elements
    for i in range(1, elen):
        nextfront = np.vstack([nextfront, edges[elist + nodenum * i, :]])

    # Remove reversed edge pairs
    nextfront = setdiff(nextfront, loop)
    flag, loc = ismember(nextfront, np.flip(nextfront, axis=1))

    id = np.where(flag)[0]
    if len(id) > 0:
        delmark = flag
        delmark[loc[loc > 0]] = 1
        nextfront = np.delete(nextfront, np.where(delmark), axis=0)

    # Reverse this loop as it is reversed relative to the input loop
    nextfront = nextfront[:, [1, 0]]

    return elist, nextfront


def ismember(A, B):
    """
    Check if rows of A are present in B.

    Returns a boolean array indicating membership and the corresponding indices.
    """
    return np.in1d(
        A.view([("", A.dtype)] * A.shape[1]), B.view([("", B.dtype)] * B.shape[1])
    ), np.where(np.in1d(A, B))


def setdiff(A, B):
    """
    Find the set difference between arrays A and B, row-wise.
    """
    dtype = np.dtype((np.void, A.dtype.itemsize * A.shape[1]))
    A_view = np.ascontiguousarray(A).view(dtype)
    B_view = np.ascontiguousarray(B).view(dtype)
    return A[~np.in1d(A_view, B_view)]


def meshreorient(node, elem):
    """
    Reorder nodes in a surface or tetrahedral mesh to ensure all elements
    are oriented consistently.

    Parameters:
    node: Node coordinates (Nx3 array for x, y, z)
    elem: Element list (Mx3 or Mx4 array for surface or volume elements)

    Returns:
    newelem: Element list with consistent ordering
    evol: Signed element volume before reorientation
    idx: Indices of the elements that had negative volume
    """

    # Calculate the signed volume of the elements
    evol = elemvolume(node, elem, "signed")

    # Find elements with negative volume
    idx = np.where(evol < 0)[0]

    # Reorder the last two nodes for elements with negative volume
    elem[idx, [-2, -1]] = elem[idx, [-1, -2]]

    newelem = elem
    return newelem, evol, idx
