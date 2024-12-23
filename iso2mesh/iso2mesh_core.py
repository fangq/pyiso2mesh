"""@package docstring
Iso2Mesh for Python - Mesh data queries and manipulations

Copyright (c) 2024 Qianqian Fang <q.fang at neu.edu>
"""
__all__ = [
    "surf2mesh",
    "vol2restrictedtri",
    "removeisolatednode",
]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import os
import re
import platform
import subprocess
import iso2mesh as im

##====================================================================================
## implementations
##====================================================================================


def v2m(img, isovalues, opt=None, maxvol=None, method=None):
    """
    Volumetric mesh generation from binary or gray-scale volumetric images.

    Parameters:
    img       : 3D numpy array, volumetric image data
    isovalues : scalar or list, isovalues to generate meshes
    opt       : options for mesh generation (default: None)
    maxvol    : maximum volume for elements (default: None)
    method    : method for surface extraction, default is 'cgalsurf'

    Returns:
    node      : generated mesh nodes
    elem      : elements of the mesh
    face      : surface triangles
    """
    if method is None:
        method = "cgalsurf"

    # Generate the mesh using vol2mesh (assumes vol2mesh exists in the Python environment)
    node, elem, face = vol2mesh(
        img,
        np.arange(img.shape[0]),
        np.arange(img.shape[1]),
        np.arange(img.shape[2]),
        opt,
        maxvol,
        1,
        method,
        isovalues,
    )

    return node, elem, face


def v2s(img, isovalues, opt=None, method=None):
    """
    Surface mesh generation from binary or grayscale volumetric images.

    Parameters:
    img       : 3D numpy array, volumetric image data
    isovalues : scalar or list, isovalues to generate meshes
    opt       : options for mesh generation (default: None)
    method    : method for surface extraction, default is 'cgalsurf'

    Returns:
    no        : generated mesh nodes
    el        : elements of the mesh
    regions   : mesh regions
    holes     : mesh holes
    """
    if method is None:
        method = "cgalsurf"

    if method == "cgalmesh":
        no, tet, el = v2m(np.uint8(img), isovalues, opt, 1000, method)
        regions = []
        fclist = np.unique(el[:, 3])

        for fc in fclist:
            pt = surfinterior(no[:, :3], el[el[:, 3] == fc, :3])
            if pt.size > 0:
                regions.append(pt)

        el = np.unique(el[:, :3], axis=0)
        no, el = removeisolatednode(no[:, :3], el[:, :3])
        holes = []
        return no, el, regions, holes

    no, el, regions, holes = vol2surf(
        img,
        np.arange(img.shape[0]),
        np.arange(img.shape[1]),
        np.arange(img.shape[2]),
        opt,
        1,
        method,
        isovalues,
    )

    return no, el, regions, holes


def s2m(
    v, f, keepratio=None, maxvol=None, method="tetgen", regions=None, holes=None, *args
):
    """
    Volumetric mesh generation from a closed surface, shortcut for surf2mesh.

    Parameters:
    v        : vertices of the surface
    f        : faces of the surface
    keepratio: ratio of triangles to preserve or a structure of options (for 'cgalpoly')
    maxvol   : maximum volume of mesh elements
    method   : method to use ('tetgen' by default or 'cgalpoly')
    regions  : predefined mesh regions
    holes    : holes in the mesh

    Returns:
    node     : generated mesh nodes
    elem     : elements of the mesh
    face     : surface triangles
    """
    if method == "cgalpoly":
        node, elem, face = cgals2m(v, f, keepratio, maxvol)
        return node, elem, face

    if regions is None:
        regions = []
    if holes is None:
        holes = []

    if args:
        node, elem, face = surf2mesh(
            v, f, [], [], keepratio, maxvol, regions, holes, 0, method, *args
        )
    else:
        node, elem, face = surf2mesh(
            v, f, [], [], keepratio, maxvol, regions, holes, 0, method
        )

    return node, elem, face


def s2v(node, face, div=50, *args):
    """
    Convert a surface mesh to a volumetric binary image.

    Parameters:
    node   : array-like, the vertices of the triangular surface (Nx3 for x, y, z)
    face   : array-like, the triangle node indices (Mx3, each row is a triangle)
    div    : int, division number along the shortest edge of the mesh (resolution)
    *args  : additional arguments for the surf2vol function

    Returns:
    img    : volumetric binary image
    v2smap : 4x4 affine transformation matrix to map voxel coordinates back to the mesh space
    """
    p0 = np.min(node, axis=0)
    p1 = np.max(node, axis=0)

    if node.shape[0] == 0 or face.shape[0] == 0:
        raise ValueError("node and face cannot be empty")

    if div == 0:
        raise ValueError("div cannot be 0")

    dx = np.min(p1 - p0) / div

    if dx <= np.finfo(float).eps:
        raise ValueError("the input mesh is in a plane")

    xi = np.arange(p0[0] - dx, p1[0] + dx, dx)
    yi = np.arange(p0[1] - dx, p1[1] + dx, dx)
    zi = np.arange(p0[2] - dx, p1[2] + dx, dx)

    img, v2smap = surf2vol(node, face, xi, yi, zi, *args)

    return img, v2smap


def sms(node, face, iter=10, alpha=0.5, method="laplacianhc"):
    """
    Simplified version of surface mesh smoothing.

    Parameters:
    node: node coordinates of a surface mesh
    face: face element list of the surface mesh
    iter: smoothing iteration number (default is 10)
    alpha: scaler, smoothing parameter, v(k+1)=alpha*v(k)+(1-alpha)*mean(neighbors) (default is 0.5)
    method: smoothing method, same as in smoothsurf (default is 'laplacianhc')

    Returns:
    newnode: the smoothed node coordinates
    """

    # Compute mesh connectivity
    conn = meshconn(face, node.shape[0])

    # Smooth surface mesh nodes
    newnode = smoothsurf(node[:, :3], None, conn, iter, alpha, method, alpha)

    return newnode


def vol2mesh(img, ix, iy, iz, opt, maxvol, dofix, method="cgalsurf", isovalues=None):
    """
    Convert a binary or multi-valued volume to a tetrahedral mesh.

    Parameters:
    img       : 3D numpy array, volumetric image data
    ix, iy, iz: indices for subvolume selection in x, y, z directions
    opt       : options for mesh generation
    maxvol    : maximum volume for mesh elements
    dofix     : boolean, whether to validate and repair the mesh
    method    : method for mesh generation ('cgalsurf', 'simplify', 'cgalmesh', 'cgalpoly')
    isovalues : list of isovalues for the levelset (optional)

    Returns:
    node      : node coordinates of the mesh
    elem      : element list of the mesh (last column is region ID)
    face      : surface elements of the mesh (last column is boundary ID)
    regions   : interior points for closed surfaces
    """

    if method == "cgalmesh":
        vol = img[np.ix_(ix, iy, iz)]
        if len(np.unique(vol)) > 64 and dofix == 1:
            raise ValueError(
                "CGAL mesher does not support grayscale images. Use 'cgalsurf' for grayscale volumes."
            )
        node, elem, face = cgalv2m(vol, opt, maxvol)
        return node, elem, face

    if isovalues is not None:
        no, el, regions, holes = vol2surf(
            img, ix, iy, iz, opt, dofix, method, isovalues
        )
    else:
        no, el, regions, holes = vol2surf(img, ix, iy, iz, opt, dofix, method)

    if method == "cgalpoly":
        node, elem, face = cgals2m(no[:, :3], el[:, :3], opt, maxvol)
        return node, elem, face

    node, elem, face = surf2mesh(no, el, [], [], 1, maxvol, regions, holes)
    return node, elem, face


def vol2surf(img, ix, iy, iz, opt, dofix=0, method="cgalsurf", isovalues=None):
    """
    Convert a 3D volumetric image to surfaces.

    Parameters:
    img: volumetric binary image. If img is empty, vol2surf will return user-defined surfaces via opt.surf if it exists.
    ix, iy, iz: subvolume selection indices in x, y, z directions.
    opt: options dict containing function parameters.
    dofix: 1 to perform mesh validation and repair, 0 to skip repairing.
    method: meshing method ('simplify', 'cgalsurf', or 'cgalpoly'), defaults to 'cgalsurf'.
    isovalues: list of isovalues for level sets.

    Returns:
    no: node list on the resulting surface mesh, with 3 columns for x, y, z.
    el: list of triangular elements on the surface [n1, n2, n3, region_id].
    regions: list of interior points for all sub-regions.
    holes: list of interior points for all holes.
    """

    print("Extracting surfaces from a volume...")

    el = []
    no = []
    holes = opt.get("holes", [])
    regions = opt.get("regions", [])

    if img is not None and len(img) > 0:
        img = img[ix, iy, iz]
        dim = img.shape
        newdim = np.array(dim) + 2
        newimg = np.zeros(newdim, dtype=img.dtype)
        newimg[1:-1, 1:-1, 1:-1] = img

        if isovalues is None:
            maxlevel = newimg.max()
            isovalues = np.arange(1, maxlevel + 1)
        else:
            isovalues = np.unique(np.sort(isovalues))
            maxlevel = len(isovalues)

        for i in range(maxlevel):
            if i < maxlevel - 1:
                levelmask = (newimg >= isovalues[i]) & (newimg < isovalues[i + 1])
            else:
                levelmask = newimg >= isovalues[i]

            levelno, levelel = binsurface(levelmask)

            if levelel.size > 0:
                if opt.get("autoregion", 0):
                    seeds = surfseeds(levelno, levelel)
                else:
                    seeds = surfinterior(levelno, levelel)

                if seeds.size > 0:
                    print(f"Region {i + 1} centroid: {seeds}")
                    regions = np.vstack((regions, seeds)) if len(regions) > 0 else seeds

        for i in range(maxlevel):
            print(f"Processing threshold level {i + 1}...")
            if method == "simplify":
                v0, f0 = binsurface(newimg >= isovalues[i])
                if dofix:
                    v0, f0 = meshcheckrepair(v0, f0)

                keepratio = (
                    opt.get("keepratio", 1)
                    if len(opt) == 1
                    else opt[i].get("keepratio", 1)
                )
                print(f"Resampling surface mesh for level {i + 1}...")
                v0, f0 = meshresample(v0, f0, keepratio)
                f0 = removeisolatedsurf(v0, f0, 3)

                if dofix:
                    v0, f0 = meshcheckrepair(v0, f0)
            else:
                radbound = (
                    opt.get("radbound", 1)
                    if len(opt) == 1
                    else opt[i].get("radbound", 1)
                )
                distbound = (
                    opt.get("distbound", radbound)
                    if len(opt) == 1
                    else opt[i].get("distbound", radbound)
                )
                maxsurfnode = (
                    opt.get("maxnode", 40000)
                    if len(opt) == 1
                    else opt[i].get("maxnode", 40000)
                )
                surfside = (
                    opt.get("side", "") if len(opt) == 1 else opt[i].get("side", "")
                )

                if surfside == "upper":
                    newimg[newimg <= isovalues[i] - 1e-9] = isovalues[i] - 1e-9
                elif surfside == "lower":
                    newimg[newimg >= isovalues[i] + 1e-9] = isovalues[i] + 1e-9

                perturb = 1e-4 * np.abs(isovalues).max()
                perturb = (
                    -perturb if np.all(newimg > isovalues[i] - perturb) else perturb
                )

                v0, f0 = vol2restrictedtri(
                    newimg,
                    isovalues[i] - perturb,
                    regions[i],
                    np.sum(newdim**2) * 2,
                    30,
                    radbound,
                    distbound,
                    maxsurfnode,
                )

            if opt.get("maxsurf", 0) == 1:
                f0 = maxsurf(finddisconnsurf(f0))

            if "A" in opt and "B" in opt:
                v0 = (opt["A"] @ v0.T + opt["B"][:, None]).T

            if "hole" in opt:
                holes = np.vstack((holes, opt["hole"]))
            if "region" in opt:
                regions = np.vstack((regions, opt["region"]))

            el = np.vstack(
                (el, np.hstack((f0 + len(no), np.ones((f0.shape[0], 1)) * (i + 1))))
            )
            no = np.vstack((no, v0)) if len(no) > 0 else v0

    if "surf" in opt:
        for surf in opt["surf"]:
            surf["elem"][:, 3] = maxlevel + 1
            el = np.vstack((el, surf["elem"] + len(no)))
            no = np.vstack((no, surf["node"]))

    print("Surface mesh generation is complete")

    return no, el, regions, holes

#_________________________________________________________________________________________________________

def surf2mesh(v,f,p0,p1,keepratio,maxvol,regions=None,holes=None,dobbx=0,method="tetgen",cmdopt=None):

    """
    Create a quality volumetric mesh from isosurface patches.

    Parameters:
    v: isosurface node list, shape (nn,3). If v has 4 columns, the last column specifies mesh density near each node.
    f: isosurface face element list, shape (be,3). If f has 4 columns, it indicates the label of the face triangles.
    p0: coordinates of one corner of the bounding box, [x0, y0, z0].
    p1: coordinates of the other corner of the bounding box, [x1, y1, z1].
    keepratio: percentage of elements kept after simplification, between 0 and 1.
    maxvol: maximum volume of tetrahedra elements.
    regions: list of regions, specified by an internal point for each region.
    holes: list of holes, similar to regions.
    forcebox: 1 to add bounding box, 0 for automatic.
    method: meshing method (default is 'tetgen').
    cmdopt: additional options for the external mesh generator.

    Returns:
    node: node coordinates of the tetrahedral mesh.
    elem: element list of the tetrahedral mesh.
    face: mesh surface element list, with the last column denoting the boundary ID.
    """
    if keepratio > 1 or keepratio < 0:
        print(
            'The "keepratio" parameter must be between 0 and 1. No simplification will be performed.'
        )

    exesuff = im.getexeext()

    # Resample surface mesh if keepratio is less than 1
    if keepratio < 1 and not isinstance(f, list):
        print("Resampling surface mesh...")
        no, el = meshresample(v[:, :3], f[:, :3], keepratio)
        el = np.unique(np.sort(el, axis=1), axis=0)
    else:
        no = v
        el = f

    # Handle regions and holes arguments
    if regions is None:
        regions = np.array([]) #[]
    if holes is None:
        holes = np.array([])

    # Warn if both maxvol and region-based volume constraints are specified
    if len(regions) > 1 and regions.shape[1] >= 4 and maxvol is not None:
        print("Warning: Both maxvol and region-based volume constraints are specified. maxvol will be ignored.")
        maxvol = None

    # Dump surface mesh to .poly file format
    if not isinstance(el, list) and no.size and el.size:
        im.saveoff(no[:, :3], el[:, :3], "post_vmesh.off")
    im.deletemeshfile(im.mwpath('post_vmesh.mtr'))
    im.savesurfpoly(no, el, holes, regions, p0, p1, mwpath("post_vmesh.poly"),forcebox=dobbx)

    moreopt = ''
    if len(no.shape) > 1 and no.shape[1] == 4:
        moreopt = moreopt + ' -m '
    # Generate volumetric mesh from surface mesh
    im.deletemeshfile(im.mwpath('post_vmesh.1.*'))
    print("Creating volumetric mesh from surface mesh...")

    if cmdopt is None:
        try:
            cmdopt = eval("ISO2MESH_TETGENOPT")
        except:
            cmdopt = ""

    if not cmdopt:
        print(im.mcpath(method,exesuff))
        status, cmdout = subprocess.getstatusoutput('"' + im.mcpath(method,exesuff) + '"'+ ' -A -q1.414a' + str(maxvol) + ' '+ moreopt + ' ' + mwpath('post_vmesh.poly'))
    else:
        status, cmdout = subprocess.getstatusoutput(f"{method} {cmdopt} post_vmesh.poly")

    if status != 0:
        raise RuntimeError(f"Tetgen command failed:\n{cmdout}")

    # Read generated mesh
    node, elem, face = im.readtetgen(im.mwpath("post_vmesh.1"))

    print("Volume mesh generation complete")
    return node, elem, face

#_________________________________________________________________________________________________________

def smoothsurf(
    node, mask, conn, iter, useralpha=0.5, usermethod="laplacian", userbeta=0.5
):
    """
    Smoothing a surface mesh.

    Parameters:
    node: node coordinates of a surface mesh
    mask: flag whether a node is movable (0 for movable, 1 for non-movable).
          If mask is None, all nodes are considered movable.
    conn: a list where each element contains a list of neighboring node IDs for a node
    iter: number of smoothing iterations
    useralpha: scalar smoothing parameter, v(k+1) = (1-alpha)*v(k) + alpha*mean(neighbors) (default 0.5)
    usermethod: smoothing method, 'laplacian', 'laplacianhc', or 'lowpass' (default 'laplacian')
    userbeta: scalar smoothing parameter for 'laplacianhc' (default 0.5)

    Returns:
    p: smoothed node coordinates
    """

    p = np.copy(node)

    # If mask is empty, all nodes are considered movable
    if mask is None:
        idx = np.arange(node.shape[0])
    else:
        idx = np.where(mask == 0)[0]

    nn = len(idx)

    alpha = useralpha
    method = usermethod
    beta = userbeta

    ibeta = 1 - beta
    ialpha = 1 - alpha

    # Remove nodes without neighbors
    idx = np.array([i for i in idx if len(conn[i]) > 0])
    nn = len(idx)

    if method == "laplacian":
        for j in range(iter):
            for i in range(nn):
                p[idx[i], :] = ialpha * p[idx[i], :] + alpha * np.mean(
                    node[conn[idx[i]], :], axis=0
                )
            node = np.copy(p)

    elif method == "laplacianhc":
        for j in range(iter):
            q = np.copy(p)
            for i in range(nn):
                p[idx[i], :] = np.mean(q[conn[idx[i]], :], axis=0)
            b = p - (alpha * node + ialpha * q)
            for i in range(nn):
                p[idx[i], :] -= beta * b[idx[i], :] + ibeta * np.mean(
                    b[conn[idx[i]], :], axis=0
                )

    elif method == "lowpass":
        beta = -1.02 * alpha
        ibeta = 1 - beta
        for j in range(iter):
            for i in range(nn):
                p[idx[i], :] = ialpha * node[idx[i], :] + alpha * np.mean(
                    node[conn[idx[i]], :], axis=0
                )
            node = np.copy(p)
            for i in range(nn):
                p[idx[i], :] = ibeta * node[idx[i], :] + beta * np.mean(
                    node[conn[idx[i]], :], axis=0
                )
            node = np.copy(p)

    return p


def surf2volz(node, face, xi, yi, zi):
    """
    Convert a triangular surface to a shell of voxels in a 3D image along the z-axis.

    Parameters:
    node: node list of the triangular surface, with 3 columns for x/y/z
    face: triangle node indices, each row represents a triangle
    xi, yi, zi: x/y/z grid for the resulting volume

    Returns:
    img: a volumetric binary image at the position of ndgrid(xi, yi, zi)
    """

    ne = face.shape[0]
    img = np.zeros((len(xi), len(yi), len(zi)), dtype=np.uint8)
    dx0 = np.min(np.abs(np.diff(xi)))
    dx = dx0 / 2
    dy0 = np.min(np.abs(np.diff(yi)))
    dy = dy0 / 2
    dz0 = np.min(np.abs(np.diff(zi)))
    dl = np.sqrt(dx**2 + dy**2)
    minz = np.min(node[:, 2])
    maxz = np.max(node[:, 2])

    # Determine the z index range
    iz = np.histogram([minz, maxz], bins=zi)[0]
    hz = np.nonzero(iz)[0]
    iz = np.arange(hz[0], min(len(zi), hz[-1] + 1))

    for i in iz:
        plane = np.array([[0, 100, zi[i]], [100, 0, zi[i]], [0, 0, zi[i]]])
        bcutpos, bcutvalue, bcutedges = qmeshcut(face[:, :3], node, node[:, 0], plane)

        if bcutpos.size == 0:
            continue

        enum = bcutedges.shape[0]

        for j in range(enum):
            e0 = bcutpos[bcutedges[j, 0], :2]
            e1 = bcutpos[bcutedges[j, 1], :2]
            length = np.ceil(np.sum(np.abs(e1 - e0)) / (np.abs(dx) + np.abs(dy))) + 1
            dd = (e1 - e0) / length

            posx = np.floor(
                (e0[0] + np.arange(length + 1) * dd[0] - xi[0]) / dx0
            ).astype(int)
            posy = np.floor(
                (e0[1] + np.arange(length + 1) * dd[1] - yi[0]) / dy0
            ).astype(int)
            pos = np.vstack((posx, posy)).T

            pos = pos[(posx > 0) & (posx <= len(xi)) & (posy > 0) & (posy <= len(yi))]

            if len(pos) > 0:
                zz = np.floor((zi[i] - zi[0]) / dz0).astype(int)
                for k in range(pos.shape[0]):
                    img[pos[k, 0], pos[k, 1], zz] = 1

    return img


def surf2vol(node, face, xi, yi, zi, **kwargs):
    """
    Convert a triangular surface to a shell of voxels in a 3D image.

    Parameters:
    node: node list of the triangular surface, 3 columns for x/y/z
    face: triangle node indices, each row is a triangle
          If face contains a 4th column, it indicates the label of the face triangles.
          If face contains 5 columns, it stores a tetrahedral mesh with labels.
    xi, yi, zi: x/y/z grid for the resulting volume
    kwargs: optional parameters:
        'fill': if set to 1, the enclosed voxels are labeled as 1.
        'label': if set to 1, the enclosed voxels are labeled by the corresponding label of the face or element.
                 Setting 'label' to 1 also implies 'fill'.

    Returns:
    img: a volumetric binary image at the position of ndgrid(xi, yi, zi)
    v2smap (optional): a 4x4 matrix denoting the Affine transformation to map voxel coordinates back to the mesh space.
    """

    opt = kwargs
    label = opt.get("label", 0)
    elabel = 1

    if face.shape[1] >= 4:
        elabel = np.unique(face[:, -1])
        if face.shape[1] == 5:
            label = 1
            el = face
            face = np.empty((0, 4))
            for lbl in elabel:
                fc = volface(el[el[:, 4] == lbl, :4])
                fc = np.hstack((fc, np.full((fc.shape[0], 1), lbl)))
                face = np.vstack((face, fc))
    else:
        fc = face

    img = np.zeros((len(xi), len(yi), len(zi)), dtype=elabel.dtype)

    for lbl in elabel:
        if face.shape[1] == 4:
            fc = face[face[:, 3] == lbl, :3]

        im = surf2volz(node[:, :3], fc[:, :3], xi, yi, zi)
        im |= np.moveaxis(surf2volz(node[:, [2, 0, 1]], fc[:, :3], zi, xi, yi), 0, 2)
        im |= np.moveaxis(surf2volz(node[:, [1, 2, 0]], fc[:, :3], yi, zi, xi), 0, 1)

        if opt.get("fill", 0) or label:
            im = imfill(im, "holes")
            if label:
                im = im.astype(elabel.dtype) * lbl

        img = np.maximum(im.astype(img.dtype), img)

    v2smap = None
    if "v2smap" in kwargs:
        dlen = np.abs([xi[1] - xi[0], yi[1] - yi[0], zi[1] - zi[0]])
        offset = np.min(node, axis=0)
        v2smap = np.eye(4)
        v2smap[:3, :3] = np.diag(np.abs(dlen))
        v2smap[:3, 3] = offset

    return img, v2smap


def binsurface(img, nface=3):
    """
    Fast isosurface extraction from 3D binary images.

    Parameters:
    img: a 3D binary image
    nface:
        - 3 or ignored: triangular faces
        - 4: square faces
        - 0: return a boundary mask image via node

    Returns:
    node: node coordinates, 3 columns for x, y, z
    elem: surface mesh face element list
    """

    dim = img.shape
    if len(dim) < 3:
        dim = (*dim, 1)

    newdim = np.array(dim) + 1

    # Find the jumps (0->1 or 1->0) for all directions
    d1 = np.diff(img, axis=0)
    d2 = np.diff(img, axis=1)
    d3 = np.diff(img, axis=2)

    ix, iy = np.where((d1 == 1) | (d1 == -1))
    jx, jy = np.where((d2 == 1) | (d2 == -1))
    kx, ky = np.where((d3 == 1) | (d3 == -1))

    # Compensate for dimension reduction due to diff
    ix = ix + 1
    iy, iz = np.unravel_index(iy, dim[1:])
    iy = np.ravel_multi_index((iy, iz), newdim[1:])

    jy, jz = np.unravel_index(jy, (dim[1] - 1, dim[2]))
    jy = np.ravel_multi_index((jy + 1, jz), newdim[1:])

    ky, kz = np.unravel_index(ky, (dim[1], dim[2] - 1))
    ky = np.ravel_multi_index((ky, kz + 1), newdim[1:])

    id1 = np.ravel_multi_index((ix, iy), newdim)
    id2 = np.ravel_multi_index((jx, jy), newdim)
    id3 = np.ravel_multi_index((kx, ky), newdim)

    if nface == 0:
        elem = np.vstack((id1, id2, id3)).T
        node = np.zeros(newdim)
        node[elem] = 1
        node = node[1:-1, 1:-1, 1:-1] - 1
        return node, elem

    # Generate triangles or boxes based on nface
    xy = newdim[0] * newdim[1]

    if nface == 3:
        elem = np.vstack(
            [
                [id1, id1 + newdim[0], id1 + newdim[0] + xy],
                [id1, id1 + newdim[0] + xy, id1 + xy],
            ]
        ).T
        elem = np.vstack(
            [elem, [id2, id2 + 1, id2 + 1 + xy], [id2, id2 + 1 + xy, id2 + xy]]
        )
        elem = np.vstack(
            [
                elem,
                [id3, id3 + 1, id3 + 1 + newdim[0]],
                [id3, id3 + 1 + newdim[0], id3 + newdim[0]],
            ]
        )
    else:
        elem = np.vstack(
            [
                [id1, id1 + newdim[0], id1 + newdim[0] + xy, id1 + xy],
                [id2, id2 + 1, id2 + 1 + xy, id2 + xy],
                [id3, id3 + 1, id3 + 1 + newdim[0], id3 + newdim[0]],
            ]
        )

    # Compress node indices
    nodemap = np.zeros(np.max(elem) + 1, dtype=int)
    nodemap[elem.ravel()] = 1
    id = np.nonzero(nodemap)[0]
    nodemap = np.zeros_like(nodemap)
    nodemap[id] = np.arange(1, len(id) + 1)
    elem = nodemap[elem]

    # Create coordinates
    xi, yi, zi = np.unravel_index(id, newdim)
    node = np.vstack([xi, yi, zi]).T - 1

    if nface == 3:
        node, elem = meshcheckrepair(node, elem)

    return node, elem


def cgalv2m(vol, opt, maxvol):
    """
    Wrapper for CGAL 3D mesher (CGAL 3.5 or up) to convert a binary or multi-valued volume to tetrahedral mesh.

    Parameters:
    vol: a volumetric binary image.
    opt: parameters for the CGAL mesher. If opt is a structure:
        opt.radbound: maximum surface element size.
        opt.angbound: minimum angle of a surface triangle.
        opt.distbound: maximum distance between the center of the surface bounding circle and the element bounding sphere.
        opt.reratio: maximum radius-edge ratio.
        If opt is a scalar, it specifies radbound.
    maxvol: target maximum tetrahedral element volume.

    Returns:
    node: node coordinates of the tetrahedral mesh.
    elem: element list of the tetrahedral mesh, the last column is the region ID.
    face: mesh surface element list of the tetrahedral mesh, the last column denotes the boundary ID.
    """

    print("Creating surface and tetrahedral mesh from a multi-domain volume...")

    if not (np.issubdtype(vol.dtype, np.bool_) or vol.dtype == np.uint8):
        raise ValueError(
            "CGAL mesher can only handle uint8 volumes. Convert your image to uint8 first."
        )

    if not np.any(vol):
        raise ValueError("No labeled regions found in the input volume.")

    exesuff = getexeext()
    exesuff = fallbackexeext(exesuff, "cgalmesh")

    ang = 30
    ssize = 6
    approx = 0.5
    reratio = 3

    if not isinstance(opt, dict):
        ssize = opt

    if isinstance(opt, dict) and len(opt) == 1:
        ssize = opt.get("radbound", ssize)
        ang = opt.get("angbound", ang)
        approx = opt.get("distbound", approx)
        reratio = opt.get("reratio", reratio)

    saveinr(vol, mwpath("pre_cgalmesh.inr"))
    deletemeshfile(mwpath("post_cgalmesh.mesh"))

    randseed = int("623F9A9E", 16)

    cmd = f'"{mcpath("cgalmesh")}{exesuff}" "{mwpath("pre_cgalmesh.inr")}" "{mwpath("post_cgalmesh.mesh")}" {ang} {ssize} {approx} {reratio} {maxvol} {randseed}'

    os.system(cmd)

    if not os.path.exists(mwpath("post_cgalmesh.mesh")):
        raise RuntimeError(f"Output file was not found. Command failed: {cmd}")

    node, elem, face = readmedit(mwpath("post_cgalmesh.mesh"))

    if isinstance(opt, dict) and len(opt) == 1:
        if "A" in opt and "B" in opt:
            node[:, :3] = (
                opt["A"] @ node[:, :3].T
                + np.tile(opt["B"][:, None], (1, node.shape[0]))
            ).T

    print(
        f"Node number: {node.shape[0]}\nTriangles: {face.shape[0]}\nTetrahedra: {elem.shape[0]}\nRegions: {len(np.unique(elem[:, -1]))}"
    )
    print("Surface and volume meshes complete")

    if node.shape[0] > 0:
        node, elem, face = sortmesh(
            node[0, :], node, elem, list(range(4)), face, list(range(3))
        )

    node += 0.5
    elem[:, :4] = meshreorient(node[:, :3], elem[:, :4])

    return node, elem, face


def cgals2m(v, f, opt, maxvol, *args):
    """
    Convert a triangular surface to a tetrahedral mesh using CGAL mesher.

    Parameters:
    v : ndarray
        Node coordinate list of a surface mesh (nn x 3)
    f : ndarray
        Face element list of a surface mesh (be x 3)
    opt : dict or scalar
        Parameters for CGAL mesher. If it's a dict, it can include:
            - radbound: Maximum surface element size
            - angbound: Minimum angle of a surface triangle
            - distbound: Max distance between surface bounding circle center and element bounding sphere center
            - reratio: Maximum radius-edge ratio
        If it's a scalar, it only specifies radbound.
    maxvol : float
        Target maximum tetrahedral element volume.
    *args : Additional arguments

    Returns:
    node : ndarray
        Node coordinates of the tetrahedral mesh.
    elem : ndarray
        Element list of the tetrahedral mesh. The last column is the region id.
    face : ndarray
        Mesh surface element list of the tetrahedral mesh. The last column denotes the boundary ID.
    """

    print("Creating surface and tetrahedral mesh from a polyhedral surface ...")

    exesuff = fallbackexeext(getexeext(), "cgalpoly")

    ang = 30
    ssize = 6
    approx = 0.5
    reratio = 3
    flags = args_to_dict(*args)

    if not isinstance(opt, dict):
        ssize = opt

    if isinstance(opt, dict) and len(opt) == 1:
        ssize = opt.get("radbound", ssize)
        ang = opt.get("angbound", ang)
        approx = opt.get("distbound", approx)
        reratio = opt.get("reratio", reratio)

    if flags.get("DoRepair", 0) == 1:
        v, f = meshcheckrepair(v, f)

    saveoff(v, f, mwpath("pre_cgalpoly.off"))
    deletemeshfile(mwpath("post_cgalpoly.mesh"))

    randseed = os.getenv("ISO2MESH_SESSION", int("623F9A9E", 16))

    cmd = (
        f'"{mcpath("cgalpoly")}{exesuff}" "{mwpath("pre_cgalpoly.off")}" "{mwpath("post_cgalpoly.mesh")}" '
        f"{ang:.16f} {ssize:.16f} {approx:.16f} {reratio:.16f} {maxvol:.16f} {randseed}"
    )

    status = subprocess.call(cmd, shell=True)

    if status != 0:
        raise RuntimeError("cgalpoly command failed")

    if not os.path.exists(mwpath("post_cgalpoly.mesh")):
        raise FileNotFoundError(
            f"Output file was not found, failure occurred when running command: \n{cmd}"
        )

    node, elem, face = readmedit(mwpath("post_cgalpoly.mesh"))

    print(f"node number:\t{node.shape[0]}")
    print(f"triangles:\t{face.shape[0]}")
    print(f"tetrahedra:\t{elem.shape[0]}")
    print(f"regions:\t{len(np.unique(elem[:, -1]))}")
    print("Surface and volume meshes complete")

    return node, elem, face


def qmeshcut(elem, node, value, cutat):
    """
    Fast tetrahedral mesh slicer. Intersects a 3D mesh with a plane or isosurface.

    Parameters:
    elem: Integer array (Nx4), indices of nodes forming tetrahedra
    node: Node coordinates (Nx3 array for x, y, z)
    value: Scalar array of values at each node or element
    cutat: Can define the cutting plane or isosurface using:
           - 3x3 matrix (plane by 3 points)
           - Vector [a, b, c, d] for plane (a*x + b*y + c*z + d = 0)
           - Scalar for isosurface at value=cutat
           - String expression for an implicit surface

    Returns:
    cutpos: Coordinates of intersection points
    cutvalue: Interpolated values at the intersection
    facedata: Indices forming the intersection polygons
    elemid: Tetrahedron indices where intersection occurs
    nodeid: Interpolation info for intersection points
    """

    if len(value) != len(node) and len(value) != len(elem) and len(value) != 0:
        raise ValueError("Length of value must match either node or elem")

    # Handle implicit plane definitions
    if isinstance(cutat, str):
        x, y, z = node[:, 0], node[:, 1], node[:, 2]
        expr = cutat.split("=")
        if len(expr) != 2:
            raise ValueError('Expression must contain a single "=" sign')
        dist = eval(expr[0]) - eval(expr[1])
    elif isinstance(cutat, list) and len(cutat) == 4:
        a, b, c, d = cutat
        dist = a * node[:, 0] + b * node[:, 1] + c * node[:, 2] + d
    else:
        dist = value - cutat

    # Determine which nodes are above/below the cut surface
    asign = np.sign(dist)

    # Find edges that cross the cut
    edges = np.vstack([elem[:, [i, j]] for i in range(3) for j in range(i + 1, 4)])
    cutedges = np.where(np.sum(asign[edges], axis=1) == 0)[0]

    # Interpolation for cut positions
    nodeid = edges[cutedges]
    cutweight = np.abs(
        dist[nodeid] / (dist[nodeid[:, 0]] - dist[nodeid[:, 1]]).reshape(-1, 1)
    )
    cutpos = (
        node[nodeid[:, 0]] * cutweight[:, 1][:, None]
        + node[nodeid[:, 1]] * cutweight[:, 0][:, None]
    )

    cutvalue = None
    if len(value) == len(node):
        cutvalue = (
            value[nodeid[:, 0]] * cutweight[:, 1]
            + value[nodeid[:, 1]] * cutweight[:, 0]
        )

    # Organize intersection polygons (faces) and element ids
    emap = np.zeros(edges.shape[0])
    emap[cutedges] = np.arange(len(cutedges))
    elemid = np.where(np.sum(emap.reshape(-1, 6), axis=1) > 0)[0]

    facedata = np.vstack([emap[cutedges].reshape(-1, 3)])

    return cutpos, cutvalue, facedata, elemid, nodeid


def meshcheckrepair(node, elem, opt=None, *args):
    """
    Check and repair a surface mesh.

    Parameters:
    node : ndarray
        Input/output, surface node list (nn x 3).
    elem : ndarray
        Input/output, surface face element list (be x 3).
    opt : str, optional
        Options include:
            'dupnode'   : Remove duplicated nodes.
            'dupelem'   : Remove duplicated elements.
            'dup'       : Both remove duplicated nodes and elements.
            'isolated'  : Remove isolated nodes.
            'open'      : Abort if open surface is found.
            'deep'      : Call external jmeshlib to remove non-manifold vertices.
            'meshfix'   : Repair closed surface using meshfix (removes self-intersecting elements, fills holes).
            'intersect' : Test for self-intersecting elements.

    Returns:
    node : ndarray
        Repaired node list.
    elem : ndarray
        Repaired element list.
    """

    extra = dict(*args)

    if opt in (None, "dupnode", "dup"):
        l1 = node.shape[0]
        node, elem = removedupnodes(node, elem, extra.get("Tolerance", 0))
        l2 = node.shape[0]
        if l2 != l1:
            print(f"{l1 - l2} duplicated nodes were removed")

    if opt in (None, "duplicated", "dupelem", "dup"):
        l1 = elem.shape[0]
        elem = removedupelem(elem)
        l2 = elem.shape[0]
        if l2 != l1:
            print(f"{l1 - l2} duplicated elements were removed")

    if opt in (None, "isolated"):
        l1 = len(node)
        node, elem = removeisolatednode(node, elem)
        l2 = len(node)
        if l2 != l1:
            print(f"{l1 - l2} isolated nodes were removed")

    if opt == "open":
        eg = surfedge(elem)
        if eg:
            raise ValueError(
                "Open surface found. You need to enclose it by padding zeros around the volume."
            )

    if opt in (None, "deep"):
        exesuff = fallbackexeext(getexeext(), "jmeshlib")
        deletemeshfile(mwpath("post_sclean.off"))
        saveoff(node[:, :3], elem[:, :3], mwpath("pre_sclean.off"))
        status, output = subprocess.getstatusoutput(
            f'"{mcpath("jmeshlib")}{exesuff}" "{mwpath("pre_sclean.off")}" "{mwpath("post_sclean.off")}"'
        )
        if status:
            raise RuntimeError(f"jmeshlib command failed: {output}")
        node, elem = readoff(mwpath("post_sclean.off"))

    if opt == "meshfix":
        exesuff = fallbackexeext(getexeext(), "meshfix")
        moreopt = extra.get("MeshfixParam", " -q -a 0.01 ")
        deletemeshfile(mwpath("pre_sclean.off"))
        deletemeshfile(mwpath("pre_sclean_fixed.off"))
        saveoff(node, elem, mwpath("pre_sclean.off"))
        status = subprocess.call(
            f'"{mcpath("meshfix")}{exesuff}" "{mwpath("pre_sclean.off")}" {moreopt}',
            shell=True,
        )
        if status:
            raise RuntimeError("meshfix command failed")
        node, elem = readoff(mwpath("pre_sclean_fixed.off"))

    if opt == "intersect":
        moreopt = f' -q --no-clean --intersect -o "{mwpath("pre_sclean_inter.msh")}"'
        deletemeshfile(mwpath("pre_sclean.off"))
        deletemeshfile(mwpath("pre_sclean_inter.msh"))
        saveoff(node, elem, mwpath("pre_sclean.off"))
        subprocess.call(
            f'"{mcpath("meshfix")}{exesuff}" "{mwpath("pre_sclean.off")}" {moreopt}',
            shell=True,
        )

    return node, elem


def removedupelem(elem):
    """
    Remove doubly duplicated (folded) elements from the element list.

    Parameters:
    elem : ndarray
        List of elements (node indices).

    Returns:
    elem : ndarray
        Element list after removing the duplicated elements.
    """

    # Sort elements and remove duplicates (folded elements)
    sorted_elem = np.sort(elem, axis=1)

    # Find unique rows and their indices
    _, idx, counts = np.unique(
        sorted_elem, axis=0, return_index=True, return_counts=True
    )

    # Histogram of element occurrences
    bins = np.bincount(counts, minlength=elem.shape[0])

    # Elements that are duplicated and their indices
    cc = bins[counts]

    # Remove folded elements
    elem = np.delete(elem, np.where((cc > 0) & (cc % 2 == 0)), axis=0)

    return elem


def removedupnodes(node, elem, tol=0):
    """
    Remove duplicated nodes from a mesh.

    Parameters:
    node : ndarray
        Node coordinates, with 3 columns for x, y, and z respectively.
    elem : ndarray or list
        Element list where each row contains the indices of nodes for each tetrahedron.
    tol : float, optional
        Tolerance for considering nodes as duplicates. Default is 0 (no tolerance).

    Returns:
    newnode : ndarray
        Node list without duplicates.
    newelem : ndarray or list
        Element list with only unique nodes.
    """

    if tol != 0:
        node = np.round(node / tol) * tol

    # Find unique rows (nodes) and map them back to elements
    newnode, I, J = np.unique(node, axis=0, return_index=True, return_inverse=True)

    if isinstance(elem, list):
        newelem = [J[e] for e in elem]
    else:
        newelem = J[elem]

    return newnode, newelem


def removeisolatednode(node, elem, face=None):
    """
    Remove isolated nodes: nodes that are not included in any element.

    Parameters:
    node : ndarray
        List of node coordinates.
    elem : ndarray or list
        List of elements of the mesh, can be a regular array or a list for PLCs (piecewise linear complexes).
    face : ndarray or list, optional
        List of triangular surface faces.

    Returns:
    no : ndarray
        Node coordinates after removing the isolated nodes.
    el : ndarray or list
        Element list of the resulting mesh.
    fa : ndarray or list, optional
        Face list of the resulting mesh.
    """

    oid = np.arange(node.shape[0])  # Old node indices

    if not isinstance(elem, list):
        idx = np.setdiff1d(oid, elem.ravel())  # Indices of isolated nodes
    else:
        el = np.concatenate(elem)
        idx = np.setdiff1d(oid, el)

    idx = np.sort(idx)
    delta = np.zeros_like(oid)
    delta[idx] = 1
    delta = -np.cumsum(
        delta
    )  # Calculate the new node index after removal of isolated nodes

    oid = oid + delta  # Map to new index

    if not isinstance(elem, list):
        el = oid[elem]  # Update element list with new indices
    else:
        el = [oid[e] for e in elem]

    if face is not None:
        if not isinstance(face, list):
            fa = oid[face]  # Update face list with new indices
        else:
            fa = [oid[f] for f in face]
    else:
        fa = None

    no = np.delete(node, idx, axis=0)  # Remove isolated nodes

    return no, el, fa


def removeisolatedsurf(v, f, maxdiameter):
    """
    Removes disjointed surface fragments smaller than a given maximum diameter.

    Args:
    v: List of vertices (nodes) of the input surface.
    f: List of faces (triangles) of the input surface.
    maxdiameter: Maximum bounding box size for surface removal.

    Returns:
    fnew: New face list after removing components smaller than maxdiameter.
    """
    fc = finddisconnsurf(f)
    for i in range(len(fc)):
        xdia = v[fc[i], 0]
        if np.max(xdia) - np.min(xdia) <= maxdiameter:
            fc[i] = []
            continue

        ydia = v[fc[i], 1]
        if np.max(ydia) - np.min(ydia) <= maxdiameter:
            fc[i] = []
            continue

        zdia = v[fc[i], 2]
        if np.max(zdia) - np.min(zdia) <= maxdiameter:
            fc[i] = []
            continue

    fnew = np.vstack([fc[i] for i in range(len(fc)) if len(fc[i]) > 0])

    if fnew.shape[0] != f.shape[0]:
        print(
            f"Removed {f.shape[0] - fnew.shape[0]} elements of small isolated surfaces"
        )

    return fnew


def surfaceclean(f, v):
    """
    Removes surface patches that are located inside the bounding box faces.

    Args:
    f: Surface face element list (be, 3).
    v: Surface node list (nn, 3).

    Returns:
    f: Faces free of those on the bounding box.
    """
    pos = v
    mi = np.min(pos, axis=0)
    ma = np.max(pos, axis=0)

    idx0 = np.where(np.abs(pos[:, 0] - mi[0]) < 1e-6)[0]
    idx1 = np.where(np.abs(pos[:, 0] - ma[0]) < 1e-6)[0]
    idy0 = np.where(np.abs(pos[:, 1] - mi[1]) < 1e-6)[0]
    idy1 = np.where(np.abs(pos[:, 1] - ma[1]) < 1e-6)[0]
    idz0 = np.where(np.abs(pos[:, 2] - mi[2]) < 1e-6)[0]
    idz1 = np.where(np.abs(pos[:, 2] - ma[2]) < 1e-6)[0]

    f = removeedgefaces(f, v, idx0)
    f = removeedgefaces(f, v, idx1)
    f = removeedgefaces(f, v, idy0)
    f = removeedgefaces(f, v, idy1)
    f = removeedgefaces(f, v, idz0)
    f = removeedgefaces(f, v, idz1)

    return f


def removeedgefaces(f, v, idx1):
    """
    Helper function to remove edge faces based on node indices.

    Args:
    f: Surface face element list.
    v: Surface node list.
    idx1: Node indices that define the bounding box edges.

    Returns:
    f: Faces with edge elements removed.
    """
    mask = np.zeros(len(v), dtype=bool)
    mask[idx1] = True
    mask_sum = np.sum(mask[f], axis=1)
    f = f[mask_sum < 3, :]
    return f


def getintersecttri(tmppath):
    """
    Get the IDs of self-intersecting elements from TetGen.

    Args:
    tmppath: Working directory where TetGen output is stored.

    Returns:
    eid: An array of all intersecting surface element IDs.
    """
    exesuff = getexeext()
    exesuff = fallbackexeext(exesuff, "tetgen")
    tetgen_path = mcpath("tetgen") + exesuff

    command = f'"{tetgen_path}" -d "{os.path.join(tmppath, "post_vmesh.poly")}"'
    status, str_output = subprocess.getstatusoutput(command)

    eid = []
    if status == 0:
        ids = re.findall(r" #([0-9]+) ", str_output)
        eid = [int(id[0]) for id in ids]

    eid = np.unique(eid)
    return eid


def delendelem(elem, mask):
    """
    Deletes elements whose nodes are all edge nodes.

    Args:
    elem: Surface/volumetric element list (2D array).
    mask: 1D array of length equal to the number of nodes, with 0 for internal nodes and 1 for edge nodes.

    Returns:
    elem: Updated element list with edge-only elements removed.
    """
    # Find elements where all nodes are edge nodes
    badidx = np.sum(mask[elem], axis=1)

    # Remove elements where all nodes are edge nodes
    elem = elem[badidx != elem.shape[1], :]

    return elem


def surfreorient(node, face):
    """
    Reorients the normals of all triangles in a closed surface mesh to point outward.

    Args:
    node: List of nodes (coordinates).
    face: List of faces (each row contains indices of nodes for a triangle).

    Returns:
    newnode: The output node list (same as input node in most cases).
    newface: The face list with consistent ordering of vertices.
    """
    newnode, newface = meshcheckrepair(node[:, :3], face[:, :3], "deep")
    return newnode, newface


def sortmesh(origin, node, elem, ecol=None, face=None, fcol=None):
    """
    Sort nodes and elements in a mesh so that indexed nodes and elements
    are closer to each other (potentially reducing cache misses during calculations).

    Args:
        origin: Reference point for sorting nodes and elements based on distance and angles.
                If None, it defaults to node[0, :].
        node: List of nodes (coordinates).
        elem: List of elements (each row contains indices of nodes that form an element).
        ecol: Columns in elem to participate in sorting. If None, all columns are used.
        face: List of surface triangles (optional).
        fcol: Columns in face to participate in sorting (optional).

    Returns:
        no: Node coordinates in the sorted order.
        el: Element list in the sorted order.
        fc: Surface triangle list in the sorted order (if face is provided).
        nodemap: New node mapping order. no = node[nodemap, :]
    """

    # Set default origin if not provided
    if origin is None:
        origin = node[0, :]

    # Compute distances relative to the origin
    sdist = node - np.tile(origin, (node.shape[0], 1))

    # Convert Cartesian to spherical coordinates
    theta, phi, R = cart2sph(sdist[:, 0], sdist[:, 1], sdist[:, 2])
    sdist = np.column_stack((R, phi, theta))

    # Sort nodes based on spherical distance
    nval, nodemap = sortrows(sdist)
    no = node[nodemap, :]

    # Sort elements based on nodemap
    nval, nidx = sortrows(nodemap)
    el = elem.copy()

    # If ecol is not provided, sort all columns
    if ecol is None:
        ecol = np.arange(elem.shape[1])

    # Update elements with sorted node indices
    el[:, ecol] = np.sort(nidx[elem[:, ecol]], axis=1)
    el = sortrows(el, ecol)

    # If face is provided, sort it as well
    fc = None
    if face is not None and fcol is not None:
        fc = face.copy()
        fc[:, fcol] = np.sort(nidx[face[:, fcol]], axis=1)
        fc = sortrows(fc, fcol)

    return no, el, fc, nodemap


def cart2sph(x, y, z):
    """Convert Cartesian coordinates to spherical (R, phi, theta)."""
    R = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / R)
    return theta, phi, R


def sortrows(matrix, cols=None):
    """Sort rows of the matrix based on specified columns."""
    if cols is None:
        return np.sort(matrix, axis=0), np.argsort(matrix, axis=0)
    else:
        return np.sort(matrix[:, cols], axis=0), np.argsort(matrix[:, cols], axis=0)


def mergemesh(node, elem, *args):
    """
    Concatenate two or more tetrahedral meshes or triangular surfaces.

    Args:
        node: Node coordinates, dimension (nn, 3).
        elem: Tetrahedral element or triangle surface, dimension (nn, 3) to (nn, 5).
        *args: Pairs of node and element arrays for additional meshes.

    Returns:
        newnode: The node coordinates after merging.
        newelem: The elements after merging.

    Note:
        Use meshcheckrepair on the output to remove duplicated nodes or elements.
        To remove self-intersecting elements, use mergesurf() instead.
    """
    # Initialize newnode and newelem with input mesh
    newnode = node
    newelem = elem

    # Check if the number of extra arguments is valid
    if len(args) > 0 and len(args) % 2 != 0:
        raise ValueError("You must give node and element in pairs")

    # Compute the Euler characteristic
    X = mesheuler(newelem)

    # Add a 5th column to tetrahedral elements if not present
    if newelem.shape[1] == 4 and X >= 0:
        newelem = np.column_stack((newelem, np.ones((newelem.shape[0], 1), dtype=int)))

    # Add a 4th column to triangular elements if not present
    if newelem.shape[1] == 3:
        newelem = np.column_stack((newelem, np.ones((newelem.shape[0], 1), dtype=int)))

    # Iterate over pairs of additional meshes and merge them
    for i in range(0, len(args), 2):
        no = args[i]  # node array
        el = args[i + 1]  # element array
        baseno = newnode.shape[0]

        # Ensure consistent node dimensions
        if no.shape[1] != newnode.shape[1]:
            raise ValueError("Input node arrays have inconsistent columns")

        # Update element indices and append nodes/elements to the merged mesh
        if el.shape[1] == 5 or el.shape[1] == 4:
            el[:, :4] += baseno
            if el.shape[1] == 4 and X >= 0:
                el = np.column_stack(
                    (el, np.ones((el.shape[0], 1), dtype=int) * (i // 2 + 1))
                )
            newnode = np.vstack((newnode, no))
            newelem = np.vstack((newelem, el))
        elif el.shape[1] == 3 and newelem.shape[1] == 4:
            el[:, :3] += baseno
            el = np.column_stack(
                (el, np.ones((el.shape[0], 1), dtype=int) * (i // 2 + 1))
            )
            newnode = np.vstack((newnode, no))
            newelem = np.vstack((newelem, el))
        else:
            raise ValueError("Input element arrays have inconsistent columns")

    return newnode, newelem


def meshrefine(node, elem, *args):
    """
    Refine a tetrahedral mesh by adding new nodes or constraints.

    Args:
        node: Existing tetrahedral mesh node list.
        elem: Existing tetrahedral element list.
        args: Optional parameters for mesh refinement. This can include a face array or an options struct.

    Returns:
        newnode: Node coordinates of the refined tetrahedral mesh.
        newelem: Element list of the refined tetrahedral mesh.
        newface: Surface element list of the tetrahedral mesh.
    """
    # Default values
    sizefield = None
    newpt = None

    # If the node array has a 4th column, treat it as sizefield and reduce node array to 3 columns
    if node.shape[1] == 4:
        sizefield = node[:, 3]
        node = node[:, :3]

    # Parse optional arguments
    face = None
    opt = {}

    if len(args) == 1:
        if isinstance(args[0], dict):
            opt = args[0]
        elif len(args[0]) == len(node) or len(args[0]) == len(elem):
            sizefield = args[0]
        else:
            newpt = args[0]
    elif len(args) >= 2:
        face = args[0]
        if isinstance(args[1], dict):
            opt = args[1]
        elif len(args[1]) == len(node) or len(args[1]) == len(elem):
            sizefield = args[1]
        else:
            newpt = args[1]
    else:
        raise ValueError("meshrefine requires at least 3 inputs")

    # Check if options struct contains new nodes or sizefield
    if isinstance(opt, dict):
        if "newnode" in opt:
            newpt = opt["newnode"]
        if "sizefield" in opt:
            sizefield = opt["sizefield"]

    # Call mesh refinement functions (external tools are required here for actual mesh refinement)
    # Placeholders for calls to external mesh generation/refinement tools such as TetGen

    newnode, newelem, newface = (
        node,
        elem,
        face,
    )  # Placeholder, actual implementation needs external tools

    return newnode, newelem, newface


def mergesurf(node, elem, *args):
    """
    Merge two or more triangular meshes and split intersecting elements.

    Args:
        node: Node coordinates, dimension (nn, 3).
        elem: Triangle surface element list (nn, 3).
        *args: Additional node-element pairs for further surfaces to be merged.

    Returns:
        newnode: The node coordinates after merging, dimension (nn, 3).
        newelem: Surface elements after merging, dimension (nn, 3).
    """
    # Initialize newnode and newelem with input node and elem
    newnode = node
    newelem = elem

    # Ensure valid number of input pairs (node, elem)
    if len(args) > 0 and len(args) % 2 != 0:
        raise ValueError("You must give node and element in pairs")

    # Iterate over each pair of node and element arrays
    for i in range(0, len(args), 2):
        no = args[i]
        el = args[i + 1]
        # Perform boolean surface merge
        newnode, newelem = surfboolean(newnode, newelem, "all", no, el)

    return newnode, newelem


def surfboolean(node, elem, *varargin):
    """
    Perform boolean operations on triangular meshes and resolve intersecting elements.

    Parameters:
    node : ndarray
        Node coordinates (nn x 3)
    elem : ndarray
        Triangle surfaces (ne x 3)
    varargin : list
        Additional parameters including operators and meshes (op, node, elem)

    Returns:
    newnode : ndarray
        Node coordinates after the boolean operations.
    newelem : ndarray
        Elements after boolean operations (nn x 4) or (nhn x 5).
    newelem0 : ndarray (optional)
        For 'self' operator, returns the intersecting element list in terms of the input node list.
    """

    len_varargin = len(varargin)
    newnode = node
    newelem = elem

    if len_varargin > 0 and len_varargin % 3 != 0:
        raise ValueError(
            "You must provide operator, node, and element in a triplet form."
        )

    try:
        exename = os.environ.get("ISO2MESH_SURFBOOLEAN", "cork")
    except KeyError:
        exename = "cork"

    exesuff = fallbackexeext(getexeext(), exename)
    randseed = int("623F9A9E", 16)  # Random seed

    # Check if ISO2MESH_RANDSEED is available
    iso2mesh_randseed = os.environ.get("ISO2MESH_RANDSEED")
    if iso2mesh_randseed is not None:
        randseed = int(iso2mesh_randseed, 16)

    for i in range(0, len_varargin, 3):
        op = varargin[i]
        no = varargin[i + 1]
        el = varargin[i + 2]
        opstr = op

        # Map operations to proper string values
        op_map = {
            "or": "union",
            "xor": "all",
            "and": "isct",
            "-": "diff",
            "self": "solid",
        }
        opstr = op_map.get(op, op)

        tempsuff = "off"
        deletemeshfile(mwpath(f"pre_surfbool*.{tempsuff}"))
        deletemeshfile(mwpath("post_surfbool.off"))

        if opstr == "all":
            deletemeshfile(mwpath("s1out2.off"))
            deletemeshfile(mwpath("s1in2.off"))
            deletemeshfile(mwpath("s2out1.off"))
            deletemeshfile(mwpath("s2in1.off"))

        if op == "decouple":
            if "node1" not in locals():
                node1 = node
                elem1 = elem
                newnode[:, 3] = 1
                newelem[:, 3] = 1
            opstr = " --decouple-inin 1 --shells 2"
            saveoff(node1[:, :3], elem1[:, :3], mwpath("pre_decouple1.off"))
            if no.shape[1] != 3:
                opstr = f"-q --shells {no}"
                cmd = f'cd "{mwpath()}" && "{mcpath("meshfix")}{exesuff}" "{mwpath("pre_decouple1.off")}" {opstr}'
            else:
                saveoff(no[:, :3], el[:, :3], mwpath("pre_decouple2.off"))
                cmd = f'cd "{mwpath()}" && "{mcpath("meshfix")}{exesuff}" "{mwpath("pre_decouple1.off")}" "{mwpath("pre_decouple2.off")}" {opstr}'
        else:
            saveoff(newnode[:, :3], newelem[:, :3], mwpath(f"pre_surfbool1.{tempsuff}"))
            saveoff(no[:, :3], el[:, :3], mwpath(f"pre_surfbool2.{tempsuff}"))
            cmd = f'cd "{mwpath()}" && "{mcpath(exename)}{exesuff}" -{opstr} "{mwpath(f"pre_surfbool1.{tempsuff}")}" "{mwpath(f"pre_surfbool2.{tempsuff}")}" "{mwpath("post_surfbool.off")}" -{randseed}'

        status, outstr = subprocess.getstatusoutput(cmd)
        if status != 0 and op != "self":
            raise RuntimeError(
                f"surface boolean command failed:\n{cmd}\nERROR: {outstr}\n"
            )

        if op == "self":
            if "NOT SOLID" not in outstr:
                print("No self-intersection was found!")
                return None, None, None
            else:
                print("Input mesh is self-intersecting")
                return np.array([1]), np.array([]), np.array([1])

    # Further processing based on the operation 'all'
    if opstr == "all":
        nnode, nelem = readoff(mwpath("s1out2.off"))
        newelem = np.hstack([nelem, np.ones((nelem.shape[0], 1))])
        newnode = np.hstack([nnode, np.ones((nnode.shape[0], 1))])
        nnode, nelem = readoff(mwpath("s1in2.off"))
        newelem = np.vstack(
            [
                newelem,
                np.hstack([nelem + newnode.shape[0], np.ones((nelem.shape[0], 1)) * 3]),
            ]
        )
        newnode = np.vstack(
            [newnode, np.hstack([nnode, np.ones((nnode.shape[0], 1)) * 3])]
        )
        nnode, nelem = readoff(mwpath("s2out1.off"))
        newelem = np.vstack(
            [
                newelem,
                np.hstack([nelem + newnode.shape[0], np.ones((nelem.shape[0], 1)) * 2]),
            ]
        )
        newnode = np.vstack(
            [newnode, np.hstack([nnode, np.ones((nnode.shape[0], 1)) * 2])]
        )
        nnode, nelem = readoff(mwpath("s2in1.off"))
        newelem = np.vstack(
            [
                newelem,
                np.hstack([nelem + newnode.shape[0], np.ones((nelem.shape[0], 1)) * 4]),
            ]
        )
        newnode = np.vstack(
            [newnode, np.hstack([nnode, np.ones((nnode.shape[0], 1)) * 4])]
        )
    else:
        newnode, newelem = readoff(mwpath("post_surfbool.off"))

    return newnode, newelem, None


def fillsurf(node, face):
    """
    Calculate the enclosed volume for a closed surface mesh.

    Args:
        node: Node coordinates (nn, 3).
        face: Surface triangle list (ne, 3).

    Returns:
        no: Node coordinates of the filled volume mesh.
        el: Element list (tetrahedral elements) of the filled volume mesh.
    """

    # Placeholder for calling an external function, typically using TetGen for surface to volume mesh conversion
    no, el = surf2mesh(node, face, None, None, 1, 1, None, None, 0, "tetgen", "-YY")

    return no, el


def mcpath(fname, ext=None):
    """
    Get full executable path by prepending a command directory path.

    Parameters:
    fname : str
        Input file name string.
    ext : str, optional
        File extension to append.

    Returns:
    binname : str
        Full file name located in the bin directory or found via system PATH.
    """

    # Check if ISO2MESH_BIN environment variable is set
    p = os.getenv("ISO2MESH_BIN")
    binname = None

    if p is None:
        # Search in the bin folder under the current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tempname = os.path.join(script_dir, "bin", fname)

        if os.path.exists(os.path.join(script_dir, "bin")):
            if ext is not None:
                if os.path.exists(tempname + ext):
                    binname = tempname + ext
                else:
                    binname = fname  # Fallback to fname without suffix
            else:
                binname = tempname
        else:
            binname = fname
    else:
        binname = os.path.join(p, fname)

    # For 64-bit Windows, check for '_x86-64.exe' suffix
    if (
        platform.system() == "Windows"
        and "64" in platform.architecture()[0]
        and not fname.endswith("_x86-64")
    ):
        w64bin = re.sub(r"(\.[eE][xX][eE])?$", "_x86-64.exe", binname)
        if os.path.exists(w64bin):
            binname = w64bin

    # If file doesn't exist, fall back to searching in system PATH
    if ext is not None and not os.path.exists(binname):
        binname = fname

    return binname


def vol2restrictedtri(vol, thres, cent, brad, ang, radbound, distbound, maxnode):
    """
    Surface mesh extraction using CGAL mesher.

    Parameters:
    vol : ndarray
        3D volumetric image.
    thres : float
        Threshold for extraction.
    cent : tuple
        A 3D position (x, y, z) inside the resulting mesh.
    brad : float
        Maximum bounding sphere squared of the resulting mesh.
    ang : float
        Minimum angular constraint for triangular elements (degrees).
    radbound : float
        Maximum triangle Delaunay circle radius.
    distbound : float
        Maximum Delaunay sphere distances.
    maxnode : int
        Maximum number of surface nodes.

    Returns:
    node : ndarray
        List of 3D nodes (x, y, z) in the resulting surface.
    elem : ndarray
        Element list of the resulting mesh (3 columns of integers).
    """

    if radbound < 1:
        print(
            "You are meshing the surface with sub-pixel size. Check if opt.radbound is set correctly."
        )

    exesuff = im.getexeext()

    # Save the input volume in .inr format
    im.saveinr(vol, im.mwpath("pre_extract.inr"))

    # Delete previous output mesh file if exists
    im.deletemeshfile(im.mwpath("post_extract.off"))

    # Random seed
    randseed = os.getenv("ISO2MESH_SESSION", int("623F9A9E", 16))

    initnum = os.getenv("ISO2MESH_INITSIZE", 50)

    # Build the system command to run CGAL mesher
    print(im.mwpath("pre_extract.inr"))
    cmd = (
        f'"{im.mcpath("cgalsurf",exesuff)}" "{im.mwpath("pre_extract.inr")}" '
        f"{thres:.16f} {cent[0]:.16f} {cent[1]:.16f} {cent[2]:.16f} {brad:.16f} {ang:.16f} {radbound:.16f} "
        f'{distbound:.16f} {maxnode} "{im.mwpath("post_extract.off")}" {randseed} {initnum}'
    )

    # Execute the system command
    status = subprocess.call(cmd, shell=True)
    if status != 0:
        raise RuntimeError(f"CGAL mesher failed with command: {cmd}")

    # Read the resulting mesh
    node, elem = im.readoff(im.mwpath("post_extract.off"))
    print(node,elem)

    # Check and repair mesh if needed
    # node, elem = meshcheckrepair(node, elem)

    # Assuming the origin [0, 0, 0] is located at the lower-bottom corner of the image
    node += 0.5

    return node, elem


def meshresample(v, f, keepratio):
    """
    Resample mesh using the CGAL mesh simplification utility.

    Parameters:
    v : ndarray
        List of nodes.
    f : ndarray
        List of surface elements (each row representing a triangle).
    keepratio : float
        Decimation rate, a number less than 1 representing the percentage of elements to keep after sampling.

    Returns:
    node : ndarray
        Node coordinates of the resampled surface mesh.
    elem : ndarray
        Element list of the resampled surface mesh.
    """

    node, elem = domeshsimplify(v, f, keepratio)

    if len(node) == 0:
        print(
            "Input mesh contains topological defects. Attempting to repair with meshcheckrepair..."
        )
        vnew, fnew = meshcheckrepair(v, f)
        node, elem = domeshsimplify(vnew, fnew, keepratio)

    # Remove duplicate nodes
    node, I, J = np.unique(node, axis=0, return_index=True, return_inverse=True)
    elem = J[elem]

    saveoff(node, elem, mwpath("post_remesh.off"))

    return node, elem


def domeshsimplify(v, f, keepratio):
    """
    Perform the actual mesh resampling using CGAL's simplification utility.

    Parameters:
    v : ndarray
        List of nodes.
    f : ndarray
        List of surface elements.
    keepratio : float
        Decimation rate, a number less than 1.

    Returns:
    node : ndarray
        Node coordinates after simplification.
    elem : ndarray
        Element list after simplification.
    """

    exesuff = getexeext()
    exesuff = fallbackexeext(exesuff, "cgalsimp2")

    # Save the input mesh in OFF format
    saveoff(v, f, mwpath("pre_remesh.off"))

    # Delete the old remeshed file if it exists
    deletemeshfile(mwpath("post_remesh.off"))

    # Build and execute the command for CGAL simplification
    cmd = f'"{mcpath("cgalsimp2")}{exesuff}" "{mwpath("pre_remesh.off")}" {keepratio} "{mwpath("post_remesh.off")}"'
    status = subprocess.call(cmd, shell=True)

    if status != 0:
        raise RuntimeError("cgalsimp2 command failed")

    # Read the resampled mesh
    node, elem = readoff(mwpath("post_remesh.off"))

    return node, elem


def remeshsurf(node, face, opt):
    """
    remeshsurf(node, face, opt)

    Remesh a triangular surface, output is guaranteed to be free of self-intersecting elements.
    This function can both downsample or upsample a mesh.

    Parameters:
        node: list of nodes on the input surface mesh, 3 columns for x, y, z
        face: list of triangular elements on the surface, [n1, n2, n3, region_id]
        opt: function parameters
            opt.gridsize: resolution for the voxelization of the mesh
            opt.closesize: if there are openings, set the closing diameter
            opt.elemsize: the size of the element of the output surface
            If opt is a scalar, it defines the elemsize and gridsize = opt / 4

    Returns:
        newno: list of nodes on the resulting surface mesh, 3 columns for x, y, z
        newfc: list of triangular elements on the surface, [n1, n2, n3, region_id]
    """

    # Step 1: convert the old surface to a volumetric image
    p0 = np.min(node, axis=0)
    p1 = np.max(node, axis=0)

    if isinstance(opt, dict):
        dx = opt.get("gridsize", None)
    else:
        dx = opt / 4

    x_range = np.arange(p0[0] - dx, p1[0] + dx, dx)
    y_range = np.arange(p0[1] - dx, p1[1] + dx)
    z_range = np.arange(p0[2] - dx, p1[2] + dx)

    img = surf2vol(node, face, x_range, y_range, z_range)

    # Compute surface edges
    eg = surfedge(face)

    closesize = 0
    if eg.size > 0 and isinstance(opt, dict):
        closesize = opt.get("closesize", 0)

    # Step 2: fill holes in the volumetric binary image
    img = fillholes3d(img, closesize)

    # Step 3: convert the filled volume to a new surface
    if isinstance(opt, dict):
        if "elemsize" in opt:
            opt["radbound"] = opt["elemsize"] / dx
            newno, newfc = v2s(img, 0.5, opt, "cgalsurf")
    else:
        opt = {"radbound": opt / dx}
        newno, newfc = v2s(img, 0.5, opt, "cgalsurf")

    # Adjust new nodes to match original coordinates
    newno[:, 0:3] *= dx
    newno[:, 0] += p0[0]
    newno[:, 1] += p0[1]
    newno[:, 2] += p0[2]

    return newno, newfc
