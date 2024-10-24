import numpy as np
import os
import re
import platform
import subprocess


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


def vol2surf(img, ix, iy, iz, opt, dofix, method="cgalsurf", isovalues=None):
    """
    Convert a 3D volumetric image to surface meshes.

    Parameters:
    img       : 3D numpy array, binary or grayscale volumetric image
    ix, iy, iz: subvolume selection indices in x, y, z directions
    opt       : options for mesh generation (dict or scalar)
    dofix     : boolean, whether to validate and repair the mesh
    method    : method ('cgalsurf', 'simplify', 'cgalpoly', default 'cgalsurf')
    isovalues : list of isovalues for levelsets (optional)

    Returns:
    no        : node coordinates of the surface mesh
    el        : element list of the surface mesh
    regions   : list of interior points for closed surfaces
    holes     : list of interior points for holes
    """
    el = []
    no = []
    holes = opt.get("holes", [])
    regions = opt.get("regions", [])

    if img is not None:
        img = img[np.ix_(ix, iy, iz)]
        dim = img.shape
        newdim = np.array(dim) + 2
        newimg = np.zeros(newdim)
        newimg[1:-1, 1:-1, 1:-1] = img

        if isovalues is None:
            maxlevel = int(np.max(newimg))
            isovalues = np.arange(1, maxlevel + 1)
        else:
            isovalues = np.unique(np.sort(isovalues))
            maxlevel = len(isovalues)

        for i in range(maxlevel):
            levelmask = (
                (newimg >= isovalues[i])
                if i == maxlevel - 1
                else (newimg >= isovalues[i]) & (newimg < isovalues[i + 1])
            )
            levelno, levelel = binsurface(levelmask.astype(np.int8))

            if levelel.size > 0:
                seeds = (
                    surfinterior(levelno, levelel)
                    if not opt.get("autoregion", False)
                    else surfinterior(levelno, levelel)
                )
                if len(seeds) > 0:
                    regions = np.vstack([regions, seeds]) if regions != [] else seeds

        for i in range(maxlevel):
            if method == "simplify":
                v0, f0 = binsurface(newimg >= isovalues[i])
                if dofix:
                    v0, f0 = meshcheckrepair(v0, f0)
                keepratio = opt.get(i, {}).get("keepratio", opt.get("keepratio", opt))
                v0, f0 = meshresample(v0, f0, keepratio)
                f0 = removeisolatedsurf(v0, f0, 3)
                if dofix:
                    v0, f0 = meshcheckrepair(v0, f0)
            else:
                radbound = opt.get(i, {}).get("radbound", opt.get("radbound", opt))
                distbound = opt.get(i, {}).get("distbound", radbound)
                surfside = opt.get(i, {}).get("side", "")
                maxsurfnode = opt.get(i, {}).get("maxnode", 40000)

                perturb = 1e-4 * abs(np.max(isovalues))
                perturb = (
                    -perturb if np.all(newimg > isovalues[i] - perturb) else perturb
                )

                v0, f0 = vol2restrictedtri(
                    newimg,
                    isovalues[i] - perturb,
                    regions[i, :],
                    np.sum(newdim**2) * 2,
                    30,
                    radbound,
                    distbound,
                    maxsurfnode,
                )

            if el == []:
                el = np.hstack([f0, (i + 1) * np.ones((f0.shape[0], 1), dtype=int)])
                no = v0
            else:
                el = np.vstack(
                    [
                        el,
                        np.hstack(
                            [
                                f0 + len(no),
                                (i + 1) * np.ones((f0.shape[0], 1), dtype=int),
                            ]
                        ),
                    ]
                )
                no = np.vstack([no, v0])

        no[:, 0:3] -= 1
        no[:, 0] = no[:, 0] * (np.max(ix) - np.min(ix) + 1) / dim[0] + (np.min(ix) - 1)
        no[:, 1] = no[:, 1] * (np.max(iy) - np.min(iy) + 1) / dim[1] + (np.min(iy) - 1)
        no[:, 2] = no[:, 2] * (np.max(iz) - np.min(iz) + 1) / dim[2] + (np.min(iz) - 1)

    if "surf" in opt:
        for surf in opt["surf"]:
            surf["elem"][:, 3] = maxlevel + len(opt["surf"])
            el = np.vstack([el, surf["elem"] + len(no)])
            no = np.vstack([no, surf["node"]])

    return no, el, regions, holes


def surf2mesh(
    v,
    f,
    p0,
    p1,
    keepratio,
    maxvol,
    regions=None,
    holes=None,
    forcebox=0,
    method="tetgen",
    cmdopt="",
):
    print("Generating tetrahedral mesh from closed surfaces...")

    if keepratio > 1 or keepratio < 0:
        print(
            "Warning: keepratio must be between 0 and 1. No simplification will be performed."
        )

    if keepratio < 1 and not isinstance(f, list):
        print("Resampling surface mesh...")
        no, el = meshresample(v[:, :3], f[:, :3], keepratio)
        el = np.unique(np.sort(el, axis=1), axis=0)
    else:
        no = v
        el = f

    if regions is None:
        regions = []
    if holes is None:
        holes = []

    if len(regions) >= 4 and maxvol is not None:
        print("Warning: maxvol will be ignored due to region-based volume constraint.")
        maxvol = None

    dobbx = forcebox

    if not isinstance(el, list) and len(no) > 0 and len(el) > 0:
        saveoff(no[:, :3], el[:, :3], "post_vmesh.off")

    savesurfpoly(no, el, holes, regions, p0, p1, "post_vmesh.poly", dobbx)

    moreopt = ""
    if no.shape[1] == 4:
        moreopt = " -m "

    # Call TetGen for mesh generation
    print("Creating volumetric mesh from surface mesh...")

    if not cmdopt:
        cmdopt = ""

    command = f'"{method}" -A -q1.414a{maxvol} {moreopt} "post_vmesh.poly"'
    status, cmdout = subprocess.getstatusoutput(command)

    if status != 0:
        raise Exception(f"TetGen command failed: {cmdout}")

    node, elem, face = readtetgen("post_vmesh.1")
    print("Volume mesh generation is complete.")

    return node, elem, face


from scipy.ndimage import binary_fill_holes


def surf2vol(node, face, xi, yi, zi, **kwargs):
    print("Converting a closed surface to a volumetric binary image...")

    # Extract options from kwargs
    label = kwargs.get("label", 0)
    elabel = 1
    img = np.zeros((len(xi), len(yi), len(zi)), dtype=int)

    # Check if face contains labels or tetrahedral elements
    if face.shape[1] >= 4:
        elabel = np.unique(face[:, -1])
        if face.shape[1] == 5:
            label = 1
            el = face
            face = []
            for i in range(len(elabel)):
                fc = volface(el[el[:, 4] == elabel[i], :4])
                fc = np.column_stack((fc, elabel[i]))
                face.append(fc)
        else:
            fc = face
    else:
        fc = face

    # Loop over the element labels
    for i in range(len(elabel)):
        if face.shape[1] == 4:
            fc = face[face[:, 3] == elabel[i], :3]
        im = surf2volz(node[:, :3], fc[:, :3], xi, yi, zi)
        im = np.logical_or(
            im, np.rollaxis(surf2volz(node[:, [2, 0, 1]], fc[:, :3], zi, xi, yi), 1)
        )
        im = np.logical_or(
            im, np.rollaxis(surf2volz(node[:, [1, 2, 0]], fc[:, :3], yi, zi, xi), 2)
        )

        if kwargs.get("fill", 0) or label:
            im = binary_fill_holes(im)
            if label:
                im = np.cast[np.dtype(elabel[i])](im) * elabel[i]

        img = np.maximum(np.cast[np.dtype(img)](im), img)

    v2smap = None
    if "v2smap" in kwargs:
        dlen = np.abs([xi[1] - xi[0], yi[1] - yi[0], zi[1] - zi[0]])
        p0 = np.min(node, axis=0)
        offset = p0
        v2smap = np.diag(np.abs(dlen))
        v2smap[3, 3] = 1
        v2smap[:3, 3] = offset.T

    return img, v2smap


def binsurface(img, nface=3):
    dim = img.shape
    if len(dim) < 3:
        dim = list(dim) + [1]
    newdim = np.array(dim) + 1

    # Find jumps (0 -> 1 or 1 -> 0) for all directions
    d1 = np.diff(img, axis=0)
    d2 = np.diff(img, axis=1)
    d3 = np.diff(img, axis=2)

    ix, iy = np.where((d1 == 1) | (d1 == -1))
    jx, jy = np.where((d2 == 1) | (d2 == -1))
    kx, ky = np.where((d3 == 1) | (d3 == -1))

    # Compensate for dimension reduction from diff
    ix += 1
    iy, iz = np.unravel_index(iy, dim[1:])
    iy = np.ravel_multi_index((iy, iz), newdim[1:])

    jy, jz = np.unravel_index(jy, [dim[1] - 1, dim[2]])
    jy += 1
    jy = np.ravel_multi_index((jy, jz), newdim[1:])

    ky, kz = np.unravel_index(ky, [dim[1], dim[2] - 1])
    kz += 1
    ky = np.ravel_multi_index((ky, kz), newdim[1:])

    id1 = np.ravel_multi_index((ix, iy), newdim)
    id2 = np.ravel_multi_index((jx, jy), newdim)
    id3 = np.ravel_multi_index((kx, ky), newdim)

    if nface == 0:
        elem = np.column_stack((id1, id2, id3))
        node = np.zeros(newdim)
        node[elem] = 1
        node = node[1:-1, 1:-1, 1:-1] - 1
        return node, elem

    xy = newdim[0] * newdim[1]

    if nface == 3:
        elem = np.vstack(
            [
                [id1, id1 + newdim[0], id1 + newdim[0] + xy],
                [id1, id1 + newdim[0] + xy, id1 + xy],
                [id2, id2 + 1, id2 + 1 + xy],
                [id2, id2 + 1 + xy, id2 + xy],
                [id3, id3 + 1, id3 + 1 + newdim[0]],
                [id3, id3 + 1 + newdim[0], id3 + newdim[0]],
            ]
        ).reshape(-1, 3)
    else:
        elem = np.vstack(
            [
                [id1, id1 + newdim[0], id1 + newdim[0] + xy, id1 + xy],
                [id2, id2 + 1, id2 + 1 + xy, id2 + xy],
                [id3, id3 + 1, id3 + 1 + newdim[0], id3 + newdim[0]],
            ]
        )

    nodemap = np.zeros(np.max(elem) + 1, dtype=int)
    nodemap[elem.ravel()] = 1
    id = np.nonzero(nodemap)[0]
    nodemap[id] = np.arange(len(id))
    elem = nodemap[elem]

    xi, yi, zi = np.unravel_index(id, newdim)
    node = np.column_stack((xi, yi, zi)) - 1

    if nface == 3:
        node, elem = meshcheckrepair(node, elem)

    return node, elem


import subprocess


def cgalv2m(vol, opt=None, maxvol=None):
    print("Creating surface and tetrahedral mesh from a multi-domain volume...")

    if not (vol.dtype == bool or vol.dtype == np.uint8):
        raise ValueError(
            "cgalmesher can only handle uint8 volumes. Convert your image to uint8 first."
        )

    if not np.any(vol):
        raise ValueError("No labeled regions found in the input volume.")

    exesuff = getexeext()
    exesuff = fallbackexeext(exesuff, "cgalmesh")

    # Default CGAL meshing parameters
    ang = 30
    ssize = 6
    approx = 0.5
    reratio = 3

    if not isinstance(opt, dict):
        ssize = opt
    else:
        ssize = opt.get("radbound", ssize)
        ang = opt.get("angbound", ang)
        approx = opt.get("distbound", approx)
        reratio = opt.get("reratio", reratio)

    saveinr(vol, mwpath("pre_cgalmesh.inr"))
    deletemeshfile(mwpath("post_cgalmesh.mesh"))

    randseed = 0x623F9A9E  # Default random seed
    randseed = get_randseed_from_base(randseed)

    format_maxvol = "%s" if isinstance(maxvol, str) else "%f"

    cmd = f'"{mcpath("cgalmesh")}{exesuff}" "{mwpath("pre_cgalmesh.inr")}" "{mwpath("post_cgalmesh.mesh")}" {ang} {ssize} {approx} {reratio} {maxvol} {randseed}'
    status, cmdout = subprocess.getstatusoutput(cmd)

    if not os.path.exists(mwpath("post_cgalmesh.mesh")):
        raise FileNotFoundError(
            f"Output file was not found. Failure encountered when running command: {cmd}"
        )

    node, elem, face = readmedit(mwpath("post_cgalmesh.mesh"))

    if isinstance(opt, dict) and "A" in opt and "B" in opt:
        node[:, :3] = (opt["A"] @ node[:, :3].T + opt["B"].reshape(-1, 1)).T

    print(
        f"Node number: {len(node)}\nTriangles: {len(face)}\nTetrahedra: {len(elem)}\nRegions: {len(np.unique(elem[:, -1]))}"
    )
    print("Surface and volume meshes complete.")

    if len(node) > 0:
        node, elem, face = sortmesh(
            node[0, :], node, elem, np.arange(4), face, np.arange(3)
        )

    node += 0.5
    elem[:, :4] = meshreorient(node[:, :3], elem[:, :4])

    return node, elem, face


def cgals2m(v, f, opt=None, maxvol=None, **kwargs):
    print("Creating surface and tetrahedral mesh from a polyhedral surface...")

    exesuff = fallbackexeext(getexeext(), "cgalpoly")

    # Default meshing parameters
    ang = 30
    ssize = 6
    approx = 0.5
    reratio = 3

    # Handle optional parameters
    if not isinstance(opt, dict):
        ssize = opt
    else:
        ssize = opt.get("radbound", ssize)
        ang = opt.get("angbound", ang)
        approx = opt.get("distbound", approx)
        reratio = opt.get("reratio", reratio)

    flags = kwargs

    # Check and repair mesh if specified
    if flags.get("DoRepair", 0) == 1:
        v, f = meshcheckrepair(v, f)

    saveoff(v, f, mwpath("pre_cgalpoly.off"))
    deletemeshfile(mwpath("post_cgalpoly.mesh"))

    # Set a random seed
    randseed = 0x623F9A9E
    randseed = get_randseed_from_base(randseed)

    cmd = f'"{mcpath("cgalpoly")}{exesuff}" "{mwpath("pre_cgalpoly.off")}" "{mwpath("post_cgalpoly.mesh")}" {ang:.16f} {ssize:.16f} {approx:.16f} {reratio:.16f} {maxvol:.16f} {randseed}'

    status, cmdout = subprocess.getstatusoutput(cmd)

    if status:
        raise RuntimeError("cgalpoly command failed")

    if not os.path.exists(mwpath("post_cgalpoly.mesh")):
        raise FileNotFoundError(f"Output file was not found. Command failed: {cmd}")

    node, elem, face = readmedit(mwpath("post_cgalpoly.mesh"))

    print(f"Node number:\t{len(node)}")
    print(f"Triangles:\t{len(face)}")
    print(f"Tetrahedra:\t{len(elem)}")
    print(f"Regions:\t{len(np.unique(elem[:, -1]))}")
    print("Surface and volume meshes complete.")

    return node, elem, face


def surf2volz(node, face, xi, yi, zi):
    ne = face.shape[0]
    img = np.zeros((len(xi), len(yi), len(zi)), dtype=np.uint8)

    dx0 = np.min(np.abs(np.diff(xi)))
    dx = dx0 / 2
    dy0 = np.min(np.abs(np.diff(yi)))
    dy = dy0 / 2
    dz0 = np.min(np.abs(np.diff(zi)))
    dl = np.sqrt(dx * dx + dy * dy)

    minz = np.min(node[:, 2])
    maxz = np.max(node[:, 2])

    iz = np.histogram([minz, maxz], bins=zi)[0]
    hz = np.nonzero(iz)[0]
    iz = np.arange(hz[0], min(len(zi), hz[-1] + 1))

    for i in iz:
        plane = np.array([[0, 100, zi[i]], [100, 0, zi[i]], [0, 0, zi[i]]])
        bcutpos, bcutvalue, bcutedges = qmeshcut(face[:, :3], node, node[:, 0], plane)

        if bcutpos is None:
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
            pos = np.column_stack((posx, posy))
            valid_idx = np.all(
                (posx > 0, posy > 0, posx < len(xi), posy < len(yi)), axis=0
            )
            pos = pos[valid_idx]

            if len(pos) > 0:
                zz = int(np.floor((zi[i] - zi[0]) / dz0))
                for k in range(pos.shape[0]):
                    img[pos[k, 0], pos[k, 1], zz] = 1

    return img


def meshcheckrepair(node, elem, opt="dup", extra=None):
    # remove duplicate nodes if opt is 'dupnode' or 'dup'
    if opt in ["dupnode", "dup"]:
        l1 = node.shape[0]
        node, elem = removedupnodes(node, elem, tolerance=0)
        l2 = node.shape[0]
        if l2 != l1:
            print(f"{l1 - l2} duplicated nodes were removed")

    # remove duplicate elements if opt is 'duplicated' or 'dupelem' or 'dup'
    if opt in ["duplicated", "dupelem", "dup"]:
        l1 = elem.shape[0]
        elem = removedupelem(elem)
        l2 = elem.shape[0]
        if l2 != l1:
            print(f"{l1 - l2} duplicated elements were removed")

    # remove isolated nodes if opt is 'isolated'
    if opt == "isolated":
        l1 = node.shape[0]
        node, elem = removeisolatednode(node, elem)
        l2 = node.shape[0]
        if l2 != l1:
            print(f"{l1 - l2} isolated nodes were removed")

    # check for open surface if opt is 'open'
    if opt == "open":
        eg = surfedge(elem)
        if len(eg) > 0:
            raise ValueError("open surface found, you need to enclose it")

    # repair mesh using external tool like jmeshlib if opt is 'deep'
    if opt == "deep":
        saveoff(node, elem, "pre_sclean.off")
        command = f"jmeshlib pre_sclean.off post_sclean.off"
        status = subprocess.run(command, shell=True)
        if status.returncode != 0:
            raise RuntimeError("jmeshlib command failed")
        node, elem = readoff("post_sclean.off")

    # repair mesh using meshfix if opt is 'meshfix'
    if opt == "meshfix":
        saveoff(node, elem, "pre_sclean.off")
        command = f"meshfix pre_sclean.off"
        status = subprocess.run(command, shell=True)
        if status.returncode != 0:
            raise RuntimeError("meshfix command failed")
        node, elem = readoff("pre_sclean_fixed.off")

    # check for self-intersections if opt is 'intersect'
    if opt == "intersect":
        saveoff(node, elem, "pre_sclean.off")
        command = f"meshfix --intersect pre_sclean.off"
        subprocess.run(command, shell=True)

    return node, elem


def meshreorient(node, elem):
    """
    Reorders nodes in a surface or tetrahedral mesh to ensure all elements are consistently oriented.

    Args:
    node: List of nodes (coordinates).
    elem: List of elements (each row contains indices of nodes for each element).

    Returns:
    newelem: Element list with consistent ordering.
    evol: Signed element volume before reorientation.
    idx: Indices of elements that had negative volume.
    """
    # Calculate the signed element volume
    evol = elemvolume(node, elem, signed=True)

    # Find elements with negative volume
    idx = np.where(evol < 0)[0]

    # Swap the last two nodes of the elements with negative volume
    elem[idx, [-2, -1]] = elem[idx, [-1, -2]]

    newelem = elem
    return newelem, evol, idx


def removedupelem(elem):
    """
    Removes duplicated (folded) elements from the element list.

    Args:
    elem: List of elements (each row contains indices of nodes for an element).

    Returns:
    elem: Element list after removing duplicated elements.
    """
    # Sort each element row-wise and find unique rows
    sorted_elem = np.sort(elem, axis=1)
    _, unique_indices, counts = np.unique(
        sorted_elem, axis=0, return_index=True, return_counts=True
    )

    # Remove elements that appear an even number of times (i.e., duplicates)
    bins = np.bincount(counts)
    cc = bins[counts]
    elem = elem[cc == 1]

    return elem


def removedupnodes(node, elem, tol=0):
    """
    Removes duplicate nodes from a mesh and adjusts the element indices accordingly.

    Args:
    node: Node coordinates, a 2D array with each row as (x, y, z).
    elem: Element connectivity, integer array with each row containing node indices.
    tol: Tolerance for considering nodes as duplicates.

    Returns:
    newnode: Nodes without duplicates.
    newelem: Elements with updated node indices.
    """
    # Apply tolerance if specified
    if tol != 0:
        node = np.round(node / tol) * tol

    # Find unique nodes and indices
    newnode, I, J = np.unique(node, axis=0, return_index=True, return_inverse=True)

    # Update element connectivity
    if isinstance(elem, list):
        newelem = [J[e] for e in elem]
    else:
        newelem = J[elem]

    return newnode, newelem


def removeisolatednode(node, elem, face=None):
    """
    Removes isolated nodes that are not included in any element.

    Args:
    node: Node coordinates, a 2D array.
    elem: Element connectivity, array or cell array (list of arrays).
    face: (optional) Surface face list.

    Returns:
    no: Node coordinates after removing isolated nodes.
    el: Element connectivity after re-indexing.
    fa: (optional) Face list after re-indexing.
    """
    oid = np.arange(node.shape[0])  # old node index

    if not isinstance(elem, list):
        idx = np.setdiff1d(oid, np.unique(elem))
    else:
        el = np.concatenate(elem)
        idx = np.setdiff1d(oid, np.unique(el))

    idx = np.sort(idx)
    delta = np.zeros(oid.shape)
    delta[idx] = 1
    delta = -np.cumsum(delta)

    # Map to new indices
    oid = oid + delta

    if not isinstance(elem, list):
        el = oid[elem]
    else:
        el = [oid[e] for e in elem]

    if face is not None:
        if not isinstance(face, list):
            fa = oid[face]
        else:
            fa = [oid[f] for f in face]
    else:
        fa = None

    # Remove isolated nodes
    no = np.delete(node, idx, axis=0)

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


def surfboolean(node, elem, *args):
    """
    Merge two or more triangular meshes and resolve intersecting elements.

    Args:
        node: Node coordinates (nn, 3).
        elem: Triangle surface elements (ne, 3).
        *args: Triplets of (operation, node, elem) for additional meshes and boolean operations.

    Returns:
        newnode: The node coordinates after boolean operations.
        newelem: Surface elements after boolean operations.
        newelem0: Intersecting element list (for self-intersection, optional).
    """
    len_args = len(args)
    newnode = node
    newelem = elem

    if len_args > 0 and len_args % 3 != 0:
        raise ValueError("You must provide operator, node, and element in triplets")

    for i in range(0, len_args, 3):
        op = args[i]
        no = args[i + 1]
        el = args[i + 2]

        # Map common operations to internal command strings
        if op in ["or", "union"]:
            opstr = "union"
        elif op == "xor":
            opstr = "all"
        elif op == "and":
            opstr = "isct"
        elif op == "-":
            opstr = "diff"
        elif op == "self":
            opstr = "solid"
        else:
            opstr = op

        # Placeholder for external command execution (like calling external tools)
        cmd = build_command(opstr, newnode, newelem, no, el)

        # Execute the command and capture output
        status, outstr = os.system(cmd)

        if status != 0:
            raise RuntimeError(
                f"Surface boolean operation failed: {cmd}\nError: {outstr}"
            )

        # Example handling for self-intersections
        if op == "self":
            if "NOT SOLID" not in outstr:
                print("No self-intersections found.")
                return [], [], []
            else:
                print("Self-intersection detected.")
                return 1, [], 1

    # Placeholder for reading mesh after boolean operations
    newnode, newelem = read_mesh("post_surfbool.off")

    return newnode, newelem, None


def build_command(opstr, node1, elem1, node2, elem2):
    """
    Build the command string for the boolean operation.

    Args:
        opstr: The operation string (e.g., 'union', 'diff').
        node1: The first set of node coordinates.
        elem1: The first set of elements.
        node2: The second set of node coordinates.
        elem2: The second set of elements.

    Returns:
        The command to be executed.
    """
    # Placeholder command construction for mesh operations
    return f"./surfboolean_tool --operation {opstr} --input1 {node1} --elem1 {elem1} --input2 {node2} --elem2 {elem2}"


def read_mesh(filename):
    """Placeholder function to read a mesh file."""
    # Actual implementation required to read .off or similar mesh file formats
    return [], []


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


def mwpath(fname=""):
    """
    Get the full temporary file path by prepending the working directory
    and current session name.

    Parameters:
    fname : str, optional
        Input file name string (default is empty string).

    Returns:
    tempname : str
        Full file name located in the working directory.
    """

    # Retrieve the ISO2MESH_TEMP and ISO2MESH_SESSION environment variables
    p = os.getenv("ISO2MESH_TEMP")
    session = os.getenv("ISO2MESH_SESSION", "")

    # Get the current user's name for Linux/Unix/Mac/Windows
    username = os.getenv("USER") or os.getenv("UserName", "")
    if username:
        username = f"iso2mesh-{username}"

    tempname = ""

    if not p:
        tdir = os.path.abspath(
            os.path.join(os.sep, "tmp")
        )  # Use default temp directory
        if username:
            tdir = os.path.join(tdir, username)
            if not os.path.exists(tdir):
                os.makedirs(tdir)

        tempname = os.path.join(tdir, session, fname)
    else:
        tempname = os.path.join(p, session, fname)

    return tempname
