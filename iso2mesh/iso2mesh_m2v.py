"""@package docstring
Iso2Mesh for Python - Mesh-to-volume mesh rasterization

Copyright (c) 2024 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = ["m2v", "mesh2vol", "mesh2mask"]

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
    # Handle node values
    if node.shape[1] == 4:
        nodeval = node[:, 3]
        node = node[:, :3]
    else:
        nodeval = None

    # Grid size determination
    if len(xi) == 1 and xi > 0:
        mn = np.min(node, axis=0)
        mx = np.max(node, axis=0)
        df = (mx[:3] - mn[:3]) / xi
    elif len(xi) == 3 and np.all(xi > 0):
        mn = np.min(node, axis=0)
        mx = np.max(node, axis=0)
        df = (mx[:3] - mn[:3]) / xi
    elif len(xi) == 3 and yi is not None and zi is not None:
        mx = [max(xi), max(yi), max(zi)]
        mn = [min(xi), min(yi), min(zi)]
        df = [np.min(np.diff(xi)), np.min(np.diff(yi)), np.min(np.diff(zi))]
    else:
        raise ValueError('Invalid input for grid dimensions.')
    
    xi = np.arange(mn[0], mx[0], df[0])
    yi = np.arange(mn[1], mx[1], df[1])
    zi = np.arange(mn[2], mx[2], df[2])

    # Initialize the 3D mask and optional weight arrays
    mask = np.zeros((len(xi) - 1, len(yi) - 1, len(zi) - 1))
    if nodeval is not None:
        weight = np.zeros((4, len(xi) - 1, len(yi) - 1, len(zi) - 1))
    else:
        weight = None
    
    fig = plt.figure()
    for i, z in enumerate(zi):
        if nodeval is not None:
            cutpos, cutvalue, facedata, elemid = qmeshcut(elem, node, nodeval, f'z={z}')
        else:
            cutpos, cutvalue, facedata, elemid = qmeshcut(elem, node, node[:, 0], f'z={z}')
        
        if cutpos is None:
            continue
        
        if nodeval is not None:
            maskz, weightz = mesh2mask(cutpos, facedata, xi, yi, fig)
            weight[:, :, :, i] = weightz
        else:
            maskz = mesh2mask(cutpos, facedata, xi, yi, fig)
        
        idx = np.where(~np.isnan(maskz))
        if nodeval is not None:
            eid = facedata[maskz[idx], :]
            maskz[idx] = (cutvalue[eid[:, 0]] * weightz[0, idx] +
                          cutvalue[eid[:, 1]] * weightz[1, idx] +
                          cutvalue[eid[:, 2]] * weightz[2, idx] +
                          cutvalue[eid[:, 3]] * weightz[3, idx])
        else:
            maskz[idx] = elemid[maskz[idx]]
        
        mask[:, :, i] = maskz

    plt.close(fig)
    return mask, weight


def mesh2mask(node, face, xi, yi=None, hf=None):
    if yi is None:
        mn = np.min(node, axis=0)
        mx = np.max(node, axis=0)
        df = (mx[:2] - mn[:2]) / xi
        xi = np.arange(mn[0], mx[0], df[0])
        yi = np.arange(mn[1], mx[1], df[1])
    else:
        mn = [min(xi), min(yi)]
        mx = [max(xi), max(yi)]
        df = [np.min(np.diff(xi)), np.min(np.diff(yi))]

    if node.shape[1] <= 1 or face.shape[1] <= 2:
        raise ValueError('node must have 2 or 3 columns; face cannot have less than 2 columns.')

    if hf is None:
        hf = plt.figure()

    plt.clf()
    plt.tripcolor(node[:, 0], node[:, 1], face, facecolors=np.arange(len(face)))
    plt.xlim(mn[0], mx[0])
    plt.ylim(mn[1], mx[1])
    plt.axis('off')
    
    output_size = (len(xi), len(yi))
    
    plt.gcf().set_size_inches(output_size[0] / 100, output_size[1] / 100)
    plt.gca().set_position([0, 0, 1, 1])
    plt.gcf().canvas.draw()

    mask = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8).reshape(output_size[1], output_size[0], 3)
    mask = np.mean(mask, axis=2)
    
    if hf is None:
        plt.close(hf)

    mask = np.rot90(mask)
    
    weight = barycentricgrid(node, face, xi, yi, mask) if np.any(np.isnan(mask)) else None

    return mask.astype(np.int32), weight