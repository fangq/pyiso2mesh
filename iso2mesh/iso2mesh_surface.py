"""@package docstring
Iso2Mesh for Python - Surface mesh processing

Copyright (c) 2024 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = ["sms"]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np

##====================================================================================
## implementations
##====================================================================================

def sms(node, face, iter=10, alpha=0.5, method='laplacianhc'):
    """
    Simplified version of surface mesh smoothing.
    
    Parameters:
    node   : array-like, node coordinates of a surface mesh
    face   : array-like, face element list of the surface mesh
    iter   : int, smoothing iteration number (default: 10)
    alpha  : float, smoothing parameter (default: 0.5)
    method : string, smoothing method (default: 'laplacianhc')

    Returns:
    newnode : array-like, smoothed node coordinates
    """
    conn = meshconn(face, node.shape[0])
    newnode = smoothsurf(node[:, :3], None, conn, iter, alpha, method, alpha)
    
    return newnode

