"""@package docstring
Iso2Mesh for Python - File I/O module

Copyright (c) 2024 Qianqian Fang <q.fang at neu.edu>
"""


__all__ = ["saveinr", "saveoff", "saveasc", "saveasc", "savestl", "savebinstl"]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np
import struct
from datetime import datetime

##====================================================================================
## implementations
##====================================================================================

def saveinr(vol, fname):
    """
    Save a 3D volume to INR format.

    Parameters:
    vol : ndarray
        Input, a binary volume.
    fname : str
        Output file name.
    """

    # Open file for writing in binary mode
    try:
        fid = open(fname, 'wb')
    except PermissionError:
        raise PermissionError("You do not have permission to save mesh files.")

    # Determine the data type and bit length of the volume
    dtype = vol.dtype.name
    if vol.dtype == np.bool_ or dtype == 'uint8':
        btype = 'unsigned fixed'
        dtype = 'uint8'
        bitlen = 8
    elif dtype == 'uint16':
        btype = 'unsigned fixed'
        bitlen = 16
    elif dtype == 'float32':
        btype = 'float'
        bitlen = 32
    elif dtype == 'float64':
        btype = 'float'
        bitlen = 64
    else:
        raise ValueError("Volume format not supported")

    # Prepare the INR header
    header = (f'#INRIMAGE-4#{{\nXDIM={vol.shape[0]}\nYDIM={vol.shape[1]}\nZDIM={vol.shape[2]}\n'
              f'VDIM=1\nTYPE={btype}\nPIXSIZE={bitlen} bits\nCPU=decm\nVX=1\nVY=1\nVZ=1\n')
    # Ensure the header has the required 256 bytes length
    header = header + '\n' * (256 - len(header) - 4) + '##}\n'

    # Write the header and the volume data to the file
    fid.write(header.encode('ascii'))
    fid.write(vol.astype(dtype).tobytes())

    # Close the file
    fid.close()



def saveoff(node, face, fname):
    """
    Save a surface mesh to an OFF (Object File Format) file.

    Parameters:
    node : ndarray
        Node list, dimension (N, 3), where N is the number of nodes.
    face : ndarray
        Face list, dimension (F, 3), where F is the number of faces.
    fname : str
        Output file name.
    """

    try:
        with open(fname, 'wt') as fid:
            # Write the header
            fid.write('OFF\n')
            fid.write(f'{node.shape[0]} {face.shape[0]} 0\n')
            
            # Write vertices (nodes)
            for vertex in node:
                fid.write(f'{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n')

            # Write faces (with a leading 3 for triangle faces)
            for f in face:
                fid.write(f'3 {f[0]} {f[1]} {f[2]}\n')

    except PermissionError:
        raise PermissionError("You do not have permission to save OFF files.")



def saveasc(v, f, fname):
    """
    Save a surface mesh to FreeSurfer ASC mesh format.

    Parameters:
    v : ndarray
        Surface node list, dimension (nn, 3), where nn is the number of nodes.
    f : ndarray
        Surface face element list, dimension (be, 3), where be is the number of faces.
    fname : str
        Output file name.
    """

    try:
        with open(fname, 'wt') as fid:
            fid.write(f'#!ascii raw data file {fname}\n')
            fid.write(f'{len(v)} {len(f)}\n')
            
            # Write vertices
            for vertex in v:
                fid.write(f'{vertex[0]:.16f} {vertex[1]:.16f} {vertex[2]:.16f} 0\n')
            
            # Write faces (subtract 1 to adjust from MATLAB 1-based indexing to Python 0-based)
            for face in f:
                fid.write(f'{face[0] - 1} {face[1] - 1} {face[2] - 1} 0\n')
    
    except PermissionError:
        raise PermissionError("You do not have permission to save mesh files.")



def saveasc(node, face=None, elem=None, fname=None):
    """
    Save a surface mesh to DXF format.

    Parameters:
    node : ndarray
        Surface node list, dimension (nn, 3), where nn is the number of nodes.
    face : ndarray, optional
        Surface face element list, dimension (be, 3), where be is the number of faces.
    elem : ndarray, optional
        Tetrahedral element list, dimension (ne, 4), where ne is the number of elements.
    fname : str
        Output file name.
    """

    if fname is None:
        if elem is None:
            fname = face
            face = None
        else:
            fname = elem
            elem = None

    try:
        with open(fname, 'wt') as fid:
            fid.write('0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n')

            if face is not None:
                fid.write(f'0\nPOLYLINE\n66\n1\n8\nI2M\n70\n64\n71\n{len(node)}\n72\n{len(face)}\n')

            if node is not None:
                node = node[:, :3]
                for vertex in node:
                    fid.write(f'0\nVERTEX\n8\nI2M\n10\n{vertex[0]:.16f}\n20\n{vertex[1]:.16f}\n30\n{vertex[2]:.16f}\n70\n192\n')

            if face is not None:
                face = face[:, :3]
                for f in face:
                    fid.write(f'0\nVERTEX\n8\nI2M\n62\n254\n10\n0.0\n20\n0.0\n30\n0.0\n70\n128\n71\n{f[0]}\n72\n{f[1]}\n73\n{f[2]}\n')

            fid.write('0\nSEQEND\n0\nENDSEC\n')

            if face is not None:
                fid.write('0\nSECTION\n2\nENTITIES\n0\nINSERT\n8\n1\n2\nMesh\n10\n0.0\n20\n0.0\n30\n0.0\n41\n1.0\n42\n1.0\n43\n1.0\n50\n0.0\n0\nENDSEC\n')

            fid.write('0\nEOF\n')
    
    except PermissionError:
        raise PermissionError("You do not have permission to save mesh files.")



def savestl(node, elem, fname, solidname=""):
    """
    Save a tetrahedral mesh to an STL (Standard Tessellation Language) file.

    Parameters:
    node : ndarray
        Surface node list, dimension (N, 3).
    elem : ndarray
        Tetrahedral element list; if size is (N, 3), it's a surface mesh.
    fname : str
        Output file name.
    solidname : str, optional
        Name of the object in the STL file.
    """

    if len(node) == 0 or node.shape[1] < 3:
        raise ValueError('Invalid node input')

    if elem is not None and elem.shape[1] >= 5:
        elem = elem[:, :4]  # Discard extra columns if necessary

    with open(fname, 'wt') as fid:
        fid.write(f'solid {solidname}\n')

        if elem is not None:
            if elem.shape[1] == 4:
                elem = volface(elem)  # Convert tetrahedra to surface triangles

            ev = surfplane(node, elem)  # Calculate the plane normals
            ev = ev[:, :3] / np.linalg.norm(ev[:, :3], axis=1)[:, np.newaxis]  # Normalize normals

            for i in range(elem.shape[0]):
                facet_normal = ev[i, :]
                vertices = node[elem[i, :3], :]
                fid.write(f'facet normal {facet_normal[0]:e} {facet_normal[1]:e} {facet_normal[2]:e}\n')
                fid.write('  outer loop\n')
                for vertex in vertices:
                    fid.write(f'    vertex {vertex[0]:e} {vertex[1]:e} {vertex[2]:e}\n')
                fid.write('  endloop\nendfacet\n')

        fid.write(f'endsolid {solidname}\n')



def savebinstl(node, elem, fname, solidname=""):
    """
    Save a tetrahedral mesh to a binary STL (Standard Tessellation Language) file.

    Parameters:
    node : ndarray
        Surface node list, dimension (N, 3).
    elem : ndarray
        Tetrahedral element list; if size(elem,2)==3, it is a surface.
    fname : str
        Output file name.
    solidname : str, optional
        An optional string for the name of the object.
    """

    if len(node) == 0 or node.shape[1] < 3:
        raise ValueError("Invalid node input")

    if elem is not None and elem.shape[1] >= 5:
        elem = elem[:, :4]  # Remove extra columns if needed

    # Open the file in binary write mode
    with open(fname, 'wb') as fid:
        # Header structure containing metadata
        header = {
            "Ver": 1,
            "Creator": "iso2mesh",
            "Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        if solidname:
            header["name"] = solidname

        headerstr = str(header).replace("\t", "").replace("\n", "").replace("\r", "")
        headerstr = headerstr[:80] if len(headerstr) > 80 else headerstr.ljust(80, "\0")
        fid.write(headerstr.encode('ascii'))

        if elem is not None:
            if elem.shape[1] == 4:
                elem = meshreorient(node, elem)
                elem = volface(elem)  # Convert tetrahedra to triangular faces

            # Compute surface normals
            ev = surfplane(node, elem)
            ev = ev[:, :3] / np.linalg.norm(ev[:, :3], axis=1, keepdims=True)

            # Write number of facets
            num_facets = len(elem)
            fid.write(struct.pack('<I', num_facets))

            # Write each facet
            for i in range(num_facets):
                # Normal vector
                fid.write(struct.pack('<3f', *ev[i, :]))
                # Vertices of the triangle
                for j in range(3):
                    fid.write(struct.pack('<3f', *node[elem[i, j], :]))
                # Attribute byte count (set to 0)
                fid.write(struct.pack('<H', 0))

