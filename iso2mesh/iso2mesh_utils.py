"""@package docstring
Iso2Mesh for Python - Mesh data queries and manipulations

Copyright (c) 2024 Qianqian Fang <q.fang at neu.edu>
"""

__all__ = ["barycentricgrid", "finddisconnsurf", "surfedge", "volface", "extractloops", "meshconn",
           "meshcentroid", "nodevolume", "elemvolume", "neighborelem", "layersurf", "faceneighbors",
           "edgeneighbors", "maxsurf", "flatsegment", "surfplane", "surfinterior", "surfpart", 
           "surfseeds", "meshquality", "meshedge", "meshface", "surfacenorm", "nodesurfnorm",
           "uniqedges", "uniqfaces", "innersurf", "outersurf", "surfvolume", "insurface"]

##====================================================================================
## dependent libraries
##====================================================================================

import numpy as np

##====================================================================================
## implementations
##====================================================================================

def barycentricgrid(node, face, xi, yi, mask):
    xx, yy = np.meshgrid(xi, yi)
    idx = np.where(~np.isnan(mask))
    eid = mask[idx].astype(int)

    t1 = node[face[eid, 0], :]
    t2 = node[face[eid, 1], :]
    t3 = node[face[eid, 2], :]

    tt = (t2[:, 1] - t3[:, 1]) * (t1[:, 0] - t3[:, 0]) + (t3[:, 0] - t2[:, 0]) * (t1[:, 1] - t3[:, 1])
    w = np.zeros((len(idx[0]), 3))
    w[:, 0] = (t2[:, 1] - t3[:, 1]) * (xx[idx] - t3[:, 0]) + (t3[:, 0] - t2[:, 0]) * (yy[idx] - t3[:, 1])
    w[:, 1] = (t3[:, 1] - t1[:, 1]) * (xx[idx] - t3[:, 0]) + (t1[:, 0] - t3[:, 0]) * (yy[idx] - t3[:, 1])
    w[:, 0:2] /= tt[:, np.newaxis]
    w[:, 2] = 1 - w[:, 0] - w[:, 1]

    weight = np.zeros((3, mask.shape[0], mask.shape[1]))
    for i in range(3):
        ww = np.zeros_like(mask, dtype=float)
        ww[idx] = w[:, i]
        weight[i] = ww

    return weight



def finddisconnsurf(surf, elem):
    # Initialize an array to keep track of disconnected surfaces
    n = len(surf)
    disconn = np.zeros(n, dtype=bool)

    for i in range(n):
        disconn[i] = np.all(np.isin(surf[i], elem[:, :4]))

    disconnected_surf = surf[~disconn]
    
    return disconnected_surf


def surfedge(face):
    # Extract unique edges from the face array
    edges = np.array([[face[:, 0], face[:, 1]], 
                      [face[:, 1], face[:, 2]], 
                      [face[:, 2], face[:, 0]]]).reshape(3, -1, 2)

    # Concatenate and sort to identify unique edges
    edges = np.sort(edges, axis=2).reshape(-1, 2)
    
    # Get unique edges and their indices
    unique_edges, idx = np.unique(edges, axis=0, return_inverse=True)
    
    # Count occurrences of each unique edge
    edge_count = np.bincount(idx)

    # Identify edges that belong to one or more faces
    edge_faces = {i: [] for i in range(len(unique_edges))}
    for i, idx in enumerate(idx):
        edge_faces[idx].append(i // 3)

    return unique_edges, edge_count, edge_faces


def volface(elem):
    n = elem.shape[0]  # Number of elements
    vol = np.zeros(n)  # Initialize volume array
    face = []  # Initialize face list

    for i in range(n):
        # Extract the vertices of the current element
        v1, v2, v3, v4 = elem[i, :4]
        
        # Calculate volume using the determinant method for tetrahedra
        vol[i] = np.abs(np.linalg.det(np.array([[1, 1, 1, 1],
                                                 [v1[0], v2[0], v3[0], v4[0]],
                                                 [v1[1], v2[1], v3[1], v4[1]],
                                                 [v1[2], v2[2], v3[2], v4[2]]])) / 6.0)

        # Add faces (triangles) of the tetrahedron
        face.append(np.array([[v1, v2, v3],
                               [v1, v2, v4],
                               [v1, v3, v4],
                               [v2, v3, v4]]))

    # Concatenate all faces and remove duplicates
    face = np.unique(np.vstack(face), axis=0)

    return vol, face



def extractloops(face):
    # Create a set to keep track of edges
    edges = {}
    for i in range(face.shape[0]):
        for j in range(3):
            # Create an edge as a tuple, sorted to avoid direction issues
            edge = tuple(sorted([face[i, j], face[i, (j + 1) % 3]]))
            if edge not in edges:
                edges[edge] = []
            edges[edge].append(i)

    # Find loops
    loops = []
    visited = set()
    for edge, faces in edges.items():
        if edge not in visited:
            loop = []
            loop.append(edge[0])
            current = edge[0]
            while len(loop) < len(faces):
                visited.add(edge)
                next_faces = [f for f in edges[edge] if f != current]
                if next_faces:
                    current = next_faces[0]
                    loop.append(current)
                else:
                    break

            loops.append(loop)

    return loops


def meshconn(elem):
    n_elem = elem.shape[0]
    conn = []

    for i in range(n_elem):
        for j in range(4):  # Assuming tetrahedra, hence 4 vertices
            # Get the current vertex and find connected elements
            current_vertex = elem[i, j]
            connected_elements = [k for k in range(n_elem) if current_vertex in elem[k]]
            
            # Store connections
            conn.append((current_vertex, connected_elements))

    return conn



def meshcentroid(node, elem):
    n_elem = elem.shape[0]
    centroids = np.zeros((n_elem, 3))

    for i in range(n_elem):
        # Get the vertices of the current element
        vertices = node[elem[i, :], :]
        # Calculate the centroid as the mean of the vertex coordinates
        centroids[i, :] = np.mean(vertices, axis=0)

    return centroids



def nodevolume(node, elem):
    n_nodes = node.shape[0]
    volume = np.zeros(n_nodes)

    for i in range(elem.shape[0]):
        v1 = node[elem[i, 0], :]
        v2 = node[elem[i, 1], :]
        v3 = node[elem[i, 2], :]
        v4 = node[elem[i, 3], :]

        # Calculate volume of tetrahedron
        vol = np.abs(np.linalg.det(np.array([
            [1, 1, 1, 1],
            [v1[0], v2[0], v3[0], v4[0]],
            [v1[1], v2[1], v3[1], v4[1]],
            [v1[2], v2[2], v3[2], v4[2]]
        ])) / 6.0)

        # Distribute volume to each vertex of the tetrahedron
        volume[elem[i, 0]] += vol / 4.0
        volume[elem[i, 1]] += vol / 4.0
        volume[elem[i, 2]] += vol / 4.0
        volume[elem[i, 3]] += vol / 4.0

    return volume


def elemvolume(elem, node):
    n_elem = elem.shape[0]
    volume = np.zeros(n_elem)

    for i in range(n_elem):
        v1 = node[elem[i, 0], :]
        v2 = node[elem[i, 1], :]
        v3 = node[elem[i, 2], :]
        v4 = node[elem[i, 3], :]

        # Calculate volume of the tetrahedron
        vol = np.abs(np.linalg.det(np.array([
            [1, 1, 1, 1],
            [v1[0], v2[0], v3[0], v4[0]],
            [v1[1], v2[1], v3[1], v4[1]],
            [v1[2], v2[2], v3[2], v4[2]]
        ])) / 6.0)

        volume[i] = vol

    return volume


def neighborelem(elem):
    n_elem = elem.shape[0]
    neighbor = {i: set() for i in range(n_elem)}

    for i in range(n_elem):
        for j in range(4):  # Assuming tetrahedral elements
            for k in range(n_elem):
                if i != k and len(set(elem[i, :]).intersection(set(elem[k, :]))) == 3:
                    neighbor[i].add(k)

    # Convert sets to sorted lists for consistency
    for key in neighbor.keys():
        neighbor[key] = sorted(list(neighbor[key]))

    return neighbor


def layersurf(node, elem):
    n_nodes = node.shape[0]
    z_values = node[:, 2]  # Assuming z-coordinates are in the third column
    layers = {}

    # Find unique z-values to identify layers
    unique_z = np.unique(z_values)

    for z in unique_z:
        layers[z] = []

    for i in range(elem.shape[0]):
        z_layer = np.mean(z_values[elem[i, :]])
        layers[z_layer].append(i)

    return layers



def faceneighbors(elem):
    n_elem = elem.shape[0]
    face_dict = {}
    
    for i in range(n_elem):
        # Create faces from the elements
        faces = [
            tuple(sorted([elem[i, 0], elem[i, 1], elem[i, 2]])),
            tuple(sorted([elem[i, 0], elem[i, 1], elem[i, 3]])),
            tuple(sorted([elem[i, 0], elem[i, 2], elem[i, 3]])),
            tuple(sorted([elem[i, 1], elem[i, 2], elem[i, 3]]))
        ]
        
        for face in faces:
            if face not in face_dict:
                face_dict[face] = []
            face_dict[face].append(i)

    neighbors = {}
    for face, elems in face_dict.items():
        if len(elems) == 2:
            neighbors[elems[0]] = elems[1]
            neighbors[elems[1]] = elems[0]

    return neighbors


def edgeneighbors(elem):
    n_elem = elem.shape[0]
    edge_dict = {}
    
    for i in range(n_elem):
        # Create edges from the elements
        edges = [
            tuple(sorted([elem[i, 0], elem[i, 1]])),
            tuple(sorted([elem[i, 0], elem[i, 2]])),
            tuple(sorted([elem[i, 0], elem[i, 3]])),
            tuple(sorted([elem[i, 1], elem[i, 2]])),
            tuple(sorted([elem[i, 1], elem[i, 3]])),
            tuple(sorted([elem[i, 2], elem[i, 3]]))
        ]
        
        for edge in edges:
            if edge not in edge_dict:
                edge_dict[edge] = []
            edge_dict[edge].append(i)

    neighbors = {}
    for edge, elems in edge_dict.items():
        if len(elems) == 2:
            neighbors[elems[0]] = elems[1]
            neighbors[elems[1]] = elems[0]

    return neighbors



def maxsurf(surf):
    # Find the maximum value in each column of the surface
    max_values = np.max(surf, axis=0)
    return max_values



def flatsegment(p1, p2, p3, p4, ndiv=5, seg=0, p5=None):
    # Default p5 to None
    if p5 is None:
        p5 = p4

    # Calculate points in each segment
    pseg = np.zeros((ndiv + 1, 3))
    for i in range(3):
        pseg[:, i] = np.linspace(p1[i], p2[i], ndiv + 1)

    # Calculate pseg2
    pseg2 = np.zeros((ndiv + 1, 3))
    for i in range(3):
        pseg2[:, i] = np.linspace(p3[i], p4[i], ndiv + 1)

    # Define vertices
    vert = np.vstack((pseg, pseg2))

    # Define faces
    face = []
    for i in range(ndiv):
        face.append([i, i + 1, ndiv + i + 1])
        face.append([i, ndiv + i + 1, ndiv + i])

    face = np.array(face)

    # Determine segments
    if seg == 1:
        pseg = np.vstack((pseg, p5))
        pseg2 = np.vstack((pseg2, p5))

        # Update vertices
        vert = np.vstack((vert, p5))

        # Update faces
        face = np.vstack((face, [ndiv, ndiv + 1, ndiv * 2 + 1]))
        face = np.vstack((face, [ndiv, ndiv * 2 + 1, ndiv * 2]))

    return vert, face



def mesheuler(vertex, face):
    # Initialize the counts for vertices and faces
    nface = face.shape[0]
    nvert = vertex.shape[0]

    # Initialize vertex count
    vertexcount = np.zeros(nvert, dtype=int)

    # Count the number of faces connected to each vertex
    for i in range(nface):
        for j in range(3):
            vertexcount[face[i, j]] += 1

    # Initialize edge and edge connectivity
    edge = []
    edgecount = []
    for i in range(nface):
        for j in range(3):
            v1 = face[i, j]
            v2 = face[i, (j + 1) % 3]
            if v1 > v2:
                v1, v2 = v2, v1
            edgepair = (v1, v2)
            if edgepair not in edge:
                edge.append(edgepair)
                edgecount.append(1)
            else:
                edgeindex = edge.index(edgepair)
                edgecount[edgeindex] += 1

    # Initialize Euler characteristics
    euler = nvert - len(edge) + nface

    return euler, vertexcount, edge, edgecount



def orderloopedge(edge, face):
    nedge = edge.shape[0]
    nface = face.shape[0]

    # Initialize an array for the edge loop
    edgeloop = []
    used = np.zeros(nedge, dtype=bool)

    # Start with the first edge
    edgeloop.append(edge[0, :])
    used[0] = True

    while len(edgeloop) < nedge:
        last_edge = edgeloop[-1]
        found_next = False

        for i in range(nedge):
            if not used[i]:
                # Check if this edge shares a vertex with the last edge
                if last_edge[0] in edge[i, :] or last_edge[1] in edge[i, :]:
                    edgeloop.append(edge[i, :])
                    used[i] = True
                    found_next = True
                    break

        if not found_next:
            break

    return np.array(edgeloop)



def bbxflatsegment(p1, p2, p3, p4, ndiv=5):
    # Calculate points in each segment
    pseg = np.zeros((ndiv + 1, 3))
    for i in range(3):
        pseg[:, i] = np.linspace(p1[i], p2[i], ndiv + 1)

    # Calculate pseg2
    pseg2 = np.zeros((ndiv + 1, 3))
    for i in range(3):
        pseg2[:, i] = np.linspace(p3[i], p4[i], ndiv + 1)

    # Define vertices
    vert = np.vstack((pseg, pseg2))

    # Define faces
    face = []
    for i in range(ndiv):
        face.append([i, i + 1, ndiv + i + 1])
        face.append([i, ndiv + i + 1, ndiv + i])

    face = np.array(face)

    return vert, face



def surfplane(n=5, p=[0, 0, 0], normal=[0, 0, 1], d=1):
    # Create a grid of points on the plane
    x = np.linspace(-d, d, n)
    y = np.linspace(-d, d, n)
    X, Y = np.meshgrid(x, y)

    # Calculate Z values based on the plane equation
    Z = p[2] + normal[2] * (X - p[0]) / normal[0] + normal[1] * (Y - p[1]) / normal[1]

    # Stack the coordinates
    vert = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    # Create faces
    face = []
    for i in range(n - 1):
        for j in range(n - 1):
            face.append([i * n + j, i * n + j + 1, (i + 1) * n + j + 1])
            face.append([i * n + j, (i + 1) * n + j + 1, (i + 1) * n + j])

    face = np.array(face)

    return vert, face



def surfinterior(surf, p0):
    # Extract vertices and faces from the surface
    vert = surf['vert']
    face = surf['face']

    # Get the number of vertices
    nvert = vert.shape[0]
    
    # Initialize interior points
    interior = []
    
    # Loop through each vertex
    for i in range(nvert):
        if np.dot(vert[i] - p0, vert[i] - p0) < np.dot(vert[i] - p0, vert[i] - p0):  # Modify condition as needed
            interior.append(vert[i])
    
    interior = np.array(interior)
    
    return interior



def surfpart(surf, p):
    vert = surf['vert']
    face = surf['face']

    # Initialize a list to hold the partitioned faces
    part = []

    # Loop through each face
    for i in range(face.shape[0]):
        f = face[i]
        if np.all(np.isin(vert[f], p)):
            part.append(f)

    part = np.array(part)

    return part



def surfseeds(surf, n):
    vert = surf['vert']
    face = surf['face']

    # Calculate the number of vertices and faces
    nvert = vert.shape[0]
    nface = face.shape[0]

    # Randomly select n vertices from the surface
    if n > nvert:
        raise ValueError("Number of seeds cannot exceed number of vertices")

    indices = np.random.choice(nvert, n, replace=False)
    seeds = vert[indices]

    return seeds



def meshquality(node, elem):
    nElem = elem.shape[0]
    quality = np.zeros(nElem)

    for i in range(nElem):
        # Get the vertices for the current element
        vertices = node[elem[i], :]
        
        # Calculate the area (for 2D) or volume (for 3D)
        if vertices.shape[1] == 2:
            # Calculate area using the shoelace formula
            area = 0.5 * np.abs(vertices[0, 0] * vertices[1, 1] - vertices[1, 0] * vertices[0, 1])
            quality[i] = area  # Quality metric for 2D (area)
        elif vertices.shape[1] == 3:
            # Calculate volume using the determinant method
            v0 = vertices[1] - vertices[0]
            v1 = vertices[2] - vertices[0]
            volume = np.abs(np.dot(v0, np.cross(v1, vertices[3] - vertices[0]))) / 6.0
            quality[i] = volume  # Quality metric for 3D (volume)

    return quality




def meshedge(node, elem):
    nElem = elem.shape[0]
    edges = []

    for i in range(nElem):
        # Get the vertices for the current element
        vertices = elem[i]
        
        # Create edges from the vertices
        edges.append([vertices[0], vertices[1]])
        edges.append([vertices[1], vertices[2]])
        edges.append([vertices[2], vertices[0]])
        
    # Remove duplicate edges
    edges = [list(edge) for edge in set(tuple(sorted(edge)) for edge in edges)]

    return np.array(edges)


def meshface(node, elem):
    nElem = elem.shape[0]
    faces = []

    for i in range(nElem):
        # Get the vertices for the current element
        vertices = elem[i]
        
        # Create faces from the vertices
        faces.append([vertices[0], vertices[1], vertices[2]])
        
    return np.array(faces)



def surfacenorm(surf):
    vert = surf['vert']
    face = surf['face']

    # Initialize normals
    normals = np.zeros(vert.shape)

    # Calculate normals for each face
    for i in range(face.shape[0]):
        f = face[i, :]
        v1 = vert[f[1]] - vert[f[0]]
        v2 = vert[f[2]] - vert[f[0]]
        n = np.cross(v1, v2)
        n = n / np.linalg.norm(n)  # Normalize the normal vector

        # Accumulate normals for each vertex
        normals[f[0]] += n
        normals[f[1]] += n
        normals[f[2]] += n

    # Normalize normals at each vertex
    for i in range(normals.shape[0]):
        normals[i] = normals[i] / np.linalg.norm(normals[i]) if np.linalg.norm(normals[i]) != 0 else normals[i]

    return normals


def nodesurfnorm(surf, p):
    vert = surf['vert']
    face = surf['face']
    
    # Initialize an array for normals
    normals = np.zeros(vert.shape)
    
    # Calculate normals for each face
    for i in range(face.shape[0]):
        f = face[i, :]
        v1 = vert[f[1]] - vert[f[0]]
        v2 = vert[f[2]] - vert[f[0]]
        n = np.cross(v1, v2)
        n = n / np.linalg.norm(n)  # Normalize the normal vector
        
        # Accumulate normals for each vertex
        normals[f[0]] += n
        normals[f[1]] += n
        normals[f[2]] += n

    # Normalize normals at each vertex
    for i in range(normals.shape[0]):
        if np.linalg.norm(normals[i]) != 0:
            normals[i] /= np.linalg.norm(normals[i])

    # Find the closest vertex to point p
    distances = np.linalg.norm(vert - p, axis=1)
    closest_index = np.argmin(distances)

    return normals[closest_index]


def uniqedges(face):
    nface = face.shape[0]
    edges = set()

    for i in range(nface):
        for j in range(3):
            # Create an edge as a sorted tuple (to avoid duplicates)
            edge = tuple(sorted((face[i, j], face[i, (j + 1) % 3])))
            edges.add(edge)

    return np.array(list(edges))


def uniqfaces(face):
    nface = face.shape[0]
    unique_faces = set()

    for i in range(nface):
        # Create a sorted tuple of the face vertices to avoid duplicates
        sorted_face = tuple(sorted((face[i, 0], face[i, 1], face[i, 2])))
        unique_faces.add(sorted_face)

    return np.array(list(unique_faces))


def innersurf(surf, offset):
    vert = surf['vert']
    face = surf['face']

    # Calculate the center of the surface
    center = np.mean(vert, axis=0)

    # Move the surface inwards by the offset
    inner_vert = vert - offset * (vert - center) / np.linalg.norm(vert - center, axis=1)[:, np.newaxis]

    return {'vert': inner_vert, 'face': face}




def outersurf(vert, face, offset):


    # Calculate the center of the surface
    center = np.mean(vert, axis=0)

    # Move the surface outwards by the offset
    outer_vert = vert + offset * (vert - center) / np.linalg.norm(vert - center, axis=1)[:, np.newaxis]

    return {'vert': outer_vert, 'face': face}


def surfvolume(vert, face):

    volume = 0.0

    # Loop through each face to calculate the volume using the divergence theorem
    for f in face:
        v0 = vert[f[0]]
        v1 = vert[f[1]]
        v2 = vert[f[2]]
        volume += np.dot(v0, np.cross(v1, v2)) / 6.0

    return abs(volume)



def insurface(vert, face, p):

    # Initialize the inside check to False
    inside = False

    # Loop through each face to check if point p is inside
    for f in face:
        v0 = vert[f[0]]
        v1 = vert[f[1]]
        v2 = vert[f[2]]

        # Create vectors from the vertices to the point p
        v0_to_p = p - v0
        v1_to_p = p - v1
        v2_to_p = p - v2

        # Calculate the volumes of the tetrahedra formed
        vol1 = np.dot(v0_to_p, np.cross(v1 - v0, v2 - v0))
        vol2 = np.dot(v1_to_p, np.cross(v2 - v1, v0 - v1))
        vol3 = np.dot(v2_to_p, np.cross(v0 - v2, v1 - v2))

        # Check if the point is inside by checking the signs of the volumes
        if np.sign(vol1) == np.sign(vol2) == np.sign(vol3):
            inside = True
            break

    return inside