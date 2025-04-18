import numpy as np
import tetgen
from igl import loop, bounding_box, grad, barycentric_coordinates_tet, barycenter
import pyvista as pv
from scipy.spatial import Delaunay
import gpytoolbox as gpy
import scipy.sparse as sp
from plyfile import PlyData

"""
This is a library of functions relevant for PSR in 3D as implented for this seminar.
It contains various functions for reading data, tetrahedralization, computing the target vector field,
solving the Poisson problem, and extracting the surface mesh.
"""

def read_ply(fpath, every_ith=200):
    """
    Read a PLY file and return the vertex positions and normals.
    """
    plydata = PlyData.read(fpath)
    X = np.array([plydata["vertex"][c] for c in ["z", "y", "x"]]).T
    X[:,1] = -X[:,1]
    N = np.array([plydata["vertex"][c] for c in ["nz", "ny", "nx"]]).T
    N[:,1] = -N[:,1]
    return X[::every_ith],N[::every_ith]

def normalize_to_origin(X):
    """
    Normalize 3D points so they are centered at the origin and fit within [-1,1]^3.
    """
    X = np.asarray(X)
    
    # Compute centroid (mean of points)
    centroid = np.mean(X, axis=0)
    
    # Center the points
    X_centered = X - centroid
    
    # Find the max absolute coordinate to scale within [-1,1]
    max_abs = np.max(np.abs(X_centered))

    if max_abs == 0:
        return np.zeros_like(X_centered)

    # Normalize to fit in the unit cube [-1,1]
    X_normalized = X_centered / max_abs
    
    return X_normalized

def generate_background_mesh(bounds, resolution=20, eps=1e-6):
    """
    Generate a background mesh with desired resolution.
    Copied from TetGen docs https://tetgen.pyvista.org/
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    grid_x, grid_y, grid_z = np.meshgrid(
        np.linspace(x_min - eps, x_max + eps, resolution),
        np.linspace(y_min - eps, y_max + eps, resolution),
        np.linspace(z_min - eps, z_max + eps, resolution),
        indexing="ij",
    )
    return pv.StructuredGrid(grid_x, grid_y, grid_z).triangulate()

def naive_sizing_function(points, focus_point=np.array([0, 0, 0]), max_size=1.0, min_size=0.2):
    distances = 1 / (np.linalg.norm(points - focus_point, axis=1) + 1e-16)
    return np.clip(max_size - distances, min_size, max_size)

def sizing_function_gaussian(points, X, sigma=0.1, max_size=1.5, min_size=0.01):
    sd = gpy.squared_distance(points, X, use_cpp=True)
    sizing_field = (1 - np.exp(-sd[0]/sigma**2))
    return np.clip(sizing_field, min_size, max_size)

def tetrahedralize_sizing_field(X, level=5, resolution=30, sigma=0.25):
    """
    Tetrahedralize a 3D point cloud with a sizing field.
    Generates a box-shaped triangle mesh around the point samples using PyVista.
    Adaptive mesh refinement is performed using TetGen with a background mesh and 
    sizing field.
    """
    
    # Generate a box-shaped triangle mesh
    bounds = (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
    box = pv.Box(bounds=bounds,level=level, quads=False)
    
    # Create the tetrahedral background mesh
    print(f'Background mesh bounds: {box.bounds}')
    bg_mesh = generate_background_mesh(bounds, resolution=resolution, eps=1e-6)
    
    # Compute sizing field
    sizing_field = sizing_function_gaussian(bg_mesh.points, X, sigma=sigma)
    bg_mesh.point_data['target_size'] = sizing_field
    
    # Create refined mesh
    tet_kwargs = dict(order=1, mindihedral=20, minratio=1.5)
    tet = tetgen.TetGen(box)
    nodes, elems = tet.tetrahedralize(bgmesh=bg_mesh, **tet_kwargs)
    
    return nodes, elems, bg_mesh
    

def tetrahedralize_regular_grid_delaunay(res=10, padding=0.1):
    """
    Tetrahedralize a regular grid.
    Uses scipy.spatial.Delaunay to compute the tetrahedralization.
    """
    xs = np.linspace(-1. - padding, 1. + padding, res)
    X = np.array(np.meshgrid(xs, xs, xs)).transpose(1, 2, 3, 0).reshape(-1, 3)
    tet = Delaunay(X, qhull_options='Qbb Qc Qz Q12 Q0') # add Qhull options to avoid degenerate tets (see https://github.com/scipy/scipy/issues/16094)
    nodes, elems = tet.points, tet.simplices
    return nodes, elems

def tetrahedralize_regular_grid(level=5):
    """
    Tetrahedralize a regular grid using PyVista and TetGen.
    Generates a box-shaped triangle mesh around the point samples.
    """
    # Generate a box-shaped triangle mesh
    bounds = (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
    box = pv.Box(bounds=bounds,level=level, quads=False)
    tet_kwargs = dict(order=1, mindihedral=20, minratio=1.5)
    tet = tetgen.TetGen(box)
    nodes, elems = tet.tetrahedralize(**tet_kwargs)
    return nodes, elems

def naive_tetrahedralize(X):
    """
    Naive tetrahedralization I initially used.
    Computes a bounding box as a triangle mesh around the 3D points and refines it by subdivision.
    This is then used for the tetrahedralization.
    """
    # Compute bounding box as triangle mesh
    v_bbox, f_bbox = bounding_box(X, pad=1.0)

    # Create tetrahedralization
    # TODO: do refinement by background mesh (see TetGen)
    v_refined, f_refined = loop(v_bbox, f_bbox, 3) # Refine bounding box mesh by subdivision (loop)
    tgen = tetgen.TetGen(v_refined, f_refined)
    nodes, elems = tgen.tetrahedralize()
    return nodes, elems

def compute_gradient_per_vertex(points, X, N, sigma=0.1):
    """
    Compute the target vector field V at each vertex.
    For each vertex in the domain, the target vector field is computed by summing the sample normals N
    weighted by a Gaussian kernel centered at the corresponding sample position.
    """
    V = np.zeros((len(points), 3))
    for i in range(len(points)):
        weights = np.exp(-np.linalg.norm(points[i] - X, axis=1)**2 / (2 * np.pi * sigma**2))
        # weights /= np.sum(weights)
        V[i] = np.dot(weights, N)
    return V

def compute_gradient_per_tet(elems, V):
    """
    Compute the gradient per tetrahedron.
    For each tetrahedron, the gradient is computed as the average of the gradients at its vertices.
    """
    F = np.zeros((len(elems), 3))
    for i, t in enumerate(elems):
        # Get the indices of the vertices of the tet
        v0, v1, v2, v3 = t

        # Get the vertex gradients
        grad0 = V[v0]
        grad1 = V[v1]
        grad2 = V[v2]
        grad3 = V[v3]

        # Compute the barycentric interpolation (uniform averaging)
        face_gradient = (grad0 + grad1 + grad2 + grad3) / 4.0

        # Store the face gradient
        F[i] = face_gradient
    return F

def compute_tetrahedron_volume(v0, v1, v2, v3):
    """
    Compute the volume of each tetrahedron using the determinant method.
    """
    matrix = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
    det = np.linalg.det(matrix)
    volume = abs(det) / 6.0
    return volume

def compute_mass_matrix(points, simplices):
    """
    Compute the simplex-wise mass matrix for a 3D tetrahedral mesh.
    """
    # Extract vertex positions for each simplex
    v0, v1, v2, v3 = points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]], points[simplices[:, 3]]

    # Compute volumes for all tetrahedra
    volumes = np.array([compute_tetrahedron_volume(v0[i], v1[i], v2[i], v3[i]) for i in range(len(simplices))])

    print(f'Volumes: {volumes.shape}')
    print(f"Number of degenerate tetrahedra: {np.sum(volumes <= 1e-12)}")

    mass_matrix = sp.diags(volumes)
    return mass_matrix

def solve_poisson(nodes, elems, V, solve_direct=False, tol=1e-8, max_iter=1000):
    """
    Solve the Poisson problem Lc = D, where L = G^T M G is the Laplacian matrix, 
    M is the mass matrix, G is the gradient matrix, c are the coefficients, and D is the right-hand side
    vector corresponding to the discrete divergence of the target vector field V.
    """
    G = grad(nodes, elems)
    M = compute_mass_matrix(nodes, elems)
    print(f'Mass matrix shape: {M.shape}')
    M_g = sp.block_diag([M, M, M]) # "stretch" mass matrix from (m, m) to (3m, 3m) for x, y and z components in G matrix
    L = G.T @ M_g @ G
    D = G.T @ M_g @ V.T.flatten()
    
    if solve_direct: # Solve using direct solver
        coeffs = sp.linalg.spsolve(L, D)
    else: # Solve using Conjugate Gradient
        # Use a Jacobi preconditioner (diagonal inverse)
        diag_L = L.diagonal()
        M_inv = sp.diags(1 / (diag_L + 1e-8))  # Avoid division by zero
        coeffs, info = sp.linalg.cg(L, D, M=M_inv, maxiter=max_iter)
        if info > 0:
            print(f"Warning: CG did not fully converge after {info} iterations.")
        elif info < 0:
            raise RuntimeError("CG solver failed.")
    return coeffs

def compute_isovalue_smoothed(coeffs, nodes, X, sigma=0.1):
    """
    Compute the isovalue of the implicit function defined by the coefficients and nodes.
    X is the set of sample points where the implicit function is evaluated.
    This is done by evaluating the implicit function at each sample point and averaging the results.
    The weights are normalized so they sum to 1.
    """
    isovalue = 0.0

    for point_sample in X:  # Evaluate implicit function at each sample point
        weights = np.exp(-np.linalg.norm(nodes - point_sample, axis=1)**2 / (2 * np.pi * sigma**2))
        weights /= np.sum(weights)  # Normalize weights to sum to 1
        isovalue += np.dot(weights, coeffs)  # Weighted sum of coefficients

    return isovalue / len(X)  # Average over sample points

def compute_isovalue(nodes, elems, X, coeffs):
    """
    Compute the isovalue of the implicit function defined by the coefficients and nodes.
    X is the set of sample points where the implicit function is evaluated.
    This is done by computing the barycentric coordinates of each barycenter of the tetrahedra,
    then finding the closest barycenter for each sample point and taking the interpolated scalar value.
    The final isovalue is the average of the interpolated values over all sample points.
    """
    barycenters = barycenter(nodes, elems)
    # Compute squared Euclidean distances efficiently using broadcasting
    distances = np.linalg.norm(X[:, np.newaxis, :] - barycenters[np.newaxis, :, :], axis=2)

    # Find the index of the closest barycenter for each point in X
    closest_indices = np.argmin(distances, axis=1)
    
    # Corners of each tetrahedron 
    # NOTE: igl.barycentric_coordinates_tet expects the corners to be in a specific format: https://github.com/libigl/libigl-python-bindings/issues/33
    corners = nodes[elems] # P x 4 x 3
    t1 = np.reshape([corners[:, 0, :]], (-1, 3))
    t2 = np.reshape([corners[:, 1, :]], (-1, 3))
    t3 = np.reshape([corners[:, 2, :]], (-1, 3))
    t4 = np.reshape([corners[:, 3, :]], (-1, 3))
    bary_coords = barycentric_coordinates_tet(
            barycenters, 
            t1, t2, t3, t4
    )
    
    isovalue = 0.0
    for i, point_sample in enumerate(X):
        bary_coords_i = bary_coords[closest_indices[i]]
        vertex_indices_i = elems[closest_indices[i]]
        isovalue +=  bary_coords_i @ coeffs[vertex_indices_i]

    return isovalue / len(X)