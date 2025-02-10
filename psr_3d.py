import numpy as np
import tetgen
from igl import loop, bounding_box
import pyvista as pv
from scipy.spatial import Delaunay
import gpytoolbox as gpy


def normalize_to_origin(X):
    """
    Normalize 3D points so they are centered at the origin and fit within [-1,1]^3.
    
    :param X: (N, 3) NumPy array of 3D points
    :return: (N, 3) NumPy array of normalized points
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
    """_summary_
    Generate a background mesh with desired resolution.
    Copied from TetGen docs https://tetgen.pyvista.org/

    Args:
        bounds (_type_): _description_
        resolution (int, optional): _description_. Defaults to 20.
        eps (_type_, optional): _description_. Defaults to 1e-6.

    Returns:
        _type_: _description_
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

def sizing_function_gaussian(points, X, sigma=0.1):
    sd = gpy.squared_distance(points, X, use_cpp=True)
    sizing_field = np.exp(-sd[0]/ sigma/sigma)
    return sizing_field

def tetrahedralize_sizing_field(X):
    
    # Load or create your PLC
    sphere = pv.Sphere(radius=1.5, center=(0, 0, 0))
    
    # Create a background mesh
    bounds = sphere.bounds
    print(f'Background mesh bounds: {bounds}')
    bg_mesh = generate_background_mesh(bounds, resolution=10, eps=1e-6)
    
    # Compute sizing field
    sizing_field = naive_sizing_function(bg_mesh.points)
    bg_mesh.point_data['target_size'] = sizing_field
    
    
    # Create refined mesh
    tet_kwargs = dict(order=1, mindihedral=20, minratio=1.5)
    breakpoint()
    tet = tetgen.TetGen(sphere)
    
    nodes, elems = tet.tetrahedralize(bgmesh=bg_mesh, **tet_kwargs) ## 
    
    return nodes, elems, bg_mesh
    

def tetrahedralize_regular_grid(res=30, padding=0.1):
    """
    Tetrahedralize a regular grid.
    Uses scipy.spatial.Delaunay to compute the tetrahedralization.
    
    :param res: int, resolution of the grid
    :return: nodes, elems
    """
    xs = np.linspace(-1. - padding, 1. + padding, res)
    X = np.array(np.meshgrid(xs, xs, xs)).transpose(1, 2, 3, 0).reshape(-1, 3)
    tet = Delaunay(X, qhull_options='Qbb Qc Qz Q12 Q0') # add Qhull options to avoid degenerate tets (see https://github.com/scipy/scipy/issues/16094)
    nodes, elems = tet.points, tet.simplices
    return nodes, elems

# Tetrahedralization I initially used
def naive_tetrahedralize(X):
    """
    Naive tetrahedralization I initially used.
    
    Computes a bounding box as a triangle mesh around the 3D points and refines it by subdivision.
    This is then used for the tetrahedralization.
    
    :param X: (N, 3) NumPy array of 3D points
    :return: (N, 3) NumPy array of normalized points
    """
    # Compute bounding box as triangle mesh
    v_bbox, f_bbox = bounding_box(X, pad=1.0)

    # Create tetrahedralization
    # TODO: do refinement by background mesh (see TetGen)
    v_refined, f_refined = loop(v_bbox, f_bbox, 3) # Refine bounding box mesh by subdivision (loop)
    tgen = tetgen.TetGen(v_refined, f_refined)
    nodes, elems = tgen.tetrahedralize()
    return nodes, elems

def tetrahedralize_sphere(radius=1.5):
    """
    Tetrahedralize a sphere.
    
    :param radius: float, radius of the sphere
    :return: nodes, elems
    """
    sphere = pv.Sphere(radius=radius)
    tet = tetgen.TetGen(sphere)
    nodes, elems = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
    return nodes, elems

def compute_grid(X, min_x, max_x, min_y, max_y, min_z, max_z, cell_size=0.1, padding=0.2):
    x = np.linspace(min_x - padding, max_x + padding, int((max_x - min_x + 2 * padding) / cell_size))
    y = np.linspace(min_y - padding, max_y + padding, int((max_y - min_y + 2 * padding) / cell_size))
    z = np.linspace(min_z - padding, max_z + padding, int((max_z - min_z + 2 * padding) / cell_size))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # 'ij' indexing for proper order
    return X, Y, Z

def compute_gradient_per_vertex(points, X, N, sigma=0.1):
    V = np.zeros((len(points), 3))
    for i in range(len(points)):
        for j in range(len(X)):
            weight = (np.exp(-np.linalg.norm(points[i] - X[j])**2 / (2 * np.pi * sigma**2)))
            V[i] += weight * N[j]
    return V

def compute_gradient_per_tet(nodes, elems, V):
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

# Compute the volume of each tetrahedron using the determinant method
def compute_tetrahedron_volume(v0, v1, v2, v3):
    matrix = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
    det = np.linalg.det(matrix)
    volume = abs(det) / 6.0
    return volume

# Compute the simplex-wise mass matrix for a 3D tetrahedral mesh.
def compute_mass_matrix(points, simplices):
    # Extract vertex positions for each simplex
    v0, v1, v2, v3 = points[simplices[:, 0]], points[simplices[:, 1]], points[simplices[:, 2]], points[simplices[:, 3]]

    # Compute volumes for all tetrahedra
    volumes = np.array([compute_tetrahedron_volume(v0[i], v1[i], v2[i], v3[i]) for i in range(len(simplices))])

    print(f'Volumes: {volumes.shape}')
    print(f"Number of degenerate tetrahedra: {np.sum(volumes <= 1e-12)}")
    
    # Create the diagonal mass matrix
    mass_matrix = np.diag(volumes)

    return mass_matrix

def compute_isovalue(coeffs, nodes, X, sigma=0.1):
    isovalue = 0.0
    for point_sample in X:
        for i, node in enumerate(nodes):
            weight = (np.exp(-np.linalg.norm(point_sample - node)**2 / (2 * np.pi * sigma**2)))
            isovalue += weight * coeffs[i]
        isovalue /= len(nodes)
    return isovalue/len(X)