import polyscope as ps
from igl import bounding_box, read_off, grad, loop
import tetgen
import scipy.sparse as sp
from psr_3d import *

# Read data
X, _, N = read_off("data/cat.off")

# Compute bounding box as triangle mesh
v_bbox, f_bbox = bounding_box(X, pad=500.0)

# Create tetrahedralization
# TODO: do refinement by background mesh (see TetGen)
v_refined, f_refined = loop(v_bbox, f_bbox, 3) # Refine bounding box mesh by subdivision (loop)
tgen = tetgen.TetGen(v_refined, f_refined)
nodes, elems = tgen.tetrahedralize()

# Compute gradient per vertex
V_vertex = compute_gradient_per_vertex(nodes, X, N)

# Compute gradient per tetrahedron
V_tet = compute_gradient_per_tet(nodes, elems, V_vertex)

# Define the Poisson problem Lc = D
G = grad(nodes, elems)
M = compute_mass_matrix(nodes, elems)
M_g = sp.block_diag([M, M, M]) # "stretch" mass matrix from (m, m) to (3m, 3m) for x, y and z components in G matrix
L = G.T @ M_g @ G
D = G.T @ M_g @ V_tet.T.flatten()
coeffs = sp.linalg.spsolve(L, D) 
breakpoint()

# Initialize polyscope
ps.init()

### Register a point cloud
pc = ps.register_point_cloud("points", X)

### Register meshes
ps.register_surface_mesh("bbox", v_bbox, f_bbox, smooth_shade=True)
ps.register_surface_mesh("triangulation", v_refined, f_refined, smooth_shade=True)
ps_vol = ps.register_volume_mesh("volume mesh", nodes, tets=elems)

# Add scalar and vector functions
pc.add_vector_quantity("normals", 
        N, color=(0.2, 0.5, 0.5))
ps_vol.add_vector_quantity("V_vertex", V_vertex,
        defined_on='vertices', enabled=False)
ps_vol.add_vector_quantity("V_tet", V_tet,
        defined_on='cells', enabled=False)
ps_vol.add_scalar_quantity("c", c, enabled=False)

# View the point cloud and mesh we just registered in the 3D UI
ps.show()