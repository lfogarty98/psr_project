import polyscope as ps
from igl import bounding_box, read_off
import tetgen
from psr_3d import *

# Read data
X, _, N = read_off("data/cat.off")

# Compute bounding box as triangle mesh
v_bbox, f_bbox = bounding_box(X, pad=500.0)

# Refine bounding box mesh by subdivision (loop)
from igl import loop
v_refined, f_refined = loop(v_bbox, f_bbox, 3)

# Create tetrahedralization
tgen = tetgen.TetGen(v_refined, f_refined)
nodes, elems = tgen.tetrahedralize()

# Compute gradient per vertex
V_vertex = compute_gradient_per_vertex(nodes, X, N)

# Compute gradient per tetrahedron
V_tet = compute_gradient_per_tet(nodes, elems, V_vertex)
breakpoint()
# Initialize polyscope
ps.init()

### Register a point cloud
ps.register_point_cloud("points", X)

### Register meshes
ps.register_surface_mesh("bbox", v_bbox, f_bbox, smooth_shade=True)
ps.register_surface_mesh("triangulation", v_refined, f_refined, smooth_shade=True)
ps_vol = ps.register_volume_mesh("volume mesh", nodes, tets=elems)

# Add scalar and vector functions
ps.get_point_cloud("points").add_vector_quantity("normals", 
        N, color=(0.2, 0.5, 0.5))
ps.get_volume_mesh("volume mesh").add_vector_quantity("V_vertex", V_vertex,
        defined_on='vertices', enabled=False)
ps.get_volume_mesh("volume mesh").add_vector_quantity("V_tet", V_tet,
        defined_on='cells', enabled=False)

# View the point cloud and mesh we just registered in the 3D UI
ps.show()