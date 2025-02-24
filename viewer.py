import polyscope as ps
from igl import read_off, grad, marching_tets
from psr_3d import *
import time

# Read data
X, _, N = read_off("data/cat.off")

# Normalize point data to [-1, 1]^3
X = normalize_to_origin(X) # NOTE: may need to tune sigma parameter for compute_gradient_per_vertex

# Compute tetrahedralization
start_time = time.time()
sizing_sigma = 0.24
nodes, elems, bg_mesh = tetrahedralize_sizing_field(X, level=1, resolution=30, sigma=sizing_sigma)
print(f'Sizing sigma: {sizing_sigma}')
end_time = time.time()
print(f'Tetrahedralization execution time: {end_time - start_time}')
print(f'nodes: {nodes.shape}, elems: {elems.shape}')

# Compute gradient per vertex
start_time = time.time()
V_vertex = compute_gradient_per_vertex(nodes, X, N, sigma=0.1) # NOTE: sigma is a parameter that needs to be tuned
end_time = time.time()
print(f'Target vector field execution time: {end_time - start_time}')

# Compute gradient per tetrahedron
V_tet = compute_gradient_per_tet(elems, V_vertex)

# Solve the Poisson problem
start_time = time.time()
coeffs = solve_poisson(nodes, elems, V_tet)
# coeffs = np.zeros_like(nodes[:, 0]) # For skipping poisson solving
end_time = time.time()
print(f'Poisson solving execution time: {end_time - start_time}')

# # Debugging the poisson problem
# G_array = G.toarray()
# L_array = L.toarray()

# Extract surface mesh
isovalue = compute_isovalue(coeffs, nodes, X, sigma=0.1)
print(f'isovalue: {isovalue}')
sv, sf, j, bc = marching_tets(nodes, elems, coeffs, isovalue)


# Initialize polyscope
ps.init()

### Register a point cloud
pc = ps.register_point_cloud("points", X, enabled=False)
background_verts = ps.register_point_cloud("background", bg_mesh.points, enabled=False)

### Register meshes
ps.register_surface_mesh("surface mesh", sv, sf, smooth_shade=True, enabled=False)
ps_vol = ps.register_volume_mesh("volume mesh", nodes, tets=elems)

bg_cells = bg_mesh.cells.reshape(-1, 5)[:, 1:] # need to reshape PyVista mesh cells to tetgen format
ps_bg_mesh = ps.register_volume_mesh("background mesh", bg_mesh.points, tets=bg_cells, enabled=False)

# Add scalar and vector functions
pc.add_vector_quantity("normals", 
        N, color=(0.2, 0.5, 0.5))
ps_vol.add_vector_quantity("V_vertex", V_vertex,
        defined_on='vertices', enabled=False)
ps_vol.add_vector_quantity("V_tet", V_tet,
        defined_on='cells', enabled=False)
ps_vol.add_scalar_quantity("coeffs", coeffs, enabled=True)
background_verts.add_scalar_quantity("sizing_field", bg_mesh.point_data['target_size'], enabled=False)

# Add a slice plane
ps_plane = ps.add_scene_slice_plane()
ps_plane.set_draw_plane(False)
ps_plane.set_draw_widget(True)

# View the point cloud and mesh we just registered in the 3D UI
ps.show()