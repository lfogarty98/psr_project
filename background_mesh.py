import pyvista as pv
import tetgen
import numpy as np
import polyscope as ps

# Generate a background mesh with desired resolution
def generate_background_mesh(bounds, resolution=30, eps=1e-6):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    grid_x, grid_y, grid_z = np.meshgrid(
        np.linspace(x_min - eps, x_max + eps, resolution),
        np.linspace(y_min - eps, y_max + eps, resolution),
        np.linspace(z_min - eps, z_max + eps, resolution),
        indexing="ij",
    )
    return pv.StructuredGrid(grid_x, grid_y, grid_z).triangulate()

# Define sizing function based on proximity to a point of interest
def sizing_function(points, focus_point=np.array([0, 0, 0]), max_size=1.0, min_size=0.1):
    distances = 1 / (np.linalg.norm(points - focus_point, axis=1) + 1e-16)
    return np.clip(max_size - distances, min_size, max_size)


# Load or create your PLC
sphere = pv.Sphere(radius=1.5, center=(0, 0, 0))
print(f'Sphere bounds: {sphere.bounds}')

# Create a background mesh
bg_mesh = generate_background_mesh(sphere.bounds)
bg_mesh.point_data['target_size'] = sizing_function(bg_mesh.points)

# Compute tetrahedralization
tet_kwargs = dict(order=1, mindihedral=20, minratio=1.5)
tet = tetgen.TetGen(sphere)
nodes, elems = tet.tetrahedralize(bgmesh=bg_mesh, **tet_kwargs)
# refined_mesh = tet.grid

print(f'nodes: {nodes.shape}, elems: {elems.shape}')

ps.init()

sphere_mesh = ps.register_volume_mesh('sphere', nodes, elems)

ps.show()