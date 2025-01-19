import polyscope as ps
from igl import bounding_box, read_off
import tetgen

# Read data
X, F, N = read_off("data/cat.off")
v_bbox, f_bbox = bounding_box(X, pad=50.0)

# Create tetrahedralization
tgen = tetgen.TetGen(v_bbox, f_bbox)
nodes, elems = tgen.tetrahedralize()



# Initialize polyscope
ps.init()

### Register a point cloud
ps.register_point_cloud("my points", X)

### Register a mesh
ps.register_surface_mesh("my mesh", X, F, smooth_shade=True)
ps.register_surface_mesh("my bbox", v_bbox, f_bbox, smooth_shade=True)
ps_vol = ps.register_volume_mesh("test volume mesh", nodes, tets=elems)

# Add a scalar function and a vector function defined on the mesh
ps.get_surface_mesh("my mesh").add_vector_quantity("my_vector", 
        N, defined_on='vertices', color=(0.2, 0.5, 0.5))

# View the point cloud and mesh we just registered in the 3D UI
ps.show()