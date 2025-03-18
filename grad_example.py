import igl
import numpy as np
import polyscope as ps

v, f = igl.read_triangle_mesh("data/cheburashka.off")
u = igl.read_dmat("data/cheburashka-scalar.dmat")

m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)

g = igl.grad(v, f)
breakpoint()

gu = g.dot(u).reshape(f.shape, order="F")
gu_mag = np.linalg.norm(gu, axis=1)












# p = plot(v, f, u, shading={"wireframe":False}, return_plot=True)
# max_size = igl.avg_edge_length(v, f) / np.mean(gu_mag)
# bc = igl.barycenter(v, f)
# bcn = bc + max_size * gu
# p.add_lines(bc, bcn, shading={"line_color": "black"});

breakpoint()

ps.init()

tri_mesh = ps.register_surface_mesh("mesh", v, f, smooth_shade=True)
u = np.array(u)
tri_mesh.add_scalar_quantity("u", u, enabled=False)
tri_mesh.add_scalar_quantity("gu_mag", gu_mag, defined_on='faces', enabled=False)

ps.show()