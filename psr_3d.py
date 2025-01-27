import numpy as np

def compute_bounding_box(X):
    min_x = np.min(X[:,0])
    max_x = np.max(X[:,0])
    min_y = np.min(X[:,1])
    max_y = np.max(X[:,1])
    min_z = np.min(X[:, 2])
    max_z = np.max(X[:, 2])
    return min_x, max_x, min_y, max_y, min_z, max_z

def compute_grid(X, min_x, max_x, min_y, max_y, min_z, max_z, cell_size=0.1, padding=0.2):
    x = np.linspace(min_x - padding, max_x + padding, int((max_x - min_x + 2 * padding) / cell_size))
    y = np.linspace(min_y - padding, max_y + padding, int((max_y - min_y + 2 * padding) / cell_size))
    z = np.linspace(min_z - padding, max_z + padding, int((max_z - min_z + 2 * padding) / cell_size))
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # 'ij' indexing for proper order
    return X, Y, Z

def compute_gradient_per_vertex(points, X, N, sigma=50.0):
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