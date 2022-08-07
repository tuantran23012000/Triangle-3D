import itertools
import matplotlib.pyplot as plt
from pylab import get_cmap
from matplotlib.tri import Triangulation, LinearTriInterpolator
import numpy as np
from scipy import stats
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def simplex(n_vals):
    base = np.linspace(0, 1, n_vals, endpoint=False)
    coords = np.asarray(list(itertools.product(base, repeat=3)))
    return coords[np.isclose(coords.sum(axis=-1), 1.0)]
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
def plot_triangle_3D():
    sim = simplex(5)
    pdf = stats.dirichlet([1.1, 1.5, 1.3]).pdf(sim.T)
    # For shorter notation we define x, y and z:
    x = sim[:, 0]
    y = sim[:, 1]
    z = sim[:, 2]
    tri = Triangulation(x, y)
    triangle_vertices = np.array([np.array([[x[T[0]], y[T[0]], z[T[0]]],
                                            [x[T[1]], y[T[1]], z[T[1]]], 
                                            [x[T[2]], y[T[2]], z[T[2]]]]) for T in tri.triangles])
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[1,0,0],[0.8,0,0.2],[0.8,0.2,0]])]),axis=0)
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[0,1,0],[0,0.8,0.2],[0.2,0.8,0]])]),axis=0)
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[0,0,1],[0.2,0,0.8],[0,0.2,0.8]])]),axis=0)
    midpoints = np.average(triangle_vertices, axis = 1)
    midx = midpoints[:, 0]
    midy = midpoints[:, 1]
    face_color_function = LinearTriInterpolator(tri, pdf)
    face_color_index = face_color_function(midx, midy)
    face_color_index[face_color_index < 0] = 0
    face_color_index /= np.max(pdf)
    cmap = get_cmap(len(face_color_index))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Creating the patches and plotting
    for i in range(len(face_color_index)):
        collection = Poly3DCollection(triangle_vertices[i], facecolors=cmap(i), edgecolors=None)
        ax.add_collection(collection)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
if __name__ == '__main__':
    plot_triangle_3D()