import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from IPython.display import HTML

def plot_spheres(ax, points, radii):
    """
    Plots spheres in 3D centered at specified points with given radii.

    Parameters:
        ax (matplotlib.axes._subplots.Axes3DSubplot): 3D axis object.
        points (list of tuples): List of four 3D coordinates [(x1, y1, z1), (x2, y2, z2), ...].
        radii (list of floats): List of four radii corresponding to the spheres.
    """
    ax.clear()
    
    # Define sphere colors
    sphere_colors = ['blue', 'darkblue', 'green', 'darkgreen']

    # Create spheres
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    for (x, y, z), r, sphere_color in zip(points, radii, sphere_colors):
        X = r * np.outer(np.cos(u), np.sin(v)) + x
        Y = r * np.outer(np.sin(u), np.sin(v)) + y
        Z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        ax.plot_surface(X, Y, Z, alpha=1.0, rstride=4, cstride=4, facecolor=sphere_color)

    # Draw line segments between the centers of the spheres
    ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], [points[0][2], points[1][2]], color='black', linestyle='--', label='Line 1')
    ax.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], [points[2][2], points[3][2]], color='red', linestyle='--', label='Line 2')

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title("3D Spheres")
    ax.legend()

def add_sphere(ax, point, radius, color_index):
    """
    Adds a single sphere to an existing plot.

    Parameters:
        ax (matplotlib.axes._subplots.Axes3DSubplot): 3D axis object.
        point (tuple): The (x, y, z) coordinates of the sphere center.
        radius (float): The radius of the sphere.
        color_index (int): An integer from 1 to 4 indicating sphere color.
    """
    sphere_colors = ['blue', 'darkblue', 'green', 'darkgreen']
    color = sphere_colors[color_index - 1]  # Convert 1-based index to 0-based

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    X = radius * np.outer(np.cos(u), np.sin(v)) + point[0]
    Y = radius * np.outer(np.sin(u), np.sin(v)) + point[1]
    Z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + point[2]
    ax.plot_surface(X, Y, Z, alpha=1.0, rstride=4, cstride=4, facecolor=color)

def update(frame, ax, points_sequence, radii):
    plot_spheres(ax, points_sequence[frame], radii)

def animate_spheres(points_sequence, radii, interval=200):
    """
    Creates an animation of spheres moving in 3D space, properly displayed in Jupyter Notebook.

    Parameters:
        points_sequence (list of lists of tuples): A sequence of point sets, each containing four 3D coordinates.
        radii (list of floats): List of four radii corresponding to the spheres.
        interval (int): Time between frames in milliseconds.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ani = animation.FuncAnimation(fig, update, frames=len(points_sequence), fargs=(ax, points_sequence, radii), interval=interval)
    
    return HTML(ani.to_jshtml())

# Example usage: Define a sequence of points over time
num_frames = 50
points_sequence = [[(np.sin(t/10) + i, np.cos(t/10) + i, i/5) for i in range(4)] for t in range(num_frames)]
radii = [1.5, 2.0, 1.0, 2.5]

# Display animation in Jupyter Notebook
animate_spheres(points_sequence, radii)
