import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

class Animator:
    def __init__(self, data) -> None:
        self.data = Animator.process_df(data)
        self.spheres = []  # Store previous sphere objects

    def animate(self):
        """
        Takes a euler df and animates it in matplotlib, assumes 3 components
        """
        # setup figure and axes
        FIG = plt.figure()
        AX = FIG.add_subplot(projection="3d")

        # setting range
        SIZE = [
            -1.3,
            1,
            -1.6,
            1,
            -1,
            1,
        ]  # plotting limits x low, x high, yl, yh, zl, zh

        AX.set(xlim3d=(SIZE[0], SIZE[1]), xlabel="X")
        AX.set(ylim3d=(SIZE[2], SIZE[3]), ylabel="Y")
        AX.set(zlim3d=(SIZE[4], SIZE[5]), zlabel="Z")
        AX.set_xticklabels([])
        AX.set_yticklabels([])
        AX.set_zticklabels([])

        steps = len(self.data[0].index)

        # animate, run update_replace at every step
        anim = FuncAnimation(
            FIG,
            self.update_replace,
            steps,
            fargs=(self.data, AX),
            interval=1,
            repeat=True,
        )

        # save animation from different angles
        AX.view_init(30, 60, 0)
        anim.save("output/XYZ.gif", writer="pillow")
        AX.view_init(90, 90, 0)
        anim.save("output/XY.gif", writer="pillow")
        AX.view_init(0, -90, 0)
        anim.save("output/XZ.gif", writer="pillow")
        AX.view_init(0, 0, 0)
        anim.save("output/YZ.gif", writer="pillow")

    @staticmethod
    def plot_ellipsoid(ax, center, radii, color="gray", alpha=1.0):
        """
        Plot a 3D ellipsoid on the given axis.
        """
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 30)

        x = radii[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2]

        surface = ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor="none")
        return surface

    @staticmethod
    def plot_sphere(ax, center, radius, color="gray", alpha=1.0):
        """
        Plot a sphere by calling plot_ellipsoid with equal radii.
        """
        return Animator.plot_ellipsoid(
            ax, center, (radius, radius, radius), color=color, alpha=alpha
        )

    def update_replace(self, frame, data, ax):
        """
        Update function for animation.
        """
        # Remove old sphere
        for sphere in self.spheres:
            sphere.remove()
        self.spheres.clear()

        # Update position for the single cell
        x = data[0]["x"][frame]
        y = data[0]["y"][frame]
        z = data[0]["z"][frame]
        
        sphere = self.plot_sphere(ax, (x, y, z), 1, color="red")
        self.spheres.append(sphere)

    @staticmethod
    def process_df(position_df: pd.DataFrame):
        if position_df is None:
            raise Exception("No data found in Euler instance.")

        cols = position_df.columns
        col_len = len(cols)
        if col_len % 3 != 0:
            raise Exception("Euler instance has invalid dimensions.")

        df_list = []
        for index in range(col_len // 3):
            obj_df = position_df[
                [cols[3 * index], cols[3 * index + 1], cols[3 * index + 2]]
            ]
            obj_df = obj_df.set_axis(["x", "y", "z"], axis=1)
            df_list.append(obj_df)
        return df_list