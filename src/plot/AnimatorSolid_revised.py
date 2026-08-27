import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from modelling.models.model_config import cell_names

colors = ["lightblue", "blue", "red", "salmon", "orange", "green", "green", "green", "green"]

class CorticalMarkers:
    def __init__(self):
        u = np.linspace(0, 2 * np.pi, 8, endpoint=False) 
        v = np.linspace(np.pi / 6, 5 * np.pi / 6, 5)
        
        theta_grid, phi_grid = np.meshgrid(u, v)

        self.theta = theta_grid.flatten()
        self.phi = phi_grid.flatten()
        
        self.p0 = np.array([
            np.cos(self.theta) * np.sin(self.phi),
            np.sin(self.theta) * np.sin(self.phi),
            np.cos(self.phi)
        ])

    def _calculate_total_angle(self, t: float, alpha: float, lam: float, radius: float) -> float:
        if lam == 0: return 0 
        cort_int = alpha * ( (1 / lam) - np.exp(-lam * t) * (t + (1 / lam)) )
        return cort_int / radius

    def _get_rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        norm = np.linalg.norm(axis)
        if norm == 0: return np.eye(3)
        axis = axis / norm
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        I = np.eye(3)
        return I + np.sin(angle) * K + (1 - np.cos(angle)) * np.matmul(K, K)

    def update_positions(self, t: float, center: list, radius: float, 
                         spindle_axis: list, alpha: float, lam: float) -> np.ndarray:
        angle = self._calculate_total_angle(t, alpha, lam, radius)
        A = self._get_rotation_matrix(np.array(spindle_axis), angle)
        rotated_vectors = A @ self.p0 
        center_vec = np.array(center).reshape(3, 1) 
        global_positions = center_vec + (1.02 * radius * rotated_vectors)
        return global_positions.T

class Animator:
    def __init__(self, data, radii_data, alpha, lam, scale_factor) -> None:
        self.data = Animator.process_df(data)
        self.radii = radii_data 
        self.spheres = []  
        self.markers = [CorticalMarkers() for _ in range(4)]
        self.alpha = alpha
        self.lam = lam
        self.scale_factor = scale_factor

    def animate(self, save_folder):
        FIG = plt.figure()
        AX = FIG.add_subplot(projection="3d") 

        SIZE = [
        -2,  
        2,   
        -2,  
        2,   
        -2,  
        2,   
        ] 

        AX.set(xlim3d=(SIZE[0], SIZE[1]), xlabel="X") 
        AX.set(ylim3d=(SIZE[2], SIZE[3]), ylabel="Y") 
        AX.set(zlim3d=(SIZE[4], SIZE[5]), zlabel="Z") 
        AX.set_xticklabels([]) 
        AX.set_yticklabels([]) 
        AX.set_zticklabels([])

        positions_vectors = [] 
        for position_index in range(len(self.data)): 
            (new_position,) = AX.plot(
                [],
                [],
                [],
                ".",
                alpha=0.4,
                markersize=3,
                label=position_index + 1,
                c=colors[position_index],
            ) 
            positions_vectors.append(new_position) 

        steps = len(self.data[0].index) 
        
        AX.legend() 
        
        custom_legend = [] 
        for i in range(len(cell_names[0:5])): 
           custom_legend.append(
                Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[i],
                markersize=8,
                label=cell_names[i],
            )) 
        
        custom_legend.append(
                Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[-1],
                markersize=8,
                label="ems",
            )) 
        
        AX.legend(handles=custom_legend) 

        (rotation_axis_1,) = AX.plot(
            [], [], [], ":", alpha=0.4, markersize=0, linewidth=1, c="black"
        ) 
        (rotation_axis_2,) = AX.plot(
            [], [], [], ":", alpha=0.4, markersize=0, linewidth=1, c="black"
        ) 
        rotation_axes = (
            rotation_axis_1,
            rotation_axis_2,
        ) 

        (wall_x1,) = AX.plot(
            [], [], [], "-o", alpha=0.4, markersize=3, linewidth=1, c="black"
        ) 
        (wall_x2,) = AX.plot(
            [], [], [], "-o", alpha=0.4, markersize=3, linewidth=1, c="black"
        ) 
        (wall_y1,) = AX.plot(
            [], [], [], "-o", alpha=0.4, markersize=3, linewidth=1, c="black"
        ) 
        (wall_y2,) = AX.plot(
            [], [], [], "-o", alpha=0.4, markersize=3, linewidth=1, c="black"
        ) 
        (wall_z1,) = AX.plot(
            [], [], [], "-o", alpha=0.4, markersize=3, linewidth=1, c="black"
        ) 
        (wall_z2,) = AX.plot(
            [], [], [], "-o", alpha=0.4, markersize=3, linewidth=1, c="black"
        ) 
        wall_vectors = (
            wall_x1,
            wall_x2,
            wall_y1,
            wall_y2,
            wall_z1,
            wall_z2,
        )

        marker_plots = []
        for _ in range(4):
            scatter = AX.scatter([], [], [], color="black", s=4, depthshade=True)
            marker_plots.append(scatter)

        anim = FuncAnimation(
            FIG,
            self.update_replace,
            steps,
            fargs=(self.data, self.radii, SIZE, positions_vectors, rotation_axes, wall_vectors, marker_plots, AX),
            interval=1,
            repeat=True,
        ) 

        AX.view_init(30, 60, 0) 
        anim.save("output/" + save_folder +  "/XYZ.gif", writer="pillow") 
        AX.view_init(90, 90, 0) 
        anim.save("output/" + save_folder + "/XY.gif", writer="pillow") 
        AX.view_init(0, -90, 0) 
        anim.save("output/" + save_folder + "/XZ.gif", writer="pillow") 
        AX.view_init(0, 0, 0) 
        anim.save("output/" + save_folder + "/YZ.gif", writer="pillow") 

    @staticmethod
    def plot_ellipsoid(ax, center, radii, color="gray", alpha=1.0):
        u = np.linspace(0, 2 * np.pi, 60) 
        v = np.linspace(0, np.pi, 30) 

        x = radii[0] * np.outer(np.cos(u), np.sin(v)) + center[0] 
        y = radii[1] * np.outer(np.sin(u), np.sin(v)) + center[1] 
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2] 

        surface = ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor="none") 
        return surface 

    @staticmethod
    def plot_sphere(ax, center, radius, color="gray", alpha=1.0):
        return Animator.plot_ellipsoid(
            ax, center, (radius, radius, radius), color=color, alpha=alpha
        ) 

    def update_replace(
        self, frame, data, radii, SIZE, positions, rotation_axes, wall_vectors, marker_plots, ax
    ):
        for sphere in self.spheres: 
            sphere.remove() 
        self.spheres.clear() 

        current_centers = []
        for curve_index in range(len(positions)): 
            x = data[curve_index]["x"][frame] 
            y = data[curve_index]["y"][frame] 
            z = data[curve_index]["z"][frame] 
            current_centers.append(np.array([x, y, z]))

            if curve_index <= 4: 
                sphere = self.plot_sphere(ax, (x, y, z), radii.loc[frame, cell_names[curve_index]], color=colors[curve_index], alpha = 1)
            else: 
                sphere = self.plot_sphere(ax, (x, y, z), radii.iloc[frame, -1], color=colors[curve_index], alpha = 1) 
            self.spheres.append(sphere) 

        t = frame * self.scale_factor #dimensionalized t
        
        spindle_axes = [
            current_centers[0] - current_centers[1], 
            current_centers[1] - current_centers[0], 
            current_centers[2] - current_centers[3], 
            current_centers[3] - current_centers[2]  
        ]
        
        for i in range(4):
            radius_val = radii.loc[frame, cell_names[i]]
            new_positions = self.markers[i].update_positions(
                t=t,
                center=current_centers[i],
                radius=radius_val,
                spindle_axis=spindle_axes[i],
                alpha=self.alpha,
                lam=self.lam
            )
            marker_plots[i]._offsets3d = (new_positions[:, 0], new_positions[:, 1], new_positions[:, 2])

        wall_vectors[0].set_data(
            [[SIZE[0], SIZE[0]], [data[0]["y"][frame], data[1]["y"][frame]]]
        ) 
        wall_vectors[0].set_3d_properties([data[0]["z"][frame], data[1]["z"][frame]])
        wall_vectors[1].set_data(
            [[SIZE[0], SIZE[0]], [data[2]["y"][frame], data[3]["y"][frame]]]
        ) 
        wall_vectors[1].set_3d_properties([data[2]["z"][frame], data[3]["z"][frame]])
        wall_vectors[2].set_data(
            [[data[0]["x"][frame], data[1]["x"][frame]], [SIZE[2], SIZE[2]]]
        ) 
        wall_vectors[2].set_3d_properties([data[0]["z"][frame], data[1]["z"][frame]]) 
        wall_vectors[3].set_data(
            [[data[2]["x"][frame], data[3]["x"][frame]], [SIZE[2], SIZE[2]]]
        ) 
        wall_vectors[3].set_3d_properties([data[2]["z"][frame], data[3]["z"][frame]]) 
        wall_vectors[4].set_data(
            [
                [data[0]["x"][frame], data[1]["x"][frame]],
                [data[0]["y"][frame], data[1]["y"][frame]],
            ]
        ) 
        wall_vectors[4].set_3d_properties([SIZE[4], SIZE[4]]) 
        wall_vectors[5].set_data(
            [
                [data[2]["x"][frame], data[3]["x"][frame]],
                [data[2]["y"][frame], data[3]["y"][frame]],
            ]
        ) 
        wall_vectors[5].set_3d_properties([SIZE[4], SIZE[4]]) 

        rotation_axes[0].set_data(
            [
                [data[0]["x"][frame], data[1]["x"][frame]],
                [data[0]["y"][frame], data[1]["y"][frame]],
            ]
        ) 
        rotation_axes[0].set_3d_properties([data[0]["z"][frame], data[1]["z"][frame]]) 
        rotation_axes[1].set_data(
            [
                [data[2]["x"][frame], data[3]["x"][frame]],
                [data[2]["y"][frame], data[3]["y"][frame]],
            ]
        ) 
        rotation_axes[1].set_3d_properties([data[2]["z"][frame], data[3]["z"][frame]]) 

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

    @staticmethod
    def generate_ball(position: tuple[float, float, float], radius: float):
        delta = np.pi / 20 
        rotation_matrix = np.array(
            [[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]]
        ) 

        center_x, center_y, center_z = position
        curr_x, curr_y, curr_z = 0, 0, radius
        ref_z = 0 
        x = [center_x + curr_x] 
        y = [center_y + curr_y] 
        z = [center_z + curr_z] 

        for _ in range(20): 
            phi_vector = np.matmul(rotation_matrix, [ref_z, curr_z]) 
            ref_z = phi_vector[0] 
            curr_z = phi_vector[1] 
            curr_x = ref_z 
            curr_y = 0 
            for _ in range(40): 
                theta_vector = np.matmul(rotation_matrix, [curr_x, curr_y]) 
                curr_x = theta_vector[0] 
                curr_y = theta_vector[1] 
                x.append(center_x + curr_x) 
                y.append(center_y + curr_y) 
                z.append(center_z + curr_z) 

        return {"x": x, "y": y, "z": z} 