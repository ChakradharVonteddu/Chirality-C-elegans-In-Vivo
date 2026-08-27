import time

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from itertools import combinations
from modelling.models.model_config import cell_names, T_FINAL_ES, T_FINAL_NO_ES


class Simulator:
    def __init__(self, model_func, y0, t_eval) -> None:
        self.y0 = y0
        self.TAU_INITIAL = 0
        self.TAU_FINAL = None
        self.t_eval = t_eval
        self.fun = model_func
        self.df = pd.DataFrame([])
        self.distance = pd.DataFrame([])
        self.angle = pd.DataFrame([])

    def run(self, type, save: bool, save_folder = "") -> None:
        """
        Uses RK45

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        """

        if type == "ES":
            self.TAU_FINAL = 1
        else:
            self.TAU_FINAL = T_FINAL_NO_ES/T_FINAL_ES
            
        start = time.time()

        def timed_run(t, y):
            if time.time() - start > 80:
                raise TimeoutError("Simulation timed out (80s limit)")
            return self.fun(t,y)

        solver = solve_ivp(
            timed_run,
            [self.TAU_INITIAL, self.TAU_FINAL],
            self.y0,
            #    method='RK23',
            method="RK45",
            t_eval = self.t_eval,
            max_step = 0.007,
        )

        if solver.status != 0:
            raise Exception("Solver Failed")

        # solver.t to get the timestamps
        format_y = np.transpose(np.array(solver.y))
        self.df = pd.DataFrame(format_y, columns=[str(index) for index in range(len(self.y0))])
        self.compute_distance()
        self.compute_angles()

        if save:
            self.df.to_csv("output/" + save_folder + "/output.csv", index=False)
            self.distance.to_csv("output/" + save_folder + "/distances.csv", index=False)
            self.angle.to_csv("output/" + save_folder + "/angles.csv", index=False)

        print("Total(s):", time.time() - start, solver.nfev, solver.njev, solver.nlu)

    def compute_distance(self):
        """ """
        if self.df.empty:
            raise (Exception("DataFrame not Found."))
        
        #assigns indices to all cells based on their position in cell_names
        cell_indices = range(len(cell_names))
        
        #calculates distances between all possible combinations of two cells
        for c1,c2 in combinations(cell_indices, 2):
            cell1_coords = self.df.iloc[:,3*c1:3*(c1+1)].to_numpy()
            cell2_coords = self.df.iloc[:,3*c2:3*(c2+1)].to_numpy()
            self.distance[f"{c1+1}{c2+1}"] = np.linalg.norm(cell1_coords - cell2_coords, axis = 1)

    def compute_angles(self):
        """
        arccos returns in range [0,pi]

        Note, for the computation of angles, see data diagrams for reference.
        """
        if self.df.empty:
            raise (Exception("DataFrame not Found."))

        # location of "0" degrees, used to dot with axis vectors to obtain angle

        axis_1to2 = pd.DataFrame(
            data={
                "x": self.df["3"] - self.df["0"],
                "y": self.df["4"] - self.df["1"],
                "z": self.df["5"] - self.df["2"],
            }
        )
        axis_4to3 = pd.DataFrame(
            data={
                "x": self.df["6"] - self.df["9"],
                "y": self.df["7"] - self.df["10"],
                "z": self.df["8"] - self.df["11"],
            }
        )

        # anterior needs 2 to 1, 3 to 4 to dot with (1,0)
        axis_1to2["-y"] = -axis_1to2["y"]
        axis_4to3["-y"] = -axis_4to3["y"]
        axis_1to2["-z"] = -axis_1to2["z"]
        axis_4to3["-z"] = -axis_4to3["z"]

        # dorsal view ; dorsal view is top down; (x,y), (-1,0) is 0 degrees. Use dot product rule to obtain.
        # -ABa['-x'] because the axis should be position 2 - position 1
        self.angle["ABa_dorsal"] = (
            np.arccos(
                -axis_1to2["x"] / np.sqrt(axis_1to2["x"] ** 2 + axis_1to2["y"] ** 2)
            )
            * 180
            / np.pi
        )
        self.angle["ABp_dorsal"] = (
            np.arccos(
                -axis_4to3["x"] / np.sqrt(axis_4to3["x"] ** 2 + axis_4to3["y"] ** 2)
            )
            * 180
            / np.pi
        )

        # anterior view ; anterior view is from the front; (y,z), (1,0) is 0 degrees. Dot with (1,0)
        self.angle["ABa_ant"] = (
            np.arccos(
                axis_1to2["-y"] / np.sqrt(axis_1to2["y"] ** 2 + axis_1to2["z"] ** 2)
            )
            * 180
            / np.pi
        )
        self.angle["ABp_ant"] = (
            np.arccos(
                axis_4to3["-y"] / np.sqrt(axis_4to3["y"] ** 2 + axis_4to3["z"] ** 2)
            )
            * 180
            / np.pi
        )

        # print(self.angle.head())
        for i in range(len(self.angle.index)):
            # print(ABa.at[i,'z'])
            if axis_1to2.at[i, "-z"] < 0:
                self.angle.at[i, "ABa_ant"] *= -1
            if axis_4to3.at[i, "-z"] < 0:
                self.angle.at[i, "ABp_ant"] *= -1
