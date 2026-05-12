import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import time


class Simulator:

    def __init__(self, model_func, y0) -> None:
        self.y0 = y0
        self.TAU_INITIAL = 0
        self.TAU_FINAL = 1 # non-dimensionalized
        self.fun = model_func
        self.df = pd.DataFrame([])
        self.distance = pd.DataFrame([])
        self.angle = pd.DataFrame([])

    def run(self, save: bool) -> None:
        """
        Uses RK45

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        """
        start = time.time()
        solver = solve_ivp(self.fun, 
                           [self.TAU_INITIAL, self.TAU_FINAL], 
                           self.y0, 
                        #    method='RK23',
                           method='RK45',
                           t_eval = np.linspace(self.TAU_INITIAL, self.TAU_FINAL, 40)
                           )
        
        if solver.status != 0:
            raise(Exception("Solver Failed"))
        

        # solver.t to get the timestamps
        format_y = np.transpose(np.array(solver.y))
        self.df = pd.DataFrame(format_y, columns=[str(index) for index in range(12)])
        self.compute_distance()
        self.compute_angles()

        if save:
            self.df.to_excel('output.xlsx', index=False)
            self.distance.to_excel('distances.xlsx', index=False)
            self.angle.to_excel('angles.xlsx', index=False)
        
        print("Total(s):", time.time()-start, solver.nfev, solver.njev, solver.nlu)
                    

    def compute_distance(self):
        """
        """
        if self.df.empty:
            raise(Exception("DataFrame not Found."))
        
        self.distance["12"] = np.sqrt((self.df["0"]-self.df["3"])**2 + (self.df["1"]-self.df["4"])**2 + (self.df["2"]-self.df["5"])**2)
        self.distance["13"] = np.sqrt((self.df["0"]-self.df["6"])**2 + (self.df["1"]-self.df["7"])**2 + (self.df["2"]-self.df["8"])**2)
        self.distance["14"] = np.sqrt((self.df["0"]-self.df["9"])**2 + (self.df["1"]-self.df["10"])**2 + (self.df["2"]-self.df["11"])**2)
        self.distance["23"] = np.sqrt((self.df["3"]-self.df["6"])**2 + (self.df["4"]-self.df["7"])**2 + (self.df["5"]-self.df["8"])**2)
        self.distance["24"] = np.sqrt((self.df["3"]-self.df["9"])**2 + (self.df["4"]-self.df["10"])**2 + (self.df["5"]-self.df["11"])**2)
        self.distance["34"] = np.sqrt((self.df["6"]-self.df["9"])**2 + (self.df["7"]-self.df["10"])**2 + (self.df["8"]-self.df["11"])**2)
    
    def compute_angles(self):
        """
        arccos returns in range [0,pi]

        Note, for the computation of angles, see data diagrams for reference. 
        """
        if self.df.empty:
            raise(Exception("DataFrame not Found."))


        # location of "0" degrees, used to dot with axis vectors to obtain angle

        axis_1to2 = pd.DataFrame(data={"x": self.df['3']-self.df['0'],
                                 "y": self.df['4']-self.df['1'],
                                 "z": self.df['5']-self.df['2']})
        axis_4to3 = pd.DataFrame(data={"x": self.df['6']-self.df['9'],
                                 "y": self.df['7']-self.df['10'],
                                 "z": self.df['8']-self.df['11']})
        

        # anterior needs 2 to 1, 3 to 4 to dot with (1,0)
        axis_1to2['-y'] = -axis_1to2['y']
        axis_4to3['-y'] = -axis_4to3['y']
        axis_1to2['-z'] = -axis_1to2['z']
        axis_4to3['-z'] = -axis_4to3['z']

        #dorsal view ; dorsal view is top down; (x,y), (-1,0) is 0 degrees. Use dot product rule to obtain. 
        #-ABa['-x'] because the axis should be position 2 - position 1
        self.angle['ABa_dorsal'] = np.arccos(-axis_1to2['x']/np.sqrt(axis_1to2['x']**2 + axis_1to2['y']**2)) * 180 / np.pi
        self.angle['ABp_dorsal'] = np.arccos(-axis_4to3['x']/np.sqrt(axis_4to3['x']**2 + axis_4to3['y']**2)) * 180 / np.pi

        #anterior view ; anterior view is from the front; (y,z), (1,0) is 0 degrees. Dot with (1,0)
        self.angle['ABa_ant'] = np.arccos(axis_1to2['-y']/np.sqrt(axis_1to2['y']**2 + axis_1to2['z']**2)) * 180 / np.pi
        self.angle['ABp_ant'] = np.arccos(axis_4to3['-y']/np.sqrt(axis_4to3['y']**2 + axis_4to3['z']**2)) * 180 / np.pi
        
        # print(self.angle.head())
        for i in range(len(self.angle.index)):
            # print(ABa.at[i,'z'])
            if axis_1to2.at[i,'-z'] < 0:
                self.angle.at[i,'ABa_ant'] *= -1
            if axis_4to3.at[i,'-z'] < 0:
                self.angle.at[i,'ABp_ant'] *= -1
        