import numpy as np
from ..model_config import T_FINAL, R0
import pandas as pd
from ..Force_Calculator import ForceCalculator


def get_velocity(params, return_data = False):
    #params[0] - mu, params[1] - alpha, params[2] - lambda
    def func(t, y):
       
      rotation_axis = np.array([1, 0, 0])

      cortical_flow = params[1]*params[2]*t*T_FINAL*np.e**(-params[2]*t*T_FINAL)
       
      u_target = np.array([0,0,1])

      cell_prime = params[0] * cortical_flow * - np.cross(-u_target,rotation_axis)

      proj_mat = np.identity(3) - np.outer(u_target,u_target)

      cell_prime_mod = np.linalg.solve(np.identity(3) + params[0]*proj_mat, cell_prime)
       
      if not return_data:
        return np.concatenate([cell_prime_mod])
       
      force_data = {"Time" : t, 
                    "cort_fric_F" : np.linalg.norm(params[0] * cortical_flow * - np.cross(-u_target,rotation_axis)),
                    "rel_vel_fric_F" : np.linalg.norm(cell_prime_mod - cell_prime),
                    "cell_velocity" : np.linalg.norm(cell_prime_mod),
                    "cortical_velocity" : cortical_flow}

      return force_data
    return func