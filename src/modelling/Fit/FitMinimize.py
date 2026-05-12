import pandas as pd
import numpy as np
from scipy.optimize import fmin, minimize 
from ..Simulator import Simulator
from .fit_config import GET_VELOCITY, INIT

def fit_fmin_model(ABa_dorsal_avg, ABp_dorsal_avg, ABa_ant_avg, ABp_ant_avg) -> tuple[float, float]:
    """
    """
        
    A, B = fmin(residual, 
                (0.1, 0.01), 
                args=(ABa_dorsal_avg, ABp_dorsal_avg, ABa_ant_avg, ABp_ant_avg))
    return (A, B)


def residual(x: tuple[float, float, float], 
        ABa_dorsal_avg: pd.DataFrame, 
        ABp_dorsal_avg: pd.DataFrame, 
        ABa_ant_avg: pd.DataFrame, 
        ABp_ant_avg: pd.DataFrame):

    # initial point not included within tau
    # sim = Simulator(fun(A, B, 195), (0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5), A=A, B=B, t_final=195,
    #                 surfaces=None)
    sim = Simulator(GET_VELOCITY(x), INIT)
    sim.run(False)

    residual = 0

    computed_dorsal_ABa = sim.angle["ABa_dorsal"].to_numpy()
    computed_ABp_dorsal = sim.angle["ABp_dorsal"].to_numpy()
    computed_ABa_anterior = sim.angle["ABa_ant"].to_numpy()
    computed_ABp_anterior = sim.angle["ABp_ant"].to_numpy()

    da_residual = ABa_dorsal_avg.to_numpy() - computed_dorsal_ABa
    dp_residual = ABp_dorsal_avg.to_numpy() - computed_ABp_dorsal
    aa_residual = ABa_ant_avg.to_numpy() - computed_ABa_anterior
    ap_residual = ABp_ant_avg.to_numpy() - computed_ABp_anterior
        
    residual += (np.linalg.norm(da_residual)
                        + np.linalg.norm(dp_residual)
                        + np.linalg.norm(aa_residual)
                        + np.linalg.norm(ap_residual))

    epsilon = 1

    # makes distance 1
    residual += epsilon * ((np.sum(sim.distance["12"].to_numpy() - np.ones(40))) +
                                (np.sum(sim.distance["23"].to_numpy() - np.ones(40))) +
                                (np.sum(sim.distance["34"].to_numpy() - np.ones(40))) +
                                (np.sum(sim.distance["14"].to_numpy() - np.ones(40))))
    
    return residual