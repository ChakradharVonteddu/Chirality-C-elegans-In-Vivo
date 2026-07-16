import time
import numpy as np

from ..least_distance.ellipsoid import min_point_ellpsoid
from .model_config import T_FINAL, E0, E1

def get_velocity(params):

    def func(t, y):
        """
        A division by 0 occurs when two cells overlap
        """
        # start = time.time()
        ABal = np.array([y[0], y[1], y[2]])
        ABar = np.array([y[3], y[4], y[5]])
        ABpr = np.array([y[6], y[7], y[8]])
        ABpl = np.array([y[9], y[10], y[11]])

        dist12 = np.linalg.norm(ABal-ABar)
        dist13 = np.linalg.norm(ABal-ABpr)
        dist14 = np.linalg.norm(ABal-ABpl)
        dist23 = np.linalg.norm(ABar-ABpr)
        dist24 = np.linalg.norm(ABar-ABpl)
        dist34 = np.linalg.norm(ABpr-ABpl)

        u12 = (ABar-ABal) / dist12 # 2-1 
        u13 = (ABpr-ABal) / dist13 # 3-1 
        u14 = (ABpl-ABal) / dist14 # 4-1
        u23 = (ABpr-ABar) / dist23 # 3-2
        u24 = (ABpl-ABar) / dist24 # 4-2
        u34 = (ABpl-ABpr) / dist34 # 4-3
        k_hat = np.array([0,0,1])

        cortical_flow_r = 0.000345*t*T_FINAL*np.e**(-0.012732*t*T_FINAL)
        cortical_flow_l = 0.00071*t*T_FINAL*np.e**(-0.0166*t*T_FINAL)
        cortical_flow_r = (cortical_flow_r + cortical_flow_l)/2
        cortical_flow_l = (cortical_flow_r + cortical_flow_l)/2

        # avg of two cortical flows
        a = 0.0005275 
        lam = 0.014666
        cortical_int = a*t*T_FINAL*(-t/(lam * T_FINAL)*np.e**(-lam * T_FINAL * t)-1/(lam**2*T_FINAL**2)*np.e**(-lam * T_FINAL*t)+1/(lam**2*T_FINAL**2))
        cortical_int_scale = 51.040149469200486 # ensures at the end, final spring length is 1.5
        cortical_int *= cortical_int_scale
        

        ABal_prime = T_FINAL * (params[0] * ((dist12 - (1 + cortical_int)) * u12 + 
                                        (dist14 - 1) * u14) + 
                                params[1] * cortical_flow_l * 
                                        (np.cross(-u14, -u34) - 
                                        np.cross(u14, u12) -
                                        np.cross(u12, k_hat)))
        ABar_prime = T_FINAL * (params[0] * ((dist12 - (1 + cortical_int)) * -u12 + 
                                        (dist23 - 1) * u23) + 
                                params[1] * cortical_flow_r * 
                                        (np.cross(-u23, u34) -
                                        np.cross(u23, -u12) -
                                        np.cross(-u12, k_hat)))

        ABpr_prime = T_FINAL * (params[0] * ((dist23 - 1) * -u23 + 
                                        (dist34 - (1 + cortical_int)) * u34) + 
                                params[1] * cortical_flow_r * 
                                        (np.cross(u23, -u12) -
                                        np.cross(-u23, u34) -
                                        np.cross(u34, k_hat)))

        ABpl_prime = T_FINAL * (params[0] * ((dist14 - 1) * -u14 +
                                        (dist34 - (1 + cortical_int)) * -u34) + 
                                params[1] * cortical_flow_l * 
                                        (np.cross(u14, u12) -
                                        np.cross(-u14, -u34) -
                                        np.cross(-u34, k_hat)))
        
        # applies spring force across cells in next iteration
        if dist13 <= 1:
                ABal_prime += T_FINAL * params[0] * (dist13 - 1) * u13
                ABpr_prime += T_FINAL * params[0] * (dist13 - 1) * -u13

        if dist24 <= 1:
                ABar_prime += T_FINAL * params[0] * (dist24 - 1) * u24
                ABpl_prime += T_FINAL * params[0] * (dist24 - 1) * -u24

        # cell wall forces 
        ABal_prime += T_FINAL * params[0] * _cell_wall_step(ABal)
        ABar_prime += T_FINAL * params[0] * _cell_wall_step(ABar)
        ABpr_prime += T_FINAL * params[0] * _cell_wall_step(ABpr)
        ABpl_prime += T_FINAL * params[0] * _cell_wall_step(ABpl)

        # print("time(s): ", time.time()-start)    

        return np.concatenate((ABal_prime, ABar_prime, ABpr_prime, ABpl_prime))
    return func


def _cell_wall_step(pos):
        """
        If the cell is outside the shell the shell pushes it away.

        Current force is Linear. Could also implemet using Van der Waals forces 
        https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/van-der-waals-force
        """

        min_point = min_point_ellpsoid(pos, E0, E1)
        distance = np.linalg.norm(min_point-pos)
        if distance < 0.5:  
                return np.array((0.5-distance)*(pos-min_point)/distance) # linear
                # return np.array(((0.5/distance)**12 - (0.5/distance)**6)*(pos-min_point)/distance) # van der waals
                # return np.array((1/distance-2)*(pos-min_point)/distance) # exponential force
        else: 
                return np.zeros(len(pos))