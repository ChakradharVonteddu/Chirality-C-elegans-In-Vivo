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


        cortical_flow_r = 0.000345*t*T_FINAL*np.e**(-0.012732*t*T_FINAL)
        cortical_flow_l = 0.00071*t*T_FINAL*np.e**(-0.0166*t*T_FINAL)

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
                                        np.cross(u14, u12)))
        ABar_prime = T_FINAL * (params[0] * ((dist12 - (1 + cortical_int)) * -u12 + 
                                        (dist23 - 1) * u23) + 
                                params[1] * cortical_flow_r * 
                                        (np.cross(-u23, u34) -
                                        np.cross(u23, -u12)))

        ABpr_prime = T_FINAL * (params[0] * ((dist23 - 1) * -u23 + 
                                        (dist34 - (1 + cortical_int)) * u34) + 
                                params[1] * cortical_flow_r * 
                                        (np.cross(u23, -u12) -
                                        np.cross(-u23, u34)))

        ABpl_prime = T_FINAL * (params[0] * ((dist14 - 1) * -u14 +
                                        (dist34 - (1 + cortical_int)) * -u34) + 
                                params[1] * cortical_flow_l * 
                                        (np.cross(u14, u12) -
                                        np.cross(-u14, -u34)))
        
        # applies spring force across cells in next iteration
        if dist13 <= 1:
                ABal_prime += T_FINAL * params[0] * (dist13 - 1) * u13
                ABpr_prime += T_FINAL * params[0] * (dist13 - 1) * -u13

        if dist24 <= 1:
                ABar_prime += T_FINAL * params[0] * (dist24 - 1) * u24
                ABpl_prime += T_FINAL * params[0] * (dist24 - 1) * -u24

        min_vector_ABal = min_point_ellpsoid(ABal, E0, E1) - ABal
        min_vector_ABar = min_point_ellpsoid(ABar, E0, E1) - ABar
        min_vector_ABpr = min_point_ellpsoid(ABpr, E0, E1) - ABpr
        min_vector_ABpl = min_point_ellpsoid(ABpl, E0, E1) - ABpl

        min_dist_ABal = np.linalg.norm(min_vector_ABal)
        min_dist_ABar = np.linalg.norm(min_vector_ABar)
        min_dist_ABpr = np.linalg.norm(min_vector_ABpr)
        min_dist_ABpl = np.linalg.norm(min_vector_ABpl)

        min_u_ABal = min_vector_ABal/min_dist_ABal
        min_u_ABar = min_vector_ABar/min_dist_ABar
        min_u_ABpr = min_vector_ABpr/min_dist_ABpr
        min_u_ABpl = min_vector_ABpl/min_dist_ABpl

        # cell wall linear forces 
        ABal_prime += T_FINAL * params[0] * _cell_wall_step_simplified(-min_vector_ABal, min_dist_ABal)
        ABar_prime += T_FINAL * params[0] * _cell_wall_step_simplified(-min_vector_ABar, min_dist_ABar)
        ABpr_prime += T_FINAL * params[0] * _cell_wall_step_simplified(-min_vector_ABpr, min_dist_ABpr)
        ABpl_prime += T_FINAL * params[0] * _cell_wall_step_simplified(-min_vector_ABpl, min_dist_ABpl)

        # cell wall friction
        ABal_prime += T_FINAL * params[1] * cortical_flow_l * np.cross(u12, min_u_ABal)
        ABar_prime += T_FINAL * params[1] * cortical_flow_r * np.cross(-u12, min_u_ABar)
        ABpr_prime += T_FINAL * params[1] * cortical_flow_r * np.cross(u34, min_u_ABpr)
        ABpl_prime += T_FINAL * params[1] * cortical_flow_l * np.cross(-u34, min_u_ABpl)

        # print("time(s): ", time.time()-start)    

        return np.concatenate((ABal_prime, ABar_prime, ABpr_prime, ABpl_prime))
    return func


def _cell_wall_step_simplified(min_vector, min_dist):
        """
        If the cell is outside the shell the shell pushes it away.

        Current force is Linear. Could also implemet using Van der Waals forces 
        https://en.wikipedia.org/wiki/Van_der_Waals_force
        """
        if min_dist < 0.5:  
                return np.array((0.5-min_dist)*(min_vector)/min_dist)
        else: 
                return np.zeros(len(min_vector))