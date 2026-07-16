import numpy as np
from .model_config import T_FINAL, P2

def get_velocity(params):

    def func(t, y):
        """
        A division by 0 occurs when two cells overlap

        ABal = 1, ABar = 2, ABpr = 3, ABpl = 4
        """
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
        dist3p2 = np.linalg.norm(P2 - ABpr)
        dist4p2 = np.linalg.norm(P2 - ABpl)

        u12 = (ABar-ABal) / dist12 # 2-1 
        u13 = (ABpr-ABal) / dist13 # 3-1 
        u14 = (ABpl-ABal) / dist14 # 4-1
        u23 = (ABpr-ABar) / dist23 # 3-2
        u24 = (ABpl-ABar) / dist24 # 4-2
        u34 = (ABpl-ABpr) / dist34 # 4-3
        u3p2 = (P2 - ABpr) / dist3p2 #p2-3
        u4p2 = (P2 - ABpl) / dist4p2 #p2-4
        k_hat = np.array([0,0,1])

        cortical_flow_r = np.multiply(0.000345*t*T_FINAL, np.e**(-0.012732*t*T_FINAL))
        cortical_flow_l = np.multiply(0.00071*t*T_FINAL, np.e**(-0.0166*t*T_FINAL))
        ABal_prime = T_FINAL * (params[0] * ((dist12 - 1) * u12 + 
                                        (dist14 - 1) * u14 - 
                                        ABal[2] * k_hat) + 
                                params[1] * cortical_flow_l * 
                                        (np.cross(-u14, -u34) - 
                                        np.cross(u14, u12) -
                                        np.cross(u12, k_hat)))
        ABar_prime = T_FINAL * (params[0] * ((dist12 - 1) * -u12 + 
                                        (dist23 - 1) * u23 - 
                                        ABar[2] * k_hat) + 
                                params[1] * cortical_flow_r * 
                                        (np.cross(-u23, u34) -
                                        np.cross(u23, -u12) -
                                        np.cross(-u12, k_hat)))

        ABpr_prime = T_FINAL * (params[0] * ((dist23 - 1) * -u23 + 
                                        (dist34 - 1) * u34 - 
                                        ABpr[2] * k_hat +
                                        (dist3p2 - 1) * u3p2
                                        ) + 
                                params[1] * cortical_flow_r * 
                                        (np.cross(u23, -u12) -
                                        np.cross(-u23, u34) -
                                        np.cross(u34, k_hat) - 
                                        np.cross(u3p2, u34))
                                )

        ABpl_prime = T_FINAL * (params[0] * ((dist14 - 1) * -u14 +
                                        (dist34 - 1) * -u34 - 
                                        ABpl[2] * k_hat + 
                                        (dist4p2 - 1) * u4p2) + 
                                params[1] * cortical_flow_l * 
                                        (np.cross(u14, u12) -
                                        np.cross(-u14, -u34) -
                                        np.cross(-u34, k_hat) - 
                                        np.cross(u4p2, -u34))
                                )
        
        # applies spring force across cells in next iteration
        if dist13 <= 1:
                ABal_prime += T_FINAL * params[0] * (dist13 - 1) * u13
                ABpr_prime += T_FINAL * params[0] * (dist13 - 1) * -u13

        if dist24 <= 1:
                ABar_prime += T_FINAL * params[0] * (dist24 - 1) * u24
                ABpl_prime += T_FINAL * params[0] * (dist24 - 1) * -u24

        return np.concatenate((ABal_prime, 
                               ABar_prime, 
                               ABpr_prime, 
                               ABpl_prime))
    return func