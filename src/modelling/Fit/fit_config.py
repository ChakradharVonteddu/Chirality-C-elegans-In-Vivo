from ..models import Combined_Model
import numpy as np

def get_init_noise(r0, d1, n, es, eta=0.0):
    ab_z = 0.25
    x_ant = r0
    x_post = -r0 
    
    #theta is the tilt in the horizontal plane and phi is the tilt in the vertical plane
    theta_a = 0  
    phi_a   = -eta
    
    theta_p = 0
    phi_p   = eta  
    
    V_ax = d1 * np.cos(phi_a) * np.sin(theta_a)
    V_ay = -d1 * np.cos(phi_a) * np.cos(theta_a)
    V_az = -d1 * np.sin(phi_a)
    
    ABar_x = x_ant + V_ax / 2
    ABar_y = 0     + V_ay / 2
    ABar_z = ab_z  + V_az / 2
    
    ABal_x = x_ant - V_ax / 2
    ABal_y = 0     - V_ay / 2
    ABal_z = ab_z  - V_az / 2

    V_px = d1 * np.cos(phi_p) * np.sin(theta_p)
    V_py = -d1 * np.cos(phi_p) * np.cos(theta_p)
    V_pz = -d1 * np.sin(phi_p)
    
    ABpr_x = x_post + V_px / 2
    ABpr_y = 0      + V_py / 2
    ABpr_z = ab_z   + V_pz / 2
    
    ABpl_x = x_post - V_px / 2
    ABpl_y = 0      - V_py / 2
    ABpl_z = ab_z   - V_pz / 2

    center_x = 0
    center_y = 0 

    v0 = 8/3*np.pi*r0**3 - np.pi/12*(4*r0 + d1)*(2*r0 - d1)**2 

    if n == 4:
        r_EMS = np.cbrt(1/np.pi * 3/11 * 36/47 * v0)
        offset = r_EMS * np.sqrt(2)/2
        ems_z = ab_z - np.sqrt(max(0, (r0+r_EMS)**2 - (r0 - offset)**2 - (d1/2)**2))

        if es:
            p2_x = -2.4*r0
            p2_y = 0
            p2_z = 0            
        else:
            p2_x = center_x
            p2_y = center_y + offset + r_EMS + r0
            p2_z = ems_z - 0.15

        init = (
            ABal_x, ABal_y, ABal_z,          
            ABar_x, ABar_y, ABar_z,          
            ABpr_x, ABpr_y, ABpr_z,          
            ABpl_x, ABpl_y, ABpl_z,          
            p2_x, p2_y, p2_z,                     
            center_x + offset, center_y, ems_z,  
            center_x, center_y + offset, ems_z,  
            center_x - offset, center_y, ems_z,  
            center_x, center_y - offset, ems_z   
        )
    elif n == 1:
        r_EMS = np.cbrt(36/47*v0*3/(4*np.pi))
        ems_z = ab_z - np.sqrt(max(0, (r0+r_EMS)**2 - (r0)**2 - (d1/2)**2))

        if es:
            p2_x = -2.4*r0
            p2_y = 0
            p2_z = 0            
        else:
            p2_x = center_x
            p2_y = center_y + r_EMS + r0
            p2_z = ems_z - 0.15
            
        init = (
            ABal_x, ABal_y, ABal_z,          
            ABar_x, ABar_y, ABar_z,          
            ABpr_x, ABpr_y, ABpr_z,          
            ABpl_x, ABpl_y, ABpl_z,          
            p2_x, p2_y, p2_z,                     
            center_x, center_y, ems_z      
        )
    else:
        print("The number of EMS spheres should either be 1 or 4")
        init = ()
        
    return init

#def get_init(r0, d1):
    #init = (0.65, d1/2, 0.25, 0.65, -d1/2, 0.25, 0.65-2*r0, d1/2, 0.25, 0.65-2*r0, -d1/2, 0.25, -2*r0, 0, 0, 0, d1/2 + 0.2, 0.25-r0-0.35, 0, d1/2 - 0.2, 0.25-r0-0.35, np.sqrt(3)/5 ,d1/2, 0.25-r0-0.35, -np.sqrt(3)/5, d1/2, 0.25-r0-0.35)
    #return init

#Overlap
#INIT = (0.65, 0.4, 0.25, 0.65, -0.4, 0.25, -0.35, -0.4, 0.25, -0.35, 0.4, 0.25, -0.9, 0, 0, 0.15, 0.2, -0.6, 0.15, -0.2, -0.6, 0.5, 0, -0.6, -0.2, 0, -0.6)

#4 EMS spheres
#INIT = (0.65, 0.5, 0.25, 0.65, -0.5, 0.25, -0.35, -0.5, 0.25, -0.35, 0.5, 0.25, -0.9, 0, 0, 0.15, 0.2, -0.6, 0.15, -0.2, -0.6, 0.5, 0, -0.6, -0.2, 0, -0.6)

#Slight tilt in x direction
#INIT = (0.6, 0.5, 0.25, 0.65, -0.5, 0.25, -0.35, -0.5, 0.25, -0.4, 0.5, 0.25, -0.9, 0, 0, 0.15, 0.2, -0.6, 0.15, -0.2, -0.6, 0.5, 0, -0.6, -0.2, 0, -0.6)

#Shorter spindle rest length
#INIT = (0.65, 0.45, 0.25, 0.65, -0.45, 0.25, -0.35, -0.45, 0.25, -0.35, 0.45, 0.25, -0.9, 0, 0, 0.15, 0.2, -0.6, 0.15, -0.2, -0.6, 0.5, 0, -0.6, -0.2, 0, -0.6)

#Shorter spindle rest length - 0.9 with slight tilt in x-direction
#INIT = (0.6, 0.45, 0.25, 0.65, -0.45, 0.25, -0.35, -0.45, 0.25, -0.4, 0.45, 0.25, -0.9, 0, 0, 0.15, 0.2, -0.6, 0.15, -0.2, -0.6, 0.5, 0, -0.6, -0.2, 0, -0.6)

#1 EMS Sphere
#INIT = (0.65, 0.5, 0.25, 0.65, -0.5, 0.25, -0.35, -0.5, 0.25, -0.35, 0.5, 0.25, -0.9, 0, 0, 0.15, 0, -0.56)

GET_VELOCITY = Combined_Model.get_velocity


