import numpy as np
from ..least_distance.ellipsoid import min_point_ellipsoid
from .model_config import T_FINAL, ASPECT_RATIO, R0
from numpy.polynomial import Polynomial

#This function applies the spring force associated with the egg-shell
def apply_shell(variables, ABal_prime, ABar_prime, ABpr_prime, ABpl_prime, P2_prime, EMS_primes):
    
    spring_const = variables["calculator"].params[0]
    ABal = variables["ABal_pos"]
    ABar = variables["ABar_pos"]
    ABpr = variables["ABpr_pos"]
    ABpl = variables["ABpl_pos"]
    p2 = variables["P2_pos"]
    EMS_pos = variables["EMS_positions"]
    r_ABa = variables["r_ABa"]
    r_ABp = variables["r_ABp"]
    r_EMS = variables["r_EMS"]
    r_P2 = variables["r_P2"]
    e1 = variables["E1"]

    #cell wall forces
    ABal_prime += T_FINAL * spring_const * _cell_wall_step(*min_vect(ABal,e1),r_ABa)
    ABar_prime += T_FINAL * spring_const  * _cell_wall_step(*min_vect(ABar,e1),r_ABa)
    ABpr_prime += T_FINAL * spring_const  * _cell_wall_step(*min_vect(ABpr,e1),r_ABp)
    ABpl_prime += T_FINAL * spring_const  * _cell_wall_step(*min_vect(ABpl,e1),r_ABp)
    P2_prime += T_FINAL * spring_const * _cell_wall_step(*min_vect(p2,e1),r_P2)
    for i in range(len(EMS_primes)):
        EMS_primes[i] += T_FINAL * spring_const * _cell_wall_step(*min_vect(EMS_pos[i],e1),r_EMS)

    return ABal_prime, ABar_prime, ABpr_prime, ABpl_prime, P2_prime, EMS_primes

#shell wall friction is ignored for P2 and EMS cells, assumed to be negligible
def apply_shell_friction(variables, ABal_prime, ABar_prime, ABpr_prime, ABpl_prime, P2_prime, EMS_primes):
     
    calc = variables["calculator"]
    fric_coef = calc.params[1]
    cort_flow_l = variables["cort_flow_l"]
    cort_flow_r = variables["cort_flow_r"]
    ABal = variables["ABal_pos"]
    ABar = variables["ABar_pos"]
    ABpr = variables["ABpr_pos"]
    ABpl = variables["ABpl_pos"]
    e1 = variables["E1"]

    uvec_mat = calc.uvec_mat
    cell_idx = calc.cell_idx

    #cell wall friction
    ABal_prime += fric_coef * cort_flow_l * np.cross(uvec_mat[cell_idx["ABar"], cell_idx["ABal"]], min_vect(ABal,e1)[0])
    ABar_prime += fric_coef * cort_flow_r * np.cross(-uvec_mat[cell_idx["ABar"], cell_idx["ABal"]], min_vect(ABar,e1)[0])
    ABpr_prime += fric_coef * cort_flow_r * np.cross(uvec_mat[cell_idx["ABpl"], cell_idx["ABpr"]], min_vect(ABpr,e1)[0])
    ABpl_prime += fric_coef * cort_flow_l * np.cross(-uvec_mat[cell_idx["ABpl"], cell_idx["ABpr"]], min_vect(ABpl,e1)[0])

    return ABal_prime, ABar_prime, ABpr_prime, ABpl_prime, P2_prime, EMS_primes

#This function applies the spring force associated with the P2 cell.
def apply_p2(variables, ABal_prime, ABar_prime, ABpr_prime, ABpl_prime, P2_prime, EMS_primes):
    
    cort_flow_l = variables["cort_flow_l"]
    cort_flow_r = variables["cort_flow_r"]
    fc = variables["calculator"]
    EMS_cells = variables["EMS_cells"]
    r_ABa = variables["r_ABa"]
    r_ABp = variables["r_ABp"]
    r_EMS = variables["r_EMS"]
    r_P2 = variables["r_P2"]

    ABal_prime += fc.get_spring_force("ABal",["p2"],[r_ABa + r_P2]) + fc.get_frictional_force("ABal",["p2"],[r_ABa + r_P2],cort_flow_l)
    ABar_prime += fc.get_spring_force("ABar",["p2"],[r_ABa + r_P2]) + fc.get_frictional_force("ABar",["p2"],[r_ABa + r_P2],cort_flow_r)
    ABpr_prime += fc.get_spring_force("ABpr",["p2"],[r_ABp + r_P2]) + fc.get_frictional_force("ABpr",["p2"],[r_ABp + r_P2],cort_flow_r)
    ABpl_prime += fc.get_spring_force("ABpl",["p2"],[r_ABp + r_P2]) + fc.get_frictional_force("ABpl",["p2"],[r_ABp + r_P2],cort_flow_l)
    for i in range(len(EMS_primes)):
        EMS_primes[i] += fc.get_spring_force(EMS_cells[i],["p2"],[r_EMS + r_P2])
    
    return ABal_prime, ABar_prime, ABpr_prime, ABpl_prime, P2_prime, EMS_primes

#This function applies the spring force associated with the EMS cell.
def apply_ems(variables, ABal_prime, ABar_prime, ABpr_prime, ABpl_prime, P2_prime, EMS_primes):
    
    cort_flow_l = variables["cort_flow_l"]
    cort_flow_r = variables["cort_flow_r"]
    fc = variables["calculator"]
    EMS_cells = variables["EMS_cells"]
    r_ABa = variables["r_ABa"]
    r_ABp = variables["r_ABp"]
    r_EMS = variables["r_EMS"]
    r_P2 = variables["r_P2"]

    ABal_prime += fc.get_spring_force("ABal",EMS_cells, np.repeat(r_EMS + r_ABa, len(EMS_cells))) + fc.get_frictional_force("ABal",EMS_cells, np.repeat(r_EMS + r_ABa, len(EMS_cells)),cort_flow_l)
    ABar_prime += fc.get_spring_force("ABar",EMS_cells, np.repeat(r_EMS + r_ABa, len(EMS_cells))) + fc.get_frictional_force("ABar",EMS_cells, np.repeat(r_EMS + r_ABa, len(EMS_cells)),cort_flow_r)
    ABpr_prime += fc.get_spring_force("ABpr",EMS_cells, np.repeat(r_EMS + r_ABp, len(EMS_cells))) + fc.get_frictional_force("ABpr",EMS_cells, np.repeat(r_EMS + r_ABp, len(EMS_cells)),cort_flow_r)
    ABpl_prime += fc.get_spring_force("ABpl",EMS_cells, np.repeat(r_EMS + r_ABp, len(EMS_cells))) + fc.get_frictional_force("ABpl",EMS_cells, np.repeat(r_EMS + r_ABp, len(EMS_cells)),cort_flow_l)
    P2_prime += fc.get_spring_force("p2", EMS_cells, np.repeat(r_EMS + r_P2, len(EMS_cells)))
    
    return ABal_prime, ABar_prime, ABpr_prime, ABpl_prime, P2_prime, EMS_primes

def min_vect(pos, E1):
    #returns the unit vector and length associated with the min_vector, E0 and E1 are assumed to be in the ratio 1.53:1
    min_point = min_point_ellipsoid(pos, ASPECT_RATIO * E1, E1)
    min_dist = np.linalg.norm(min_point - pos)
    min_u_vect = (pos - min_point)/min_dist
    return min_u_vect, min_dist
               
def _cell_wall_step(min_u_vect, min_dist, radius):
        """
        If the cell is outside the shell the shell pushes it away.

        Current force is Linear. Could also implemet using Van der Waals forces 
        https://en.wikipedia.org/wiki/Van_der_Waals_force
        """
        if min_dist < radius:  
                return np.array((radius - min_dist)*(min_u_vect))
        else: 
                return np.zeros(len(min_u_vect))

#calculates radius of dividing cells (ABal,ABpl,ABar,ABpr) based on distance between cells, volume to be conserved is calculated based on initial distance between cells which is d1
def calculate_d_cell_radius(d1,d):
    v0 =  8/3*np.pi*R0**3 - np.pi/12*(4*R0 + d1)*(2*R0 - d1)**2
    r1 = np.cbrt(3*v0/(8*np.pi))
    #Cubic equation to be solved is 4*pi/3*r^3 + pi*d*r^2 - pi*d^3/12 = v0, where LHS represents volume of ABa/ABp with overlap
    p = Polynomial([-np.pi*d**3/12 - v0, 0, np.pi*d, 4*np.pi/3])
    sol2 = np.array([np.real_if_close(r) for r in p.roots()]) #processing of roots to get rid of negligible imaginary parts introduced due to numerical solver
    r2 = sol2[np.isreal(sol2)][0].real 
    #Appending solution to sol_list if solution is valid i.e. distance between cells must be greater than 2*radius if they don't overlap and less than 2*radius if they do.
    if 2*r1 < d:
        radius = r1
    elif 2*r2 >= d:
        radius = r2
    else:
        print("Solutions returned by root solver don't satisfy specified conditions.")
    return radius
            

modifier_map = {
    "include_shell" : apply_shell,
    "include_p2" : apply_p2,
    "include_shell_friction" : apply_shell_friction,
    "include_ems" : apply_ems}