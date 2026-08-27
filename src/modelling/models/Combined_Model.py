import numpy as np
from .model_config import T_FINAL_ES, T_FINAL_NO_ES, cell_names, NUM_EMS, R0, param_loc, LAM, ALPHA
from .All_Modifiers import modifier_map, calculate_d_cell_radius, _cell_wall_step, min_vect
import pandas as pd
from .Force_Calculator import ForceCalculator

def get_velocity(params, modifiers, return_data = False):
    
    def func(t, y):
        """
        A division by 0 occurs when two cells overlap

        ABal = 1, ABar = 2, ABpr = 3, ABpl = 4
        """
        #params[0] - spring constant/gamma (1/s), params[1] - frictional constant/gamma (unitless), params[2] - E1 (in R0), params[3] - d1 (in R0), params[4] - d2_es (in R0), params[5] - adhesion constant (unitless), params[6] - d2_no_es (in R0), params[7] - t0 (in units of 195s) 

        cell_pos = np.array([y[3*k:3*(k+1)] for k in range(len(cell_names))])
        d_cells = cell_names[0:4] #dividing cells
        EMS_cells = cell_names[5:len(cell_names)]
        
        #Array of differences in position vectors of spheres
        diff_array = cell_pos[:,np.newaxis] - cell_pos[np.newaxis,:]
        #Obtain the norm for each difference of position vectors
        distances = np.linalg.norm(diff_array, axis=2)
        with np.errstate(divide = "ignore", invalid = "ignore"):
            uvec_array = diff_array / distances[:, :, np.newaxis] #Obtain unit vectors between spheres
            uvec_array = np.nan_to_num(uvec_array) #Replace NaN values with 0
        
        # Create dictionary map for numpy array indexing
        cell_idx = {name: i for i, name in enumerate(cell_names)}

        d_list = [distances[cell_idx["ABal"], cell_idx["ABar"]], distances[cell_idx["ABpl"], cell_idx["ABpr"]]]

        sol_list = []

        v0 =  8/3*np.pi*R0**3 - np.pi/12*(4*R0 + params[param_loc["d1"]])*(2*R0 - params[param_loc["d1"]])**2 #Initial volume of ABa cell
        #Calculation of EMS radius - Volume of 4 EMS spheres(accounting for 2-sphere overlaps) is given as 4*pi*d*r^2 - pi*d^3/3, where d is the initial distance between centers of EMS spheres. 
        #We set this equal to 36/47*initial volume of ABa and solve for r_EMS
        if NUM_EMS == 4:
            r_EMS = np.cbrt(1/np.pi * 3/11 * 36/47 * v0)
            EMS2EMS_REST_L = r_EMS
        elif NUM_EMS == 1:
            r_EMS = np.cbrt(36/47*v0*3/(4*np.pi))
        else:
            print("The number of EMS spheres should either be 1 or 4")
       
        for d in d_list:
            r = calculate_d_cell_radius(params[param_loc["d1"]],d)
            sol_list.append(r)
        
        r_ABa = sol_list[0]
        r_ABp = sol_list[1]
            
        if modifiers["include_shell"]:
            cortical_flow_r = T_FINAL_ES*LAM*ALPHA*t*T_FINAL_ES*np.e**(-LAM*(t*T_FINAL_ES))
            cortical_flow_l = cortical_flow_r
            spindle_length = params[param_loc["d1"]] + t*(params[param_loc["d2_es"]] - params[param_loc["d1"]])
        elif not modifiers["include_shell"] and t < params[param_loc["t0"]]:
            cortical_flow_r = 0
            cortical_flow_l = 0
            spindle_length = params[param_loc["d1"]]
        else:
            cortical_flow_r = T_FINAL_ES*LAM*ALPHA*(t - params[param_loc["t0"]])*T_FINAL_ES*np.e**(-LAM*(t - params[param_loc["t0"]])*T_FINAL_ES)
            cortical_flow_l = cortical_flow_r
            spindle_length = params[param_loc["d1"]] + (t - params[param_loc["t0"]])/(T_FINAL_NO_ES/T_FINAL_ES - params[param_loc["t0"]])*(params[param_loc["d2_no_es"]] - params[param_loc["d1"]]) 
            #cortical_flow_r = T_FINAL_ES*LAM*ALPHA*t*T_FINAL_ES*np.e**(-LAM*(t*T_FINAL_ES))
            #cortical_flow_l = cortical_flow_r
            #spindle_length = params[param_loc["d1"]] + t/(T_FINAL_NO_ES/T_FINAL_ES)*(params[param_loc["d2_no_es"]] - params[param_loc["d1"]])
        
        #Rotational axis of non-rotating cells is set to the zero vector for convenience when calculating frictional force
        rotation_axes = {
            "ABal" : uvec_array[cell_idx["ABar"], cell_idx["ABal"]], 
            "ABar" : uvec_array[cell_idx["ABal"], cell_idx["ABar"]],
            "ABpr" : uvec_array[cell_idx["ABpl"], cell_idx["ABpr"]], 
            "ABpl" : uvec_array[cell_idx["ABpr"], cell_idx["ABpl"]]
        } | dict.fromkeys(["p2"] + EMS_cells, np.zeros(3))

        fc = ForceCalculator(params, distances, uvec_array, rotation_axes, cell_idx)
        
        #Rest_length at time t is set equal to d1 + t*(d2 - d1) between ABa/ABp spheres, which allows for d(t) to be linear in t. t here is normalized to be between 0 and 1.
        ABal_prime = fc.get_spring_force("ABal",r_ABa,["ABar","ABpr", "ABpl"],[r_ABa,r_ABp,r_ABp],[spindle_length,r_ABp+r_ABa,r_ABp+r_ABa]) + fc.get_frictional_force("ABal",["ABpr","ABpl"],[r_ABp+r_ABa,r_ABp+r_ABa],cortical_flow_l)
        
        ABar_prime = fc.get_spring_force("ABar",r_ABa,["ABal", "ABpr", "ABpl"],[r_ABa,r_ABp,r_ABp],[spindle_length,r_ABp+r_ABa,r_ABp+r_ABa]) + fc.get_frictional_force("ABar",["ABpr","ABpl"],[r_ABp+r_ABa,r_ABp+r_ABa],cortical_flow_r)
        
        ABpr_prime = fc.get_spring_force("ABpr",r_ABp,["ABal","ABar","ABpl"],[r_ABa,r_ABa,r_ABp],[r_ABp+r_ABa,r_ABp+r_ABa,spindle_length]) + fc.get_frictional_force("ABpr",["ABal","ABar"],[r_ABp+r_ABa,r_ABp+r_ABa],cortical_flow_r)
        
        ABpl_prime = fc.get_spring_force("ABpl",r_ABp,["ABal","ABar","ABpr"],[r_ABa,r_ABa,r_ABp],[r_ABp+r_ABa,r_ABp+r_ABa,spindle_length]) + fc.get_frictional_force("ABpl",["ABar","ABal"],[r_ABp+r_ABa,r_ABp+r_ABa],cortical_flow_l)

        P2_prime = (fc.get_spring_force("p2",R0,d_cells,[r_ABa,r_ABa,r_ABp,r_ABp],[R0 + r_ABa,R0+ r_ABa, R0 + r_ABp, R0 + r_ABp]) - 
                    fc.get_frictional_force("ABal",["p2"],[R0 + r_ABa],cortical_flow_l) -
                    fc.get_frictional_force("ABar",["p2"],[R0 + r_ABa],cortical_flow_r) -
                    fc.get_frictional_force("ABpr",["p2"],[R0 + r_ABp],cortical_flow_r) -
                    fc.get_frictional_force("ABpl",["p2"],[R0 + r_ABp],cortical_flow_l))

        EMS_primes = []
        for ems_cell in EMS_cells:
            EMS_prime = (fc.get_spring_force(ems_cell, r_EMS, d_cells + [c for c in EMS_cells if c != ems_cell], [r_ABa,r_ABa,r_ABp,r_ABp] + [r_EMS]*(len(EMS_cells)-1), np.concatenate([[r_EMS + r_ABa,r_EMS + r_ABa, r_EMS + r_ABp, r_EMS + r_ABp],np.repeat(EMS2EMS_REST_L,len(EMS_cells) - 1)])) -
                        fc.get_frictional_force("ABal",[ems_cell],[r_EMS + r_ABa],cortical_flow_l) -
                        fc.get_frictional_force("ABar",[ems_cell],[r_EMS + r_ABa],cortical_flow_r) -
                        fc.get_frictional_force("ABpr",[ems_cell],[r_EMS + r_ABp],cortical_flow_r) -
                        fc.get_frictional_force("ABpl",[ems_cell],[r_EMS + r_ABp],cortical_flow_l))
            EMS_primes.append(EMS_prime)
            
        variables = {
            "ABal_pos": cell_pos[cell_names.index("ABal")],
            "ABar_pos" : cell_pos[cell_names.index("ABar")],
            "ABpr_pos" : cell_pos[cell_names.index("ABpr")],
            "ABpl_pos" : cell_pos[cell_names.index("ABpl")],
            "P2_pos" : cell_pos[cell_names.index("p2")],
            "r_ABa" : r_ABa,
            "r_ABp" : r_ABp,
            "r_EMS" : r_EMS,
            "r_P2" : R0,
            "E1" : params[param_loc["E1"]],
            "EMS_positions" : cell_pos[cell_names.index("ems_a"):],
            "EMS_cells" : EMS_cells,
            "d_cells" : d_cells,
            "cort_flow_r" : cortical_flow_r,
            "cort_flow_l" : cortical_flow_l,
            "calculator" : fc}
        
        for modifier, active in modifiers.items():
              if active:
                    ABal_prime, ABar_prime, ABpr_prime, ABpl_prime, P2_prime, EMS_primes = modifier_map[modifier](variables, ABal_prime, ABar_prime, ABpr_prime, ABpl_prime, P2_prime, EMS_primes)
              else:
                    continue        
       
        v_tuple = (ABal_prime, ABar_prime, ABpr_prime, ABpl_prime, P2_prime) + tuple(EMS_primes)
        
        #Solve frictional force matrix system in the 6 cell model
        if modifiers["include_p2"] and modifiers["include_ems"]:
            forces_dict = {"ABal" : ABal_prime, "ABar" : ABar_prime, "ABpr" : ABpr_prime, "ABpl" : ABpl_prime, "p2" : P2_prime} | dict(zip(EMS_cells, EMS_primes))
            radii_dict = dict(zip(d_cells + ["p2"],[r_ABa,r_ABa,r_ABp,r_ABp,R0]))| dict.fromkeys(EMS_cells, r_EMS)
            neighbors_dict = {"ABal": ["ABpr","ABpl","p2"] + EMS_cells, "ABar": ["ABpr","ABpl","p2"] + EMS_cells, "ABpr": ["ABal","ABar","p2"] + EMS_cells, "ABpl": ["ABal","ABar","p2"] + EMS_cells, "p2" : d_cells + EMS_cells} | dict.fromkeys(EMS_cells, d_cells + ["p2"])
            v_tuple_mod = fc.solve_frictional_force_system(cell_names, forces_dict, radii_dict, neighbors_dict)

        if not return_data:
            return np.concatenate(v_tuple_mod)
       
        #d_cells_dict = {"ABal": ["ABpr","ABpl"], "ABar": ["ABpr","ABpl"], "ABpr": ["ABal","ABar"], "ABpl": ["ABal","ABar"]}
        
        if modifiers["include_shell"]:
            spring_force_shell = T_FINAL_ES * params[param_loc["spring_constant"]] * _cell_wall_step(*min_vect(cell_pos[cell_idx["ABal"]],params[param_loc["E1"]]),r_ABa)
        else:
            spring_force_shell = np.zeros(3)
            
        force_data = {"Time" : t, 
                    "Spring_force_dividing": fc.get_spring_force("ABal",r_ABa,["ABar","ABpr", "ABpl"],[r_ABa,r_ABp,r_ABp],[spindle_length ,r_ABp+r_ABa,r_ABp+r_ABa]),
                    "Spring_force_shell" : spring_force_shell,
                    "Rotational_frictional_force_dividing" : fc.get_frictional_force("ABal",["ABpr","ABpl"],[r_ABp+r_ABa,r_ABp+r_ABa],cortical_flow_l),
                    "Spring_force_EMS" : fc.get_spring_force("ABal",r_ABa, EMS_cells, [r_EMS]*len(EMS_cells), np.repeat(r_EMS + r_ABa, len(EMS_cells))),
                    "Frictional_force_EMSa" : fc.params[param_loc["frictional_constant"]]*(fc.get_frictional_force("ABal",["ems_a"],[r_EMS + r_ABa],cortical_flow_l)/fc.params[param_loc["frictional_constant"]] + np.heaviside(r_ABa + r_EMS - distances[cell_idx["ABal"],cell_idx["ems_a"]],0) * np.matmul(np.identity(3) - np.outer(uvec_array[cell_idx["ems_a"],cell_idx["ABal"]],uvec_array[cell_idx["ems_a"],cell_idx["ABal"]]),v_tuple_mod[cell_idx["ems_a"]] - v_tuple_mod[cell_idx["ABal"]])),
                    "Frictional_force_EMSb" : fc.params[param_loc["frictional_constant"]]*(fc.get_frictional_force("ABal",["ems_b"],[r_EMS + r_ABa],cortical_flow_l)/fc.params[param_loc["frictional_constant"]] + np.heaviside(r_ABa + r_EMS - distances[cell_idx["ABal"],cell_idx["ems_b"]],0) * np.matmul(np.identity(3) - np.outer(uvec_array[cell_idx["ems_b"],cell_idx["ABal"]],uvec_array[cell_idx["ems_b"],cell_idx["ABal"]]),v_tuple_mod[cell_idx["ems_b"]] - v_tuple_mod[cell_idx["ABal"]])),
                    "ABal_vel": np.linalg.norm(fc.get_frictional_force("ABal",["ems_a"],[r_EMS + r_ABa],cortical_flow_l)/fc.params[param_loc["frictional_constant"]] + v_tuple_mod[cell_idx["ABal"]]),
                    "EMSa_vel": np.linalg.norm(v_tuple_mod[cell_idx["ems_a"]])}
        
        #for c1, c2 in itertools.product(cell_names,cell_names):
            #temp_data[c1 + "_" + c2 + "_rel_vel"] = np.linalg.norm(v_tuple[cell_names.index(c1)] - v_tuple[cell_names.index(c2)])
        
        #vel_data = {}
        #for cell in d_cells:
            #for neighbor in d_cells_dict[cell]:
                #u_cell = fc.uvec_mat[cell_idx[cell],cell_idx[neighbor]]
                #proj_mat = np.identity(3) - np.outer(u_cell, u_cell)
                #vel_data[cell + "_" + neighbor + "_rot_vel"] =  np.linalg.norm(np.cross(-u_cell, fc.rotation_axes[cell]))
                #vel_data[cell + "_" + neighbor + "_rel_vel"] = np.linalg.norm(np.matmul(proj_mat, v_tuple[cell_names.index(cell)]) - np.matmul(proj_mat, v_tuple[cell_names.index(neighbor)]))
                 
        radii_data = {"time" : t, "ABal" : r_ABa, "ABar" : r_ABa, "ABpr" : r_ABp, "ABpl" : r_ABp, "p2" : R0, "EMS" : r_EMS}

        return force_data, radii_data
        #return vel_data, radii_data
    return func
        