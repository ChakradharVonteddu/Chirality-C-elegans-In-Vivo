import numpy as np
from .model_config import T_FINAL_ES, param_loc

class ForceCalculator:
    def __init__(self, params, distance_data, unitvect_data, rotation_axis_data, cell_idx):
        self.params = params
        self.dist_mat = distance_data
        self.uvec_mat = unitvect_data
        self.rotation_axes = rotation_axis_data
        self.cell_idx = cell_idx

    #calculates spring force applied on target by neighboring cells
    def get_spring_force(self, target, target_radius, neighbors, neighbors_radii, rest_lengths):
        spring_f = np.zeros(3)
        t_idx = self.cell_idx[target]
        for i in range(len(neighbors)):
            n_idx = self.cell_idx[neighbors[i]]
            distance = self.dist_mat[t_idx, n_idx] #distance between target and neighbor
            u_target = self.uvec_mat[t_idx, n_idx] #unit vector from neighbor to target
            spring_f += T_FINAL_ES * self.params[param_loc["spring_constant"]] * (rest_lengths[i]- distance) * u_target * np.heaviside(rest_lengths[i] + self.params[param_loc["adhesion_constant"]]*(target_radius + neighbors_radii[i])/2 - distance,0)
        return spring_f
    
    #calculates frictional force applied on target by neighboring cells, cortical flow is kept the same for both target and neighbor for now but changes might be necessary if it is different
    def get_frictional_force(self, target, neighbors, contact_lengths, cortical_flow):
        frictional_f = np.zeros(3)
        t_idx = self.cell_idx[target]
        for i in range(len(neighbors)):
            n_idx = self.cell_idx[neighbors[i]]
            distance = self.dist_mat[t_idx, n_idx] #distance between target and neighbor
            u_target = self.uvec_mat[t_idx, n_idx] #unit vector from neighbor to target
            frictional_f += self.params[param_loc["frictional_constant"]] * cortical_flow * np.heaviside(contact_lengths[i] - distance,0) * (np.cross(u_target, self.rotation_axes[neighbors[i]]) - np.cross(-u_target, self.rotation_axes[target]))
        return frictional_f
    
    def solve_frictional_force_system(self, targets, y, radii, neighbors):
        hblock_list = []
        #For each target in the list, we construct a horizontal block that represents the vector equation corresponding to that target's frictional force.
        #Then, these horizontal blocks are stacked to create a matrix after which the matrix equation Ax = b is solved.
        for target in targets:
           t_idx = self.cell_idx[target]
           neighbors_list = neighbors[target]
           mod_mat = dict.fromkeys(targets, np.zeros((3,3)))
           for neighbor in neighbors_list:
                n_idx = self.cell_idx[neighbor]
                contact_length = radii[neighbor] + radii[target]
                #The projection matrix onto the span of the vector pointing from the target to the neighbor is given by the outer product of the unit vector. 
                #To get the projection matrix onto the tangent plane which is the orthogonal complement, we subtract the outer product from the identity matrix.
                u_target = self.uvec_mat[t_idx, n_idx] #unit vector from neighbor to target
                proj_mat = np.identity(3) - np.outer(u_target, u_target)
                #Multiply projection matrix with corresponding heaviside functions and frictional coefficient
                mod_mat[neighbor] = -self.params[param_loc["frictional_constant"]] * np.heaviside(contact_length - self.dist_mat[n_idx, t_idx],0) * proj_mat
           mod_mat[target] = np.identity(3) - np.sum([mod_mat[n] for n in neighbors_list], axis = 0)
           hblock = np.hstack([mod_mat[t] for t in targets])
           #The matrices are horizontally stacked in the same order for each loop i.e. the order of the elements in the targets list
           hblock_list.append(hblock)

        A = np.vstack(hblock_list)
        x = np.linalg.solve(A, np.concatenate([y[t] for t in targets]))
        
        f_vec = [x[3*i:3*(i+1)] for i in range(len(targets))]

        return tuple(f_vec)