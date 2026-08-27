import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution, LinearConstraint, NonlinearConstraint
from scipy.stats import t
from ..Simulator import Simulator
from .fit_config import GET_VELOCITY, get_init_noise
from ..models.model_config import ES_TIMES, NO_ES_TIMES, T_FINAL_C, NUM_EMS, R0, R0_ALT, ETA, ALPHA, LAM, modifiers_w_es, modifiers_wo_es, param_loc 
from ..models.All_Modifiers import calculate_d_cell_radius
import warnings

def objective_function(params, es_data, es_sigma, no_es_data, no_es_sigma):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
        es_pred, no_es_pred = fit_model_curve(*params)
        
        #Standardized residuals denoting how far the prediction is from the data in terms of standard deviations
        es_residuals = (es_data - es_pred)/es_sigma
        no_es_residuals = (no_es_data - no_es_pred)/no_es_sigma
                
        sse = np.mean(es_residuals**2) + np.mean(no_es_residuals**2)

        if np.isnan(sse):
            return np.inf
            
        return sse
            
    except Exception as e:
        return np.inf

def fit_model_whole(angle_data_es, angle_data_no_es, cort_data):
    """
    """
    ABa_dorsal = angle_data_es['ABa_dorsal_avg'].to_numpy()
    ABp_dorsal = angle_data_es['ABp_dorsal_avg'].to_numpy()
    ABa_ant = angle_data_es['ABa_ant_avg'].to_numpy()
    ABp_ant = angle_data_es['ABp_ant_avg'].to_numpy()
    ABa_dorsal_no_es = angle_data_no_es['ABa_dorsal_avg'].to_numpy()
    ABp_dorsal_no_es = angle_data_no_es['ABp_dorsal_avg'].to_numpy()
    #cvel_data = 1/R0_ALT * cort_data['cort_avg'].to_numpy()

    ABa_dorsal_stdmean = angle_data_es['ABa_dorsal_stdeofmean'].to_numpy()
    ABp_dorsal_stdmean = angle_data_es['ABp_dorsal_stdeofmean'].to_numpy()
    ABa_dorsal_stdmean_no_es = angle_data_no_es['ABa_dorsal_stdeofmean'].to_numpy()
    ABp_dorsal_stdmean_no_es = angle_data_no_es['ABp_dorsal_stdeofmean'].to_numpy()
    ABa_ant_stdmean = angle_data_es['ABa_ant_stdeofmean'].to_numpy()
    ABp_ant_stdmean = angle_data_es['ABp_ant_stdeofmean'].to_numpy()
    #cvel_error = 1/R0_ALT * cort_data['cort_stdeofmean'].to_numpy()


    es_data = np.concatenate((ABa_dorsal, ABp_dorsal, ABa_ant, ABp_ant))
    es_error = np.concatenate((ABa_dorsal_stdmean, 
                                      ABp_dorsal_stdmean, 
                                      ABa_ant_stdmean, 
                                      ABp_ant_stdmean))
    
    no_es_data = np.concatenate((ABa_dorsal_no_es, ABp_dorsal_no_es))
    no_es_error = np.concatenate((ABa_dorsal_stdmean_no_es, ABp_dorsal_stdmean_no_es))
    
    es_sigma = np.array([10**(-6) if error == 0 else error for error in es_error])
    #cvel_sigma = np.array([10**(-6) if error == 0 else error for error in cvel_error])
    no_es_sigma = np.array([10**(-6) if error == 0 else error for error in no_es_error])

    #bounds for parameters in the order : spring constant/gamma, frictional constant/gamma, E1, d1, d2_es, adhesion constant, d2_no_es
    bounds = [
        (0, 40),   
        (0, 40), 
        (1, 3), 
        (0, 2),  
        (1, 3),
        (0, 0.05),
        (1, 3)]

    #capping the extended spring length change to be less than 30% of the initial spring length for the shelled-case
    def extended_spring_length_change_es(params):
        final_radius = calculate_d_cell_radius(params[param_loc["d1"]],params[param_loc["d2_es"]])
        return (2*final_radius + params[param_loc["d2_es"]] - 2*R0 - params[param_loc["d1"]])/(2*R0 + params[param_loc["d1"]])

    #capping the extended spring length change to be less than 30% of the initial spring length for the no-shelled-case
    def extended_spring_length_change_no_es(params):
        final_radius = calculate_d_cell_radius(params[param_loc["d1"]],params[param_loc["d2_no_es"]])
        return (2*final_radius + params[param_loc["d2_no_es"]] - 2*R0 - params[param_loc["d1"]])/(2*R0 + params[param_loc["d1"]])


    #linear constraint to allow for d2_es to be greater than d1
    lc1 = LinearConstraint([0,0,0,-1,1,0,0], 0, np.inf)
    #linear constraint to allow for d2_no_es to be greater than d1
    lc2 = LinearConstraint([0,0,0,-1,0,0,1], 0, np.inf)
    #Non-linear constraint to ensure extended spring length increases and increase is less than 30%
    nlc1 = NonlinearConstraint(extended_spring_length_change_es,0,0.3)
    nlc2 = NonlinearConstraint(extended_spring_length_change_no_es,0,0.3)

    #Displays optimization progress after each iteration
    def progress(xk, convergence):
        error = objective_function(xk, es_data, es_sigma, no_es_data, no_es_sigma)
        print(f"Error: {error:>8.3f} | Convergence: {convergence * 100:>5.2f}% | Best params vector: {xk}")
        #Prints out error associated with fit, convergence of error values and best params vector

    print("Starting global optimization (Differential Evolution). This may take a few hours...")
    global_result = differential_evolution(objective_function, bounds, args = (es_data, es_sigma, no_es_data, no_es_sigma), disp = False, workers = 8, polish = False, callback = progress, constraints = (lc1,lc2,nlc1,nlc2), tol = 0.02)

    print("\nGlobal Search found an approximate minimum at:", global_result.x)
    print("Polishing the fit...\n")

    #Pass result of global optimization into local gradient-free optimizer as an initial guess for fine-tuning
    result = minimize(objective_function, x0 = global_result.x, args = (es_data, es_sigma, no_es_data, no_es_sigma), method = "Nelder-Mead", bounds = bounds, options={"maxiter": 100})
    popt = result.x

    #curve_fit was removed because it doesn't handle weights well. In particular, the only allowable weights are standard error based ones. A different approach to calculate CIs of best-fit parameters must be considered.

    #TODO, this needs to be fixed; using average to fit now
    #alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
    #n = len(es_data) + len(no_es_data) + len(cvel_data)  #number of data points
    #p = len(popt) # number of parameters
    #df = max(0, n - p) # number of degrees of freedom
    #tval = t.ppf(1.0-alpha/2., df) # student-t value for the df and confidence level
    
    return (popt)


def fit_model_curve(*params):
    """
    """
    sim_es = Simulator(GET_VELOCITY(params, modifiers_w_es), get_init_noise(R0, params[param_loc["d1"]], NUM_EMS, modifiers_w_es["include_shell"], ETA), ES_TIMES)
    sim_es.run("ES", False)

    sim_no_es = Simulator(GET_VELOCITY(params, modifiers_wo_es), get_init_noise(R0, params[param_loc["d1"]], NUM_EMS, modifiers_wo_es["include_shell"], ETA), NO_ES_TIMES)
    sim_no_es.run("NO_ES", False)

    #time = np.linspace(0, T_FINAL_C, int(T_FINAL_C/5 + 1))
    #cort_vel_pred = LAM * ALPHA * time * np.exp(-LAM*time)

    return np.concatenate((sim_es.angle["ABa_dorsal"], 
                            sim_es.angle["ABp_dorsal"],
                            sim_es.angle["ABa_ant"],
                            sim_es.angle["ABp_ant"])), np.concatenate((sim_no_es.angle["ABa_dorsal"], sim_no_es.angle["ABp_dorsal"]))


    
     