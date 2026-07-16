import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, differential_evolution, LinearConstraint, NonlinearConstraint
from numpy.polynomial import Polynomial
from scipy.stats import t
from collections.abc import Sequence
from ..Simulator import Simulator
from .fit_config import GET_VELOCITY, get_init_noise
from ..models.model_config import T_FINAL, T_FINAL_C, NUM_EMS, R0, R0_ALT, ETA
from ..models.All_Modifiers import calculate_d_cell_radius
import warnings

def objective_function(params, y_data, sigma):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
        y_pred = fit_model_curve((), *params)
        
        #Standardized residuals denoting how far the prediction is from the data in terms of standard deviations
        residuals = (y_data - y_pred)/sigma
                
        sse = np.sum(residuals**2)

        if np.isnan(sse):
            return np.inf
            
        return sse
            
    except Exception as e:
        return np.inf

def fit_model_whole(angle_data, cort_data):
    """
    """
    ABa_dorsal = angle_data['ABa_dorsal_avg'].to_numpy()
    ABp_dorsal = angle_data['ABp_dorsal_avg'].to_numpy()
    ABa_ant = angle_data['ABa_ant_avg'].to_numpy()
    ABp_ant = angle_data['ABp_ant_avg'].to_numpy()
    cort_vel = T_FINAL/R0_ALT * cort_data['cort_avg'].to_numpy()

    ABa_dorsal_stdmean = angle_data['ABa_dorsal_stdeofmean'].to_numpy()
    ABp_dorsal_stdmean = angle_data['ABp_dorsal_stdeofmean'].to_numpy()
    ABa_ant_stdmean = angle_data['ABa_ant_stdeofmean'].to_numpy()
    ABp_ant_stdmean = angle_data['ABp_ant_stdeofmean'].to_numpy()
    cort_stdmean = T_FINAL/R0_ALT * cort_data['cort_stdeofmean'].to_numpy()

    y_data = np.concatenate((ABa_dorsal, ABp_dorsal, ABa_ant, ABp_ant, cort_vel))
    y_error = np.concatenate((ABa_dorsal_stdmean, 
                                      ABp_dorsal_stdmean, 
                                      ABa_ant_stdmean, 
                                      ABp_ant_stdmean, 
                                      cort_stdmean))
    
    sigma = np.array([10**(-6) if error == 0 else error for error in y_error])
    
    #bounds for parameters in the order : spring constant/gamma, frictional constant/gamma, lambda, E1, d1, d2 and alpha
    bounds = [
        (0, 10),   
        (0, 50),  
        (0, 1),  
        (1.5, 3), 
        (0, 2),  
        (1, 3),
        (0, 2)
    ]
    
    def extended_spring_length_change(params):
        final_radius = calculate_d_cell_radius(params[4],params[5])
        return (2*final_radius + params[5] - 2*R0 - params[4])/(2*R0 + params[4])
    
    #def compression(params):
        #final_radius = calculate_d_cell_radius(params[4],params[5])
        #return (2*final_radius + params[5] - 2*params[3])/(2*final_radius + params[5])
    
    #linear constraint to allow for d2 to be greater than d1
    lc1 = LinearConstraint([[0,0,0,0,-1,1,0]], 0, np.inf)
    #Non-linear constraint to ensure extended spring length increases and increase is less than 30%
    nlc1 = NonlinearConstraint(extended_spring_length_change,0,0.3)
    #Non-linear constraint to have at-most 10% compression under maximum extended spring length (calculated as 2*final_radius + d2)
    #nlc2 = NonlinearConstraint(compression,-np.inf,0.1)
 
    #Displays optimization progress after each iteration
    def progress(xk, convergence):
        error = objective_function(xk, y_data, sigma)
        print(f"Error: {error:>8.3f} | Convergence: {convergence * 100:>5.2f}% | Best params vector: {xk}")
        #Prints out error associated with fit, convergence of error values and best params vector

    print("Starting global optimization (Differential Evolution). This may take a few hours...")
    global_result = differential_evolution(objective_function, bounds, args = (y_data, sigma), disp = False, workers = 8, polish = False, callback = progress, constraints = (lc1,nlc1))

    print("\nGlobal Search found an approximate minimum at:", global_result.x)
    print("Polishing the fit...\n")
    
    #Decompose into lower and upper bounds in the same order
    lower, upper = zip(*bounds)
    
    #Pass result of global optimization into local optimizer as an initial guess for fine-tuning
    popt, pcov = curve_fit(
        fit_model_curve, 
        (), 
        y_data, 
        p0 = global_result.x, 
        bounds = (lower, upper), 
        sigma = sigma
    )

    #TODO, this needs to be fixed; using average to fit now
    alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
    n = len(y_data)    # number of data points
    p = len(popt) # number of parameters
    df = max(0, n - p) # number of degrees of freedom
    tval = t.ppf(1.0-alpha/2., df) # student-t value for the df and confidence level
    
    return (popt, pcov, ((np.diag(pcov)[0]**0.5)*tval,
                         (np.diag(pcov)[1]**0.5)*tval))


def fit_model_curve(x: Sequence[float], *params):
    """
    """
    sim = Simulator(GET_VELOCITY(params), get_init_noise(R0, params[4], NUM_EMS, ETA))
    sim.run(False)

    time = np.linspace(0, T_FINAL_C, int(T_FINAL_C/5 + 1))
    cort_vel_pred = params[2] * params[6] * time * np.exp(-params[2]*time)

    return np.concatenate((sim.angle["ABa_dorsal"], 
                            sim.angle["ABp_dorsal"],
                            sim.angle["ABa_ant"],
                            sim.angle["ABp_ant"],
                            cort_vel_pred))


    
     