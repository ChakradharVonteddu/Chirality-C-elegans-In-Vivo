import numpy as np
import pandas as pd
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from scipy.stats import linregress

def normal_negative_log_likelihood(params, dt, dtheta):
    v, sigma = params
    n = len(dtheta)
    nll = 0.5 * n * np.log(2*np.pi*sigma**2*dt) + np.sum((dtheta - v*dt)**2)/(2*sigma**2*dt)
    return nll

#fits arithmetic brownian motion parameters for data at regular time intervals using maximum likelihood estimation
def fit_abm_mle(angle_data):
    results = {"type" : [], "theta0": [], "v" : [], "sigma" : [], "t-stat" : []}

    grouped = angle_data.groupby(['condition','interval','embryo_id','cell'])

    for (condition, interval, embryo_id, cell), groupdf in grouped:
        fit_data = groupdf.sort_values('time')
        t = fit_data["time"].to_numpy()
        dt = np.diff(t)[0] #assumes regular time intervals
        dtheta = np.diff(fit_data['angle'].to_numpy())

        opt_params_res = minimize(normal_negative_log_likelihood, x0=[1, 1], args = (dt, dtheta), bounds = [(None, None), (1e-6, None)]) #Bounds on sigma to ensure it is positive
        opt_params = opt_params_res.x

        oi_mat = opt_params_res.hess_inv.todense() #Observed information matrix
        SE_v = np.sqrt(oi_mat[0,0])
        t_stat_v = opt_params[0]/SE_v

        results["type"].append(f"{condition}_{interval}s")
        results["theta0"].append(fit_data['angle'].iloc[0])
        results["v"].append(opt_params[0])
        results["sigma"].append(opt_params[1])
        results["t-stat"].append(t_stat_v)
    return pd.DataFrame(results)

def simulate_abm(type, v_hat, sigma_hat, theta0, n_steps = 40, n_simulations = 50, dt = 5):
    for _ in range(n_simulations):
        epsilon = np.random.normal(0, 1, n_steps)
        dtheta = v_hat * dt + sigma_hat * np.sqrt(dt) * epsilon
        theta_array = np.concatenate([[theta0], theta0 + np.cumsum(dtheta)])
        time_array = np.arange(0, (n_steps + 1)*dt, dt)
        plt.plot(time_array, theta_array, linewidth = 1.5, alpha = 0.5, color = "grey")
        
    expected_theta = theta0 + v_hat * time_array
    plt.plot(time_array, expected_theta, color = "black", linewidth = 2, label = "Expected trajectory")
    plt.title(f"Angle Trajectory (v_hat={round(v_hat,2)}, sigma_hat={round(sigma_hat,2)})")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.savefig(f"./brownian_motion/{type}_v_{round(v_hat,2)}_sigma_{round(sigma_hat,2)}.png")
    plt.close()

def fit_embryo_pair(theta_ABx1, theta_ABx2, dt):
    S = theta_ABx1 + theta_ABx2
    D = theta_ABx1 - theta_ABx2

    dS = np.diff(S)
    opt_params_res = minimize(normal_negative_log_likelihood, x0 = [1,1], args = (dt, dS), bounds = [(None, None),(1e-6, None)])
    v_hat = opt_params_res.x[0]/2 #account for different coefficient on v term
    sigma2_S = opt_params_res.x[1]**2/2 #account for different coefficient on the sigma term

    D0, D1 = D[:-1], D[1:]
    phi_hat = np.sum(D0 * D1)/np.sum(D0**2)
    resid = D1 - phi_hat*D0
    resid_var = np.sum(resid**2)/(len(resid) - 1)
    se_phi = np.sqrt(resid_var/np.sum(D0**2))
    if phi_hat <= 0:
        k_hat = np.nan
        sigma2_D = np.nan
        se_k = np.nan
    elif phi_hat == 1.0:
        k_hat = 0
        sigma2_D = resid_var/(2*dt)
        se_k = se_phi/(2*dt)
    else:
        k_hat = -np.log(phi_hat)/(2*dt)
        sigma2_D = (resid_var * 2 * k_hat)/(1 - np.exp(-4*k_hat*dt))
        se_k = se_phi/(2*dt*np.abs(phi_hat)) #se of k computed using delta method
    scaled_resid = resid * np.sqrt(2 * k_hat/(1 - np.exp(-4*k_hat*dt)))

    return dict(v = v_hat, k = k_hat, sigma2_S = sigma2_S, sigma2_D = sigma2_D, se_k = se_k, dS = dS, resid_D = scaled_resid)












