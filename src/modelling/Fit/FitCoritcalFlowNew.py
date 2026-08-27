import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from modelling.models.model_config import R0_ALT

def model_func(t, alpha, lam):
    return alpha * lam * t * np.exp(-lam * t)

def fit_cortical(data, fig_name):
    t = data["t"].to_numpy()
    cvel_data = 1/R0_ALT * data['cort_avg'].to_numpy() #Normalize by R0_ALT
    cvel_error =  1/R0_ALT * data['cort_stdeofmean'].to_numpy() #Normalize by R0_ALT

    cvel_sigma = np.array([10**(-6) if error == 0 else error for error in cvel_error])

    # Fit model
    popt, pcov = curve_fit(model_func, t, cvel_data, sigma=cvel_sigma, p0=(0.01, 0.015))
    alpha, lam = popt
    perr = np.sqrt(np.diag(pcov))  # 1-sigma standard errors

    # Plot
    fig, ax = plt.subplots()
    ax.plot(t, model_func(t, *popt), label="model", color='blue')
    ax.plot(t, cvel_data, label = "data", color = "orange")
    ax.set_title(f"Cortical Velocity Fit (alpha={alpha}, lambda={lam:.4f})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cortical Flow Velocity")
    ax.legend()
    ax.legend(loc = "upper left")
    ax.grid(True)
    plt.savefig(fig_name)
    print(f"alpha = {alpha:.6f} ± {perr[0]:.6f}, lambda = {lam:.6f} ± {perr[1]:.6f}")
