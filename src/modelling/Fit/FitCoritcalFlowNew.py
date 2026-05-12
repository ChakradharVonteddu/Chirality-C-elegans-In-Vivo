import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model_func(t, alpha, lam):
    return alpha * t * np.exp(-lam * t)

def fit_cortical(data, fig_name):
    t = data["t"].to_numpy()
    cortical_data = data.drop(columns=["t"])

    # Flip y-values if filename includes 'cortical_l'
    if "cortical_l" in fig_name:
        cortical_data *= -1

    # Flatten data to fit all replicates
    t_all = np.tile(t, cortical_data.shape[1])         # Repeat t for each replicate
    y_all = cortical_data.to_numpy().flatten()         # Flattened y-values

    # Fit model
    popt, pcov = curve_fit(model_func, t_all, y_all, p0=(1, 0.01))
    alpha, lam = popt
    perr = np.sqrt(np.diag(pcov))  # 1-sigma standard errors

    if "cortical_l" in fig_name:
        label = "Left side"
    elif "cortical_r" in fig_name:
        label = "Right side"
        
    # Plot
    fig, ax = plt.subplots()
    ax.scatter(t_all, y_all, s=10, alpha=0.5, label=label)
    t_fit = np.linspace(t.min(), t.max(), 200)
    ax.plot(t_fit, model_func(t_fit, *popt), label=f"Fit: α={alpha:.6f}±{perr[0]:.6f}, λ={lam:.6f}±{perr[1]:.6f}", color='red')
    ax.set_xlabel("Time")
    ax.set_ylabel("Cortical Flow Velocity")
    ax.legend()
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.savefig(fig_name)
    print(f"alpha = {alpha:.6f} ± {perr[0]:.6f}, lambda = {lam:.6f} ± {perr[1]:.6f}")
