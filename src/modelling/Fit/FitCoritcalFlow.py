from dataclasses import dataclass
from enum import Enum, auto
from collections.abc import Sequence
import pandas as pd 
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt

@dataclass
class CorticalFlowFit(Enum):
    LINEAR = auto()
    QUADRATIC = auto()


def fit_cortical(data, fig_name) -> None:
    """
    Fit cortical data to αte^(-λt)
    """
    t = data["t"]
    cortical_data = data.drop(["t"], axis=1).apply(lambda x: abs(x))
    
    # fit data using αte^(-λt)
    alpha, lam = fmin(fit_cortical_func, [1, 1], args=(t, cortical_data, CorticalFlowFit.LINEAR))
    # average angles
    cortical_avg = cortical_data.mean(axis=1).to_numpy()

    # initialize figure and plot
    fig, ax = plt.subplots()
    ax.plot(t, cortical_avg)
    ax.plot(t, np.multiply(alpha*t,np.e**(-lam * t)))
    plt.savefig(fig_name)
    print(f"alpha = {alpha}, lambda = {lam}")


def fit_cortical_func(x: tuple[float, float], t: Sequence, data: pd.DataFrame, fit_type: CorticalFlowFit) -> float:
    """
    Fit function that calculates the residuals of the cortical flow model to the raw data at a set of 
    parameters; used for fitting. 

    Returns the residual squared score
    """

    alpha, lam = x
    t = np.array(t)

    if fit_type == CorticalFlowFit.LINEAR:
        # αte^(-λt)
        output = alpha * t * np.e**(-lam * t)
    elif fit_type == CorticalFlowFit.QUADRATIC:
        # αt^2e^(-λt)
        output = alpha * t**2 * np.e**(-lam * t)

    residual_squared = 0 
    for column_index in range(10):
        residual_squared += np.linalg.norm(data.iloc[:, column_index] - output) ** 2
    
    return residual_squared