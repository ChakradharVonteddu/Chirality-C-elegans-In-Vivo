from matplotlib.pyplot import Axes
import pandas as pd

def plot_distance(ax: Axes, data: pd.DataFrame) -> None:
    """
    Plots the inter-cell distances given the data on the given Axes. 
    """
    t = range(len(data))
    ax.plot(t, data["12"].to_numpy(), "-o", label="12", c="blue", markersize=1)
    ax.plot(t, data["13"].to_numpy(), "-o", label="13", c="orange", markersize=1)
    ax.plot(t, data["14"].to_numpy(), "-o", label="14", c="green", markersize=1)
    ax.plot(t, data["23"].to_numpy(), "-o", label="23", c="red", markersize=1)
    ax.plot(t, data["24"].to_numpy(), "-o", label="24", c="yellow", markersize=1)
    ax.plot(t, data["34"].to_numpy(), "-o", label="34", c="pink", markersize=1)
    ax.legend()