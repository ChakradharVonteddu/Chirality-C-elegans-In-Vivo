from matplotlib.pyplot import Axes
import pandas as pd




def plot_angles(ax: Axes, 
                angles_df: pd.DataFrame, 
                ABa_dorsal, 
                ABp_dorsal, 
                ABa_ant, 
                ABp_ant
                ) -> None:
    """
    Plots the Theta (anterior) vs Phi (dorsal) graph on the given Axes.
    """

    # plots model theta vs phi
    ax.plot(angles_df["ABa_dorsal"].to_numpy(), 
                angles_df["ABa_ant"].to_numpy(), 
                "-o", 
                markersize=2, 
                c="blue")
    
    ax.plot(angles_df["ABp_dorsal"].to_numpy(), 
                angles_df["ABp_ant"].to_numpy(), 
                "-o", 
                markersize=2, 
                c="red")
    
    # plot data theta vs phi 
    for i in range(10):
        ax.plot(ABa_dorsal.iloc[:,i].to_numpy(), 
                ABa_ant.iloc[:,i].to_numpy(), 
                "-o", 
                markersize=2, 
                c="skyblue", 
                alpha=0.4)
        ax.plot(ABp_dorsal.iloc[:,i].to_numpy(), 
                ABp_ant.iloc[:,i].to_numpy(), 
                "-o", 
                markersize=2, 
                c="lightcoral", 
                alpha=0.4)