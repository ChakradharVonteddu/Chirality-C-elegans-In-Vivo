from matplotlib.pyplot import Axes


#TODO
def plot_xz(axX: Axes, axZ: Axes, position_df) -> None:
    """
    Plots the xz graph of the cell position-vectors on the given Axes.
    """    
    # plots x and z coordinates of the model position vectors 
    for index in range(4):
        t = range(len(position_df))
        x = position_df[str(3*index)]
        z = position_df[str(3*index+2)]
        color = ["blue", "orange", "green", "red"]
        axX.plot(t, x, "-o", label=index+1, c = color[index], markersize = 1)
        axZ.plot(t, z, "-o", label=index+1, c = color[index], markersize = 1)

    axX.legend()
    axZ.legend()
