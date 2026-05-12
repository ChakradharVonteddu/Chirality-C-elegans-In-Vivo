import matplotlib.pyplot as plt
import pandas as pd

from modelling.fit.fit_config import GET_VELOCITY, INIT
from modelling.fit.FitCoritcalFlow import fit_cortical
from modelling.fit.FitCurveFit import fit_model_whole
from modelling.Simulator import Simulator
from plot.AnimatorSolid import Animator
from plot.plot_angles import plot_angles
from plot.plot_distances import plot_distance
from plot.plot_fit import plot_fit
from plot.plot_xz import plot_xz


def fit():
    "FIT; make sure to customize the model in the fit functions"
    # print(fit_fmin_model(get_angular_data()))
    data_stat = pd.read_excel("./src/data/data_stat.xlsx")
    print(data_stat.shape)
    model_fit = fit_model_whole(data_stat)
    print(model_fit)
    return model_fit


def run(params):
    "RUN, SAVE, AND ANIMATE"
    sim = Simulator(GET_VELOCITY(params), INIT)
    sim.run(True)
    animator = Animator(sim.df)
    animator.animate()


def plot_data():
    "PLOTTING"
    distances = pd.read_excel("./output/distances.xlsx")
    angles = pd.read_excel("./output/angles.xlsx")
    output = pd.read_excel("./output/output.xlsx")
    ABa_dorsal = pd.read_excel("./src/data/data_ABa_dorsal.xlsx").drop(["t"], axis=1)
    ABp_dorsal = pd.read_excel("./src/data/data_ABp_dorsal.xlsx").drop(["t"], axis=1)
    ABa_ant = pd.read_excel("./src/data/data_ABa_ant.xlsx").drop(["t"], axis=1)
    ABp_ant = pd.read_excel("./src/data/data_ABp_ant.xlsx").drop(["t"], axis=1)
    #    data_stat = pd.read_excel("./src/data/data_stat.xlsx")
    data_stat = pd.read_excel("./src/data/data_stat_new.xlsx")

    fig, ((axX, axZ), (axDist, axDegree)) = plt.subplots(2, 2)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    axX.title.set_text("X Plot")
    axZ.title.set_text("Z Plot")
    axDist.title.set_text("Distances")
    axDegree.title.set_text("Theta vs Phi")

    # run plotting helper functions; saves figure
    plot_distance(axDist, distances)
    plot_angles(axDegree, angles, ABa_dorsal, ABp_dorsal, ABa_ant, ABp_ant)
    plot_xz(axX, axZ, output)
    plt.savefig("./output/xz.png")

    plot_fit(data_stat, angles)


def fit_cortical_flow():
    "FIT CORTICAL FLOW"
    cortical_l = pd.read_excel("./src/data/data_cortical_l.xlsx")
    cortical_r = pd.read_excel("./src/data/data_cortical_r.xlsx")
    fit_cortical(cortical_l, "cortical_l")
    fit_cortical(cortical_r, "cortical_r")


if __name__ == "__main__":
    opt_params = fit()[0]
    run(opt_params)
    #    run([1,0])
    print(opt_params)
    plot_data()
