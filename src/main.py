import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from modelling.models.model_config import NUM_EMS, R0, ETA, modifiers_w_es, modifiers_wo_es
from modelling.Fit.fit_config import GET_VELOCITY, get_init_noise
from modelling.Fit.FitCoritcalFlow import fit_cortical
from modelling.Fit.FitCurveFit import fit_model_whole
from modelling.Simulator import Simulator
from plot.AnimatorSolid import Animator
from plot.plot_angles import plot_angles
from plot.plot_distances import plot_distance
from plot.plot_fit import plot_fit, plot_cort_fit
from plot.plot_xz import plot_xz


def fit():
    "FIT; make sure to customize the model in the fit functions"
    # print(fit_fmin_model(get_angular_data()))
    data_stat = pd.read_excel("./src/data/data_stat.xlsx")
    data_cort = pd.read_excel("./src/data/data_cortical_all.xlsx")
    print(data_stat.shape)
    model_fit = fit_model_whole(data_stat, data_cort)
    print(model_fit)
    return model_fit


def run(params, modifiers, save_folder):
    "RUN, SAVE, AND ANIMATE"
    sim = Simulator(GET_VELOCITY(params, modifiers), get_init_noise(R0, params[4], NUM_EMS, ETA))
    sim.run(True, save_folder)

    evaluator = GET_VELOCITY(params, modifiers, return_data=True)
    radii_data = []
    force_data = []
    
    times = np.linspace(sim.TAU_INITIAL, sim.TAU_FINAL, 40)
    
    for i in range(len(sim.df)):
        t = times[i]
        y = sim.df.iloc[i].values
        force_at_t, radii_at_t = evaluator(t, y) 
        radii_data.append(radii_at_t)
        force_data.append(force_at_t)

    radii_df = pd.DataFrame(radii_data)
    force_df = pd.DataFrame(force_data)
        
    save_path = "/Users/chakradhar/Desktop/Cellular Chirality model/chirality_C_elegans_In_Vivo/output/" + save_folder + "/force_data.xlsx"
    force_df.to_excel(save_path, index=False)
    print(f"Force data saved successfully to {save_path}")

    animator = Animator(sim.df, radii_df)
    animator.animate(save_folder)


def plot_data(params):
    "PLOTTING"
    distances = pd.read_csv("./output/distances.csv")
    angles = pd.read_csv("./output/angles.csv")
    output = pd.read_csv("./output/output.csv")
    ABa_dorsal = pd.read_excel("./src/data/data_ABa_dorsal.xlsx").drop(["t"], axis=1)
    ABp_dorsal = pd.read_excel("./src/data/data_ABp_dorsal.xlsx").drop(["t"], axis=1)
    ABa_ant = pd.read_excel("./src/data/data_ABa_ant.xlsx").drop(["t"], axis=1)
    ABp_ant = pd.read_excel("./src/data/data_ABp_ant.xlsx").drop(["t"], axis=1)
    #    data_stat = pd.read_excel("./src/data/data_stat.xlsx")
    data_stat = pd.read_excel("./src/data/data_stat_new.xlsx")
    data_cort = pd.read_excel("./src/data/data_cortical_all.xlsx")

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

    plot_fit(data_stat, angles, params[3])
    plot_cort_fit(data_cort, alpha = params[6], lam = params[2])


def fit_cortical_flow():
    "FIT CORTICAL FLOW"
    cortical_l = pd.read_excel("./src/data/data_cortical_l.xlsx")
    cortical_r = pd.read_excel("./src/data/data_cortical_r.xlsx")
    fit_cortical(cortical_l, "cortical_l")
    fit_cortical(cortical_r, "cortical_r")


if __name__ == "__main__":
    opt_params = fit()[0]
    print("The optimal parameters are: ", opt_params)
    run(opt_params , modifiers_w_es, "es")
    run(opt_params, modifiers_wo_es, "without_es")
    plot_data(opt_params) #plots for eggshell model only (for now)
    
    #[9.83505885e-02, 7.05053333e+01, 1.07613064e-02, 2.4, 1.62967004e+00, 1.66688578e+00] - solution 1, no relative velocity based friction
    #[3.32324622e+00 9.90929901e+01 1.07573262e-02 2.18514148e+00 1.87793458e+00 1.99636674e+00 4.40406747e-01] - incomplete run, solution 2, relative velocity friction parameter
    #[5.37476580e+00, 9.32340217e+01, 1.07570959e-02, 2.15426394e+00, 1.49900671e+00, 1.87011063e+00, 4.27844576e-01] - solution 3a, different friction parameter only for EMS and P2
    #[9.15682734e+00, 8.55949710e+01, 1.09190825e-02, 2.17003114e+00, 1.56867694e+00, 1.87328333e+00, 2.77911881e-01] - solution 3b, different friction parameter only for EMS and P2
    
    #[4.66141965, 2.39936276, 0.01101217, 1.47150761, 1.41123144, 1.94792636] - new 6 params, with eggshell and 4 EMS spheres
    #This optimum shows no movement in the without-eggshell simulation as the high spring forces push the cells apart and the cortical forces no longer apply once the cells are pushed apart.
    #[7.74264387e-01, 1.77556376e+01, 1.06543547e-02, 1.26516449e+00, 1.56148184e+00, 1.75729263e+00] - k - [0,1], mu - [5,20], not a good optimum
    #[1.86002229, 5.07617368, 0.01116509, 2.5375156, 0.10113376, 2.29080625] - k - [0,2], mu - [5,20]
    #[9.40156350e-01, 2.62286314e+01, 1.12268415e-02, 1.24735063e+00, 1.52225734e+00, 2.02755594e+00] - k - [0,4], mu - [10,30], not a good optimum
    #[1.82552326, 8.08061855, 0.01084875, 1.44370301, 1.63846917, 2.11038811] - 3 degree tilt supporting, k - [0,10], mu - [0,10]
    #[2.83905177, 9.16518423, 0.01089593, 1.49071318, 1.2148078, 2.14106484] - 3 degree tilt supporting, k - [0,10], mu - [0,10], additional linear constraint to limit initial compression

#[0.91039903, 1.57152351, 0.01094358, 0.58646896, 0.97212214, 1.22329519] - old 6 params, with eggshell and cort data, parametrized EMS and P2 radius, 4 EMS spheres
#[2.92106273, 1.91422549, 0.01098747, 0.63696686, 0.98422745, 1.31556727] - old  6 params, with eggshell and cort data, parametrized EMS and P2 radius, 1 EMS sphere

#[7.38393839 0.58001898 0.79172621 1.86493726 0.4174721  1.80931021] - incomplete run, 3 degree tilt opposing, k - [0,10], mu - [0,10]

