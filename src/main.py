import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from modelling.models.model_config import NUM_EMS, R0, ETA, T_FINAL_ES, T_FINAL_NO_ES, ALPHA, LAM, modifiers_w_es, modifiers_wo_es, param_loc
from modelling.Fit.fit_config import GET_VELOCITY, get_init_noise
from modelling.Fit.FitCoritcalFlowNew import fit_cortical
from modelling.Fit.FitCurveFit import fit_model_whole
from modelling.Fit.hypothesis_testing import fit_spline, permutation_test, t_test, welch_t_test, mannwhitneyu_test, within_corr, cross_corr, levene_test, wilcoxon_signed_rank_test
from modelling.Fit.brownian_motion import fit_abm_mle, simulate_abm, fit_embryo_pair
from modelling.Simulator import Simulator
from plot.AnimatorSolid_revised import Animator
from plot.plot_angles import plot_angles
from plot.plot_distances import plot_distance
from plot.plot_fit import plot_es_fit, plot_cort_fit, plot_no_es_fit
from plot.plot_xz import plot_xz


def fit():
    "FIT; make sure to customize the model in the fit functions"
    # print(fit_fmin_model(get_angular_data()))
    data_stat_es = pd.read_excel("./src/data/data_stat.xlsx", sheet_name = "ES")
    data_stat_no_es = pd.read_excel("./src/data/data_stat.xlsx", sheet_name = "NO_ES")
    data_cort = pd.read_excel("./src/data/data_cortical_all.xlsx")
    model_fit = fit_model_whole(data_stat_es, data_stat_no_es, data_cort)
    print(model_fit)
    return model_fit


def run(params, modifiers, save_folder):
    "RUN, SAVE, AND ANIMATE"
    if save_folder == "ES":
        #times = np.linspace(0, 1, T_FINAL_ES + 1)
        times = np.linspace(0, 1, int(T_FINAL_ES/5 + 1))
        t_final = T_FINAL_ES
    else:
        #times = np.linspace(0, T_FINAL_NO_ES/T_FINAL_ES, T_FINAL_NO_ES + 1)
        times = np.linspace(0, T_FINAL_NO_ES/T_FINAL_ES, int(T_FINAL_NO_ES/7 + 1)) #works for 7s increment data, needs changes to account for 15s increment data. If considering a mix of 15s and 7s increment data, combine both series for this variable.
        t_final = T_FINAL_NO_ES

    sim = Simulator(GET_VELOCITY(params, modifiers), get_init_noise(R0, params[param_loc["d1"]], NUM_EMS, modifiers["include_shell"], ETA), times)
    sim.run(type = save_folder, save = True, save_folder = save_folder)

    evaluator = GET_VELOCITY(params, modifiers, return_data=True)
    radii_data = []
    force_data = []
        
    for i in range(len(sim.df)):
        t = times[i]
        y = sim.df.iloc[i].values
        force_at_t, radii_at_t = evaluator(t, y) 
        radii_data.append(radii_at_t)
        force_data.append(force_at_t)

    radii_df = pd.DataFrame(radii_data)
    force_df = pd.DataFrame(force_data)
        
    save_path = "./output/" + save_folder + "/force_data.csv"
    force_df.to_csv(save_path, index=False)
    print(f"Force data saved successfully to {save_path}")

    animator = Animator(sim.df, radii_df, alpha = ALPHA, lam = LAM, scale_factor = t_final/(len(times)-1))
    animator.animate(save_folder)


def plot_data(params, save_folder):
    "PLOTTING"
    distances = pd.read_csv(f"./output/{save_folder}/distances.csv")
    angles = pd.read_csv(f"./output/{save_folder}/angles.csv")
    output = pd.read_csv(f"./output/{save_folder}/output.csv")
    ABa_dorsal = pd.read_excel("./src/data/data_ABa_dorsal.xlsx").drop(["t"], axis=1)
    ABp_dorsal = pd.read_excel("./src/data/data_ABp_dorsal.xlsx").drop(["t"], axis=1)
    ABa_ant = pd.read_excel("./src/data/data_ABa_ant.xlsx").drop(["t"], axis=1)
    ABp_ant = pd.read_excel("./src/data/data_ABp_ant.xlsx").drop(["t"], axis=1)
    data_stat = pd.read_excel("./src/data/data_stat.xlsx", sheet_name = save_folder)
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
    if save_folder == "ES":
        plot_angles(axDegree, angles, ABa_dorsal, ABp_dorsal, ABa_ant, ABp_ant) #Theta vs phi graph only for ES condition, since no anterior angle data exists for no_es
    plot_xz(axX, axZ, output)
    plt.savefig(f"./output/{save_folder}/xz.png")
    if save_folder == "ES":
        plot_es_fit(data_stat, angles, params[param_loc["E1"]], save_folder)
    else:
        plot_no_es_fit(data_stat, angles, params[param_loc["E1"]], save_folder, ["ABa_dorsal","ABp_dorsal"])
    plot_cort_fit(data_cort, alpha = ALPHA, lam = LAM)

def fit_cortical_flow():
    "FIT CORTICAL FLOW"
    cortical_all = pd.read_excel("./src/data/data_cortical_all.xlsx")
    fit_cortical(cortical_all, "./output/cortical_fit.png")

def run_permutation_test(groups, T, n_permutations = 1000):
    combined_data = pd.read_excel("./src/data/angle_data_combined.xlsx")
    splines = fit_spline(combined_data, groups)
    p_values = permutation_test(splines, groups, T, n_permutations)
    return p_values

def run_brownian_motion():
    combined_data = pd.read_excel("./src/data/angle_data_combined.xlsx")
    results = fit_abm_mle(combined_data)
    results.to_csv("./brownian_motion/brownian_motion_fit_results.csv", index=False)
    for index, row in results.iterrows():
        simulate_abm(row["type"],row["v"],row["sigma"],row["theta0"])

def run_angle_change_tests():
    combined_data = pd.read_excel("./src/data/angle_data_combined.xlsx").drop('interval', axis = 1)
    embryo_data = pd.pivot_table(combined_data, values = 'angle', columns = 'cell', index = ['condition','embryo_id','time'])
    embryo_data['angle_avg'] = (embryo_data['ABx1'] + embryo_data['ABx2'])/2
    embryo_data = embryo_data.drop(['ABx1','ABx2'], axis = 1)
    embryo_data.reset_index(inplace = True)
    es_data = embryo_data[(embryo_data['condition'] == "es") & (embryo_data['time'] <= 195)]
    no_es_data = embryo_data[(embryo_data['condition'] == "no_es") & (embryo_data['time'] <= 196) ]
    es_angle_changes = t_test(es_data)
    no_es_angle_changes = t_test(no_es_data)
    wilcoxon_signed_rank_test(es_angle_changes,0)
    wilcoxon_signed_rank_test(no_es_angle_changes,0)
    welch_t_test(es_angle_changes, no_es_angle_changes)
    mannwhitneyu_test(es_angle_changes, no_es_angle_changes)

def calculate_correlation():
    combined_data = pd.read_excel("./src/data/angle_data_combined.xlsx")
    pivot_data = pd.pivot_table(combined_data, values = 'angle', columns = 'cell', index = ['condition','embryo_id','time','interval']).reset_index()
    pivot_data = pivot_data.sort_values(['time'])
    pivot_data[['ABx1_diff','ABx2_diff']] = pivot_data.groupby(['condition','interval','embryo_id'])[['ABx1','ABx2']].diff()
    diff_data = pivot_data.dropna(subset = ['ABx1_diff','ABx2_diff']).copy()
    within_corr_df = within_corr(diff_data)
    cross_corr_df = cross_corr(diff_data)
    print(within_corr_df)
    print(cross_corr_df)

def fit_all_embryos():
    combined_data = pd.read_excel("./src/data/angle_data_combined.xlsx")
    pivot_data = pd.pivot_table(combined_data, values = 'angle', columns = 'cell', index = ['condition','embryo_id','time','interval']).reset_index()
    pivot_data = pivot_data.sort_values(['time'])
    fit_res = {"es" : {"v" : [], "k" : [], "sigma2_S" : [], "sigma2_D" : [], "se_k" : [], "dS" : [], "resid_D" : []}, 
               "no_es" : {"v" : [], "k" : [], "sigma2_S" : [], "sigma2_D" : [], "se_k" : [], "dS" : [], "resid_D" : []}
    }
    for condition, groupdf in pivot_data.groupby('condition'):
        embryos = groupdf['embryo_id'].unique()
        for embryo in embryos:
            fit_data = groupdf[groupdf['embryo_id'] == embryo].sort_values('time')
            ABx1_theta = fit_data['ABx1'].to_numpy()
            ABx2_theta = fit_data['ABx2'].to_numpy()
            dt = np.diff(fit_data['time'].to_numpy())[0]
            res_dict = fit_embryo_pair(ABx1_theta, ABx2_theta, dt)
            for key, value in res_dict.items():
                fit_res[condition][key].append(value)
    print(np.array(fit_res['es']['se_k'])/np.array(fit_res['es']['k']))
    print(np.array(fit_res['no_es']['se_k'])/np.array(fit_res['no_es']['k']))
    es_k_mean = np.nanmean(fit_res['es']['k'])
    no_es_k_mean = np.nanmean(fit_res['no_es']['k'])
    es_k_std = np.nanstd(fit_res['es']['k'], ddof = 1)
    no_es_k_std = np.nanstd(fit_res['no_es']['k'], ddof = 1)
    print(f"es_k_mean: {es_k_mean}, es_k_std: {es_k_std}")
    print(f"no_es_k_mean: {no_es_k_mean}, no_es_k_std: {no_es_k_std}")
    print(f"es_v_mean: {np.nanmean(fit_res['es']['v'])}")
    welch_t_test(fit_res['es']['k'], fit_res['no_es']['k'])
    mannwhitneyu_test(fit_res['es']['k'], fit_res['no_es']['k'])


if __name__ == "__main__":
    fit_all_embryos()
    #calculate_correlation()
    #run_angle_change_tests()
    #run_brownian_motion()
    #[6.95070665e+00, 2.72663760e+01, 1.57850248e+00, 1.78399916e+00, 2.04042424e+00, 2.02866637e-02, 2.30445677e+00, 3.54847729e-01] - with t0
    #[2.4717867, 11.04045702, 1.56806558, 1.48865608, 2.1438866, 0.02764666, 2.46013253] - without t0
    #p_values = run_permutation_test(["no_es_15s", "no_es_7s", "es_5s"], {"no_es_15s": 360, "no_es_7s": 238, "es_5s": 195})
    #opt_params = fit()
    #print("The optimal parameters are: ", opt_params)
    #opt_params = [6.95070665e+00, 2.72663760e+01, 1.57850248e+00, 1.78399916e+00, 2.04042424e+00, 2.02866637e-02, 2.30445677e+00, 3.54847729e-01]
    #run(opt_params, modifiers_w_es, "ES")
    #run(opt_params, modifiers_wo_es, "NO_ES")
    #plot_data(opt_params, "ES")
    #plot_data(opt_params, "NO_ES")

    #[5.35023653e+00, 1.93259627e+01, 1.49389179e-02, 1.60258734e+00, 1.58282714e+00, 2.07246103e+00, 3.60939324e-03, 7.16053181e-02]

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

