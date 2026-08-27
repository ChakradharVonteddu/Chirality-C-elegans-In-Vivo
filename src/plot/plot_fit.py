import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy

from modelling.models.model_config import T_FINAL_C, ASPECT_RATIO, R0_ALT, ES_TIMES, T_FINAL_ES, time_configs, T_FINAL_NO_ES


def plot_es_fit(data, angles, E1, save_folder, angle_types=None):
    """
    Plot eggshell model fit against raw data.
 
    angle_types: optional list of which angle series to plot, chosen from
        ["ABa_dorsal", "ABp_dorsal", "ABa_ant", "ABp_ant"]. Defaults to all
        four. Use this to skip series whose data is not available.
    """
    all_angle_types = ["ABa_dorsal", "ABp_dorsal", "ABa_ant", "ABp_ant"]
    if angle_types is None:
        angle_types = all_angle_types

    angle_data = angles

    t = data["t"].to_numpy()
 
    # z-score for a 95% CI, used to convert standard error of the mean to a CI half-width
    z_0975 = scipy.stats.norm.ppf(0.975)
 
    series_color = {"ABa_dorsal": "blue", "ABp_dorsal": "red", "ABa_ant": "blue", "ABp_ant": "red"}
 
    # initialize graph, set Axes titles
    fig, (axD, axA) = plt.subplots(2)
    plt.subplots_adjust(hspace=0.35)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    axD.set_title("Dorsal-view angles", fontsize=20)
    axA.set_title("Anterior-view angles", fontsize=20)
 
    for angle_type in angle_types:
        ax = axD if angle_type.endswith("_dorsal") else axA
        color = series_color[angle_type]
        label_prefix = angle_type.split("_")[0]
        average = data[f"{angle_type}_avg"].to_numpy()
        CIhalfwidth = z_0975 * data[f"{angle_type}_stdeofmean"].to_numpy()
        computed = angle_data[angle_type].to_numpy()
 
        ax.plot(t, computed, label=f"{label_prefix} model", markersize=2, color=color)
        ax.plot(
            t,
            average,
            "o",
            label=f"{label_prefix} data (shading is CI)",
            markersize=4,
            color="none",
            markeredgecolor=color,
        )
        ax.fill_between(t, average - CIhalfwidth, average + CIhalfwidth, alpha=0.2, color=color)
 
    axD.tick_params(axis="both", labelsize=18)
 
    axA.axhline(y=0, color="grey", linestyle="--", linewidth=1)
 
    axA.tick_params(axis="both", labelsize=20)
 
    axA.set_ylim(-40, 40)
 
    # plot legend
    axD.legend(fontsize=18)
 
    # save figure
    filename = f"./output/{save_folder}/fit_E0_{ASPECT_RATIO*E1:.2f}_E1_{E1:.2f}.pdf"
    plt.savefig(filename)

def plot_no_es_fit(data, angles, E1, save_folder, angle_types = None):
    """
        Plot no eggshell model fit against raw data.
     
        angle_types: optional list of which angle series to plot, chosen from
            ["ABa_dorsal", "ABp_dorsal", "ABa_ant", "ABp_ant"]. Defaults to all
            four. Use this to skip series whose data is not available.
    """

    z_0975 = scipy.stats.norm.ppf(0.975)

    all_angle_types = ["ABa_dorsal", "ABp_dorsal", "ABa_ant", "ABp_ant"]
    if angle_types is None:
        angle_types = all_angle_types
    
    series_color = {"ABa_dorsal": "blue", "ABp_dorsal": "red", "ABa_ant": "blue", "ABp_ant": "red"}
 
    for config in time_configs:
        t_plot = np.arange(config["t_min"], config["t_max"] + 1, config["interval"])
 
        data_sub = data.loc[data["t"].isin(t_plot)].reset_index(drop=True)
 
        angles_sub = angles
 
        fig, (axABa, axABp) = plt.subplots(2)
        plt.subplots_adjust(hspace=0.35)
        fig.set_figheight(7)
        fig.set_figwidth(15)
 
        axes = {"ABa_dorsal": axABa, "ABp_dorsal": axABp}
 
        for angle_type in angle_types:
            ax = axes[angle_type]
            color = series_color[angle_type]
            label_prefix = angle_type.split("_")[0]
 
            ax.set_title(f"{label_prefix} dorsal-view angle", fontsize=20)
 
            average = data_sub[f"{angle_type}_avg"].to_numpy()
            CIhalfwidth = z_0975 * data_sub[f"{angle_type}_stdeofmean"].to_numpy()
            computed = angles_sub[angle_type].to_numpy()
 
            ax.plot(t_plot, computed, label=f"{label_prefix} model", markersize=2, color=color)
            ax.plot(
                t_plot,
                average,
                "o",
                label=f"{label_prefix} data (shading is CI)",
                markersize=4,
                color="none",
                markeredgecolor=color,
            )
            ax.fill_between(t_plot, average - CIhalfwidth, average + CIhalfwidth, alpha=0.2, color=color)
            ax.tick_params(axis="both", labelsize=18)
            ax.legend(fontsize=18)
 
        filename = (
            f"./output/{save_folder}/fit_no_es_{config['label']}"
            f"_E0_{ASPECT_RATIO*E1:.2f}_E1_{E1:.2f}.pdf"
        )
        plt.savefig(filename)

def plot_cort_fit(cort_data, lam, alpha):
    time = np.linspace(0, T_FINAL_C, int(T_FINAL_C/5 + 1))
    cort_vel_pred = alpha * lam * time * np.exp(-lam*time)
    cort_vel = 1/R0_ALT * cort_data['cort_avg'].to_numpy()
    
    fig, ax = plt.subplots()
    ax.plot(time, cort_vel_pred, label = "model")
    ax.plot(time, cort_vel, label = "data")

    ax.set_title(f"Cortical Velocity Fit (alpha={alpha}, lambda={lam:.4f})")
    ax.legend()

    plt.savefig(f"./output/cort_fit_alpha_{alpha}_lam_{lam:.4f}.pdf")

def t_test(sample1: np.ndarray, sample2: np.ndarray):
    """
    Performs the t_test on two data samples for the null hypothesis that 2 independent samples
    have identical average (expected) values.
    """
    t_test_result = scipy.stats.ttest_ind(sample1, sample2)
    return t_test_result
