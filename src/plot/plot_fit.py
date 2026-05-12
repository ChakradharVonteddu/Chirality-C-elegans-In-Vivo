import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy

from modelling.models.model_config import E0, E1, FINAL_SPRING_LENGTH, LAM


def plot_fit(data, angles):
    """
    Plot model fit against raw data.
    """
    t = data["t"].to_numpy()

    da_average = data["ABa_dorsal_avg"].to_numpy()
    dp_average = data["ABp_dorsal_avg"].to_numpy()
    aa_average = data["ABa_ant_avg"].to_numpy()
    ap_average = data["ABp_ant_avg"].to_numpy()

    # get standard error of the mean
    da_std = data["ABa_dorsal_stdeofmean"].to_numpy()
    dp_std = data["ABp_dorsal_stdeofmean"].to_numpy()
    aa_std = data["ABa_ant_stdeofmean"].to_numpy()
    ap_std = data["ABp_ant_stdeofmean"].to_numpy()

    # load data
    computed_ABa_dorsal = angles["ABa_dorsal"].to_numpy()
    computed_ABp_dorsal = angles["ABp_dorsal"].to_numpy()
    computed_ABa_ant = angles["ABa_ant"].to_numpy()
    computed_ABp_ant = angles["ABp_ant"].to_numpy()

    # t-test

    # boolean array; True if t test passes, False if not
    dorsal_t_test = data["dorsal_t_test"].to_numpy()
    anterior_t_test = data["ant_t_test"].to_numpy()

    # True if not pass, False if pass
    dorsal_t_ntest = ~np.array(dorsal_t_test)
    anterior_t_ntest = ~np.array(anterior_t_test)

    # initialize graph, set Axes titles
    fig, (axD, axA) = plt.subplots(2)
    plt.subplots_adjust(hspace=0.35)
    fig.set_figheight(7)
    fig.set_figwidth(15)
    axD.set_title("Dorsal-view angles", fontsize=20)
    axA.set_title("Anterior-view angles", fontsize=20)

    dorsal_t_passed = list(itertools.compress(t, dorsal_t_test))
    dorsal_t_npassed = list(itertools.compress(t, dorsal_t_ntest))
    anterior_t_passed = list(itertools.compress(t, anterior_t_test))
    anterior_t_npassed = list(itertools.compress(t, anterior_t_ntest))

    da_average_passed = list(itertools.compress(da_average, dorsal_t_test))
    dp_average_passed = list(itertools.compress(dp_average, dorsal_t_test))
    da_average_npassed = list(itertools.compress(da_average, dorsal_t_ntest))
    dp_average_npassed = list(itertools.compress(dp_average, dorsal_t_ntest))

    aa_average_passed = list(itertools.compress(aa_average, anterior_t_test))
    ap_average_passed = list(itertools.compress(ap_average, anterior_t_test))
    aa_average_npassed = list(itertools.compress(aa_average, anterior_t_ntest))
    ap_average_npassed = list(itertools.compress(ap_average, anterior_t_ntest))

    # plot data
    axD.plot(t, computed_ABa_dorsal, label="ABa model", markersize=2, color="blue")
    # axD.plot(dorsal_t_passed, da_average_passed, "o", label="ABa data - passed", markersize=4, color="blue")
    axD.plot(
        dorsal_t_passed,
        da_average_passed,
        "o",
        markersize=4,
        color="none",
        markeredgecolor="blue",
    )
    axD.plot(
        dorsal_t_npassed,
        da_average_npassed,
        "o",
        label="ABa data (shading is CI)",
        markersize=4,
        color="none",
        markeredgecolor="blue",
    )

    axD.plot(t, computed_ABp_dorsal, label="ABp model", markersize=2, color="red")
    # axD.plot(dorsal_t_passed, dp_average_passed, "o", label="ABp_dorsal data - passed", markersize=4, color="red")
    axD.plot(
        dorsal_t_passed,
        dp_average_passed,
        "o",
        markersize=4,
        color="none",
        markeredgecolor="red",
    )
    axD.plot(
        dorsal_t_npassed,
        dp_average_npassed,
        "o",
        label="ABp data (shading is CI)",
        markersize=4,
        color="none",
        markeredgecolor="red",
    )

    axD.tick_params(axis="both", labelsize=18)

    axA.plot(t, computed_ABa_ant, label="ABa model", markersize=2, color="blue")
    # axA.plot(anterior_t_passed, aa_average_passed, "o", label="ABa_ant data - passed", markersize=4, color="blue")
    axA.plot(
        anterior_t_passed,
        aa_average_passed,
        "o",
        markersize=4,
        color="none",
        markeredgecolor="blue",
    )
    axA.plot(
        anterior_t_npassed,
        aa_average_npassed,
        "o",
        label="ABa data (shading is CI)",
        markersize=4,
        color="none",
        markeredgecolor="blue",
    )

    axA.plot(t, computed_ABp_ant, label="ABp model", markersize=2, color="red")
    # axA.plot(anterior_t_passed, ap_average_passed, "o", label="ABp_ant data - passed", markersize=4, color="red")
    axA.plot(
        anterior_t_passed,
        ap_average_passed,
        "o",
        markersize=4,
        color="none",
        markeredgecolor="red",
    )
    axA.plot(
        anterior_t_npassed,
        ap_average_npassed,
        "o",
        label="ABp data (shading is CI)",
        markersize=4,
        color="none",
        markeredgecolor="red",
    )

    axA.axhline(y=0, color="grey", linestyle="--", linewidth=1)

    axA.tick_params(axis="both", labelsize=20)

    axA.set_ylim(-40, 40)

    # plot confidence bands
    axD.fill_between(
        t, da_average - da_std, da_average + da_std, alpha=0.2, color="blue"
    )
    axD.fill_between(
        t, dp_average - dp_std, dp_average + dp_std, alpha=0.2, color="red"
    )
    axA.fill_between(
        t, aa_average - aa_std, aa_average + aa_std, alpha=0.2, color="blue"
    )
    axA.fill_between(
        t, ap_average - ap_std, ap_average + ap_std, alpha=0.2, color="red"
    )

    # plot legend
    axD.legend(fontsize=18)
    # axA.legend(fontsize=24)

    # save figure
    #    filename = f"./output/fit_E0_{E0:.2f}_E1_{E1:.2f}_FSL_{FINAL_SPRING_LENGTH:.2f}_LAM_{LAM:.3f}.png"
    filename = f"./output/fit_E0_{E0:.2f}_E1_{E1:.2f}_FSL_{FINAL_SPRING_LENGTH:.2f}_LAM_{LAM:.3f}.pdf"
    plt.savefig(filename)


def t_test(sample1: np.ndarray, sample2: np.ndarray):
    """
    Performs the t_test on two data samples for the null hypothesis that 2 independent samples
    have identical average (expected) values.
    """
    t_test_result = scipy.stats.ttest_ind(sample1, sample2)
    return t_test_result
