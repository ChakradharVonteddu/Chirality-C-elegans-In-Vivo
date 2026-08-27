import pandas as pd
import numpy as np
from scipy.interpolate import make_smoothing_spline, BSpline
from scipy.integrate import quad
from scipy.stats import ttest_1samp, ttest_ind, mannwhitneyu, pearsonr, levene, wilcoxon
from itertools import combinations, permutations
import random

def fit_spline(angle_data, groups):
    splines = {group: [] for group in groups}
    grouped = angle_data.groupby(['condition','interval','embryo_id','cell'])
    for (condition,interval,embryo_id,cell),group in grouped:
        fit_data = group.sort_values('time')
        t = fit_data["time"].to_numpy()
        y = np.absolute(fit_data['angle'].to_numpy() - fit_data['angle'].iloc[0]) #get absolute dorsal tilt
        #Fit a smoothing spline to the data
        spl = make_smoothing_spline(t, y, lam = None)
        splines[f"{condition}_{interval}s"].append(spl)
    return splines

def permutation_test(splines, groups, T, n_permutations = 1000):

   #function to evaluate the squared difference between mean-curves of two groups at a given time
   def eval_diff_sq(t, spA, spB):
        mean_A = np.mean([s(t) for s in spA])
        mean_B = np.mean([s(t) for s in spB])
        return (mean_A - mean_B)**2
   
   p_values = {}
   for groupA, groupB in combinations(groups,2):
        splinesA = splines[groupA]
        splinesB = splines[groupB]
        t_max = min(T[groupA],T[groupB])
        test_stat = quad(lambda t: eval_diff_sq(t, splinesA, splinesB), 0, t_max, limit = 500)[0] #integrate the squared difference over time interval to get test statistic

        #perform permutation test
        combined_splines = np.array(splinesA + splinesB)
        nA = len(splinesA)
        perm_stats = []
        for _ in range(n_permutations):
            shuffled_idx = np.random.permutation(len(combined_splines))
            permA = combined_splines[shuffled_idx[:nA]]
            permB = combined_splines[shuffled_idx[nA:]]
            perm_stat = quad(lambda t: eval_diff_sq(t, permA, permB), 0, t_max, limit = 500)[0] #integrate the squared difference of permuted groups to obtain empirical distribution
            perm_stats.append(perm_stat)
        #calculate p-value based on the number of permutations that have a test statistic greater than or equal to the observed test statistic
        p_value = np.sum(np.array(perm_stats) >= test_stat) / n_permutations
        p_values[(groupA, groupB)] = p_value
        print(f"Permutation test between {groupA} and {groupB}: p-value = {p_value}")

   return p_values
 
def t_test(angle_data):
    grouped = angle_data.groupby('embryo_id')
    angle_changes = []
    for _, group in grouped:
        data = group.sort_values('time')
        angle_changes.append(data['angle_avg'].iloc[-1] - data['angle_avg'].iloc[0])
    res = ttest_1samp(angle_changes, 0)
    print(f"Mean: {np.mean(angle_changes):.2f}, t_stat: {res.statistic:.4f}, p_value: {res.pvalue:.4e}")
    return angle_changes

def wilcoxon_signed_rank_test(a,hyp_med):
    res = wilcoxon(np.array(a) - hyp_med)
    print(f"test_stat: {res.statistic:.4f}, p_value: {res.pvalue:.4f}")

def welch_t_test(a, b):
    res = ttest_ind(a, b, equal_var = False)
    print(f"test_stat: {res.statistic:.4f}, p_value: {res.pvalue:.4f}")

def mannwhitneyu_test(a,b):
    res = mannwhitneyu(a,b)
    print(f"test_stat: {res.statistic:.4f}, p_value: {res.pvalue:.4f}")

def levene_test(arr_list):
    res = levene(*arr_list, center = 'median')
    print(f"test_stat: {res.statistic:.4f}, p_value: {res.pvalue:.4f}")

def within_corr(diff_data):
    corr_df = diff_data.groupby(['condition','interval','embryo_id']).apply(lambda g: pearsonr(g['ABx1_diff'],g['ABx2_diff']).statistic, include_groups = False).reset_index(name = 'r')
    corr_df['r_transformed'] = np.arctanh(corr_df['r'].to_numpy())
    corr_agg = corr_df.groupby(['condition','interval']).apply(lambda g: np.tanh(np.mean(g['r_transformed'])), include_groups = False).reset_index(name = 'within_pair_r')
    return corr_agg

def cross_corr(diff_data):
    results = []
    diff_data.loc[:,'emb_angle_diff'] = (diff_data.loc[:,'ABx1_diff'] + diff_data.loc[:,'ABx2_diff'])/2
    for (condition, interval), groupdf in diff_data.groupby(['condition','interval']):
        embryos = groupdf['embryo_id'].unique()
        cross_rs = []

        for emb_i, emb_j in combinations(embryos, 2):
            ser_i = groupdf.loc[groupdf['embryo_id'] == emb_i].sort_values('time')['emb_angle_diff']
            ser_j = groupdf.loc[groupdf['embryo_id'] == emb_j].sort_values('time')['emb_angle_diff']
            cross_rs.append(pearsonr(ser_i.to_numpy(),ser_j.to_numpy()).statistic)

        cross_r_trans = np.arctanh(np.array(cross_rs))
        cross_r_agg = np.tanh(np.mean(cross_r_trans))
        #cross_r_agg = np.mean(cross_rs)

        results.append({"condition" : condition, "interval" : interval, "cross_pair_r" : cross_r_agg})

    return pd.DataFrame(results)


