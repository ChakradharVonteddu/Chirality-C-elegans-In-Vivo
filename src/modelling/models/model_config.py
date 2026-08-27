import numpy as np
import pandas as pd

T_FINAL_ES = 195
T_FINAL_NO_ES = 238

#T_START_NO_ES = 0
#T_END_NO_ES = 360

burn_in = False
# P2 = np.array([-1.1660254, 0, 0]) # 0 degrees
#P2 = np.array([-1.05, 0, -0.433012702]) # 30 degrees
#P2_0 = np.array([-0.75, 0, -0.33]) # eye balled in Desmos
#P2 = np.array([-0.91237244, 0, -0.612372436]) # 45 degrees

#EMS = np.array([-0.91237244, 0, -0.612372436]) # 45 degrees

# E0 is the length of the major axis and E1 is the length of the other two (minor) axes
#E0, E1 = (1.8, 0.87)
#E0, E1 = (2.6, 1.7)  # actual ratios, from Cao, J., Guan, G., Ho, V.W.S. et al. Establishment of a morphological atlas of the Caenorhabditis elegans embryo using deep-learning-based 4D segmentation. Nat Commun 11, 6254 (2020). https://doi.org/10.1038/s41467-020-19863-x
# All 17 embryos with segmented cell morphologies were embodied by a unified cylindroid, approximately with a height of 18 μm, a semimajor axis of 27 μm and a semiminor axis of 18 μm
T_FINAL_C = 250

R0 = 1 #The initial radius of an ABal/ABar/ABpr/ABpl cell is approx 8.95 micro meters.
R0_ALT = 8.95 #R0 in micro meters

ES_TIMES = pd.read_excel("./src/data/data_stat.xlsx", sheet_name = "ES")["t"].to_numpy()/T_FINAL_ES #non-dimensionalized eggshell times
NO_ES_TIMES = pd.read_excel("./src/data/data_stat.xlsx", sheet_name = "NO_ES")["t"].to_numpy()/T_FINAL_ES #non-dimensionalized no eggshell times

ASPECT_RATIO = 1.53
#E0, E1 = (2.6, 1.7) #to be used during burn-in phase

ETA = 0 #Initial tilt

ALPHA = 0.003496
LAM = 0.0152

#params[0] - spring constant/gamma (1/s), params[1] - frictional constant/gamma (unitless), params[2] - E1 (in R0), params[3] - d1 (in R0), params[4] - d2_es (in R0), params[5] - adhesion constant (unitless), params[6] - d2_no_es (in R0), params[7] - t0 (in units of 195s) 
param_loc = {"spring_constant" : 0, "frictional_constant" : 1, "E1" : 2, "d1" : 3, "d2_es" : 4, "adhesion_constant" : 5, "d2_no_es" : 6, "t0" : 7}
              
time_configs = [{"label": "7s","t_min": 0, "t_max": 238, "interval": 7}]
        #{"label": "15s", "t_min" : 0, "t_max": 360, "interval": 15}
 
NUM_EMS = 4

cell_names = ["ABal","ABar","ABpr","ABpl","p2","ems_a","ems_b","ems_c","ems_d"]
#cell_names = ["ABal","ABar","ABpr","ABpl","p2","ems_a"]
#cell_names = ["ABpr","ABpl","p2"]

# The time constant in the cortical flow function
##LAM = 0.014666  # Marcus' fit value
##LAM = 0.002  # Marcus' fit value
#LAM = 0.0107  # Marcus' fit value

modifiers_w_es = {
"include_shell" : True,
"include_p2" : True,
"include_shell_friction" : False,
"include_ems" : True}

modifiers_wo_es = {
"include_shell" : False,
"include_p2" : True,
"include_shell_friction" : False,
"include_ems" : True}