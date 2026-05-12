import numpy as np

T_FINAL = 195
# P2 = np.array([-1.1660254, 0, 0]) # 0 degrees
#P2 = np.array([-1.05, 0, -0.433012702]) # 30 degrees
P2 = np.array([-0.75, 0, -0.33]) # eye balled in Desmos
#P2 = np.array([-0.91237244, 0, -0.612372436]) # 45 degrees

#EMS = np.array([-0.91237244, 0, -0.612372436]) # 45 degrees

# E0 is the length of the major axis and E1 is the length of the other two (minor) axes
#E0, E1 = (1.8, 0.87)
#E0, E1 = (2.6, 1.7)  # actual ratios, from Cao, J., Guan, G., Ho, V.W.S. et al. Establishment of a morphological atlas of the Caenorhabditis elegans embryo using deep-learning-based 4D segmentation. Nat Commun 11, 6254 (2020). https://doi.org/10.1038/s41467-020-19863-x
# All 17 embryos with segmented cell morphologies were embodied by a unified cylindroid, approximately with a height of 18 μm, a semimajor axis of 27 μm and a semiminor axis of 18 μm
E0, E1 = (1.3, 0.85)

# The additional length of the mitotic axis at the end (t=195) (total final length is 1+ this)
#FINAL_SPRING_LENGTH = 0.2
FINAL_SPRING_LENGTH = 0.1

# The time constant in the cortical flow function
#LAM = 0.014666  # Marcus' fit value
#LAM = 0.002  # Marcus' fit value
LAM = 0.008  # Marcus' fit value
