# %% Libraries and functions
import sys
sys.path.append('G:/My Drive/PhD/CAPTUR3D_personal/03 Software e utilities/v0.2.0/Libraries')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import flim_module as flim
import seaborn as sns
import sklearn
import sklearn.model_selection
import os

##

# load data
# g,s,intensity, filename = flim.decode.import_R64()
# gnew, snew=flim.IntensityThresholdCorrection(g, s, intensity, 
                                             # threshold_low=30)
                                             
# draw phasor plot
# flim.phasorplot(gnew, snew)
# flim.decode.savefigure()

# show intensity image
# flim.imshow(intensity, cmap='gray', vmin=0, vmax=30)

# plt.imshow(intensity[0,:,:], cmap='gray', interpolation='none', 
#             vmin=0, vmax=np.max(intensity))
# plt.axis('off')
# flim.decode.savefigure(extension='svg')
# plt.colorbar(location='left')


