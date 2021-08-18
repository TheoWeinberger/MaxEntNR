##########################################################################################
 # @file rescale.py                                                 
 # @brief Code to rescale the reflectivity data from the MDPD simulations to match up with 
 # ISIS data and MaxEnt reconstructions 
##########################################################################################
 #
 # @author Theo Weinberger
 # @version 0.1
 # @date 2021-08-03
 # 
 # @copyright Copyright (c) 2021
 #
##########################################################################################

import pandas as pd 
import numpy as np

#load in data
#can change file name here
reflectivity = pd.read_csv("reflectivityl9Average", header = None, sep = '   ')
reflectivity[2] = (reflectivity[0]*2*np.pi)/(10*len(reflectivity))

#get normalisation value where Q value for normalisation must be read off from data
normValue = 2.53405758793970e-02 #set this 
data = pd.read_csv("data", header = None, sep = '   ')
normalisation = data.iloc[(data[0] - normValue).abs().argsort()[:1]][1].astype(np.float64).values[0]

#determine renormalisation
renormalisation = reflectivity.iloc[(reflectivity[2] - normValue).abs().argsort()[:1]][1].astype(np.float64).values[0]
renormalisation = normalisation/renormalisation

#renormalisation factor
reflectivity[1] = reflectivity[1]*renormalisation

#output data
with open("reflectivityScaled", "w") as file:
    for i in range(len(reflectivity)):
        file.write(str(reflectivity[2][i]))
        file.write(" ")
        file.write(str(reflectivity[1][i]))
        file.write(" ")
        file.write("\n")