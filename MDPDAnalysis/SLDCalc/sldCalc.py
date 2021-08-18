##########################################################################################
 # @file sldCalc.py                                                 
 # @brief Code to calculate the SLD value from bead distribution of MDPD simulations.
 # requires .dat files for the substrate (E), A (fluorocarbon), B (linker), C (side chain), 
 # W (water). .dat files contain z index and last 3 simulation time steps of the data
##########################################################################################
 #
 # @author Theo Weinberger
 # @version 1.0
 # @date 2021-08-03
 # 
 # @copyright Copyright (c) 2021
 #
##########################################################################################

import pandas as pd 
import numpy as np

#load in data
substrate = pd.read_csv("E", header = None, sep = ' ')
fluorocarbon = pd.read_csv("A", header = None, sep = ' ')
linker = pd.read_csv("B", header = None, sep = ' ')
side = pd.read_csv("C", header = None, sep = ' ')
water = pd.read_csv("W", header = None, sep = ' ')

sldSubstrate = 3.665 #assuming a density of 2.20 g/cm^3 for Carbon - 7.334, or 3.665 for quartz assuming a density of 2.32 g/cm^3
sldFluorocarbon = 4.325 #assuming a density of 2.00 g/cm^3 for CF2
sldLinker = 4.331 #assuming a density of 2.00 g/cm^3 for OCF2C(CF3)FOCF2
sldSide = 1.916 #assuming a density of 2.00 g/cm^3 for CF2SO3H(H2O3)
sldWater = -0.561 #assuming a density of 1.00 g/cm^3 for H2O

#list of components and SLDs
substances = [substrate, fluorocarbon, linker, side, water]
substanceAverage = [None]*len(substances)
substanceSLD = [sldSubstrate, sldFluorocarbon, sldLinker, sldSide, sldWater]

#average larst three simulation cells
for i in range(len(substances)):
    substances[i][4] = (substances[i][3] + substances[i][2] + substances[i][1])/3

#create rolling average substance distributions to match up with MaxEnt resolution
for i in range(len(substances)):
    substanceAverage[i] = substances[i].rolling(4,min_periods=1).mean()

with open("EFinal", "w") as file:
    for i in range(len(substances[0])):
        file.write(str(substances[0][0][i]))
        file.write(" ")
        file.write(str(substances[0][4][i]))
        file.write(" ")
        file.write("\n")

with open("AFinal", "w") as file:
    for i in range(len(substances[1])):
        file.write(str(substances[1][0][i]))
        file.write(" ")
        file.write(str(substances[1][4][i]))
        file.write(" ")
        file.write("\n")

with open("BFinal", "w") as file:
    for i in range(len(substances[2])):
        file.write(str(substances[2][0][i]))
        file.write(" ")
        file.write(str(substances[2][4][i]))
        file.write(" ")
        file.write("\n")

with open("CFinal", "w") as file:
    for i in range(len(substances[3])):
        file.write(str(substances[3][0][i]))
        file.write(" ")
        file.write(str(substances[3][4][i]))
        file.write(" ")
        file.write("\n")

with open("WFinal", "w") as file:
    for i in range(len(substances[4])):
        file.write(str(substances[4][0][i]))
        file.write(" ")
        file.write(str(substances[4][4][i]))
        file.write(" ")
        file.write("\n")

with open("EAverage", "w") as file:
    for i in range(len(substanceAverage[0])):
        file.write(str(substanceAverage[0][0][i]))
        file.write(" ")
        file.write(str(substanceAverage[0][4][i]))
        file.write(" ")
        file.write("\n")

with open("AAverage", "w") as file:
    for i in range(len(substanceAverage[1])):
        file.write(str(substanceAverage[1][0][i]))
        file.write(" ")
        file.write(str(substanceAverage[1][4][i]))
        file.write(" ")
        file.write("\n")

with open("BAverage", "w") as file:
    for i in range(len(substanceAverage[2])):
        file.write(str(substanceAverage[2][0][i]))
        file.write(" ")
        file.write(str(substanceAverage[2][4][i]))
        file.write(" ")
        file.write("\n")

with open("CAverage", "w") as file:
    for i in range(len(substanceAverage[3])):
        file.write(str(substanceAverage[3][0][i]))
        file.write(" ")
        file.write(str(substanceAverage[3][4][i]))
        file.write(" ")
        file.write("\n")

with open("WAverage", "w") as file:
    for i in range(len(substanceAverage[4])):
        file.write(str(substanceAverage[4][0][i]))
        file.write(" ")
        file.write(str(substanceAverage[4][4][i]))
        file.write(" ")
        file.write("\n")


#Iteration one of calculation is for SLD calc
##############################################################################
#sum over all bead densities for volumetric normalisation
totalDensity = pd.DataFrame().reindex_like(water)
totalDensity = totalDensity.fillna(0)

for i in range(len(substances)):
    substances[i], water = substances[i].align(water, fill_value = 0)
    totalDensity += substances[i]

#reindex
totalDensity[0] = water[0]

#find max density for normalisation
maxDensity = totalDensity[4].max()

#renormalise system for volumetric normalisation
for i in range(len(substances)):
    substances[i][4] = substances[i][4]/maxDensity

#create SLD profile 
SLD = pd.DataFrame().reindex_like(water)
SLD = SLD.fillna(0)

for i in range(len(substances)):
    SLD += substances[i]*substanceSLD[i]

SLD[0] = water[0]

sldProfile  = pd.DataFrame(columns=[0,1])
sldProfile[0] = SLD[0]
sldProfile[1] = SLD[4]
######################################################################
######################################################################
#Iterations 2 is for rolling average calc, this is lazy coding but 
#as this is not going to be upscaled is passable
totalDensity = pd.DataFrame().reindex_like(water)
totalDensity = totalDensity.fillna(0)

for i in range(len(substances)):
    substanceAverage[i], water = substanceAverage[i].align(water, fill_value = 0)
    totalDensity += substanceAverage[i]

#reindex
totalDensity[0] = water[0]

#find max density for normalisation
maxDensity = totalDensity[4].max()

#renormalise system for volumetric normalisation
for i in range(len(substanceAverage)):
    substanceAverage[i][4] = substanceAverage[i][4]/maxDensity

#create SLD profile 
SLD = pd.DataFrame().reindex_like(water)
SLD = SLD.fillna(0)

for i in range(len(substances)):
    SLD += substanceAverage[i]*substanceSLD[i]

SLD[0] = water[0]

sldAverage  = pd.DataFrame(columns=[0,1])
sldAverage[0] = SLD[0]
sldAverage[1] = SLD[4]

sldRollingAverage = sldProfile.rolling(4,min_periods=1).mean()

with open("total", "w") as file:
    for i in range(len(totalDensity)):
        file.write(str(totalDensity[0][i]))
        file.write(" ")
        file.write(str(totalDensity[4][i]))
        file.write(" ")
        file.write("\n")


with open("sld", "w") as file:
    for i in range(len(sldProfile)):
        file.write(str(sldProfile[0][i]))
        file.write(" ")
        file.write(str(sldProfile[1][i]))
        file.write(" ")
        file.write("\n")

with open("sldAverage", "w") as file:
    for i in range(len(sldAverage)):
        file.write(str(sldAverage[0][i]))
        file.write(" ")
        file.write(str(sldAverage[1][i]))
        file.write(" ")
        file.write("\n")

with open("sldRollingAverage", "w") as file:
    for i in range(len(sldRollingAverage)):
        file.write(str(sldRollingAverage[0][i]))
        file.write(" ")
        file.write(str(sldRollingAverage[1][i]))
        file.write(" ")
        file.write("\n")

