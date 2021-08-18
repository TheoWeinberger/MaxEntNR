##########################################################################################
 # @file average.py                                                 
 # @brief Code to perform basic data analysis. The standard deviation and mean values 
 # are calculared for the reduced chi squared, minimum reduced chi squared, test value
 # and minimum test value and appended to the averages file. Input data should be in 
 # convData file
##########################################################################################
 #
 # @author Theo Weinberger
 # @version 1.0
 # @date 2021-07-09
 # 
 # @copyright Copyright (c) 2021
 #
##########################################################################################
import pandas as pd 
import numpy as np

#load in data
data = pd.read_csv("convData", header = None, sep = ' ')

#calculate standard deviations and mean
means = data.mean(axis = 0)
standardDev = data.std(axis = 0)

#output data to file in order, redChi (mean, standard deviation), minRedChi (mean, standard deviation), test (mean, standard deviation), minTest (mean, standard deviation)
with open("averages", "a") as file:
    for i in range(len(means)):
        file.write(str(means[i]))
        file.write(" ")
        file.write(str(standardDev[i]))
        file.write(" ")
    file.write("\n")
