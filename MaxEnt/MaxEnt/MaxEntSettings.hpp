/**
 * @file MaxEntSettings.hpp
 * @author Theo Weinberger
 * @brief This header file defines all the relevant functions for reading in the configuration data to perform MaxEnt method
 * @version 2.0
 * @date 2021-05-01
 * 
 * @copyright Copyright (c) 2021
 * 
 */

/**
 * @brief Method to read file settings into MaxEnt simulation
 * 
 * @param fileName String data containing the name of the file containing configuration data.
 * @param totalIterations Total number of search iterations
 * @param maximumSearchIter The maximum number of potential lagrange multipliers to be recorded
 * @param numBasisVectors Number of basis vectors to be used in the search
 * @param zeroLevel Minimum value that the matrix is allowed to contain otherwise it is set to \a zeroLevel
 * @param minVar Minimum value that the variance of the image is allowed to take, below this threshold it is set to \a minVar
 * @param dataScale Value by which reflectivity data is scaled, have found a value of 10.0 tends to work quite well
 * @param sldMaxBound Maximum bound for the SLD profile if known (only used for scaling)
 * @param sldMinBound Minimum bound of the SLD profile if known (only used for scaling)
 * @param propagationSLD The SLD of the propogation region of the neutrons (typically air with a value of 0.0)
 * @param substrateSLD The SLD of the substrate region 
 * @param lengthPropagation The length (in vector indeces) of the region of air propagation that should be fixed
 * @param lengthSubstrate The length (in vector indeces) of the region of the substrate that should be fixed
 * @param smoothProfile Use the smooth incrementation of the profile generator which can help produce more physical profiles
 * @param sldScaling Scale the SLD profile to the known max and min values
 * @param useEdgeConstraints Use the constraints on the edges of the SLD profile to constrain the system, this required sldScaling = True
 * @param chiSquaredScale Linear scaling the chisquared gradients to weight the fitting to change relative weight of the data fitting vs the entropy maximisation in the Langrangian
 * @param spikeCharge Spike the initial charge distribution near the substrate to encourage SLD formation in that region
 * @param spikePortion Fraction of charge distribution that is spiked (measured from the tail of the charge array)
 * @param spikeAmount Amount by which charge distribution is spiked - note this is relative to an intial distribution which is random uniform in the range [0:1]
 * @return int Exit Code
 */
int ReadFile(const std::string& fileName, int& totalIterations, int& maximumSearchIter, int& numBasisVectors, double& zeroLevel, double& minVar, double& dataScale, double& sldMaxBound, double& sldMinBound, double& propagationSLD, double& substrateSLD, int& lengthPropagation, int& lengthSubstrate, bool& smoothProfile, bool& sldScaling, bool& useEdgeConstraints, double& chiSquaredScale, bool& spikeCharge, double& spikePortion, double& spikeAmount);

