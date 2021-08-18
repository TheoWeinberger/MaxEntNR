/**
 * @file NewtonSettings.hpp
 * @author Theo Weinberger
 * @brief Header file fo NewtonSettings which contains all the relevant functions for reading in the configuration data to perform Newton fitting method
 * @version 1.0
 * @date 2021-06-30
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef NEWTONSETTINGS_HPP
#define NEWTONSETTINGS_HPP

/**
 * @brief Function to read in data that is to be stored in an armadillo vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 * @param numSubstances Number of different substances in this system
 */
void ReadVector(arma::vec& myVector, const libconfig::Setting& mySetting, const int& numSubstances);

/**
 * @brief Method to read file settings into MaxEnt simulation
 * 
 * @param fileName String data containing the name of the file containing configuration data.
 * @param totalIterations Total number of search iterations
 * @param zeroLevel Minimum value that the matrix is allowed to contain otherwise it is set to \a zeroLevel
 * @param minVar Minimum value that the variance of the image is allowed to take, below this threshold it is set to \a minVar
 * @param dataScale Value by which reflectivity data is scaled, have found a value of 10.0 tends to work quite well
 * @param numSubstances Number of different substances in this system
 * @param propagationSLD The SLD of the propogation region of the neutrons (typically air with a value of 0.0)
 * @param substrateSLD The SLD of the substrate region 
 * @param lengthPropagation The length (in vector indeces) of the region of air propagation that should be fixed
 * @param lengthSubstrate The length (in vector indeces) of the region of the substrate that should be fixed
 * @param smoothIncrement Use the smooth incrementation of the profile generator which can help produce more physical profiles
 * @param useEdgeConstraints Use the constraints on the edges of the SLD profile to constrain the system, this required sldScaling = True
 * @param sldVal The SLD of this material
 * @param useDamping Use gradient scaling to produce better convergence, implemented via Armijo backtracing
 * @param alphaInit Initial value of alpha in Armijo backtracing
 * @param alphaFactor Scaling factor for alpha used in backtracing
 * @param gammaInit Initial value of gamma used in Armijo backtracing
 * @param gammaFactor Scaling factor for gamma used in backtracing
 * @param smoothProfile Use this to recalculate the charges as N_i = 1/4 N_i-1 + 1/2 N_i + 1/4 N_i+1 every smoothInterval steps
 * @param smoothInterval The number of steps inbetween smoothing operations
 * @param forceZero Forces values below a fraction of the max charge to be set to 0
 * @param fracMax The fraction of the maximum charge for a given species that defined the cutoff where below this cutoff the value will be forced to 0
 * @param chargeScale value by which the charge total is scaled
 * @param forceInterval Interval at which _forceZero is applied
 * @param volumetricNormalisation Boolean to determine whether to use volumetric normalisation or not
 * @param boundSLD Boolean variable stating whether or not SLD scaling to known max and min bounds is being used
 * @param sldMaxBound The max bound of the SLD if known
 * @param sldMinBound The min bound of the SLD if known
 * @param qOffset The q index where the data is read from
 * @param qCutOff the q index where the data is cropped
 * @param error whether dataset contains real errors
 * @return int Exit Code
 */
int ReadFile(const std::string& fileName, int& totalIterations, double& zeroLevel, double& minVar, double& dataScale, int& numSubstances, double& propagationSLD, double& substrateSLD, int& lengthPropagation, int& lengthSubstrate, bool& smoothIncrement, bool& useEdgeConstraints,  arma::vec& sldVal, bool& useDamping, double& alphaInit, double& alphaFactor, double& gammaInit, double& gammaFactor, bool& forceZero, double& fracMax, int& forceInterval, bool& smoothProfile, int& smoothInterval, bool& volumetricNormalisation, bool& boundSLD, double& sldMaxBound, double& sldMinBound, int& qOffset, int& qCutOff, bool& error);

#endif