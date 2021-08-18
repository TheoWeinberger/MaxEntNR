/**
 * @file MaxEntSettings.hpp
 * @author Theo Weinberger
 * @brief This header file defines all the relevant functions for reading in the configuration data to perform MaxEnt method
 * @version 3.0
 * @date 2021-07-01
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef MAXENTSETTINGS_H
#define MAXENTSETTINGS_H

/**
 * @brief Function to read in data that is to be stored in an armadillo vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 * @param numSubstances Number of different substances in this system
 */
void ReadVector(arma::vec& myVector, const libconfig::Setting& mySetting, const int& numSubstances);

/**
 * @brief Function to read in data that is to be stored in a vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 * @param numSubstances Number of different substances in this system
 */
void ReadVector(std::vector<int>& myVector, const libconfig::Setting& mySetting, const int& numSubstances);

/**
 * @brief Function to read in data that is to be stored in a vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 * @param numSubstances Number of different substances in this system
 */
void ReadVector(std::vector<double>& myVector, const libconfig::Setting& mySetting, const int& numSubstances);

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
 * @param numSubstances Number of different substances in this system
 * @param propagationSLD The SLD of the propogation region of the neutrons (typically air with a value of 0.0)
 * @param substrateSLD The SLD of the substrate region 
 * @param lengthPropagation The length (in vector indeces) of the region of air propagation that should be fixed
 * @param lengthSubstrate The length (in vector indeces) of the region of the substrate that should be fixed
 * @param smoothIncrement Use the smooth incrementation of the profile generator which can help produce more physical profiles
 * @param useEdgeConstraints Use the constraints on the edges of the SLD profile to constrain the system, this required sldScaling = True
 * @param chiSquaredScale Linear scaling the chisquared gradients to weight the fitting to change relative weight of the data fitting vs the entropy maximisation in the Langrangian
 * @param spikeCharge Spike the initial charge distribution near the substrate to encourage SLD formation in that region
 * @param spikePortion Fraction of charge distribution that is spiked (measured from the tail of the charge array)
 * @param spikeAmount Amount by which charge distribution is spiked - note this is relative to an intial distribution which is random uniform in the range [0:1]
 * @param sldVal The SLD of this material
 * @param total The total amount of substance in the whole system (used for normalisation)
 * @param toyModel Boolean to determine whether the model being studied is a 'toy' model
 * @param volumetricNormalisation Boolean to determine whether to use volumetric normalisation or not
 * @param error Boolean variable determining whether input data has error values
 * @param boundSLD Boolean variable stating whether or not SLD scaling to known max and min bounds is being used
 * @param sldMaxBound The max bound of the SLD if known
 * @param sldMinBound The min bound of the SLD if known
 * @param smoothProfile Use this to recalculate the charges as N_i = 1/4 N_i-1 + 1/2 N_i + 1/4 N_i+1 every smoothInterval steps
 * @param smoothInterval The number of steps inbetween smoothing operations
 * @param forceZero Forces values below a fraction of the max charge to be set to 0
 * @param fracMax The fraction of the maximum charge for a given species that defined the cutoff where below this cutoff the value will be forced to 0
 * @param chargeScale value by which the charge total is scaled
 * @param forceInteral Interval at which _forceZero is applied
 * @param qOffset The parameter offset at which the data starts being read
 * @param qCutOff The index at which the data is then zeroed
 * @param initChargeSpecific Boolean to determine which type of charge initialisation to use
 * @param noise Boolean to determine whether to add noise to the specific charge distribution
 * @param lengthPropagation The length (in real space) of the region of air propagation that should be fixed
 * @param lengthSubstrate The length (in real space) of the region of the substrate that should be fixed
 * @param initLayerStart Vector containing the index of the start of the initial charge distributions
 * @param realWorldScaling Whether real world scaling is sued in the system
 * @param initLayerStop Vector containing the index of the end of the initial charge distributions
 * @param qOffsetReal The parameter offset at which the data starts being read in real world units
 * @param qCutOffReal The index at which the data is then zeroed in real world units
 * @param initLayerStartReal Vector containing the value of the start of the initial charge distributions
 * @param initLayerStopReal Vector containing the value of the end of the initial charge distributions
 * @return int Exit Code
 */
int ReadFile(const std::string& fileName, int& totalIterations, int& maximumSearchIter, int& numBasisVectors, double& zeroLevel, double& minVar, double& dataScale, int& numSubstances, double& propagationSLD, double& substrateSLD, int& lengthPropagation, int& lengthSubstrate, bool& smoothIncrement, bool& useEdgeConstraints, double& chiSquaredScale, bool& spikeCharge, double& spikePortion, double& spikeAmount, arma::vec& sldVal, arma::vec& total, bool& toyModel, bool& volumetricNormalisation, bool& error, bool& boundSLD, double& sldMinBound, double& sldMaxBound, bool& smoothProfile, int& smoothInterval, bool& forceZero, double& fracMax, double& chargeScale, int& forceInterval, int& qOffset, int& qCutOff, bool& initChargeSpecific, bool& noise, std::vector<int>& initLayerStart, std::vector<int>& initLayerStop, double& lengthPropagationReal, double& lengthSubstrateReal, double& qOffsetReal, double& qCutOffReal, std::vector<double>& initLayerStartReal, std::vector<double>& initLayerStopReal, bool& realWorldScaling);

#endif