/**
 * @file DataMaskSettings.hpp
 * @author Theo Weinberger
 * @brief Settings file for masking functions
 * @version 1.0
 * @date 2021-06-29
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef DATAMASKSETTINGS_HPP
#define DATAMASKSETTINGS_HPP

/**
 * @brief Function to read in data that is to be stored in an armadillo vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 * @param numGauss The number of gaussians used to mask the system
 */
void ReadVector(arma::vec& myVector, const libconfig::Setting& mySetting, const int& numGauss);

/**
 * @brief Method to read file settings into MaxEnt simulation
 * 
 * @param fileName String data containing the name of the file containing configuration data.
 * @param qOffset The q index offset from which the data is outputted
 * @param lowFreqMask Boolean determining if low frequency mask is used
 * @param highFreqMask Boolean determining if high frequency mask is used
 * @param userDefMask Boolean determining if user defined mask is used
 * @param sigmas The standard deviations of the user def gaussians
 * @param means The means of the user def gaussians
 * @param numGauss The number of gaussians in the system
 * @return int Exit Code
 */
int ReadFile(const std::string& fileName, int& qOffset, bool& lowFreqMask, bool& highFreqMask, bool& userDefMask, arma::vec& sigmas, arma::vec& means, int& numGauss);
#endif