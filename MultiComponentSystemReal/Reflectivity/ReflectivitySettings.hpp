/**
 * @file ReflectivitySettings.hpp
 * @author Theo Weinberger
 * @brief Functions to read in data from a configurations file to create and SLD profile from which the reflectivity 
 * can be approximated by the Fourier approximation
 * @version 0.1
 * @date 2021-05-12
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef REFLECTIVITYSETTINGS_H
#define REFLECTIVITYSETTINGS_H

/**
 * @brief Function to read in data that is to be stored in an armadillo vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 * @param numSubstances Number of different substances in this system
 */
void ReadVectorDouble(std::vector<double>& myVector, const libconfig::Setting& mySetting, const int& numSubstances);

/**
 * @brief Function to read in data that is to be stored in an armadillo vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 * @param numSubstances Number of different substances in this system
 */
void ReadVectorDouble(std::vector<arma::vec>& myVector, const libconfig::Setting& mySetting, const int& numSubstances);
/**
 * @brief Function to read in data that is to be stored in an armadillo vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 * @param numSubstances Number of different substances in this system
 */
void ReadVectorInt(std::vector<arma::vec>& myVector, const libconfig::Setting& mySetting, const int& numSubstances);

/**
 * @brief Function to read in data from a configurations file. This function reads in arrays containing the
 * depths of each layer and the SLDs of each layer. From this the full SLD profile can be constructed.
 * 
 * @param fileName The name of the configurations file containing the data
 * @param numSubstances Number of different substances in this system
 * @param depths Vector containing thickness of each layer
 * @param sldVal Value of the SLD of this substance
 * @param n Vector containing the amount of substance of each layer
 * @param total Total amount of this substance
 * @param substrateSLD The value of the SLD of the substrate
 * @param lengthSubstrate The length of the substrate in index units
 * @param volumetricNormalisation Boolean to determine whether to use volumetric normalisation or not
 * @param toyModel Boolean to determine whether the model being studied is a 'toy' model
 * @param boundSLD Boolean variable stating whether or not SLD scaling to known max and min bounds is being used
 * @param sldMaxBound The max bound of the SLD if known
 * @param sldMinBound The min bound of the SLD if known
 * @param qOffset The q space offset of the reflectivity
 * @return int Exit code
 */
int ReadFile(const std::string& fileName, int& numSubstances, std::vector<arma::vec>& depths, std::vector<double>& sldVal, std::vector<arma::vec>& n, std::vector<double>& total, double& substrateSLD, int& lengthSubstrate, bool& volumetricNormalisation, bool& toyModel, bool& boundSLD, double& sldMinBound, double& sldMaxBound, int& qOffset);

#endif
