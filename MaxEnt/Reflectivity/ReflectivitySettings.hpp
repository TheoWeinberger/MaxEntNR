/**
 * @file ReflectivitySettings.hpp
 * @author Theo Weinberger
 * @brief Functions to read in data from a configurations file to create and SLD profile from which the reflectivity 
 * can be approximated by the Fourier approximation
 * @version 2.0
 * @date 2021-04-16
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
 */
void ReadVectorDouble(arma::vec& myVector, const libconfig::Setting& mySetting);

/**
 * @brief Function to read in data that is to be stored in an armadillo vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 */
void ReadVectorInt(arma::vec& myVector, const libconfig::Setting& mySetting);

/**
 * @brief Function to read in data from a configurations file. This function reads in arrays containing the
 * depths of each layer and the SLDs of each layer. From this the full SLD profile can be constructed.
 * 
 * @param fileName The name of the configurations file containing the data
 * @param depths A vector containing the depths of each layer in order
 * @param slds A vector containing the SLDs of each layer in order
 * @return int 
 */
int ReadFile(const std::string& fileName, arma::vec& depths, arma::vec& slds);

#endif
