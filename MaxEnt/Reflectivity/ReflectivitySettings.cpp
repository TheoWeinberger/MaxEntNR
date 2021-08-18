/**
 * @file ReflectivitySettings.cpp
 * @author Theo Weinberger
 * @brief Functions to read in data from a configurations file to create and SLD profile from which the reflectivity 
 * can be approximated by the Fourier approximation
 * @version 2.0
 * @date 2021-04-16
 * 
 * @copyright Copyright (c) 2021
 * 
 */


#include <iostream>
#include <armadillo>
#include <libconfig.h++>
#include "ReflectivitySettings.hpp"

/**
 * @brief Function to read in data that is to be stored in an armadillo vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 */
void ReadVectorDouble(arma::vec& myVector, const libconfig::Setting& mySetting)
{
	int length = mySetting.getLength();
	
	myVector.set_size(length);

	for(int i = 0; i < length; i++)
	{
		double val  = mySetting[i];
		myVector[i] = val;
	}
}

/**
 * @brief Function to read in data that is to be stored in an armadillo vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 */
void ReadVectorInt(arma::vec& myVector, const libconfig::Setting& mySetting)
{
	int length = mySetting.getLength();
	
	myVector.set_size(length);

	for(int i = 0; i < length; i++)
	{
		int val  = mySetting[i];
		myVector[i] = val;
	}
}
/**
 * @brief Function to read in data from a configurations file. This function reads in arrays containing the
 * depths of each layer and the SLDs of each layer. From this the full SLD profile can be constructed.
 * 
 * @param fileName The name of the configurations file containing the data
 * @param depths A vector containing the depths of each layer in order
 * @param slds A vector containing the SLDs of each layer in order
 * @return int 
 */
int ReadFile(const std::string& fileName, arma::vec& depths, arma::vec& slds)
{
	//define storage variables
	libconfig::Config cfg;
	
	//read in diffraction configuration file, reporting error if it does not exist
	try
	{
		cfg.readFile(fileName.c_str());
	}
  	catch(const libconfig::FileIOException &fioex)
	{
		std::cerr << "I/O error while reading file." << std::endl;
		exit(EXIT_FAILURE);
	}
  	catch(const libconfig::ParseException &pex)
	{
		std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
				<< " - " << pex.getError() << std::endl;
		exit(EXIT_FAILURE);
	}

  

	const libconfig::Setting& root = cfg.getRoot();

	//Get data from file 
	try
	{
		const libconfig::Setting& depthsSetting = root["SLD"].lookup("depths");
		ReadVectorInt(depths, depthsSetting);

		const libconfig::Setting& sldsSetting = root["SLD"].lookup("slds");
		ReadVectorDouble(slds, sldsSetting);
	}
  	catch(const libconfig::ParseException &pex)
	{
		std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
					<< " - " << pex.getError() << std::endl;
		exit(EXIT_FAILURE);
	}

	//check dimensionality of input file, all dimensions must match of data importing will fail
	if(slds.n_elem == depths.n_elem)
	{
	 	return(EXIT_SUCCESS);
	}
  	else
	{
		std::cerr << "Array lengths are not equal, check configuration file" << std::endl;
		exit(EXIT_FAILURE);
	}
}

  

  











