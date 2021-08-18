/**
 * @file ReflectivitySettings.cpp
 * @author Theo Weinberger
 * @brief Functions to read in data from a configurations file to create and SLD profile from which the reflectivity 
 * can be approximated by the Fourier approximation
 * @version 1.0
 * @date 2021-05-12
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
 * @param numSubstances Number of different substances in this system
 */
void ReadVectorDouble(std::vector<double>& myVector, const libconfig::Setting& mySetting, const int& numSubstances)
{
	int length = mySetting.getLength();
	
	myVector.resize(numSubstances);

	if(numSubstances != length)
	{
		std::cerr << "Number of elements of the array is not equal to the number of materials in the substance" << std::endl;
		exit(EXIT_FAILURE);
	}

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
 * @param numSubstances Number of different substances in this system
 */
void ReadVectorDouble(std::vector<arma::vec>& myVector, const libconfig::Setting& mySetting, const int& numSubstances)
{
	int length = mySetting.getLength();
	
	myVector.resize(numSubstances);

	if(numSubstances != length)
	{
		std::cerr << "Number of elements of the array is not equal to the number of materials in the substance" << std::endl;
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < length; i++)
	{
		const libconfig::Setting& subSetting  = mySetting[i];
		
		myVector[i].set_size(subSetting.getLength());

		for(int j = 0; j < subSetting.getLength(); j++)
		{
			double val = subSetting[j];
			myVector[i][j] = val;
		}
	}
}

/**
 * @brief Function to read in data that is to be stored in an armadillo vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 * @param numSubstances Number of different substances in this system
 */
void ReadVectorInt(std::vector<arma::vec>& myVector, const libconfig::Setting& mySetting, const int& numSubstances)
{
	int length = mySetting.getLength();
	
	myVector.resize(numSubstances);

	if(numSubstances != length)
	{
		std::cerr << "Number of elements of the array is not equal to the number of materials in the substance" << std::endl;
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < length; i++)
	{
		const libconfig::Setting& subSetting  = mySetting[i];

		myVector[i].set_size(subSetting.getLength());

		for(int j = 0; j < subSetting.getLength(); j++)
		{
			int val = subSetting[j];
			myVector[i][j] = val;
		}
	}
}

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
int ReadFile(const std::string& fileName, int& numSubstances, std::vector<arma::vec>& depths, std::vector<double>& sldVal, std::vector<arma::vec>& n, std::vector<double>& total, double& substrateSLD, int& lengthSubstrate, bool& volumetricNormalisation, bool& toyModel, bool& boundSLD, double& sldMinBound, double& sldMaxBound, int& qOffset)
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
		numSubstances = root["Components"].lookup("numSubstances");

		const libconfig::Setting& sldSetting = root["Components"].lookup("sldVal");
		ReadVectorDouble(sldVal, sldSetting, numSubstances);

		const libconfig::Setting& totalSetting = root["Components"].lookup("total");
		ReadVectorDouble(total, totalSetting, numSubstances);

		const libconfig::Setting& depthsSetting1 = root["Components"].lookup("depths");
		ReadVectorInt(depths, depthsSetting1, numSubstances);

		const libconfig::Setting& sldsSetting1 = root["Components"].lookup("n");
		ReadVectorDouble(n, sldsSetting1, numSubstances);

		substrateSLD = root["Substrate"]["substrateSLD"];

		lengthSubstrate = root["Substrate"]["lengthSubstrate"];

		volumetricNormalisation = cfg.lookup("volumetricNormalisation");

		toyModel = cfg.lookup("toyModel");

		boundSLD = cfg.lookup("boundSLD");

		qOffset = cfg.lookup("qOffset");

		//if sld is bounded then the max and min bounds must also be read in
		if(boundSLD == true)
		{
			sldMaxBound = cfg.lookup("sldMaxBound");
			sldMinBound = cfg.lookup("sldMinBound");
		}
	}
  	catch(const libconfig::ParseException &pex)
	{
		std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
					<< " - " << pex.getError() << std::endl;
		exit(EXIT_FAILURE);
	}

	//check dimensionality of input file, all dimensions must match of data importing will fail
	if(n[0].n_elem == depths[0].n_elem && n[1].n_elem == depths[1].n_elem)
	{
	 	return(EXIT_SUCCESS);
	}
  	else
	{
		std::cerr << "Array lengths are not equal, check configuration file" << std::endl;
		exit(EXIT_FAILURE);
	}
}

  

  











