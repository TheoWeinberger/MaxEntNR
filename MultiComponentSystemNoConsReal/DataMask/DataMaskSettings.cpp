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

#include <iostream>
#include <armadillo>
#include <libconfig.h++>
#include "DataMaskSettings.hpp"

/**
 * @brief Function to read in data that is to be stored in an armadillo vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 * @param numGauss The number of gaussians used to mask the system
 */
void ReadVector(arma::vec& myVector, const libconfig::Setting& mySetting, const int& numGauss)
{
	int length = mySetting.getLength();
	
	myVector.set_size(numGauss);

	if(numGauss != length)
	{
		std::cerr << "Number of elements of the array is not equal to the number of gaussians being used" << std::endl;
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < length; i++)
	{
		double val  = mySetting[i];
		myVector[i] = val;
	}
}

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
int ReadFile(const std::string& fileName, int& qOffset, bool& lowFreqMask, bool& highFreqMask, bool& userDefMask, arma::vec& sigmas, arma::vec& means, int& numGauss)
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

  
	//Get data from file 
	try
	{
		qOffset = cfg.lookup("qOffset");
		lowFreqMask = cfg.lookup("lowFreqMask");
		highFreqMask = cfg.lookup("highFreqMask");
		userDefMask = cfg.lookup("userDefMask");

		//only one mask type can be used at a time 
		//calls to multiple mask types throw and error
		if(lowFreqMask == true && highFreqMask == true)
		{
			std::cout << "Only one type of masking can be used at a time." << std::endl;
			std::cout << "Please check configurations file." << std::endl;
			exit(EXIT_FAILURE);
		}	

		if(lowFreqMask == true && userDefMask == true)
		{
			std::cout << "Only one type of masking can be used at a time." << std::endl;
			std::cout << "Please check configurations file." << std::endl;
			exit(EXIT_FAILURE);
		}		

		if(userDefMask == true && highFreqMask == true)
		{
			std::cout << "Only one type of masking can be used at a time." << std::endl;
			std::cout << "Please check configurations file." << std::endl;
			exit(EXIT_FAILURE);
		}	

		//if the userDefMask is called then arrays containing standard deviation and mean data 
		//must be found. The number of gaussians in the system must be specified in the settings file.
		//This is used for sense checking
		if(userDefMask == true)
		{
			numGauss = cfg.lookup("numGauss");

			const libconfig::Setting& meanSetting = cfg.lookup("means");
			ReadVector(means, meanSetting, numGauss);

			const libconfig::Setting& sigmaSetting = cfg.lookup("sigmas");
			ReadVector(sigmas, sigmaSetting, numGauss);
		}
		//If regular masking is used the system will contain one gaussian with the standard deviation
		//specified in the sigmas array
		else
		{
			numGauss = cfg.lookup("numGauss");

			//if regular masking is being used but the number of gaussians is not 
			//one then throw an error
			if(numGauss != 1)
			{
				std::cout << "Wrong number of Gaussians for a regular mask." << std::endl;
				std::cout << "Please check configurations file." << std::endl;
			}

			const libconfig::Setting& sigmaSetting = cfg.lookup("sigmas");
			ReadVector(sigmas, sigmaSetting, numGauss);
		}
		
		return(EXIT_SUCCESS);
	}
  	catch(const libconfig::ParseException &pex)
	{
		std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
					<< " - " << pex.getError() << std::endl;
		exit(EXIT_FAILURE);
	}
}
