/**
 * @file InterpolatorSettings.cpp
 * @author Theo Weinberger
 * @brief This file contains all the relevant functions for reading in the configuration data to run interpolator
 * @version 1.0
 * @date 2021-05-22
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include <armadillo>
#include <libconfig.h++>
#include "InterpolatorSettings.hpp"


/**
 * @brief Method to read file settings into MaxEnt simulation
 * 
 * @param fileName String data containing the name of the file containing configuration data.
 * @param sectionsOut Number of segments of the interpolated data
 * @param errors Whether input data has errors to be interpolated
 * @param rangeTimes Factor by which to scale the max range by
 * @param accountError Whether to accout for experimnetal limitation in the data
 * @param errorKnown Whether the experimental limit of the apparatus is known
 * @param resLimit What the experimnetal limit of the apparatus is 
 * @return int Exit Code
 */
int ReadFile(const std::string& fileName, int& sectionsOut, bool& errors, double& rangeTimes, bool& accountError, bool& errorKnown, double& resLimit)
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
		//get standard 
		sectionsOut = cfg.lookup("sectionsOut");
		errors = cfg.lookup("errors");
		rangeTimes = cfg.lookup("rangeTimes");
		accountError = cfg.lookup("accountError");
		errorKnown = cfg.lookup("errorKnown");

		//resolution limit is only a valid setting if the experimental errro is known
		if(errorKnown == true)
		{
			resLimit = cfg.lookup("resLimit");
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
