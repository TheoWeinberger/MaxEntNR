/**
 * @file NewtonSettings.cpp
 * @author Theo Weinberger
 * @brief This file contains all the relevant functions for reading in the configuration data to perform Newton fitting method
 * @version 2.0
 * @date 2021-06-30
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include <armadillo>
#include <libconfig.h++>
#include "NewtonSettings.hpp"

/**
 * @brief Function to read in data that is to be stored in an armadillo vector
 * 
 * @param myVector Generic vector into which data should be put.
 * @param mySetting The configuration data that is to be read into \a myVector
 * @param numSubstances Number of different substances in this system
 */
void ReadVector(arma::vec& myVector, const libconfig::Setting& mySetting, const int& numSubstances)
{
	int length = mySetting.getLength();
	
	myVector.set_size(numSubstances);

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
 * @param realWorldScaling Whether real world scaling is used in the system
 * @param qOffsetReal The parameter offset at which the data starts being read in real world units
 * @param qCutOffReal The index at which the data is then zeroed in real world units
 * @param initLayerStartReal Vector containing the value of the start of the initial charge distributions
 * @param initLayerStopReal Vector containing the value of the end of the initial charge distributions
 * @param error whether dataset contains real errors
 * @return int Exit Code
 */
int ReadFile(const std::string& fileName, int& totalIterations, double& zeroLevel, double& minVar, double& dataScale, int& numSubstances, double& propagationSLD, double& substrateSLD, int& lengthPropagation, int& lengthSubstrate, bool& smoothIncrement, bool& useEdgeConstraints,  arma::vec& sldVal, bool& useDamping, double& alphaInit, double& alphaFactor, double& gammaInit, double& gammaFactor, bool& forceZero, double& fracMax, int& forceInterval, bool& smoothProfile, int& smoothInterval, bool& volumetricNormalisation, bool& boundSLD, double& sldMaxBound, double& sldMinBound, int& qOffset, int& qCutOff, bool& realWorldScaling, double& lengthPropagationReal, double& lengthSubstrateReal, double& qOffsetReal, double& qCutOffReal, bool& error)
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
		totalIterations = cfg.lookup("numIterations");
		minVar = cfg.lookup("minVariance");
		zeroLevel = cfg.lookup("zeroLevel");
		dataScale = cfg.lookup("dataScale");
		smoothIncrement =cfg.lookup("smoothIncrement");
		useEdgeConstraints = cfg.lookup("useEdgeConstraints"); 
		numSubstances = root["Components"].lookup("numSubstances");
        useDamping = cfg.lookup("useDamping");
		smoothProfile = cfg.lookup("smoothProfile");
		forceZero = cfg.lookup("forceZero");
		volumetricNormalisation = cfg.lookup("volumetricNormalisation");
		boundSLD = cfg.lookup("boundSLD");
		error = cfg.lookup("error");
		realWorldScaling =cfg.lookup("realWorldScaling");

		//if damping is being used then the gamma and alpha parameters are needed
		if(useDamping == true)
		{
			alphaInit = cfg.lookup("alphaInit");
			alphaFactor = cfg.lookup("alphaFactor");
			gammaFactor = cfg.lookup("gammaFactor");
			gammaInit = cfg.lookup("gammaInit");
		}
		
		//if real world scaling is used the offset values should be real
		if(realWorldScaling == true)
		{
			qOffsetReal = cfg.lookup("qOffset");
			qCutOffReal = cfg.lookup("qCutOff");
		}
		//else they will be index values
		else
		{
			qOffset = cfg.lookup("qOffset");
			qCutOff = cfg.lookup("qCutOff");
		}

		//If edgec onstraints are being used load these in
		if(useEdgeConstraints == true)
		{
			substrateSLD = root["Substrate"]["substrateSLD"];
			propagationSLD = root["Propagation"]["propagationSLD"];

			//if realWorldScaling is being used then the lengths of the substrate should be in real units
			if(realWorldScaling == true)
			{
				lengthSubstrateReal = root["Substrate"]["lengthSubstrate"];
				lengthPropagationReal = root["Propagation"]["lengthPropagation"];
			}
			//else they are index lengths
			else
			{
				lengthSubstrate = root["Substrate"]["lengthSubstrate"];
				lengthPropagation = root["Propagation"]["lengthPropagation"];
			}
		}

		//get values associated with the various system constraint techniques
		if(smoothProfile == true)
		{
			smoothInterval = cfg.lookup("smoothInterval");
		}

		if(forceZero == true)
		{
			fracMax = cfg.lookup("fracMax");
			forceInterval = cfg.lookup("forceInterval");
		}

		if(boundSLD == true)
		{
			sldMaxBound = cfg.lookup("sldMaxBound");
			sldMinBound = cfg.lookup("sldMinBound");
		}

		//only one normalisation method can be used at one
		if(boundSLD == true && volumetricNormalisation == true)
		{
			std::cout << "Only on of boundSLD and volumetricNormalisation can be used, please edit settings.cfg" << std::endl;
			exit(EXIT_FAILURE);
		}

		const libconfig::Setting& sldSetting = root["Components"].lookup("sldVal");
		ReadVector(sldVal, sldSetting, numSubstances);
		
		return(EXIT_SUCCESS);
	}
  	catch(const libconfig::ParseException &pex)
	{
		std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
					<< " - " << pex.getError() << std::endl;
		exit(EXIT_FAILURE);
	}
}
