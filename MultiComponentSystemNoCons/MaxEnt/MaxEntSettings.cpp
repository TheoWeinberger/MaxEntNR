/**
 * @file MaxEntSettings.cpp
 * @author Theo Weinberger
 * @brief This file contains all the relevant functions for reading in the configuration data to perform MaxEnt method
 * @version 1.0
 * @date 2021-06-04
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include <armadillo>
#include <libconfig.h++>
#include "MaxEntSettings.hpp"

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
 * @param forceInterval Interval at which _forceZero is applied
 * @param qOffset The parameter offset at which the data starts being read
 * @param qCutOff The index at which the data is then zeroed
 * @return int Exit Code
 */
int ReadFile(const std::string& fileName, int& totalIterations, int& maximumSearchIter, int& numBasisVectors, double& zeroLevel, double& minVar, double& dataScale, int& numSubstances, double& propagationSLD, double& substrateSLD, int& lengthPropagation, int& lengthSubstrate, bool& smoothIncrement, bool& useEdgeConstraints, double& chiSquaredScale, bool& spikeCharge, double& spikePortion, double& spikeAmount, arma::vec& sldVal, arma::vec& total, bool& toyModel, bool& volumetricNormalisation, bool& error, bool& boundSLD, double& sldMinBound, double& sldMaxBound, bool& smoothProfile, int& smoothInterval, bool& forceZero, double& fracMax, double& chargeScale, int& forceInterval, int& qOffset, int& qCutOff)
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
		maximumSearchIter = cfg.lookup("numLagrangeSearches");
		numBasisVectors = cfg.lookup("numBasisVectors");
		error = cfg.lookup("error");
		minVar = cfg.lookup("minVariance");
		zeroLevel = cfg.lookup("zeroLevel");
		dataScale = cfg.lookup("dataScale");
		chargeScale = cfg.lookup("chargeScale");
		smoothIncrement =cfg.lookup("smoothIncrement");
		useEdgeConstraints = cfg.lookup("useEdgeConstraints"); 
		chiSquaredScale = cfg.lookup("chiSquaredScale");
		spikeCharge = cfg.lookup("spikeCharge");
		spikePortion = cfg.lookup("spikePortion");
		spikeAmount = cfg.lookup("spikeAmount");
		toyModel = cfg.lookup("toyModel");
		volumetricNormalisation = cfg.lookup("volumetricNormalisation");
		numSubstances = root["Components"].lookup("numSubstances");
		substrateSLD = root["Substrate"]["substrateSLD"];
		lengthSubstrate = root["Substrate"]["lengthSubstrate"];
		propagationSLD = root["Propagation"]["propagationSLD"];
		lengthPropagation = root["Propagation"]["lengthPropagation"];
		boundSLD = cfg.lookup("boundSLD");
		smoothProfile = cfg.lookup("smoothProfile");
		forceZero = cfg.lookup("forceZero");
		qOffset = cfg.lookup("qOffset");
		qCutOff = cfg.lookup("qCutOff");

		if(boundSLD == true)
		{
			sldMaxBound = cfg.lookup("sldMaxBound");
			sldMinBound = cfg.lookup("sldMinBound");
		}

		if(boundSLD == true && volumetricNormalisation == true)
		{
			std::cout << "Only on of boundSLD and volumetricNormalisation can be used, please edit settings.cfg" << std::endl;
			exit(EXIT_FAILURE);
		}

		if(smoothProfile == true)
		{
			smoothInterval = cfg.lookup("smoothInterval");
		}

		if(forceZero == true)
		{
			fracMax = cfg.lookup("fracMax");
			forceInterval = cfg.lookup("forceInterval");
		}

		const libconfig::Setting& sldSetting = root["Components"].lookup("sldVal");
		ReadVector(sldVal, sldSetting, numSubstances);

		const libconfig::Setting& totalSetting = root["Components"].lookup("total");
		ReadVector(total, totalSetting, numSubstances);
		
		return(EXIT_SUCCESS);
	}
  	catch(const libconfig::ParseException &pex)
	{
		std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
					<< " - " << pex.getError() << std::endl;
		exit(EXIT_FAILURE);
	}
}
