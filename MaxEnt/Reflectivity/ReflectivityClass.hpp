/**
 * @file ReflectivityClass.hpp
 * @author Theo Weinberger
 * @brief Class containing anayltical method of producuing reflectivity spectra from scattering
 * length density profile
 * @version 2.0
 * @date 2021-04-15
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef REFLECTIVITYCLASS_H
#define REFLECTIVITYCLASS_H

/**
 * @brief Reflectivity class used to generate reflectivity spectrum from inputted SLD profile
 * 
 */
class Reflectivity{   
public:

    /**
     * @brief Construct a new Reflectivity class from user input
     * 
     */
    Reflectivity();

    /**
     * @brief Construct a new Reflectivity object from configuration file
     * 
     */
    Reflectivity(const std::string&, const std::string&);

    /**
     * @brief Class function to get the scaled reflectivity spectrum
     * 
     * @return arma::vec _reflectivity the reflectivity spectrum
     */
    arma::vec GetRef()const;

    /**
     * @brief Class function to get the scaled reflectivity spectrum
     * 
     * @return arma::mat _reflectivityScaled the scaled reflectivity spectrum
     */
    arma::mat GetRefScaled()const;

    /**
     * @brief Class function to get the SLD
     * 
     * @return arma::vec _sld the SLD profile
     */
    arma::vec GetSLD()const;

    /**
     * @brief Class function to get the scaled SLD
     * 
     * @return arma::mat _sldScaled the scaled SLD profile
     */
    arma::mat GetSLDScaled()const;


private:

    /**
     * @brief Class function to transform the SLD profile to the reflectivity according to R[Q] = \frac{1}{Q^4} * |FFT(SLD)[Q]*[1-exp{-Q*I*(2*PI/totalDepth)}] +
     * SLD[totalDepth - 1] - SLD[0]|^2
     * (Weinberger 2021) Eqn. 11/67
     * 
     */
    void _ReflectivityCalc();

    /**
     * @brief Class function to create scaled x axis for reflectivity data
     * 
     */    
    void _ScaleReflectivity();

    /**
     * @brief Class function to create scaled x axis for SLD data
     * 
     */    
    void _ScaleSLD();

    /**
     * @brief Vector containing total SLD profile
     * 
     */
    arma::vec _sld; 

    /**
     * @brief Vector containing scaled SLD profile
     * 
     */
    arma::mat _sldScaled; 

    /**
     * @brief Vector containing derivate of SLD profile used to create reflectometry spectrum
     * 
     */
    arma::vec _sldPrime;

    /**
     * @brief Vector containing reflectivity spectrum
     * 
     */
    arma::vec _reflectivity;


    /**
     * @brief Matrix containing reflectivity spectrum and corresponding x-axis
     * 
     */
    arma::mat _reflectivityScaled;

    /**
     * @brief Total depth of thin film
     * 
     */
    int _depth;

    /**
     * @brief Number of layers of thin film
     * 
     */
    int _numLayers;

    /**
     * @brief Scaling factor for x-axis
     * 
     */
    double _depthScale;

    /**
     * @brief Vector containing thickness of each layer
     * 
     */
    arma::vec _depths;

    /**
     * @brief Vector containing SLDs of each layer
     * 
     */
    arma::vec _slds;

};

#endif


