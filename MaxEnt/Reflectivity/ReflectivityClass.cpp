/**
 * @file ReflectivityClass.cpp
 * @author Theo Weinberger
 * @brief File containing functions used by ReflectivityClass.hpp used to 
 * generate the reflectivity spectra
 * @version 2.0
 * @date 2021-04-15
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>
#include <armadillo>
#include <cmath>
#include <complex.h>
#include <fftw3.h>
#include <libconfig.h++>
#include <assert.h>
#include "ReflectivityClass.hpp"
#include "ReflectivitySettings.hpp"

//debug mode
//#define DEBUG

/**
 * @brief Construct a new Reflectivity:: Reflectivity object
 * 
 */
Reflectivity::Reflectivity()
{

    //get number of layers as user input
    std::cout << "Enter number of layers in substance" << std::endl;
    std::cin >> _numLayers;
    std::cout << std::endl; 

    int totalDepth = 0;

    _slds.set_size(_numLayers);
    _depths.set_size(_numLayers);

    //repeatedly get user inputs for layer thicknesses and layer SLDs
    for(int i = 0; i < _numLayers; i++)
    {
        int layerDepth;
        double layerSLD;

        std::cout << "Enter thickness of layer: " << i << std::endl;
        std::cin >> layerDepth;
        std::cout << std::endl;

        std::cout << "Enter SLD of layer: " << i << std::endl;
        std::cin >> layerSLD;
        std::cout << std::endl;

        _slds[i] = layerSLD;
        _depths[i] = layerDepth;
        
        //summation to get total thickness of thin film
        totalDepth += layerDepth;
    }

    _depth = totalDepth;

    _reflectivity.set_size(_depth);
    _sld.set_size(_depth);

    int currentThickness = 0;

    //loop that iterates over the _depths vector and _slds vector to create overall _sld profile
    for(int i = 0; i < _numLayers; i++)
    {
        int finalThickness = currentThickness + _depths[i];

        //iteration to input values of slds into sld vector
        //slds are input from the initial thickness of that layer to the final thickness
        for(int j = currentThickness; j < finalThickness; j++)
        {
            _sld[j] = _slds[i];
        }

        currentThickness = finalThickness;
    }



    _ReflectivityCalc();
    _ScaleReflectivity();
    _ScaleSLD();

}

/**
 * @brief Construct a new Reflectivity:: Reflectivity object
 * 
 * @param configFile Configuration file containg information to create an SLD profile
 * @param dataType What type of data is being used as the input data. Options are either a file containing an actual SLD curve, 'SLD',
 *  or a configurations file containing a series of SLDs and their corresponding thicknesses 
 *  
 */
Reflectivity::Reflectivity(const std::string& configFile, const std::string& dataType)
{
    //if data type is data then direct SLD data is being read in 
    if(dataType == "data")
    {
        _sld.load(configFile, arma::raw_ascii);
    }
    //if datatpye is config then data is being read in from a configuration file
    if(dataType == "config")
    {
        ReadFile(configFile, _depths, _slds);

        //depth, which is number of elements, is equal to the total number of depths of the SLD profile
        _depth = (int)arma::accu(_depths);

        _sld.set_size(_depth);        

        //the number of layers in the susbtance is equal to the of elements in the depth/sld array
        _numLayers = _depths.n_elem;

        int currentThickness = 0;
        for(int i = 0; i < _numLayers; i++)
        {
            int finalThickness = currentThickness + _depths[i];

            //iteration to input values of slds into sld vector
            //slds are input from the initial thickness of that layer to the final thickness
            for(int j = currentThickness; j < finalThickness; j++)
            {
                _sld[j] = _slds[i];
            }

            currentThickness = finalThickness;
        }
    }

    //recalculate depth as number of sld elements for consistency
    _depth = _sld.n_elem;
    _reflectivity.set_size(_depth);

    //check input profile is as expected
    #ifdef DEBUG
    std::cout << "Depth Input: "<< _depth << " Profile Depth: " << _sld.n_elem << std::endl;
    assert(_depth == _sld.n_elem);
    int totalDepth = 0;
    for(int i = 0; i < _numLayers; i++)
    {
        std::cout << "SLD Input: "<< _slds[i] << " SLD Profile: " << _sld[totalDepth] << std::endl;
        assert(_slds[i] == _sld[totalDepth]);
        totalDepth += _depths[i];
    }
    #endif

    _ReflectivityCalc();
    _ScaleReflectivity();
    _ScaleSLD();
}

/**
 * @brief Class function to get the reflectivity spectrum
 * 
 * @return arma::vec _reflectivity the reflectivity spectrum
 */
arma::vec Reflectivity::GetRef()const
{
    return _reflectivity;
}


/**
 * @brief Class function to get the scaled reflectivity spectrum
 * 
 * @return arma::mat _reflectivityScaled the scaled reflectivity spectrum
 */
arma::mat Reflectivity::GetRefScaled()const
{
    return _reflectivityScaled;
}

/**
 * @brief Class function to get the SLD
 * 
 * @return arma::vec _sld the SLD profile
 */
arma::vec Reflectivity::GetSLD()const
{
    return _sld;
}


/**
 * @brief Class function to get the scaled SLD
 * 
 * @return arma::mat _sldScaled the scaled SLD profile
 */
arma::mat Reflectivity::GetSLDScaled()const
{
    return _sldScaled;
}

/**
 * @brief Class function to create scaled x axis for reflectivity data
 * 
 */
void Reflectivity::_ScaleReflectivity()
{
    _reflectivityScaled.set_size(_depth,2);
    
    for(int i = 0; i < _depth; i++)
    {
        //reflectivity is unscaled, x-axis is just the index
        _reflectivityScaled(i,0) = i;
        _reflectivityScaled(i,1) = _reflectivity[i];
    }
}

/**
 * @brief Class function to create scaled x axis for sld data
 * 
 */
void Reflectivity::_ScaleSLD()
{
    _sldScaled.set_size(_depth,2);
    
    for(int i = 0; i < _depth; i++)
    {
        //sld is unscaled, x-axis is just the index
        _sldScaled(i,0) = i;
        _sldScaled(i,1) = _sld[i];
    }
}

/**
 * @brief Class function to transform the SLD profile to the reflectivity according to R[Q] = \frac{1}{Q^4} * |FFT(SLD)[Q]*[1-exp{-Q*I*(2*PI/totalDepth)}] +
 * SLD[totalDepth - 1] - SLD[0]|^2
 * (Weinberger 2021) Eqn. 11/67
 * 
 */
void Reflectivity::_ReflectivityCalc()
{

    //temporary array to store direct transform of the SLD in
    arma::cx_vec reflectivityTemp;

	reflectivityTemp.copy_size(_sld);

    //create data for a complex array containing the SLD profile to be transformed via the fourier approximation to calculate the reflectivity
    arma::vec zerosFill;
    zerosFill.copy_size(_sld);
    zerosFill.zeros();
    arma::cx_vec sldComplex = arma::cx_vec(_sld, zerosFill);

    //check matrices for transforms
    #ifdef DEBUG
    std::cout << "relectivityTemp length: " << reflectivityTemp.n_elem <<  " Profile length: " << _sld.n_elem << std::endl;
    assert(reflectivityTemp.n_elem == _sld.n_elem);

    std::cout << "relectivityTemp length: " << reflectivityTemp.n_elem <<  " Profile length: " << sldComplex.n_elem << std::endl;
    assert(sldComplex.n_elem == _sld.n_elem);

    assert(arma::approx_equal(arma::real(sldComplex), _sld, "absdiff", 0.01));
    assert(arma::approx_equal(arma::imag(sldComplex), zerosFill, "absdiff", 0.01)); 
    #endif


    //create FFTW plan determining how to apply DFT
    fftw_plan plan=fftw_plan_dft_1d(sldComplex.n_elem,(double(*)[2])&sldComplex(0), (double(*)[2])&reflectivityTemp(0), FFTW_FORWARD, FFTW_ESTIMATE);

    //perform DFT
    fftw_execute(plan);

    //delete plan
    fftw_destroy_plan(plan);


    //normalisation factor such that reflectivity spectrum is normalised to one
    double normalisation;

    //Factor for DFT derivative rule
    arma::cx_vec dftDeriv;

    dftDeriv.set_size(_depth);

    for(int i = 0; i < _depth; i++)
    {
        dftDeriv[i] = 1-cexp(-i*_Complex_I*(2*M_PI/(double)_depth));
    }

    //Below is the methodology to tranform the DFT of the SLD to the reflectivity pattern

    reflectivityTemp = reflectivityTemp%dftDeriv + _sld[_depth-1] - _sld[0];

    arma::vec reflectivityTempInt = abs(reflectivityTemp);

    //check matrix for lengths
    #ifdef DEBUG
    std::cout << "Exponential factor length: " << dftDeriv.n_elem <<  " Profile length: " << _sld.n_elem << std::endl;
    assert(dftDeriv.n_elem == _sld.n_elem);

    std::cout << "relectivity length: " << reflectivityTempInt.n_elem <<  " Profile length: " << _sld.n_elem << std::endl;
    assert(reflectivityTempInt.n_elem == _sld.n_elem);
    #endif
    
    for(int i = 0; i < _depth; i++)
    {
        //Value at Q = 1 is undefined so set equal to 1 such that it is at the maximum value
        if(i == 0)
        {
            _reflectivity[0] = 1.0;
        }
        //define the first measureable value of reflectivity as being 1
        else if(i == 1)
        {
            _reflectivity[1] = 1.0;
            //Factor for DFT derivative rule
            normalisation = pow(reflectivityTempInt[1], 2.0);
        }
        //Calculate remaining values of reflectivity according to reflectivity formula
        else
        {
            _reflectivity[i] = pow(reflectivityTempInt[i]/(double)(i*i), 2.0)/normalisation;
        }
    }

    //check matrix for lengths
    #ifdef DEBUG
    std::cout << "relectivity length: " << _reflectivity.n_elem <<  " Profile length: " << _sld.n_elem << std::endl;
    assert(_reflectivity.n_elem == _sld.n_elem);
    #endif
   
}



