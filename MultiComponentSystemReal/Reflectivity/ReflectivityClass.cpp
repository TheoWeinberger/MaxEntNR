/****************************************************************************************
 * @file ReflectivityClass.cpp                                                 
 * @brief Code to generate the reflectivity pattern from a SLD profile using the 
 * Fourier form of the reflectivity relationship which is defined to be
 * 
 *          R[Q] = \frac{1}{Q^4} * |FFT(SLD)[Q]*[1-exp{-Q*I*(2*PI/totalDepth)}] +
 *                           SLD[totalDepth - 1] - SLD[0]|^2
 *
 * The paper regarding this transform and its applications to Nafion thin film systems is
 * 
 *    (1) Theodore Weinberger, The Inversion Problem, Reconstructing Scattering 
 *        Length Density Profiles from Neutron Reflectivity Spectra
 *        via the Principle of Maximum-Entropy (2021), Preprint submitted to 
 *        MPhil in Scientific Computing
 * 
 * This program also uses the following packages
 *
 * For matrix manipulation: Armadillo from http://arma.sourceforge.net/docs.html
 *
 *    (1)  Conrad Sanderson and Ryan Curtin.
 *         Armadillo: a template-based C++ library for linear algebra.
 *         Journal of Open Source Software, Vol. 1, pp. 26, 2016.
 *
 *    (2)  Conrad Sanderson and Ryan Curtin.
 *         A User-Friendly Hybrid Sparse Matrix Class in C++.
 *         Lecture Notes in Computer Science (LNCS), Vol. 10931, pp. 422-430, 2018.
 *
 * For discrete fast fourier transforms: FFTW3 from http://www.fftw.org/
 *
 *    (1) Frigo M, Johnson SG (1998) FFTW: An adaptive software architecture for the FFT.
 *        In: Proceedings IEEE international conference acoustics, speech, and signal
 *        processing (ICASSP), 3, pp 13811384
 *
 *
 *    (2) Frigo M (1999) A fast Fourier transform compiler.
 *        In Proceedings of the ACM SIGPLAN 1999 conference on programming language
 *        design and implementation (PLDI 1999). ACM, New York, pp 169180
 *
 *    (3) Frigo M, Johnson SG (2005) The design and implementation of FFTW3.
 *        Proceedings of the IEEE 93(2):216231
 *
 * For accessing settings files: libconfig from http://hyperrealm.github.io/libconfig/
 * 
 ****************************************************************************************
 *
 * @author Theo Weinberger
 * @version 2.0
 * @date 2021-05-15
 * 
 * @copyright Copyright (c) 2021
 *
 ****************************************************************************************
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
        ReadFile(configFile, _numSubstances, _depths, _sldVal, _n, _total, _substrateSLD, _lengthSubstrate, _volumetricNormalisation, _toyModel, _boundSLD, _sldMinBound, _sldMaxBound, _qOffset);

        //depth, which is number of elements, is equal to the total number of depths of the SLD profile
        _depth = (int)arma::accu(_depths[0]) + _lengthSubstrate;

        for(int i = 0; i < _numSubstances; i++)
        {
            if((int)arma::accu(_depths[0]) !=  (int)arma::accu(_depths[i]))
            {
                std::cerr << "Substance depths do not sum to the same value" << std::endl;
                exit(EXIT_FAILURE);
            } 
        }

        _sld.set_size(_depth);
        _scalingVector.set_size(_depth);
        _sld.zeros();   
        _scalingVector.zeros();  

        /**
        //physical scaling
        if(_volumetricNormalisation == false)
        {
            int divisibility = 0;

            for(int i = 0; i < _numSubstances; i++)
            {
                _total[i] *= _nA; 
            }

            double rem = *std::max_element(_total.begin(), _total.end());

            while(rem > 10.0)
            {
                rem /= 10.0;
                divisibility++;
            }

            for(int i = 0; i < _numSubstances; i++)
            {
                _total[i] /= pow(10.0, divisibility - 2); //scale total to be of order 100 as this works best for algo
                _sldVal[i] *= pow(10.0, divisibility - 25); //scale sld val along with this, note the factor of -9 to account for the fm scaling and the cm scaling

            }
        }
        **/
        if(_toyModel == true)
        {
            //check the settings file is for toy data
            if(_depths.size() != 1 || _numSubstances != 1 || _volumetricNormalisation != false)
            {
                std::cout << "Input configuration has more than 1 substance." << std::endl;
                std::cout << "Subsequent simulations are for a toy model and will only use values for the first substance." << std::endl;
                std::cout << "Press X to quit or any other button to continue with the simulation" << std::endl;
                char answer;
                std::cin >> answer;

                if(answer == 'X' || answer == 'x')
                {
                    exit(EXIT_SUCCESS);
                }
            }

            //force required value
            _volumetricNormalisation = false;

            //get thickness and layer counters
            int numLayers = _depths[0].n_elem;
            int currentThickness = 0;

            for(int i = 0; i < numLayers; i++)
            {
                int finalThickness = currentThickness + _depths[0][i];

                //iteration to input values of slds into sld vector
                //slds are input from the initial thickness of that layer to the final thickness
                for(int j = currentThickness; j < finalThickness; j++)
                {
                    _sld[j] += _n[0][i]; //for toy model n is the sld value 
                }

                currentThickness = finalThickness;
            }

        }
        else
        {
            for(_componentNum = 0; _componentNum < _numSubstances; _componentNum++)
            {
                _SLDGenerate(_depths[_componentNum], _n[_componentNum], _total[_componentNum], _sldVal[_componentNum]);
            } 
            
            if(_volumetricNormalisation == true)
            {
                double scalingValue = _scalingVector.max();
                _sld /= scalingValue;
            }

            if(_boundSLD == true)
            {
                _BoundSLD();
            }
        }

        //append the substrate SLD
        _sld.tail(_lengthSubstrate) += _substrateSLD;
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
 * @brief Class member funciton used to generate the SLD profile from the component parts
 * 
 * @param depths Array containing substance depths
 * @param n Array containing susbtance amunts
 * @param total Total amount of substance
 * @param sldVal SLD of substance
 */
void Reflectivity::_SLDGenerate(const arma::vec& depths, const arma::vec& n, const double& total, const double& sldVal)
{
    //the number of layers in the susbtance is equal to the of elements in the depth/sld array
    int numLayers = depths.n_elem;
    arma::vec scalingVector; 
    scalingVector.set_size(_depth);

    int currentThickness = 0;
    double normalisation = total/arma::accu(n % depths);
    arma::mat component;

    //used for component analysis 
    component.set_size(_depth,2);
    component.zeros();

    for(int i = 0; i < numLayers; i++)
    {
        int finalThickness = currentThickness + depths[i];

        //iteration to input values of slds into sld vector
        //slds are input from the initial thickness of that layer to the final thickness
        for(int j = currentThickness; j < finalThickness; j++)
        {
            _sld[j] += n[i]*normalisation*sldVal;
            component(j,1) = n[i]*normalisation;
            _scalingVector[j] += n[i]*normalisation;
        }

        currentThickness = finalThickness;
    }

    for(int i = 0; i < _depth; i++)
    {
        component(i,0) = 2*M_PI*i/((double)_depth);
    }

    component.save("charge" + std::to_string(_componentNum + 1), arma::raw_ascii);
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
        _sldScaled(i,0) = 2*M_PI*i/((double)_depth);
        _sldScaled(i,1) = _sld[i];
    }
}

/**
 * @brief Scale charge to physical SLD parameters so that value constraints can be imposed on the system
 * 
 */
void Reflectivity::_BoundSLD()
{
    //rescale the magnituyde of the reconstructed SLD so that the max value is the max bound
    arma::vec sldTemp;
    sldTemp.set_size(_depth);

    sldTemp = _sld;


    //get min and max values of the array
    double maxVal = sldTemp.max();
    double minVal = sldTemp.min();


    double minValRatio = 0.0;
    double maxValRatio = 0.0;

    //determine which ratio to scale the array by
    if(sign(minVal) == sign(_sldMinBound) && minVal != 0.0)
    {
        minValRatio = _sldMinBound/minVal;
    }

    if(sign(maxVal) == sign(_sldMaxBound) && maxVal != 0.0)
    {
        maxValRatio = _sldMaxBound/maxVal;
    }

    //scale the SLD
    if(minValRatio < maxValRatio || maxValRatio == 0.0)
    {
        if(minValRatio != 0.0 )
        {
            sldTemp *= minValRatio;
        }
    }

    if(maxValRatio < minValRatio || minValRatio == 0.0)
    {
        if(maxValRatio != 0.0)
        {
            sldTemp *= maxValRatio;
        }
    }


    //check SLD is still bounded
    #ifdef DEBUG
    std::cout << "Max initial value: " << chargeTemp.max() <<  std::endl;
    assert(sldTemp.max() <= _sldMaxBound);

    std::cout << "Min initial value: " << chargeTemp.min() <<  std::endl;
    assert(sldTemp.min() >= _sldMinBound);
    #endif

    _sld  = sldTemp;
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

    reflectivityTemp /= (double)_depth;


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

    reflectivityTemp = reflectivityTemp%dftDeriv + (_sld[_depth-1] - _sld[0])/(double)_depth;

    arma::vec reflectivityTempInt = abs(reflectivityTemp);

    //check matrix for lengths
    #ifdef DEBUG
    std::cout << "Exponential factor length: " << dftDeriv.n_elem <<  " Profile length: " << _sld.n_elem << std::endl;
    assert(dftDeriv.n_elem == _sld.n_elem);

    std::cout << "relectivity length: " << reflectivityTempInt.n_elem <<  " Profile length: " << _sld.n_elem << std::endl;
    assert(reflectivityTempInt.n_elem == _sld.n_elem);
    #endif
    
    //System is undefined if _qOffset = 0, therefore _qOffset = 0 is defined to be equivalent to _qOffset = 1
    if(_qOffset == 0)
    {
        _qOffset = 1;
    }

    for(int i = 0; i < _depth; i++)
    {
        //define the first measureable value of reflectivity as being 1
        if(i <= _qOffset)
        {
            _reflectivity[i] = 1.0;
            //Factor for DFT derivative rule
            normalisation = pow(reflectivityTempInt[i], 2.0)/pow(i, 4.0);
        }
        //Calculate remaining values of reflectivity according to reflectivity formula
        else
        {
            _reflectivity[i] = pow(reflectivityTempInt[i]/((double)(i*i)), 2.0)/normalisation;
        }
    }

    //check matrix for lengths
    #ifdef DEBUG
    std::cout << "relectivity length: " << _reflectivity.n_elem <<  " Profile length: " << _sld.n_elem << std::endl;
    assert(_reflectivity.n_elem == _sld.n_elem);
    #endif
   
}



