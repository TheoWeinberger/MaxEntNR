/****************************************************************************************
 * 
 * @file DataMask.cpp
 * @author Theo Weinberger
 * 
 *****************************************************************************************
 *
 * @brief Method to clean experimental data by using either user defined or preset Gaussian frequency masks
 * This code uses simple gaussian forms to pick out the desired parts of the frequency spectrum
 * to produce cleaned data. 
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
 *****************************************************************************************
 * 
 * @version 1.0
 * @date 2021-06-29
 * 
 * @copyright Copyright (c) 2021
 * 
 *****************************************************************************************
 */

#include <iostream>
#include <armadillo>
#include <cmath>
#include <complex.h>
#include <fftw3.h>
#include <libconfig.h++>
#include <assert.h>
#include "DataMask.hpp"
#include "DataMaskSettings.hpp"

/**
 * @brief Construct a new Mask object using just the input datafile
 * 
 * @param datafile The name of the file containing the reflectivity data
 */
Mask::Mask(const std::string& datafile)
{
    //load in data file and data to relevant parameters
    _dataFile.load(datafile, arma::raw_ascii);
    _data = _dataFile.col(1);

    //set system variables for data cutoffs and renormalisations
    _length = _data.n_elem;
    _qOffset = 30;
    _reflectivityNorm = _data[_qOffset];

    //remove 1/q^4 dependence from the data so that FT can be applied
    for(int i = 0; i < _length; i++)
    {
        _data[i] *= pow(_dataFile.col(0)[i],4.0);
    }

    //output initial system data for comparison
    _data.save("dataInit", arma::raw_ascii);

    //Set size of system containers
    _dataCleaned.copy_size(_dataFile);
    _dataCleaned.col(0) = _dataFile.col(0);

    _dataCleanedComplex.set_size(_length);
    _dataComplex.set_size(_length);
    _dataFT.set_size(_length);
    _mask.set_size(_length);

}

/**
 * @brief Construct a new Mask object
 * 
 * @param datafile The name of the file containing the reflectivity data
 * @param configfile The name of the file containing the settings data
 */
Mask::Mask(const std::string& datafile, const std::string& configfile)
{
    //Read in variables from configurations file 
    ReadFile(configfile, _qOffset, _lowFreqMask, _highFreqMask, _userDefMask, _sigmas, _means, _numGauss);

    //load in data file and data to relevant parameters
    _dataFile.load(datafile, arma::raw_ascii);
    _data = _dataFile.col(1);

    //get system parameters such as length and renormalisation value
    _length = _data.n_elem;
    _reflectivityNorm = _data[_qOffset];

    //remove 1/q^4 dependence for the system
    for(int i = 0; i < _length; i++)
    {
        _data[i] *= pow(_dataFile.col(0)[i],4.0);
    }
    
    //save initial data for comparison
    _data.save("dataInit", arma::raw_ascii);

    //set containers for variable data
    _dataCleaned.copy_size(_dataFile);
    _dataCleaned.col(0) = _dataFile.col(0);

    _dataCleanedComplex.set_size(_length);
    _dataComplex.set_size(_length);
    _dataFT.set_size(_length);
    _mask.set_size(_length);

}

/**
 * @brief Private Member function used to create either the default 
 * high frequency or low frequency mask
 * 
 */
void Mask::_GaussianMask()
{
    //mean of low frequency mask is at center so that the gaussian is centered arround the 
    //high frequency middle so it masks off the low frequency regions
    double mean = round(_length/2.0);

    for(int i = 0; i < _length; i++)
    {
        //Mask form is a gaussian with value 1 at the peak
        _mask[i] = exp(-0.5*pow((i - mean)/(_sigma), 2.0));
    }

    //if a high frequency mask is being used the gaussian is cyclically shifted
    //by half the length of the array so that it is now centered on the origin
    if(_highFreqMask == true)
    {
        arma::vec maskTemp = _mask;
        _mask = shift(maskTemp, (int)mean);
    }

}

/**
 * @brief Private Member function used to create and array of Gaussians as specified by the 
 * user in the settings file
 * 
 */
void Mask::_UserDefMask()
{
    //zero all the values in the mask
    _mask.zeros();

    //place a number of gaussians with parameters specified by the 
    //mean and standard deviation array in the configurations file
    for(int j = 0; j < _numGauss; j++)
    {
        for(int i = 0; i < _length; i++)
        {
            _mask[i] += exp(-0.5*pow((i - _means[j])/(_sigmas[j]), 2.0));
        }
    }
}

/**
 * @brief Member function used to perform masking operations 
 * 
 */
void Mask::MaskData()
{
    //make and perform DFTs to get frequency distribution for system 
    _MakeDFT();

    _MakeComplex();

    fftw_execute_dft(_plan, (double(*)[2])&_dataComplex(0), (double(*)[2])&_dataFT(0));

    //Take absolute value of the FT of the data to determine
    //the frequency distribution
    _absDataFT = abs(_dataFT);

    _absDataFT.save("dataFT", arma::raw_ascii); 

    //Apply desired mask
    if(_userDefMask)
    {
        _UserDefMask();
    } 
    else
    {
        _sigma = _sigmas[0];
        _GaussianMask();
    }

    //Apply mask
    _dataFT = _dataFT%_mask;

    //get masked frequency distribution and inver
    _absDataFT = abs(_dataFT);

    _absDataFT.save("dataFTMasked", arma::raw_ascii); 

    //Inverse FT to return to frequency spectrum
    fftw_execute_dft(_planBack, (double(*)[2])&_dataFT(0), (double(*)[2])&_dataCleanedComplex(0));

    //output and renormalise cleaned data
    _dataCleaned.col(1) = abs(_dataCleanedComplex);

    double normalisation = _dataCleaned.col(1)[_qOffset]/pow(_dataCleaned.col(0)[_qOffset],4.0);

    for(int i = 0; i < _length; i++)
    {
        if(i < _qOffset)
        {
            _dataCleaned.col(1)[i] = 0.0;
        }
        else
        {
            _dataCleaned.col(1)[i] *= _reflectivityNorm/(normalisation*pow(_dataCleaned.col(0)[i],4.0));
        }
    }

    //save final data
    _dataCleaned.save("dataMasked", arma::raw_ascii);

}

/**
 * @brief Private member function to create DFT plans 
 * used in this code
 * 
 */
void Mask::_MakeDFT()
{
    _plan = fftw_plan_dft_1d(_length, (double(*)[2])&_dataComplex(0), (double(*)[2])&_dataFT(0), FFTW_FORWARD, FFTW_MEASURE);
    _planBack = fftw_plan_dft_1d(_length, (double(*)[2])&_dataComplex(0), (double(*)[2])&_dataFT(0), FFTW_BACKWARD, FFTW_MEASURE);
}

/**
 * @brief Private member function to destroy DFT plans used in function
 * 
 */
void Mask::_DestroyDFT()
{
    fftw_destroy_plan(_plan);
    fftw_destroy_plan(_planBack); 
}

/**
 * @brief Private member function to make arrays complex for use in DFTs
 * 
 */
void Mask::_MakeComplex()
{
    //create a vector of the same size as the input data but filled with zeros
    arma::vec zerosFill;
    zerosFill.copy_size(_data);
    zerosFill.zeros();

    //combine the _data as the real part and the zeros as the imaginary part to make a complex matrix for DFT
    _dataComplex = arma::cx_vec(_data, zerosFill);
}

/**
 * @brief Use Mask class to clean data
 * 
 * @return int 
 */
int main()
{

    //create and run mask class
    Mask data("data", "settings.cfg");
    data.MaskData();

    return 0;
}
