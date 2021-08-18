/****************************************************************************************
 * 
 * @file DataMask.hpp
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


#ifndef DATAMASK_H
#define DATAMASK_H

/**
 * @brief Mask class which contains functions and variables used to mask 
 * frequency domains in the reflectivity spectrum using either predefined 
 * low/high frequency masks or a user-defined Gaussian distribution
 * 
 */
class Mask{
public:

    /**
     * @brief Construct a new Mask object using just the input datafile
     * 
     * @param datafile The name of the file containing the reflectivity data
     */
    Mask(const std::string&);

    /**
     * @brief Construct a new Mask object
     * 
     * @param datafile The name of the file containing the reflectivity data
     * @param configfile The name of the file containing the settings data
     */
    Mask(const std::string& datafile, const std::string& configfile);

    /**
     * @brief Member function used to perform masking operations 
     * 
     */
    void MaskData();

private:

    /**
     * @brief Private Member function used to create either the default 
     * high frequency or low frequency mask
     * 
     */
    void _GaussianMask();

    /**
     * @brief Private Member function used to create and array of Gaussians as specified by the 
     * user in the settings file
     * 
     */
    void _UserDefMask();

    /**
     * @brief Private member function to create DFT plans 
     * used in this code
     * 
     */
    void _MakeDFT();

    /**
     * @brief Private member function to destroy DFT plans used in function
     * 
     */
    void _DestroyDFT();

    /**
     * @brief Private member function to make arrays complex for use in DFTs
     * 
     */
    void _MakeComplex();

    /**
     * @brief The length of the array containing the data being masked
     * 
     */
    double _length;

    /**
     * @brief Matrix containing full set of datas
     * 
     */
    arma::mat _dataFile;

    /**
     * @brief Vector containing reflectivity data converted to a complex array
     * 
     */
    arma::cx_vec _dataComplex;

    /**
     * @brief Vector containing raw reflectivity data, no scales
     * 
     */
    arma::vec _data;

    /**
     * @brief Fourier transform of the data
     * 
     */
    arma::cx_vec _dataFT;

    /**
     * @brief Absolute value of the fourier transform of the data, used for plotting and
     * visualisation
     * 
     */
    arma::vec _absDataFT;

    /**
     * @brief Complex array containing the cleaned data
     * 
     */
    arma::cx_vec _dataCleanedComplex;

    /**
     * @brief Matrix containing rescaled, cleaned data.
     * 
     */
    arma::mat _dataCleaned;

    /**
     * @brief FFTW plan for forwards transforms
     * 
     */
    fftw_plan _plan;

    /**
     * @brief FFTW plan for backwards transforms 
     * 
     */
    fftw_plan _planBack;

    /**
     * @brief Normalisation for the reflectivity data.
     * This ios used to renormalise the output data which will have
     * picked up factors in the FT and masking
     * 
     */
    double _reflectivityNorm;

    /**
     * @brief q index offset of the reflectivity data to select only the physical region of the transform
     * 
     */
    int _qOffset; 

    /**
     * @brief Masking function
     * 
     */
    arma::vec _mask; 

    /**
     * @brief Standard deviation for regular masks
     * 
     */
    double _sigma;

    /**
     * @brief Vector containing standard deviations for multi-gauss mask
     * 
     */
    arma::vec _sigmas;

    /**
     * @brief Vector containing means for multi-gauss mask
     * 
     */
    arma::vec _means;

    /**
     * @brief Number of Gaussians in the system
     * 
     */
    int _numGauss;

    /**
     * @brief Boolean specifying a low frequency mask is being used
     * 
     */
    bool _lowFreqMask = true;

    /**
     * @brief Boolean specifying a high frequency mask is being used
     * 
     */
    bool _highFreqMask;


    /**
     * @brief Boolean specifying a user defined mask is being used
     * 
     */
    bool _userDefMask;


};

#endif