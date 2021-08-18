/****************************************************************************************
 * @file ReflectivityClass.hpp                                                 
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
     * @brief Construct a new Reflectivity:: Reflectivity object
     * 
     * @param configFile Configuration file containg information to create an SLD profile
     * @param dataType What type of data is being used as the input data. Options are either a file containing an actual SLD curve, 'SLD',
     *  or a configurations file containing a series of SLDs and their corresponding thicknesses 
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
     * @brief Class member funciton used to generate the SLD profile from the component parts
     * 
     * @param depths Array containing substance depths
     * @param n Array containing susbtance amunts
     * @param total Total amount of substance
     * @param sldVal SLD of substance
     */
    void _SLDGenerate(const arma::vec&, const arma::vec&, const double&, const double&);

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
     * @brief Scale charge to physical SLD parameters so that value constraints can be imposed on the system
     * 
     */
    void _BoundSLD();

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
     * @brief Scaling factor for x-axis
     * 
     */
    double _depthScale;

    /**
     * @brief Number of different substances in this system
     * 
     */
    int _numSubstances; 

    /**
     * @brief Vector containing thickness of each layer
     * 
     */
    std::vector<arma::vec> _depths;

    /**
     * @brief Vector containing the amount of substance of each layer
     * 
     */
    std::vector<arma::vec> _n;

    /**
     * @brief Value of the SLD of this substance
     * 
     */
    std::vector<double> _sldVal;

    /**
     * @brief Total amount of this substance
     * 
     */
    std::vector<double> _total; 

    /**
     * @brief Vector containing thickness of each layer
     * 
     */

    /**
     * @brief The value of the SLD of the substrate
     * 
     */
    double _substrateSLD;

    /**
     * @brief The length of the substrate in index units
     * 
     */
    int _lengthSubstrate;

    /**
     * @brief Boolean to determine whether to use volumetric normalisation or not
     * 
     */
    bool _volumetricNormalisation = false; 

    /**
     * @brief Vector to store total number of species in each element for scaling
     * 
     */
    arma::vec _scalingVector;

    /**
     * @brief 
     * 
     */
    bool _toyModel = true;

    /**
     * @brief Boolean stating whether sld should be scaled
     * 
     */
    bool _boundSLD;

    /**
     * @brief Minimum bound of profile if known
     * 
     */
    double _sldMinBound;

    /**
     * @brief Maximum bound of profile if known
     * 
     */
    double _sldMaxBound;

    /**
     * @brief Avogadros Number
     * 
     */
    double _nA = 6.02214076e+23; 

    /**
     * @brief Offset in Q space
     * 
     */
    int _qOffset = 0;
    
    /**
     * @brief Components in the material
     * 
     */
    int _componentNum;

};

/**
 * @brief Template to determine the sign of a number
 * 
 * @tparam T 
 * @param x
 * @return int 
 */
template <typename T> int sign(T x) {
    return (T(0) < x) - (x < T(0));
}

#endif


