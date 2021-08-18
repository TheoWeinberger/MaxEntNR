/****************************************************************************************
 * 
 * @file InterpolatorClass.hpp
 * @author Theo Weinberger
 * 
 ******************************************************************************************
 * 
 * @brief C++ class to run interpolation script to interpolate data so it is on a uniform line
 * uses the cubic spline method from the ALGLIB package. Often input data is has a non uniform
 * reflectivity scale and this serves to fix it.
 * 
 * The number of segments that it is divided into is specified by the user/settings file
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
 * For further Matrix manipulation and cubic spline: ALGLIB from www.alglib.net, Sergey Bochkanov
 * 
 * For accessing settings files: libconfig from http://hyperrealm.github.io/libconfig/
 * 
 *****************************************************************************************
 *
 * @version 1.0
 * @date 2021-05-31
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef INTERPOLATORCLASS_H
#define INTERPOLATORCLASS_H

/**
 * @brief Class object to run interpolation script for data
 * 
 */
class Interpolator{
public:

    /**
     * @brief Construct a new Interpolator object from reflectivity dataset 
     * 
     * @param filename The name of the data file containing the original spacing and the reflectivity data
     */
    Interpolator(const std::string&);

    /**
     * @brief Construct a new Interpolator object from reflectivity dataset where the number of sections is sepcified by the user
     * 
     * @param filename The name of the data file containing the original spacing and the reflectivity data
     * @param settings Settings file for the system
     */
    Interpolator(const std::string&, const std::string&);

private:

    /**
     * @brief Private Member function used to run interpolation script
     * 
     */
    void _Interpolate();

    /**
     * @brief Matrix to contain interpolated data
     * 
     */
    arma::mat _dataOut;

    /**
     * @brief Array to contain input data
     * 
     */
    arma::mat _dataIn;

    /**
     * @brief input coordinates - these may not be regularly spaced
     * 
     */
    alglib::real_1d_array _coordinatesIn;

    /**
     * @brief Input reflectivity data
     * 
     */
    alglib::real_1d_array _reflectivityIn;

    /**
     * @brief Input error data
     * 
     */
    alglib::real_1d_array _errorsIn;

    /**
     * @brief Output coordinates - these will be regularly spaced
     * 
     */
    alglib::real_1d_array _coordinatesOut;

    /**
     * @brief Output reflectivity data
     * 
     */
    alglib::real_1d_array _reflectivityOut;

    /**
     * @brief Output error data
     * 
     */
    alglib::real_1d_array _errorsOut;

    /**
     * @brief The number of sections in the input data
     * 
     */
    int _sectionsIn;

    /**
     * @brief The number of sections that the interpolator divides the system into
     * 
     */
    int _sectionsOut;

    /**
     * @brief Boolean value determining whether or not the input data contains error data
     * 
     */
    bool _errors = false;

    /**
     * @brief factor by which to scale the range of the data
     * 
     */
    double _rangeTimes = 1.0;

    /**
     * @brief Boolean determining whether experimental errors should be accounted for 
     * 
     */
    bool _accountError = true;

    /**
     * @brief Boolean stating if experimental error is to be accounted for whether the known resolution limit should be used 
     * or whether it should be approximated from the data range
     * 
     */
    bool _errorKnown = false;

    /**
     * @brief If the error is known this is then the limit of the resolution given in angstroms
     * 
     */
    double _resLimit;


};

#endif
