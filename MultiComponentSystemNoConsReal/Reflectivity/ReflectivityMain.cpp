/****************************************************************************************
 * @file ReflectivityMain.cpp                                                 
 * @brief Code to generate the reflectivity pattern from a SLD profile using the 
 * Fourier form of the reflectivity relationship which is defined to be
 * 
 *          R[Q] = \frac{1}{Q^4} * |FFT(SLD)[Q]*[1-exp{-Q*I*(2*PI/totalDepth)}] +
 *                           SLD[totalDepth - 1] - SLD[0]|^2
 *
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
 * @version 1.0
 * @date 2021-05-13
 * 
 * @copyright Copyright (c) 2021
 *
 ****************************************************************************************
 */

#include <iostream>
#include <armadillo>
#include <cmath>
#include <complex.h>
#include <libconfig.h++>
#include <fftw3.h>
#include "ReflectivityClass.hpp"

/**
 * @brief main function to execture reflectivity code
 * 
 * @return int Return 0 if code executes correctly
 */
int main()
{

    //initialise SLDs
    Reflectivity reflectionSpectrum("ReflectivitySettings.cfg", "config");

    //Get final reflectivity data
    arma::mat reflectivity = reflectionSpectrum.GetRefScaled();
    arma::mat reflectivityRaw = reflectionSpectrum.GetRef();
    arma::mat sld = reflectionSpectrum.GetSLDScaled();
    arma::vec sldRaw = reflectionSpectrum.GetSLD();

    //save reflectivity data in txt file
    reflectivity.save("reflectivitySpectrum", arma::raw_ascii);
    reflectivityRaw.save("data", arma::raw_ascii);
    sld.save("SLD", arma::raw_ascii);
    sldRaw.save("SLDRaw", arma::raw_ascii);

    return 0;

}