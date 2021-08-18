/****************************************************************************************
 * @file main.cpp                                                 
 * @brief Code to first generate a reflectivity spectrum from a known SLD profile using the 
 * FOurier form the the transform this is done by the ReflectivityClass.hpp method.
 * Then it reconstructs the SLD profile from the reflectivity spectrum via the MaxEnt method using the 
 * Fourier form of the reflectivity relationship which is defined to be
 * 
 *          R[Q] = \frac{1}{Q^4} * |FFT(SLD)[Q]*[1-exp{-Q*I*(2*PI/totalDepth)}] +
 *                           SLD[totalDepth - 1] - SLD[0]|^2
 * 
 * This is done by the MaxEntClass.hpp method
 * 
 * This method does not remove 1/q^4 dependence but does scale the system gradients down to
 * the new scaled charges
 * 
 * The code used the Cambridge Algorithm for maximum entropy
 *
 *    (1)  John Skilling and S.F. Gull, in Maximum-Entropy and
 *         Bayesian Methods in Inverse Problems Ed. C.Ray Smith
 *         and W.T.Grandy Jr., (1985), pp83-132, D. Reidel
 *         Publishing Company.
 *
 *    (2)  John Skilling, in Maximum Entropy and Bayesian Methods 
 *         in Applied Statistics, Ed. James H. Justice, (1986), 
 *         pp179-193, CUP.
 *
* The code here was aided by previous work by Elliott and Hanna applying MaxEnt techniques
 * to Fraunhofer diffraction based systems
 * 
 *    (1) James Elliott and Simon Hanna, A model-independent maximum-entropy method for the
 *        inversion of small-angle X-ray diffraction patterns (1999). 
 *        J. Appl. Cryst. 32, 1069-1083.
 *
 * The paper regarding this algorithm and its applications to Nafion thin film systems is
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
 * @version 1.0
 * @date 2021-07-01
 * 
 * @copyright Copyright (c) 2021
 *
 ****************************************************************************************
 */
#include <iostream>
#include <armadillo>
#include <cmath>
#include <omp.h>
#include <stdio.h>
#include <complex.h>
#include <libconfig.h++>
#include <fftw3.h>
#include "Reflectivity/ReflectivityClass.hpp"
#include "MaxEnt/MaxEntClass.hpp"

/**
 * @brief main function to execute all code
 * 
 * @return int Return 0 if code executes correctly
 */
int main()
{  
    //initialise SLDs
    Reflectivity reflectionSpectrum("settings.cfg", "config");

    //Get final reflectivity data
    arma::mat reflectivity = reflectionSpectrum.GetRefScaled();
    arma::mat reflectivityRaw = reflectionSpectrum.GetRef();
    arma::mat sld = reflectionSpectrum.GetSLDScaled();

    //save reflectivity data in txt file
    reflectivity.save("reflectivitySpectrum", arma::raw_ascii);
    reflectivityRaw.save("data", arma::raw_ascii);
    sld.save("SLD", arma::raw_ascii);

    //matrices to store output data in
    arma::mat SLDRecon, SpectrumRecon;
    MaxEnt SLDReconstruction("reflectivitySpectrum", "settings.cfg");

    //apply max ent method
    SLDReconstruction.Solve();

    //create scaled output data
    SLDRecon = SLDReconstruction.GetSLDScaled();
    SpectrumRecon = SLDReconstruction.GetReflectivityScaled();

    SLDRecon.save("reconstructedSLD", arma::raw_ascii);
    SpectrumRecon.save("reconstructedSpectrum", arma::raw_ascii);

    return 0; 
}