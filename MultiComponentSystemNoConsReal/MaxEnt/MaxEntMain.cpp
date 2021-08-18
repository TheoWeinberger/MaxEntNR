/****************************************************************************************
 * @file MaxEntMain.cpp                                                 
 * @brief Code to reconstruct the SLD profile from the reflectivity spectrum via the MaxEnt method using the 
 * Fourier form of the reflectivity relationship which is defined to be
 * 
 *          R[Q] = \frac{1}{Q^4} * |FFT(SLD)[Q]*[1-exp{-Q*I*(2*PI/totalDepth)}] +
 *                           SLD[totalDepth - 1] - SLD[0]|^2
 * 
 * This method get ride of the 1/q^4 dependence
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
 * @version 3.0
 * @date 2021-07-01
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
#include "MaxEntClass.hpp"

/**
 * @brief Main code used to run Maximum Entropy, Cambridge algorithm on a data set
 * 
 * @return int Exit Code
 */
int main()
{
    //matrices to store output data in
    MaxEnt sldReconstruction("data", "settings.cfg");

    //apply max ent method
    sldReconstruction.Solve();
    return 0;  
}
