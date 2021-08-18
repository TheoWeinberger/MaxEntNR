/****************************************************************************************
 * 
 * @file NewtonClass.cpp
 * @author Theo Weinberger 
 * 
 ****************************************************************************************
 
 * @brief Chisquared minimiser that employs the Newton method of optimisation to fit 
 * a partially fit initial profile to a final SLD profile. This version does not include the
 * 1/q^4 dependence in the data.
 * It reconstructs the SLD profile from the reflectivity spectrum method using the 
 * Fourier form of the reflectivity relationship which is defined to be
 * 
 *          R[Q] = \frac{1}{Q^4} * |FFT(SLD)[Q]*[1-exp{-Q*I*(2*PI/totalDepth)}] +
 *                           SLD[totalDepth - 1] - SLD[0]|^2
 * 
 * The Newton method iteratively minimises the fit within a local region where 
 * it assumes a quadratic fit where each step follows the relation
 * 
 *                      SLD{_n+1} = SLD_n - H^{-1} \nabla C
 * 
 *      (1) Avriel, Mordecai (2003). Nonlinear Programming: 
 *          Analysis and Methods. Dover Publishing. 
 *          ISBN 0-486-43227-0.
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
 * @version 2.0
 * @date 2021-06-30
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
#include "NewtonClass.hpp"
#include "NewtonSettings.hpp"

/**
 * @brief Main code used to run Newton algorithm on a data set
 * 
 * @return int Exit Code
 */

int main()
{
    //create Newton object to be solved for
    Newton sldReconstruction("reflectivitySpectrum", "settings.cfg");

    //apply Newton's method
    sldReconstruction.Solve();

    return 0;  
}
