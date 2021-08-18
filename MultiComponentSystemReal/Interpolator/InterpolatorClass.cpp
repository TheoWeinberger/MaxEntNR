/****************************************************************************************
 * 
 * @file InterpolatorClass.cpp
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

#include <libalglib/interpolation.h>
#include <armadillo>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "InterpolatorClass.hpp"
#include "InterpolatorSettings.hpp"

/**
 * @brief Construct a new Interpolator object from reflectivity dataset 
 * 
 * @param filename The name of the data file containing the original spacing and the reflectivity data
 */
Interpolator::Interpolator(const std::string& filename)
{
    //read in the datafile
    _dataIn.load(filename, arma::raw_ascii);

    //For the default system the number of output sections is the same as the number of inputs
    //the output is just regularly spaced
    _sectionsIn = _dataIn.n_rows; 
    _sectionsOut = _sectionsIn;

    //run the interpolation
    _Interpolate();
}

/**
 * @brief Construct a new Interpolator object from reflectivity dataset where the number of sections is sepcified by the user
 * 
 * @param filename The name of the data file containing the original spacing and the reflectivity data
 * @param settings Settings file for the system
 */
Interpolator::Interpolator(const std::string& filename, const std::string& settings)
{
    //read in the datafile
    _dataIn.load(filename, arma::raw_ascii);

    ReadFile(settings, _sectionsOut, _errors, _rangeTimes, _accountError, _errorKnown, _resLimit);

    //For this constructor the number of output sections is specified by the suer
    //again the output is on a regularly spaced grid
    _sectionsIn = _dataIn.n_rows; 
    

    //run the interpolation
    _Interpolate();

}

/**
 * @brief Private Member function used to run interpolation script
 * 
 */
void Interpolator::_Interpolate()
{
    //get the number of columns of the input data
    //if there are two columns that is just the Q and reflectivity
    //a third column corresponds to the inclusion of errors
    int numCols = _dataIn.n_cols;

    //if there are three columns, set the boolean _errors to true so that the rest of the 
    //code uses the error containing sections
    if(numCols == 3)
    {
        _errors = true;
    }
    else
    {
        _errors = false;
    }

    //set input and output array sizes
    _coordinatesIn.setlength(_sectionsIn);
    _reflectivityIn.setlength(_sectionsIn);
    _errorsIn.setlength(_sectionsIn);

    _coordinatesOut.setlength(_sectionsOut);
    _reflectivityOut.setlength(_sectionsOut);
    _errorsOut.setlength(_sectionsOut);

    //seperate 2d array containing all the information
    //as read in to _dataIn, into two/three arrays containing the separated data
    for(int i = 0; i < _sectionsIn; i++)
    {
        _coordinatesIn[i] = _dataIn(i,0);
        _reflectivityIn[i] = _dataIn(i,1);
        if(_errors == true)
        {
            _errorsIn[i] = _dataIn(i,2);
        }
    }
    
    //get the max and min Q values for the system
    double maxCoord = _coordinatesIn[_sectionsIn -1];
    double minCoord = _coordinatesIn[0];

    //determine the max range of the data for this interpolation
    double dataRange = _rangeTimes*maxCoord;

    //calculate the uniform spacing between elements
    double delta = (dataRange)/((double)_sectionsOut - 1);
    arma::vec sincMod; 
    sincMod.set_size(_sectionsOut);

    //determine the scale of the sinc modulating the reflectivity data
    double sincScaling;

    if(_errorKnown == true)
    {
        sincScaling = _resLimit/((2.0*M_PI));
    }
    else
    {
        sincScaling = maxCoord - minCoord;
    }

    //create the uniform coordinate grid
    for(int i = 0; i < _sectionsOut; i++)
    {
        _coordinatesOut[i] = i*delta;
        sincMod[i] = pow((sincScaling)/(2.0*M_PI)*sin((2.0*M_PI*i*delta)/(sincScaling))/((double)i*delta),2.0);
    }
    sincMod[0] = 1.0;

    //calculate interpolated reflectivity 
    alglib::spline1dconvcubic(_coordinatesIn, _reflectivityIn, _coordinatesOut, _reflectivityOut);

    //if the datafile contains errors interpolate these two (although this doesn't really make sense)
    if(_errors == true)
    {
        alglib::spline1dconvcubic(_coordinatesIn, _errorsIn, _coordinatesOut, _errorsOut);
    }

    //create output arma::mat to store data in
    if(_errors == true)
    {
        _dataOut.set_size(_sectionsOut,3);
    }
    else
    {
        _dataOut.set_size(_sectionsOut,2);
    }
    
    //read in output data to arma::mat
    for(int i = 0; i < _sectionsOut; i++)
    {
        _dataOut(i,0) = _coordinatesOut[i];

        if(_dataOut(i,0) < minCoord)
        {
            _dataOut(i,1) = 0.0;
        }
        else if(_dataOut(i,0) > maxCoord)
        {
            _dataOut(i,1) = 0.0;
        }
        else
        {
            //account for experimental errors in the interpolated data or not
            if(_accountError == true)
            {
                if(_dataOut(i,0) < maxCoord/3.0)
                {   
                    _dataOut(i,1) = _reflectivityOut[i]/sincMod[i];
                }
                else
                {
                    _dataOut(i,1) = 0.0;
                }
            }
            else
            {
                _dataOut(i,1) = _reflectivityOut[i];
            }

            if(_errors == true)
            {
                _dataOut(i,2) = _errorsOut[i];
                //account for experimental errors in the interpolated data or not
                if(_accountError == true)
                {
                    if(_dataOut(i,0) < maxCoord/3.0)
                    {   
                        _dataOut(i,2) = _errorsOut[i]/sincMod[i];
                    }
                    else
                    {
                        _dataOut(i,2) = 0.0;
                    }
                }
            }
        }
    }

    //store interpolated data to an ascii file
    _dataOut.save("InterpolatedData", arma::raw_ascii);
}

//run interpolator
int main()
{
    //note here the settings file is being used
    Interpolator("data", "settings.cfg");
    return 0;
}