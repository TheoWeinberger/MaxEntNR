/****************************************************************************************
 * 
 * @file NewtonClass.cpp
 * @author Theo Weinberger 
 * 
 ****************************************************************************************

 * @brief Chisquared minimiser that employs the Newton method of optimisation to fit 
 * a partially fit initial profile to a final SLD profile. This version includes the
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
#include <fftw3.h>
#include <libconfig.h++>
#include <assert.h>
#include "NewtonClass.hpp"
#include "NewtonSettings.hpp"

/**
 * @brief Construct a Newton object which reads in data from a string which specifies the file in 
 * which the data is stored. The data file should contain the reflectivity data in a one dimensionol column without
 * scaled axes (the algorirthm uses unscaled data as any scaling is normalised out).
 * 
 */
Newton::Newton(const std::string& dataFile)
{

    //load in file containing reflectivity data
    arma::mat dataTemp;
    dataTemp.load(dataFile, arma::raw_ascii);

    _dataFit = dataTemp.col(1);

    //Determine system scaling
    _reflectivityScale = dataTemp.col(0);
    _deltaQ = _reflectivityScale[1] - _reflectivityScale[0];
    _reflectivityNorm = _dataFit[_qOffset];


    //System is undefined if _qOffset = 0, therefore _qOffset = 0 is defined to be equivalent to _qOffset = 1
    if(_qOffset == 0)
    {
        _qOffset = 1;
    }

    //define system _depth which is equal to the number of elements in the array
    _depth = _dataFit.n_elem;

    //for default system _qCutOff = _depth so there is no cropping
    _qCutOff = _depth;

    //Load in data, data outside of qOFfset and qCutOff is cropped
    for(int i = 0; i < _depth; i++)
    {
        if(i < _qOffset)
        {
            _dataFit[i] = _dataFit[_qOffset];
        }
        if( i > _qCutOff)
        {
            _dataFit[i] = _dataFit[_qCutOff];
        }
        //transformation from real to reciprocal space is 1/deltaQ
        _sldScale[i] = 2*M_PI*i/(_deltaQ*_depth) ;
    }



    //renormalise input data so max value of intensity is 1
    double maxData =  _dataFit.max();
    _dataFit /= maxData/_dataScale; 


    //output data used for fitting
    _dataFit.save("dataInit", arma::raw_ascii);

    //set data vector sizes
    _inverseVar.set_size(_depth);
    _sld.set_size(_depth);
    _sldTransform.set_size(_depth);
    _sldTransformConj.set_size(_depth);
    _sldImage.set_size(_depth);
    _gradChiSquared.set_size(_depth);
    _hessian.set_size(_depth,_depth);

    _charge.resize(_numSubstances);
    _sldVal.set_size(_numSubstances);
    _total.set_size(_numSubstances);

    for(int i = 0; i < _numSubstances; i++)
    {
        _charge[i].set_size(_depth);
    }
}

/**
 * @brief Construct a new Newton object which reads in data from a string which specifies the file in 
 * which the data is stored. The data file should contain the reflectivity data in a one dimensionol column without
 * scaled axes (the algorirthm uses unscaled data as any scaling is normalised out). The second string refers to the 
 * settings file, which is a .cfg file containing the simulation specifics such as run number that are used in
 * the Newton algortihm
 * 
 */
Newton::Newton(const std::string& dataFile, const std::string& configFile)
{
    //read in data for diffraction grating to be built from
    ReadFile(configFile, _totalIterations, _zeroLevel, _minVar, _dataScale, _numSubstances, _propagationSLD, _substrateSLD, _lengthPropagation, _lengthSubstrate, _smoothIncrement, _useEdgeConstraints, _sldVal, _useDamping, _alphaInit, _alphaFactor, _gammaInit, _gammaFactor, _forceZero, _fracMax, _forceInterval, _smoothProfile, _smoothInterval, _volumetricNormalisation, _boundSLD, _sldMaxBound, _sldMinBound, _qOffset, _qCutOff, _realWorldScaling, _lengthPropagationReal, _lengthSubstrateReal, _qOffsetReal, _qCutOffReal, _error);

    //get datafile
    arma::mat dataTemp;
    dataTemp.load(dataFile, arma::raw_ascii);

    //load in file containing reflectivity data
    _dataFit = dataTemp.col(1);

    //Determine system scaling
    _reflectivityScale = dataTemp.col(0);
    _deltaQ = _reflectivityScale[1] - _reflectivityScale[0];

    //scale cutoffs to real world if needed
    if(_realWorldScaling == true)
    {
        _RealToIndex(_reflectivityScale, _qOffsetReal, _qOffset);
        _RealToIndex(_reflectivityScale, _qCutOffReal, _qCutOff);
    }

    //System is undefined if _qOffset = 0, therefore _qOffset = 0 is defined to be equivalent to _qOffset = 1
    if(_qOffset == 0)
    {
        _qOffset = 1;
    }

    //renormalise data and get the first value of the fitting data - this will be used for 
    //renormalisation at ouput
    _reflectivityNorm = _dataFit[_qOffset];
    _dataFit /= _reflectivityNorm;

    //check whether system is valid for using errors
    if(dataTemp.n_cols != 3 && _error == true)
    {
        std::cout << "No errors in the dataset, assuming Poissonian statistics" << std::endl;
        std::cout << "Press X to quit or any other button to continue with the simulation" << std::endl;
        char answer;
        std::cin >> answer;
        if(answer == 'X' || answer == 'x')
        {
            exit(EXIT_SUCCESS);
        }
        _error = false;
    }

    //if all criteria for holds for error usage, load in errors
    if(dataTemp.n_cols == 3 && _error == true)
    {
        _inverseVar = dataTemp.col(2);
    }

    //define system _depth which is equal to the number of elements in the array
    _depth = _dataFit.n_elem;

    //Maximum value of qCutOff is the depth of the system
    if(_qCutOff > _depth)
    {
        _qCutOff = _depth;
    }

    _sldScale.set_size(_depth);

    //Load in data, data outside of qOffset and qCutOff is cropped
    for(int i = 0; i < _depth; i++)
    {
        if(i < _qOffset)
        {
            _dataFit[i] = _dataFit[_qOffset];
            if(_error == true)
            {
                _inverseVar[i] = _inverseVar[_qOffset];
            }
        }
        if(i > _qCutOff)
        {
            _dataFit[i] = _dataFit[_qCutOff];
            if(_error == true)
            {
                _inverseVar[i] = _inverseVar[_qCutOff];
            }
        }
        //transformation from real to reciprocal space is 1/deltaQ
        _sldScale[i] = 2*M_PI*i/(_deltaQ*_depth) ;
    }

    //convert scaling if required
    if(_realWorldScaling == true)
    {
        _RealToIndexLength(_sldScale, _lengthSubstrateReal, _lengthSubstrate);
        _RealToIndexLength(_sldScale, _lengthPropagationReal, _lengthPropagation);
    }


    //renormalise input data so max value of intensity is 1
    double maxData =  _dataFit.max();
    _dataFit /= maxData/_dataScale;

    if(_error == true)
    {
        _inverseVar /= maxData/_dataScale;
    }

    //Matrix to output scaled data to 
    arma::mat dataFitScaled;
    dataFitScaled.set_size(_depth,2);
    
    for(int i = 0; i < _depth; i++)
    {
        dataFitScaled(i,0) = i;
        dataFitScaled(i,1) = _dataFit[i];
    }

    //output data used for fitting
    dataFitScaled.save("dataInit", arma::raw_ascii);

    //set data vector sizes
    _inverseVar.set_size(_depth);
    _sld.set_size(_depth);
    _sldTransform.set_size(_depth);
    _sldTransformConj.set_size(_depth);
    _sldImage.set_size(_depth);
    _gradChiSquared.set_size(_depth);
    _delta.set_size(_depth);
    _tempCharge.set_size(_depth);
    _hessian.set_size(_depth,_depth);

    _charge.resize(_numSubstances);
    _total.set_size(_numSubstances);

    for(int i = 0; i < _numSubstances; i++)
    {
        _charge[i].set_size(_depth);
    }
}

/**
 * @brief Class member function to apply one step of the MaxEnt algorithm
 * 
 * @param charge The charge of one of the components of the system
 * @param norm The normalisation of the component of the system
 * @param def A parameter used to define the contrained entropy such that the total charge can be kept constant without need for an additional lagrange multiplier
 */
void Newton::_Step(arma::vec& charge, double& norm, double& sld)
{

    _ConjSLD();

    //get chisquared data for this iterations
    _ChiSquared();
    _GradChiSquared(sld);
    _Hessian(sld);

    //calculate the next step for ths system
    _CalcNewCharge(charge, sld);

    //if the matrix equation is not solvable, end the fitting and output the current profile and images
    if(_solvable != true)
    {
        arma::mat sldOutput = GetSLDScaled();
	    arma::mat imageOutput = GetReflectivityScaled();

        for(int i = 0; i < _numSubstances; i++)
        {
            arma::mat chargeOutput = GetChargeScaled(_charge[i]);
            chargeOutput.save("charge" + std::to_string(i + 1) + "FitFinal", arma::raw_ascii);
        }

        sldOutput.save("sldFitFinal", arma::raw_ascii);
        imageOutput.save("imageFitFinal", arma::raw_ascii);
        std::cout << "--------------- Newton Algorithm - end fitting ----------------" << std::endl;
        std::cout << "Newton's algorithm ended early due to unsolvable matrix equation" << std::endl;
        exit(EXIT_FAILURE);
    }

    _StepNewCharge(charge, norm);

    //generate current SLD for system
    _SLDGenerate();

    //claculate current charge image, and FTs
    _Reflectivity();
}

/**
 * @brief Class member function to print out data from Newton algortihm
 * 
 */ 
void Newton::_Print()
{
    std::cout << "Iteration Number --- Chi Squared --- Reduced Chi Squared" << std::endl;
    std::cout << "-----" << _iterationCount << "-----------------" << _chiSquared << "-------------" << _redChi  << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
}

/**
 * @brief Class member function to save data from Newton algorithm
 * 
 */
void Newton::_Store()
{
    //output charge and image data for the specified iteration numbers
    //this allows for evaluation of how the charge distribution and image fitting
    //process evolve
    if(_iterationCount == 10 || _iterationCount == 100 || _iterationCount == 1000)
	{
        arma::mat sldOutput = GetSLDScaled();
        arma::mat imageOutput = GetReflectivityScaled();

        for(int i = 0; i < _numSubstances; i++)
        {
            arma::mat chargeOutput = GetChargeScaled(_charge[i]);
            chargeOutput.save("chargeFit" + std::to_string(i + 1) + "_" + std::to_string(_iterationCount), arma::raw_ascii);
        }

        sldOutput.save("sldFit" + std::to_string(_iterationCount), arma::raw_ascii);
        imageOutput.save("imageFit" + std::to_string(_iterationCount) , arma::raw_ascii);
	}
    if(_iterationCount == _totalIterations)
    {
        arma::mat sldOutput = GetSLDScaled();
	    arma::mat imageOutput = GetReflectivityScaled();

        for(int i = 0; i < _numSubstances; i++)
        {
            arma::mat chargeOutput = GetChargeScaled(_charge[i]);
            chargeOutput.save("charge" + std::to_string(i + 1) + "FitFinal", arma::raw_ascii);
        }

        sldOutput.save("sldFitFinal", arma::raw_ascii);
        imageOutput.save("imageFitFinal", arma::raw_ascii);
    }
}

/**
 * @brief Employs Newton algorithm to further fit the maximum entropy SLD profile consistent with the 
 * input data that has been produced by the MaxEnt code
 * 
 */
void Newton::Solve()
{
    std::cout << "-------------- Newton Algorithm - beginning fitting --------------" << std::endl;

    //Initialise system so that Newton's algortihm can be employed
    _InitStatQuant();
    _MakeDFTPlans();
    //generate current SLD for system
    _SLDGenerate();
    //claculate current charge image, and FTs
    _Reflectivity();

    arma::mat sldOutput = GetSLDScaled();
    arma::mat imageOutput = GetReflectivityScaled();

    sldOutput.save("sldInit", arma::raw_ascii);
    imageOutput.save("imageInit", arma::raw_ascii);

    //main body of Newton
    //at each overall step the Newton for that step is first employed
    //Next the relevant data to check simulation progression is outpute
    //Then the image and charge are stored for certain iteration counts
    //This occurs until the iteration counter maxima as specified in the settings file is reached
    while(_iterationCount <= _totalIterations)
	{
        for(int _currentCharge = 0; _currentCharge < _numSubstances; _currentCharge++)
        {
            _Step(_charge[_currentCharge], _total[_currentCharge], _sldVal[_currentCharge]);
            _Print();
        }
        _Store();

        _iterationCount += 1;
	}

    std::cout << "--------------- Newton Algorithm - end fitting ----------------" << std::endl;
    
    _DeleteDFTPlans();
}

/**
 * @brief Get the scaled SLD 
 * 
 * @return arma::mat @param sldScaled the scaled SLD
 */
arma::mat Newton::GetSLDScaled()const
{
    //Matrix to store SLD data in 
    arma::mat sldScaled;
    sldScaled.set_size(_depth,2);


    //scale the SLD    
    for(int i = 0; i < _depth; i++)
    {
        sldScaled(i,0) = _sldScale[i];
        sldScaled(i,1) = _sld[i];
    }

    return sldScaled;
}

/**
 * @brief Get the scaled charge
 * 
 * @param charge The charge of one of the components of the system
 * @return arma::mat @param chargeScaled the scaled charge
 */
arma::mat Newton::GetChargeScaled(arma::vec& charge)const
{
    //Matrix to store SLD data in 
    arma::mat chargeScaled;
    chargeScaled.set_size(_depth,2);


    //scale the SLD    
    for(int i = 0; i < _depth; i++)
    {
        chargeScaled(i,0) = _sldScale[i];
        chargeScaled(i,1) = charge[i];
    }

    return chargeScaled;
}

/**
 * @brief Get the scaled Reflectivity
 * 
 * @return arma::mat @param reflectivityScaled the scaled reflectivity
 */
arma::mat Newton::GetReflectivityScaled()const
{
    //Matrix to store reflectivity data in
    arma::mat reflectivityScaled;
    reflectivityScaled.set_size(_depth,2);
    double normalisation =_sldImage[_qOffset];

    
    //apply scaling for output data so it matches magnitude of input data
    for(int i = 0; i < _depth; i++)
    {
        reflectivityScaled(i,0) = _reflectivityScale[i];
        
        if(i < _qOffset)
        {
            //set the first value
            reflectivityScaled(i,1) = 0.0;
        }
        else 
        {

            reflectivityScaled(i,1) = (_sldImage[i]*_reflectivityNorm)/normalisation;
        }
        
       //reflectivityScaled(i,1) = _sldImage[i];
    }
    return reflectivityScaled;
}

/**
 * @brief Class member function used to initialise relevant statistical quantities to be used throughout Newton algorithm
 * 
 */
void Newton::_InitStatQuant()
{
    //assign matrix for inverse variance and calculate each value
    //if errors are being used, use them. If not, Poissonian varaince is assumed
    if(_error == true)
    {
        for(int i = 0; i < _depth; i++)
        {
            if (_inverseVar[i] > _minVar)
            {
                _inverseVar[i] = 1.0/_inverseVar[i];   
            }
            else
            {
                _inverseVar[i] = 1.0/_minVar;
            }
        }
    }
    else
    {
        for(int i = 0; i < _depth; i++)
        {
            if (_dataFit[i] > _minVar)
            {
                _inverseVar[i] = 1.0/_dataFit[i];   
            }
            else
            {
                _inverseVar[i] = 1.0/_minVar;
            }
        }
    }

    //data normalisation is determined to be the sum over the input data
    _norm = accu(_dataFit);


    for(int i = 0; i < _numSubstances; i++)
    {
        //load in file containing initial charge data
        arma::mat chargeTemp;
        chargeTemp.load("charge" + std::to_string(i + 1) + "Chi", arma::raw_ascii);
        _charge[i] = chargeTemp.col(1);
        _total[i] = accu(_charge[i]);
    }

    //check stat quant values, data size and range
    #ifdef DEBUG	
    std::cout << "Data Height: " << _dataFit.n_cols << " Data Width: " << _dataFit.n_rows << std::endl; 

    std::cout << "Data Max: " << _dataFit.max() << " Data Min: " << _dataFit.min() << std::endl; 
    assert(_dataFit.max() <= _dataScale && _dataFit.min() >= 0.0);
    #endif			
}

/**
 * @brief Convert real world value to an index
 * 
 * @param scale The real world scale for this
 * @param value The real world value of the scale
 * @param index the index 
 */
void Newton::_RealToIndex(arma::vec& scale, double& value, int& index)
{
    //determine the scale separation which is used to convert the value
    arma::vec scaleDiff = abs(scale - value);

    index = (int)scaleDiff.index_min();
}

/**
 * @brief Convert real world value to an index
 * 
 * @param scale The real world scale for this
 * @param value The real world value of the scale
 * @param index the index 
 */
void Newton::_RealToIndexLength(arma::vec& scale, double& value, int& index)
{
    //determine the scale separtion which is used to convert the value
    double scaleDiff = scale[1] - scale[0];

    index = (int)round(value/scaleDiff);
}

/**
 * @brief Class member function used to calculate the conjugate charge for the system
 * 
 */
void Newton::_ConjSLD()
{
  _sldTransformConj = conj(_sldTransform);
}

/**
 * @brief Class member function used to calculate quantities relevant to the chisquared of the system
 * this overloaded funciton allows for the potential updated chisquared to be calculated
 * 
 * @param charge The charge of one of the components of the system
 * @param sld the SLD value of one of the system components
 */
double Newton::_ChiSquared(arma::vec& charge, double& sld)
{
    //temporary variables for storing potential updated - ***********these could be made global and stored over the run to reduce calculations*********
    arma::vec sldTemp = _sld;
    arma::vec sldImageTemp; 
    sldImageTemp.set_size(_depth);

    //create updated sld profile
    _SLDGenerate(sldTemp, charge, sld);

    //calculate the updated reflectiviy
    _Reflectivity(sldImageTemp, sldTemp);

    //Calculate the chisquared value for the update
    arma::vec chiSquaredMat = pow(_sldImage - _dataFit,2.0)%_inverseVar;

    double chiSquared = accu(chiSquaredMat);

    return chiSquared;
}

/**
 * @brief Class member function to calculate the curret chisquared of the system
 * 
 */
void Newton::_ChiSquared()
{
    //calculate chi-Squared
    arma::vec chiSquaredMat = pow(_sldImage - _dataFit,2)%_inverseVar;

    _chiSquared = accu(chiSquaredMat);
    
    _redChi = _chiSquared/((double)(_depth));
}

/**
 * @brief Class member funciton used to generate the SLD profile from the component parts
 * 
 */
void Newton::_SLDGenerate()
{
    //calculate the sld profile by summing over the individual charge components
    _sld.zeros();
    for(int i = 0; i < _depth; i++)
    {
        for(int j = 0; j < _numSubstances; j++)
        {
            _sld[i] += _charge[j][i]*_sldVal[j];
        }
    }

    //Apply system contraints as required
    if(_volumetricNormalisation == true)
    {
        _GetMaxCharge();
        _sld /= _chargeMax;
    }

    if(_boundSLD == true)
    {
        _BoundSLD();
    }

    if(_useEdgeConstraints == true)
    {
        //temp vectors to store propogation and substrate SLDs to allow for substraction
        arma::vec propagationSLD(_lengthPropagation);
        arma::vec substrateSLD(_lengthSubstrate);

        propagationSLD.fill(_propagationSLD);
        substrateSLD.fill(_substrateSLD);
        
        _sld.head(_lengthPropagation) = propagationSLD;
        _sld.tail(_lengthSubstrate) = substrateSLD;
    }
}

/**
 * @brief Get the maximum total charge of the system and its index
 * 
 */
void Newton::_GetMaxCharge()
{

    //get the total charge at each element
    arma::vec totalCharge;
    totalCharge.set_size(_depth);
    totalCharge.zeros();

    for(int i = 0; i < _numSubstances; i++)
    {
        totalCharge += _charge[i];
    }

    //get max value of charge and the index of its occurence
    _indexMax = totalCharge.index_max();
    _chargeMax = totalCharge.max();

    //regular scaling for most standard elements
    _regularScaling =  1.0/_chargeMax;

    //scaling for the element j = jmax
    _maxIndexScaling = 0;
    for(int i = 0; i < _numSubstances; i++)
    {
        if(i != _currentCharge)
        {
            _maxIndexScaling = _sldVal[_currentCharge]*_charge[i][_indexMax] - _sldVal[i]*_charge[i][_indexMax];
        }
    }
    _maxIndexScaling /= (_chargeMax*_chargeMax);
}

/**
 * @brief Scale charge to physical SLD parameters so that value constraints can be imposed on the system
 * 
 */
void Newton::_BoundSLD()
{
    //rescale the magnituyde of the reconstructed SLD so that the max value is the max bound
    arma::vec sldTemp;
    sldTemp.set_size(_depth);

    sldTemp = _sld;

    //get min and max values of the array
    double maxVal = sldTemp.max();
    double minVal = sldTemp.min();

    //max and min indeces
    int indexMax = sldTemp.index_max();
    int indexMin = sldTemp.index_min();

    double minValRatio = 0.0;
    double maxValRatio = 0.0;

    //determine which ratio to scale the array by
    if(sgn(minVal) == sgn(_sldMinBound) && minVal != 0.0)
    {
        minValRatio = _sldMinBound/minVal;
    }

    if(sgn(maxVal) == sgn(_sldMaxBound) && maxVal != 0.0)
    {
        maxValRatio = _sldMaxBound/maxVal;
    }

    //scale the SLD
    if(minValRatio < maxValRatio || maxValRatio == 0.0)
    {
        if(minValRatio != 0.0 )
        {
            sldTemp *= minValRatio;

            //regular scaling for most standard elements
            _regularScaling =  minValRatio;

            //set the extremum index
            _indexMax = indexMin;
        }
    }

    if(maxValRatio < minValRatio || minValRatio == 0.0)
    {
        if(maxValRatio != 0.0)
        {
            sldTemp *= maxValRatio;

            //regular scaling for most standard elements
            _regularScaling =  maxValRatio;

            //set the extremum index
            _indexMax = indexMax;
        }
    }


    //scaling for the element j = jmax
    _maxIndexScaling = 0;


    //check SLD is still bounded
    #ifdef DEBUG
    std::cout << "Max initial value: " << sldTemp.max() <<  std::endl;
    assert(sldTemp.max() <=  _sldMaxBound + 0.1);

    std::cout << "Min initial value: " << sldTemp.min() <<  std::endl;
    assert(sldTemp.min() >= _sldMinBound - 0.1);
    #endif

    _sld  = sldTemp;
}

/**
 * @brief Class member function used to generate the SLD profile from the component parts
 * 
 * @param sldTemp The temporary SLD profile used for backtracing
 * @param charge The charge of one of the components of the system
 * @param sld the SLD value of the system component
 */
void Newton::_SLDGenerate(arma::vec& sldTemp, arma::vec& charge, double& sld)
{
    for(int i = 0; i < _depth; i++)
    {
        sldTemp[i] -= charge[i]*sld;
        sldTemp[i] += _tempCharge[i]*sld;
    }

    //Apply system contraints as required
    if(_volumetricNormalisation == true)
    {
        //get the total charge at each element
        arma::vec totalCharge;
        totalCharge.set_size(_depth);
        totalCharge.zeros();

        for(int i = 0; i < _numSubstances; i++)
        {
            totalCharge += _charge[i];
        }
        totalCharge -= charge;
        totalCharge += _tempCharge;

        //get max value of charge and the index of its occurence
        double chargeMax = totalCharge.max();
        sldTemp /= chargeMax;
    }

    if(_boundSLD == true)
    {
        //get min and max values of the array
        double maxVal = sldTemp.max();
        double minVal = sldTemp.min();

        double minValRatio = 0.0;
        double maxValRatio = 0.0;

        //determine which ratio to scale the array by
        if(sgn(minVal) == sgn(_sldMinBound) && minVal != 0.0)
        {
            minValRatio = _sldMinBound/minVal;
        }

        if(sgn(maxVal) == sgn(_sldMaxBound) && maxVal != 0.0)
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
    }

    if(_useEdgeConstraints == true)
    {
        //temp vectors to store propogation and substrate SLDs to allow for substraction
        arma::vec propagationSLD(_lengthPropagation);
        arma::vec substrateSLD(_lengthSubstrate);

        propagationSLD.fill(_propagationSLD);
        substrateSLD.fill(_substrateSLD);
        
        sldTemp.head(_lengthPropagation) = propagationSLD;
        sldTemp.tail(_lengthSubstrate) = substrateSLD;
    }
}

/**
 * @brief Class member function  to renormalise the charge or the charge image so that the total charge of the system remains constant at the initial total intensity of reflectivity spectrum
 * 
 * @param unnormalisedVector The unormalised input to be renormalised
 * @param norm The normalisation of the vector
 */
void Newton::_Renormalise(arma::vec& unnormalisedVector, double& norm)
{
  //renormalise the charge so that total charge = matrix size
  double totalCharge = fabs(accu(unnormalisedVector));

  unnormalisedVector = norm*unnormalisedVector/(totalCharge);//total charge normalised to size of matrix

  //check renormalisation is working as expected
  #ifdef DEBUGPLUS
  std::cout << "Normalisation: " << norm << " Total Charge " << accu(unnormalisedVector) << std::endl;
  #endif
  
  #ifdef DEBUG
  assert(norm <= accu(unnormalisedVector) + 0.1 && norm >= accu(unnormalisedVector) - 0.1);
  #endif 
}

/**
 * @brief Make a real vector into a complex vector for FT purposes
 * 
 * @param in The input real vector
 * @param out The output complex vector
 */
void Newton::_MakeComplex(arma::vec& in, arma::cx_vec& out)
{
    //make vector of zeros mathcing input size
    arma::vec zerosFill;
    zerosFill.copy_size(in);
    zerosFill.zeros();

    //combine zeros and in to make complex array
    out = arma::cx_vec(in, zerosFill);
}

/**
 * @brief Make all the DFT plans to be used in the simulations
 * 
 */
void Newton::_MakeDFTPlans()
{
    //temporary vectors for plan sizes
    arma::cx_vec temp1,temp2;
    temp1.set_size(_depth);
    temp2.set_size(_depth); 

    _inPlacePlan = fftw_plan_dft_1d(_depth, (double(*)[2])&temp1(0), (double(*)[2])&temp1(0), FFTW_FORWARD, FFTW_MEASURE);
    _outOfPlacePlan = fftw_plan_dft_1d(_depth, (double(*)[2])&temp1(0), (double(*)[2])&temp2(0), FFTW_FORWARD, FFTW_MEASURE);
}

/**
 * @brief Delete all the DFT plans for system cleanup
 * 
 */
void Newton::_DeleteDFTPlans()
{
    fftw_destroy_plan(_inPlacePlan);

    fftw_destroy_plan(_outOfPlacePlan);
}


/**
 * @brief ComplexFT method that uses the FFTW3 library to perform a fourier transform on a complex input matrix and output and complex matrix containing the fourier transform. This function allows for in place transforms. Normalisation is defined so that there is no normalisation on forwards transforms and a 1/N factor on backwards transforms
 * 
 * @param in Input matrix containing complex values
 * @param out Output matrix containing complex values
 * @param direction Direction of the transform which corresponds to the sign in the exponent, can take values -1, +1, -2, +2
 */
void Newton::_ComplexFT(arma::cx_vec& in, arma::cx_vec& out, const int& direction)
{

    out.copy_size(in);

    //check matrices for transforms
    #ifdef DEBUG
    assert(out.n_elem == in.n_elem);
    assert(out.n_elem == in.n_elem);
    #endif
    
    #ifdef DEBUGPLUS
    std::cout << "out length: " << out.n_elem <<  " in length: " << in.n_elem << std::endl;
    std::cout << "out length: " << out.n_elem <<  " in length: " << in.n_elem << std::endl;
    #endif
    
    //Forward DFT
    if(direction == -1)
	{
        //create FFTW plan determining how to apply DFT
        fftw_plan plan=fftw_plan_dft_1d(_depth, (double(*)[2])&in(0), (double(*)[2])&out(0), FFTW_FORWARD, FFTW_ESTIMATE);

        //perform DFT
        fftw_execute(plan);

        //delete plan
        fftw_destroy_plan(plan);
	}
    //Backward DFT
    else if (direction == 1)
	{
        //create FFTW plan determining how to apply DFT
        fftw_plan plan=fftw_plan_dft_1d(_depth, (double(*)[2])&in(0), (double(*)[2])&out(0), FFTW_BACKWARD, FFTW_ESTIMATE);

        //perform DFT
        fftw_execute(plan);

        //delete plan
        fftw_destroy_plan(plan);
	}
    //forwards dft factor 2
    else if(direction == -2)
	{
        //create FFTW plan determining how to apply DFT
        fftw_plan plan=fftw_plan_dft_1d(_depth, (double(*)[2])&in(0), (double(*)[2])&out(0), -2, FFTW_ESTIMATE);

        //perform DFT
        fftw_execute(plan);

        //delete plan
        fftw_destroy_plan(plan);
	}
    //Backward DFT factor 2
    else if (direction == 2)
	{
        //create FFTW plan determining how to apply DFT
        fftw_plan plan=fftw_plan_dft_1d(_depth, (double(*)[2])&in(0), (double(*)[2])&out(0), 2, FFTW_ESTIMATE);

        //perform DFT
        fftw_execute(plan);

        //delete plan
        fftw_destroy_plan(plan);
	}
    else
	{
        std::cerr << "Undefined value for transform direction" << std::endl;
        exit (EXIT_FAILURE);
	}
}

/**
 * @brief RealFT method that uses the FFTW3 library to perform a fourier transform on a real input matrix and output and complex matrix containing the fourier transform. This function does not allow for in place transforms. Normalisation is defined so that there is no normalisation on forwards transforms and a 1/N factor on backwards transforms
 * 
 * @param in Input matrix containing real values
 * @param out Output matrix containing complex values
 * @param direction Direction of the transform which corresponds to the sign in the exponent, can take values -1, +1, -2, +2
 */
void Newton::_RealFT(const arma::vec& in, arma::cx_vec& out, const int& direction)
{

    arma::vec zerosFill;
    zerosFill.copy_size(in);
    zerosFill.zeros();
    arma::cx_vec inComplex = arma::cx_vec(in, zerosFill);

    out.copy_size(in);

    //check matrices for transforms
    #ifdef DEBUG
    assert(out.n_elem == in.n_elem);
    assert(out.n_elem == in.n_elem);

    assert(arma::approx_equal(arma::real(inComplex), in, "absdiff", 0.01));
    assert(arma::approx_equal(arma::imag(inComplex), zerosFill, "absdiff", 0.01)); 
    #endif
    
    #ifdef DEBUGPLUS
    std::cout << "out length: " << out.n_elem <<  " in length: " << in.n_elem << std::endl;
    std::cout << "out length: " << out.n_elem <<  " in length: " << in.n_elem << std::endl;
    #endif
    
    //Forward DFT
    if(direction == -1)
	{
        //create FFTW plan determining how to apply DFT
        fftw_plan plan=fftw_plan_dft_1d(inComplex.n_elem, (double(*)[2])&inComplex(0), (double(*)[2])&out(0), FFTW_FORWARD, FFTW_ESTIMATE);

        //perform DFT
        fftw_execute(plan);

        //out /= sqrt(out.n_elem);

        //delete plan
        fftw_destroy_plan(plan);
	}
    //Backward DFT
    else if (direction == 1)
	{
        //create FFTW plan determining how to apply DFT
        fftw_plan plan=fftw_plan_dft_1d(inComplex.n_elem, (double(*)[2])&inComplex(0), (double(*)[2])&out(0), FFTW_BACKWARD, FFTW_ESTIMATE);

        //perform DFT
        fftw_execute(plan);

        //delete plan
        fftw_destroy_plan(plan);
	}
    else
	{
        std::cerr << "Undefined value for transform direction" << std::endl;
        exit (EXIT_FAILURE);
	}
}

/**
 * @brief Class member function that finds the increment of the charge via the Newton method
 * and implement backtracing
 * 
 * @param charge The charge of one of the components of the system
 * @param sld the SLD value of the system component
 */
void Newton::_CalcNewCharge(arma::vec& charge, double& sld)
{
    //boolean determining whether step is accepted
    bool accept = false;

    double tempChiSquared;
    int iter = 0; //max cutoff for backtracing

    //scaling factors for backtracing
    double gamma = _gammaInit; 
    double alpha = _alphaInit;

    //solve H (x_k+1 - x_k) = - gradChi
    _solvable = solve(_delta, _hessian, -_gradChiSquared);    

    //Armijo backtracing 
    if (_useDamping == true)
    {
        while(accept == false)
        {

            //create temporary variables containing updates
            _tempCharge = gamma*_delta + charge;

            tempChiSquared = _ChiSquared(charge, sld);

            //Implement Armijo criteria
            if(tempChiSquared <= _chiSquared - abs(alpha*gamma*(dot(_gradChiSquared,_delta))) || iter >= 10)
            {
                accept = true;
            }
            else
            {
                gamma *= _gammaFactor;
                alpha *= _alphaFactor;
                iter++;
            }
        }
    }
    else
    {
        _tempCharge = gamma*_delta + charge;
    } 
}

/**
 * @brief Class member function that increments the charge by the search vectors found using the lagrange multipliers to find an updated charge to be used in the next step (iteratively)
 * 
 * @param charge The charge of one of the components of the system
 * @param norm The total amount of the charge in the system for normalisation
 * 
 */
void Newton::_StepNewCharge(arma::vec& charge, double& norm)
{

    //this constraint determines whether or not limitations should be put on the charge incrementation which smooths the profile to make it more physical
    if(_smoothIncrement == true)
    {
        _SmoothIncrement(charge);
    }
    else
    {
        _RegularIncrement(charge);
    }


    //this constrains fixes the edge regions if the propagation region and substrate constraints are known 
    if(_useEdgeConstraints == true)
    {
        _Constraints(charge);
    }

    //smooth profile constraints
    if(_smoothProfile == true && _iterationCount%_smoothInterval == 0)
    {
        _SmoothProfile(charge);
    }

    //Force zero constraints
    if(_forceZero == true && _iterationCount%_forceInterval == 0)
    {
        _ForceZero(charge);
    }

    _SetZero(charge);
    _Renormalise(charge, norm);
}

/**
 * @brief Smoothes the charge profile according to N_i = 1/4 N_i-1 + 1/2 N_i + 1/4 N_i+1 every smoothInterval steps
 * 
 */
void Newton::_SmoothProfile(arma::vec& charge)
{
    arma::vec oldCharge = charge;

    for(int i = 1; i < _depth - 1; i++)
    { 
        if(oldCharge[i] < oldCharge[i + 1] && oldCharge[i] < oldCharge[i -1])
        {
            if(abs(oldCharge[i] - oldCharge[i+1]) < abs(oldCharge[i] - oldCharge[i-1]))
            {
                charge[i] =  0.5*oldCharge[i + 1] + 0.25*oldCharge[i] + 0.25*oldCharge[i-1];
            }
            else
            {
                charge[i] = 0.25*oldCharge[i + 1] + 0.25*oldCharge[i] + 0.5*oldCharge[i-1];
            }
        }
        else if(oldCharge[i] > oldCharge[i + 1] && oldCharge[i] > oldCharge[i -1])
        {
            if(abs(oldCharge[i] - oldCharge[i+1]) < abs(oldCharge[i] - oldCharge[i-1]))
            {
                charge[i] =  0.5*oldCharge[i + 1] + 0.25*oldCharge[i] + 0.25*oldCharge[i-1];
            }
            else
            {
                charge[i] = 0.25*oldCharge[i + 1] + 0.25*oldCharge[i] + 0.5*oldCharge[i-1];
            }
        }
    }
}

/**
 * @brief Forces values of charge below a cutoff of the max charge to 0
 * 
 * @param charge The charge of one of the components of the system
 */
void Newton::_ForceZero(arma::vec& charge)
{
    //get the max of this charge group
    double chargeMax = charge.max();
    int chargeIndexMax = charge.index_max();

    //force values below the threshold to be 0
    for(int i = 0; i < chargeIndexMax; i++)
    {
        if(charge[i] < _fracMax*chargeMax)
        {
            charge[i] = 0;
        }
    }
}

/**
 * @brief Class member function that sets values of array below a threshold to the threshold value
 * 
 * @param charge The charge of one of the components of the system
 */
void Newton::_SetZero(arma::vec& charge)
{
    for(auto& val : charge)
	{
	    if(val < _zeroLevel)
		{
		    val = _zeroLevel;
		}
	}

    //check limits 
    #ifdef DEBUG
    assert(charge.min() >= _zeroLevel);
    #endif

    #ifdef DEBUGPLUS
    std::cout << "Minimum Charge: " << charge.min() << std::endl;
    #endif 
}

/**
 * @brief Regular incrementation of the charge as defined by the Cambridge Algorithm by Skilling and Gull
 * 
 * @param charge The charge of one of the components of the system
 */
void Newton::_RegularIncrement(arma::vec& charge)
{
    charge = _tempCharge;
}

/**
 * @brief Charge incrementation with smoothness constraints that produce a more physical SLD profile 
 * 
 * @param charge The charge of one of the components of the system
 */
void Newton::_SmoothIncrement(arma::vec& charge)
{

    for(int j = 0; j < _depth; j++)
    {
        if(j!=0 && j!=_depth-1 && charge[j] < charge[j-1] && charge[j] < charge[j+1] && _tempCharge[j] <= charge[j])
        {

        }
        else if( j!=0 && j!=_depth-1 && charge[j] > charge[j-1] && charge[j] > charge[j+1]  && _tempCharge[j] >= charge[j])
        {

        }
        else
        {
            charge[j] = _tempCharge[j];
        }
	}
}

/**
 * @brief Constraints on starting (air region) SLD value and final (substrate SLD) values
 * 
 * @param charge The charge of one of the components of the system
 */
void Newton::_Constraints(arma::vec& charge)
{
    //temp vectors to store propogationa and substrate SLDs to allow for substraction
    arma::vec propagationSLD(_lengthPropagation);
    arma::vec substrateSLD(_lengthSubstrate);

    propagationSLD.zeros();
    substrateSLD.zeros();
    
    charge.head(_lengthPropagation) = propagationSLD;
    charge.tail(_lengthSubstrate) = substrateSLD;
}

/**
 * @brief Function to calculate the reflectivity (Image) from a charge distribution (in this case the reflectivity spectrum from the SLD profile)
 * 
 */
void Newton::_Reflectivity()
{
    arma::cx_vec sldComplex;

    //calculation for Fourier transform
    _MakeComplex(_sld, sldComplex);
    fftw_execute_dft(_outOfPlacePlan, (double(*)[2])&sldComplex(0), (double(*)[2])&_sldTransform(0));

    //Factor for DFT derivative rule
    arma::cx_vec dftDeriv;

    dftDeriv.set_size(_depth);

    for(int i = 0; i < _depth; i++)
    {
        dftDeriv[i] = 1-cexp(-i*_Complex_I*(2*M_PI/(double)_depth));
    }

    //combine factor and transform to calculate full transform 
    _sldTransform = (_sldTransform%dftDeriv + _sld[_depth-1] - _sld[0])/((double)_depth);
    
    //1/q^2 scaling
    for(int i = 0; i < _depth; i++)
    {
        _sldTransform[i] /= pow(_deltaQ*i, 2.0);
    }

    //crop data
    for(int i = 0; i < _qOffset; i++)
    {
        _sldTransform[i] = _sldTransform[_qOffset];
    }
    for(int i = _qCutOff; i < _depth; i++)
    {
        _sldTransform[i] = _sldTransform[_qCutOff];
    }

    arma::vec reflectivityTempInt = abs(_sldTransform);

    //crop image
    for(int i = 0; i < _depth; i++)
    {
        _sldImage[i] = pow(reflectivityTempInt[i], 2.0);
    }

    for(int i = 0; i < _qOffset; i++)
    {
        _sldImage[i] = _sldImage[_qOffset];
    }
    for(int i = _qCutOff; i < _depth; i++)
    {
        _sldImage[i] = _sldImage[_qCutOff];
    }


    _Renormalise(_sldImage, _norm);

}

/**
 * @brief Function to calculate the reflectivity (Image) from a charge distribution (in this case the reflectivity spectrum from the SLD profile)
 * This overloaded function is used in chisquared update calculation
 * 
 * @param sldImageTemp Temporary sld image
 * @param sldTemp Temporary sld profile
 */
void Newton::_Reflectivity(arma::vec& sldImageTemp, arma::vec& sldTemp)
{
    arma::cx_vec sldTransformTemp;
    sldTransformTemp.set_size(_depth);
    arma::cx_vec sldComplex;

    //calculation for Fourier transform
    _MakeComplex(sldTemp, sldComplex);
    fftw_execute_dft(_outOfPlacePlan, (double(*)[2])&sldComplex(0), (double(*)[2])&sldTransformTemp(0));

    //Factor for DFT derivative rule
    arma::cx_vec dftDeriv;

    dftDeriv.set_size(_depth);

    for(int i = 0; i < _depth; i++)
    {
        dftDeriv[i] = 1-cexp(-i*_Complex_I*(2*M_PI/(double)_depth));
    }

    //combine factor and transform to calculate full transform
    sldTransformTemp = (sldTransformTemp%dftDeriv + sldTemp[_depth-1] - sldTemp[0])/((double)_depth);

    for(int i = 0; i < _depth; i++)
    {
        sldTransformTemp[i] /= pow(_deltaQ*i, 2.0);
    }

    //crop data
    for(int i = 0; i < _qOffset; i++)
    {
        sldTransformTemp[i] = sldTransformTemp[_qOffset];
    }
    for(int i = _qCutOff; i < _depth; i++)
    {
        sldTransformTemp[i] = sldTransformTemp[_qCutOff];
    }

    arma::vec reflectivityTempInt = abs(_sldTransform);

    for(int i = 0; i < _depth; i++)
    {
        sldImageTemp[i] = pow(reflectivityTempInt[i], 2.0);
    }

    //crop image
    for(int i = 0; i < _qOffset; i++)
    {
        sldImageTemp[i] = sldImageTemp[_qOffset];
    }
    for(int i = _qCutOff; i < _depth; i++)
    {
        sldImageTemp[i] = sldImageTemp[_qCutOff];
    }


    _Renormalise(sldImageTemp, _norm);
}

/**
 * @brief Class member function used to calculate the chisquared gradient of the system
 * 
 */
void Newton::_GradChiSquared(double& sld)
{
    //calculate gradient of Chi-Squared 
    arma::cx_vec temp1 = 2.0*(_sldTransformConj%(_sldImage - _dataFit))%_inverseVar;


    //get 1/i^2 scaling and set _qOffset region
    for(int i = 0; i < _depth; i++)
    {
        temp1[i] /= pow(_deltaQ*i, 2.0);
    }
    
    for(int i = 0; i < _qOffset; i++)
    {
        temp1[i] = temp1[_qOffset];
    }
    for(int i = _qCutOff; i < _depth; i++)
    {
        temp1[i] = temp1[_qCutOff];
    }


    //sum temp value as required for the delta function components in the gradient of the chisquared
	arma::cx_double temp1Sum = accu(temp1);

    //half of the gradient of the chi squared, the other half is just the complex conjugate of this part
	arma::cx_vec gradChiSquaredPart;

    //fourier transform temp
	fftw_execute_dft(_inPlacePlan, (double(*)[2])&temp1(0), (double(*)[2])&temp1(0));

	gradChiSquaredPart.set_size(_depth);

    //apply chi squared gradient equation
	for(int i = 0; i < _depth ; i++)
	{
		gradChiSquaredPart[i] = temp1[i] - temp1[(i+1)%_depth];
	}

	gradChiSquaredPart[0] -= temp1Sum; 
	gradChiSquaredPart[_depth-1] += temp1Sum; 

	_gradChiSquared = sld*2*arma::real(gradChiSquaredPart)/((double)_depth);

    if(_volumetricNormalisation == true || _boundSLD == true)
    {
        _gradChiSquared *= _regularScaling;
        _gradChiSquared[_indexMax] *= _maxIndexScaling/(_regularScaling*sld);
    }
}

/**
 * @brief Class member function to calculate the hessian for the system
 * 
 */
void Newton::_Hessian(double& sld)
{
    arma::vec temp1;
    arma::cx_vec temp2, temp1FT, temp2FT, temp1Complex;
    temp1FT.set_size(_depth);
    temp2FT.set_size(_depth);
    arma::cx_mat hessianTemp;
    hessianTemp.set_size(_depth, _depth);

    temp1 = 2.0*_sldImage%_inverseVar; 
    temp2 =2.0*_sldTransformConj%_sldTransformConj%_inverseVar; 

    //get 1/i^4 scaling and set _qOffset region
    for(int i = 0; i < _depth; i++)
    {
        temp1[i] /= pow(_deltaQ*i, 4.0);
        temp2[i] /= pow(_deltaQ*i, 4.0);
    }

    
    for(int i = 0; i < _qOffset; i++)
    {
        temp1[i] = temp1[_qOffset];
        temp2[i] = temp2[_qOffset];
    }
    for(int i = _qCutOff; i < _depth; i++)
    {
        temp1[i] = temp1[_qCutOff];
        temp2[i] = temp1[_qCutOff];
    }



    double temp1Sum = accu(temp1);
    arma::cx_double temp2Sum = accu(temp2);

    _MakeComplex(temp1, temp1Complex);

    fftw_execute_dft(_outOfPlacePlan, (double(*)[2])&temp1Complex(0), (double(*)[2])&temp1FT(0));
    fftw_execute_dft(_outOfPlacePlan, (double(*)[2])&temp2(0), (double(*)[2])&temp2FT(0));

    //compute values of Hessian
    for(int i = 0; i < _depth; i++)
	{
        for(int j = 0; j <= i; j++)
		{
            hessianTemp(j,i) =  2.0*temp1FT[abs(j-i)%_depth]- temp1FT[abs(j-i-1)%_depth] - temp1FT[abs(j-i+1)%_depth] + temp2FT[(j+i)%_depth] - 2.0*temp2FT[(j+i+1)%_depth] + temp2FT[(j+i+2)%_depth];
            if(i == 0)
            {
                hessianTemp(j,i) +=  temp1FT[(j+1)%_depth] - temp1FT[(j)%_depth] + temp2FT[(j+1)%_depth] - temp2FT[(j)%_depth];
            }
            if(j == 0)
            {
                hessianTemp(j,i) +=  temp1FT[(i+1)%_depth] - temp1FT[(i)%_depth] + temp2FT[(i+1)%_depth] - temp2FT[(i)%_depth];
            }
            if(i == _depth - 1)
            {
                hessianTemp(j,i) -=  temp1FT[(j+1)%_depth] - temp1FT[(j)%_depth] + temp2FT[(j+1)%_depth] - temp2FT[(j)%_depth];
            }
            if(j ==_depth -1)
            {
                hessianTemp(j,i) -=  temp1FT[(i+1)%_depth] - temp1FT[(i)%_depth] + temp2FT[(i+1)%_depth] - temp2FT[(i)%_depth];
            }
            if(i == 0 && j == 0)
            {
                hessianTemp(j,i) += temp1Sum + temp2Sum;
            }
            if(i == _depth - 1 && j == _depth - 1)
            {
                hessianTemp(j,i) += temp1Sum + temp2Sum;
            }
            if(i == 0 && j == _depth - 1)
            {
                hessianTemp(j,i) -= temp1Sum + temp2Sum;
            }
            if(i == _depth - 1 && j == 0)
            {
                hessianTemp(j,i) -= temp1Sum + temp2Sum;
            }
            if(_volumetricNormalisation == true || _boundSLD == true)
            {
                if(i == _indexMax)
                {
                    hessianTemp(j,i) *= _maxIndexScaling/(_regularScaling*sld);
                }
                if(j == _indexMax)
                {
                    hessianTemp(j,i) *= _maxIndexScaling/(_regularScaling*sld);
                }
            }
            if(i != j)
			{
                hessianTemp(i,j) = hessianTemp(j,i);
			}
		}
	}

    //apply rescaling if required
    if(_volumetricNormalisation == true || _boundSLD == true)
    {
        _hessian *= (_regularScaling*_regularScaling);
    }

    _hessian = 2.0*sld*sld*arma::real(hessianTemp)/((double)_depth*_depth) + accu(4.0*(_sldImage - _dataFit)%_inverseVar);
}


