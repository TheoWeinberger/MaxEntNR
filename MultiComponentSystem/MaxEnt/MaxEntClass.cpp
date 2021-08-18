/****************************************************************************************
 * @file MaxEntClass.cpp                                                 
 * @brief Code to reconstruct the SLD profile from the reflectivity spectrum via the MaxEnt method using the 
 * Fourier form of the reflectivity relationship which is defined to be
 * 
 *          R[Q] = \frac{1}{Q^4} * |FFT(SLD)[Q]*[1-exp{-Q*I*(2*PI/totalDepth)}] +
 *                           SLD[totalDepth - 1] - SLD[0]|^2
 * 
 * This method gets rid of the 1/q^4 dependence and scales the system gradients down to
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
 * @version 5.0
 * @date 2021-06-04
 * 
 * @copyright Copyright (c) 2021
 *
 ****************************************************************************************
 */

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <omp.h>
#include <armadillo>
#include <cmath>
#include <complex.h>
#include <fftw3.h>
#include <libconfig.h++>
#include <assert.h>
#include "MaxEntClass.hpp"
#include "MaxEntSettings.hpp"

//debug mode
//#define DEBUG
//Debug mode including inner loops, outputs a lot of information but allows for iterative checking 
//#define DEBUGPLUS
//benchmarking bottlenecks of the code
//#define BENCHMARK
//benchmarking all of code 
//#define BENCHMARKFULL

/**
 * @brief Construct a new Max Ent object which reads in data from a string which specifies the file in 
 * which the data is stored. The data file should contain the reflectivity data in a one dimensionol column without
 * scaled axes (the algorirthm uses unscaled data as any scaling is normalised out).
 * 
 * @param dataFile The name of the file containing the reflectivity data
 */
MaxEnt::MaxEnt(const std::string& dataFile)
{
    //get datafile
    arma::mat dataTemp;
    dataTemp.load(dataFile, arma::raw_ascii);

    //System is undefined if _qOffset = 0, therefore _qOffset = 0 is defined to be equivalent to _qOffset = 1
    if(_qOffset == 0)
    {
        _qOffset = 1;
    }

    //load in file containing reflectivity data
    _dataFit = dataTemp.col(1);
    _reflectivityScale = dataTemp.col(0);
    _deltaQ = _reflectivityScale[1] - _reflectivityScale[0];
    _reflectivityNorm = _dataFit[_qOffset];

    _dataFit /= _reflectivityNorm;

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


    if(dataTemp.n_cols == 3 && _error == true)
    {
        _inverseVar = dataTemp.col(2);
    }

    //define system _depth which is equal to the number of elements in the array
    _depth = _dataFit.n_elem;

    _qCutOff = _depth;

    _sldScale.set_size(_depth);

    //remove Q^4 dependence from data, this simplifies the tranform and also creates larger
    //data perturbations which should produce better fitting
    for(int i = 0; i < _depth; i++)
    {
        if(i < _qOffset)
        {
            _dataFit[i] = 0.0;
            if(_error == true)
            {
                _inverseVar[i] = 0.0;
            }
            
        }
        else if(i >= _qOffset && i <= _qCutOff)
        {
            //_dataFit.print();
            _dataFit[i] = _dataFit[i]*pow((i*_deltaQ), 4.0);///_dataFit[_qOffset];

            if(_error == true)
            {
                _inverseVar[i] = _inverseVar[i]*pow((i*_deltaQ), 4.0);
            }
        }
        else if(i > _qCutOff)
        {
            _dataFit[i] = 0.0;
            if(_error == true)
            {
                _inverseVar[i] = 0.0;
            }
        }

        //transformation from real to reciprocal space is 1/deltaQ
        _sldScale[i] = 2*M_PI*i/(_deltaQ*_depth) ;
    }



    //renormalise input data so max value of intensity is 1
    double maxData =  _dataFit.max();
    _dataFit /= maxData/_dataScale;


    //scale errors 
    if(_error == true)
    {
        _inverseVar /= maxData/_dataScale;
    }

    //output data without Q^4 dependence to a file called dataInit so that fitting data can be checked for consistency
    _dataFit.save("dataInit", arma::raw_ascii);

    //set data vector sizes
    _inverseVar.set_size(_depth);
    _sld.set_size(_depth);
    _sldTransform.set_size(_depth);
    _sldTransformConj.set_size(_depth);
    _sldImage.set_size(_depth);
    _gradChiSquared.set_size(_depth);
    _ggChiSquared.set_size(_depth);
    _gradEntropy.set_size(_depth);

    _charge.resize(_numSubstances);
    _def.set_size(_numSubstances);
    _l0Squared.set_size(_numSubstances);
    _sldVal.set_size(_numSubstances);

    for(int i = 0; i < _numSubstances; i++)
    {
        _charge[i].set_size(_depth);
    }

    //DFT Containers
    _temp1.set_size(_depth);
    _temp2.set_size(_depth);
    _sldComplex.set_size(_depth);

    for(int i = 0; i < _numBasisVectors; i++)
    {
        _eVecTempA[i].set_size(_depth);
        _eVecComplex[i].set_size(_depth);
    }

}

/**
 * @brief Construct a new Max Ent object which reads in data from a string which specifies the file in 
 * which the data is stored. The data file should contain the reflectivity data in a one dimensionol column without
 * scaled axes (the algorirthm uses unscaled data as any scaling is normalised out). The second string refers to the 
 * settings file, which is a .cfg file containing the simulation specifics such as run number that are used in
 * the Cambridge algortihm
 * 
 * @param dataFile The name of the file containing the reflectivity data
 * @param configFile The name of the file containing the settings for fitting
 */
MaxEnt::MaxEnt(const std::string& dataFile, const std::string& configFile)
{
    //get datafile
    arma::mat dataTemp;
    dataTemp.load(dataFile, arma::raw_ascii);

    //read in data for diffraction grating to be built from
    ReadFile(configFile, _totalIterations, _maximumSearchIter, _numBasisVectors, _zeroLevel, _minVar, _dataScale, _numSubstances, _propagationSLD,
     _substrateSLD, _lengthPropagation, _lengthSubstrate, _smoothIncrement, _useEdgeConstraints, _chiSquaredScale, _spikeCharge, _spikePortion, _spikeAmount,
      _sldVal, _total, _toyModel, _volumetricNormalisation, _error, _boundSLD, _sldMinBound, _sldMaxBound, _smoothProfile, _smoothInterval, _forceZero, _fracMax, _chargeScale, _forceInterval, _qOffset, _qCutOff);

    //System is undefined if _qOffset = 0, therefore _qOffset = 0 is defined to be equivalent to _qOffset = 1
    if(_qOffset == 0)
    {
        _qOffset = 1;
    }

    //load in file containing reflectivity data
    _dataFit = dataTemp.col(1);

    //get system scaling
    _reflectivityScale = dataTemp.col(0);
    _deltaQ = _reflectivityScale[1] - _reflectivityScale[0];
    _reflectivityNorm = _dataFit[_qOffset];

    _dataFit /= _reflectivityNorm;

    //check whether system contains real errors
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

    //load in errors
    if(dataTemp.n_cols == 3 && _error == true)
    {
        _inverseVar = dataTemp.col(2);
    }

    //define system _depth which is equal to the number of elements in the array
    _depth = _dataFit.n_elem;

    //bound cutoffs
    if(_qCutOff > _depth)
    {
        _qCutOff = _depth;
    }


    _sldScale.set_size(_depth);

    //remove Q^4 dependence from data, this simplifies the tranform and also creates larger
    //data perturbations which should produce better fitting
    for(int i = 0; i < _depth; i++)
    {
        if(i < _qOffset)
        {
            _dataFit[i] = 0.0;
            if(_error == true)
            {
                _inverseVar[i] = 0.0;
            }
            
        }
        else if(i >= _qOffset && i <= _qCutOff)
        {
            //_dataFit.print();
            _dataFit[i] = _dataFit[i]*pow((i*_deltaQ), 4.0);///_dataFit[_qOffset];

            if(_error == true)
            {
                _inverseVar[i] = _inverseVar[i]*pow((i*_deltaQ), 4.0);
            }
        }
        else if(i > _qCutOff)
        {
            _dataFit[i] = 0.0;
            if(_error == true)
            {
                _inverseVar[i] = 0.0;
            }
        }

        //transformation from real to reciprocal space is 1/deltaQ
        _sldScale[i] = 2*M_PI*i/(_deltaQ*_depth) ;
    }


    //renormalise input data so max value of intensity is 1
    double maxData =  _dataFit.max();
    _dataFit /= maxData/_dataScale;

    //scale errors 
    if(_error == true)
    {
        _inverseVar /= maxData/_dataScale;
    }

    //Matrix to output scaled data to 
    arma::mat dataFitScaled;
    dataFitScaled.set_size(_depth,2);
    
    for(int i = 0; i < _depth; i++)
    {
        dataFitScaled(i,0) = _reflectivityScale[i];
        dataFitScaled(i,1) = _dataFit[i];
    }

    dataFitScaled.save("dataInit", arma::raw_ascii);

    //set size of matrices and vectors used in simulation
    _s = arma::vec(_numBasisVectors);
    _c = arma::vec(_numBasisVectors);
    _x = arma::vec(_numBasisVectors);
    _g = arma::mat(_numBasisVectors,_numBasisVectors);
    _h = arma::mat(_numBasisVectors,_numBasisVectors);    
    _gamma = arma::vec(_numBasisVectors);
    _delta = arma::vec(_numBasisVectors);
    _aVector = arma::vec(_maximumSearchIter);
    _bVector = arma::vec(_maximumSearchIter);  
    _eVec.resize(_numBasisVectors);

    //set data vector sizes
    _inverseVar.set_size(_depth);
    _sld.set_size(_depth);
    _sldTransform.set_size(_depth);
    _sldTransformConj.set_size(_depth);
    _sldImage.set_size(_depth);
    _gradChiSquared.set_size(_depth);
    _ggChiSquared.set_size(_depth);
    _gradEntropy.set_size(_depth);

    _charge.resize(_numSubstances);
    _def.set_size(_numSubstances);
    _l0Squared.set_size(_numSubstances);

    for(int i = 0; i < _numSubstances; i++)
    {
        _charge[i].set_size(_depth);
    }

    //DFT Containers
    _temp1.set_size(_depth);
    _temp2.set_size(_depth);
    _sldComplex.set_size(_depth);

    _eVecTempA.resize(_numBasisVectors);
    _eVecComplex.resize(_numBasisVectors);

    for(int i = 0; i < _numBasisVectors; i++)
    {
        _eVecTempA[i].set_size(_depth);
        _eVecComplex[i].set_size(_depth);
    }

}

/**
 * @brief Class member fucntion to initialise the data for the MaxEnt algorithm
 * 
 */
void MaxEnt::_Init()
{
    //redefine the normalisation of the charges as a fraction of the total of the data to match regular MaxEnt
    if(_volumetricNormalisation == true || _boundSLD == true)
    {
        double totalSum = accu(_total);
        double normalisation = _chargeScale*accu(_dataFit);

        for(int i = 0; i < _numSubstances; i++)
        {
           _total[i] =  normalisation * (_total[i]/totalSum);
        }
    }

    _MakeDFTPlans();

    /**
    //scaling for system if not using volumetric normalisation
    else
    {
        int divisibility = 0;
        _total *= _nA;
        double rem = _total.max();
        while(rem > 10.0)
        {
            rem /= 10.0;
            divisibility++;
        }
        for(int i = 0; i < _numSubstances; i++)
        {
            _total[i] /= pow(10.0, divisibility - 2); //scale total to be of order 100 as this works best for algo
            _sldVal[i] *= pow(10.0, divisibility - 25); //scale sld val along with this, note the factor of -9 to account for the fm scaling and the cm scaling
        }
    }
    **/




    //Initialise statistical quantities for the simulation 
    _InitStatQuant();

    //initialise the charge distribution for the system
    for(int i = 0; i < _numSubstances; i++)
    {
        _InitCharge(_charge[i], _total[i]);
    }

    _SLDGenerate();

    //output initial charge disitribution to file called chargeInit
    arma::mat sldOutput = GetSLDScaled();

    for(int i = 0; i < _numSubstances; i++)
    {
        arma::mat chargeOutput = GetChargeScaled(_charge[i]);
        chargeOutput.save("chargeInit" + std::to_string(i + 1) , arma::raw_ascii);

    }

    sldOutput.save("sldInit", arma::raw_ascii);

    //calculate current charge FTs and image
    _Reflectivity();

    //arma::mat imageOutput = GetReflectivityScaled();

    //imageOutput.save("imageInit", arma::raw_ascii);

    //calculate the DEF parameter for the system - the weighted average of the cells
    for(int i = 0; i < _numSubstances; i++)
    {
        _DEF(_charge[i],_def[i]);
    }
}

/**
 * @brief Class member fucntion to initialise the data for the MaxEnt algorithm
 * 
 */
void MaxEnt::_InitToy()
{
    _numSubstances = 1;
    
    _charge.resize(_numSubstances);
    _def.set_size(_numSubstances);
    _l0Squared.set_size(_numSubstances);
    _sldVal.set_size(_numSubstances);
    _total.set_size(_numSubstances);

    _MakeDFTPlans();

    for(int i = 0; i < _numSubstances; i++)
    {
        _charge[i].set_size(_depth);
    }

    _sldVal[0] = 1.0;

    //Initialise statistical quantities for the simulation 
    _InitStatQuant();

    _total[0] = _norm;

    //initialise the charge distribution for the system
    for(int i = 0; i < _numSubstances; i++)
    {
        _InitCharge(_charge[i], _total[i]);
    }

    _SLDGenerate();

    //_sld.print();

    //output initial charge disitribution to file called chargeInit
    arma::mat sldOutput = GetSLDScaled();

    for(int i = 0; i < _numSubstances; i++)
    {
        arma::mat chargeOutput = GetChargeScaled(_charge[i]);
        chargeOutput.save("chargeInit" + std::to_string(i + 1) , arma::raw_ascii);

    }

    sldOutput.save("sldInit", arma::raw_ascii);

    //calculate current charge FTs and image
    _Reflectivity();

    arma::mat imageOutput = GetReflectivityScaled();

    imageOutput.save("imageInit", arma::raw_ascii);

    //calculate the DEF parameter for the system - the weighted average of the cells
    for(int i = 0; i < _numSubstances; i++)
    {
        _DEF(_charge[i],_def[i]);
    }
}

/**
 * @brief Class member function to apply one step of the MaxEnt algorithm
 * 
 * @param charge The charge of one of the components of the system
 * @param norm The normalisation of the component of the system
 * @param def A parameter used to define the contrained entropy such that the total charge can be kept constant without need for an additional lagrange multiplier
 * @param sld SLD value of substance
 * @param l0squared The length of yhe trust region radius
 */
void MaxEnt::_Step(arma::vec& charge, double& norm, double& def, double& sld, double& l0Squared)
{

    #ifdef BENCHMARKFULL
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    //calculate starting _entropy and chisquared
    _Entropy(charge, def);

    #ifdef BENCHMARKFULL
    auto finish = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "Time taken by function Entropy: " << runtime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARKFULL
    start = std::chrono::high_resolution_clock::now();
    #endif

    _ConjSLD();

    #ifdef BENCHMARKFULL
    finish = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "Time taken by function ConjSLD: " << runtime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARK
    auto beginning = std::chrono::high_resolution_clock::now();
    #endif

    //get ChiSquared related quantities
    _ChiSquared(sld);

    #ifdef BENCHMARK
    auto end = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    auto executiontime = std::chrono::duration_cast<std::chrono::microseconds>(end - beginning);
    std::cout << "Time taken by function ChiSquared: " << executiontime.count() << " microseconds" << std::endl;
    #endif

    //_Renormalise(charge,_norm); //is this required/does this help --> few questions about normalisation

    //calculate basis set and calculate search directions including transformation to diagonalised basis sets 
    #ifdef BENCHMARK
    beginning = std::chrono::high_resolution_clock::now();
    #endif

    _BasisFunctions(charge);

    #ifdef BENCHMARK
    end = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    executiontime = std::chrono::duration_cast<std::chrono::microseconds>(end - beginning);
    std::cout << "Time taken by function BasisFunctions: " << executiontime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARKFULL
    start = std::chrono::high_resolution_clock::now();
    #endif

    _DistCalc();

    #ifdef BENCHMARKFULL
    finish = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "Time taken by function DistCalc: " << runtime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARKFULL
    start = std::chrono::high_resolution_clock::now();
    #endif

    _TESTCalc();

    #ifdef BENCHMARKFULL
    finish = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "Time taken by function TestCalc: " << runtime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARKFULL
    start = std::chrono::high_resolution_clock::now();
    #endif

    _NormBasisVec();

    #ifdef BENCHMARKFULL
    finish = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "Time taken by function NormBasisVec: " << runtime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARK
    beginning = std::chrono::high_resolution_clock::now();
    #endif

    _DiagG(charge);

    #ifdef BENCHMARK
    end = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    executiontime = std::chrono::duration_cast<std::chrono::microseconds>(end - beginning);
    std::cout << "Time taken by function DiagG: " << executiontime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARK
    beginning = std::chrono::high_resolution_clock::now();
    #endif

    _DiagH(sld);

    #ifdef BENCHMARK
    end = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    executiontime = std::chrono::duration_cast<std::chrono::microseconds>(end - beginning);
    std::cout << "Time taken by function DiagH: " << executiontime.count() << " microseconds" << std::endl;
    #endif


    //_NormBasisVec();

    //calculate subspaces for seaching

    #ifdef BENCHMARKFULL
    start = std::chrono::high_resolution_clock::now();
    #endif

    _SubspaceS();

    #ifdef BENCHMARKFULL
    finish = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "Time taken by function SubspaceS: " << runtime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARKFULL
    start = std::chrono::high_resolution_clock::now();
    #endif

    _SubspaceC();

    #ifdef BENCHMARKFULL
    finish = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "Time taken by function SubspaceC: " << runtime.count() << " microseconds" << std::endl;
    #endif

    //search for lagrange multipliers

    #ifdef BENCHMARK
    beginning = std::chrono::high_resolution_clock::now();
    #endif

    _LagrangeSearch(l0Squared);

    #ifdef BENCHMARK
    end = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    executiontime = std::chrono::duration_cast<std::chrono::microseconds>(end - beginning);
    std::cout << "Time taken by function LagrangeSearch: " << executiontime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARK
    beginning = std::chrono::high_resolution_clock::now();
    #endif

    _ChooseAB();

    #ifdef BENCHMARK
    end = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    executiontime = std::chrono::duration_cast<std::chrono::microseconds>(end - beginning);
    std::cout << "Time taken by function ChooseAB: " << executiontime.count() << " microseconds" << std::endl;
    #endif

    //Calculate charge distribution and image for update system

    #ifdef BENCHMARK
    beginning = std::chrono::high_resolution_clock::now();
    #endif

    _CalcNewCharge(charge, norm);

    #ifdef BENCHMARK
    end = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    executiontime = std::chrono::duration_cast<std::chrono::microseconds>(end - beginning);
    std::cout << "Time taken by function CalcNewCharge: " << executiontime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARKFULL
    start = std::chrono::high_resolution_clock::now();
    #endif

    _SLDGenerate(); //note this is redundant and should probably be removed

    #ifdef BENCHMARKFULL
    finish = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "Time taken by function SLDGenerate: " << runtime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARKFULL
    start = std::chrono::high_resolution_clock::now();
    #endif

    _Reflectivity(); //note this is redundant and should probably be removed

    #ifdef BENCHMARKFULL
    finish = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "Time taken by function Reflectivity: " << runtime.count() << " microseconds" << std::endl;
    #endif
}

/**
 * @brief Class member function to print out data from MaxEnt algortihm
 * 
 */ 
void MaxEnt::_Print()
{
    std::cout << "Iteration Number --- Entropy --- Chi Squared --- Reduced Chi Squared --- TEST" << std::endl;
    std::cout << "-----" << _iterationCount << "----------" << _entropy << "------" << _chiSquared << "---------" << _redChi << "----------" << _test << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;

    std::ofstream myfile;
    myfile.open ("chiSquared", std::ios_base::app);
    myfile << _redChi << std::endl;
    myfile.close();
}

/**
 * @brief Class member function to save data from MaxEnt algorithm
 * 
 */
void MaxEnt::_Store()
{
    //output charge and image data for the specified iteration numbers
    //this allows for evaluation of how the charge distribution and image fitting
    //process evolve
    if(_iterationCount == 100 || _iterationCount == 1000 || _iterationCount == 10000)
	{
        arma::mat sldOutput = GetSLDScaled();
        arma::mat imageOutput = GetReflectivityScaled();

        for(int i = 0; i < _numSubstances; i++)
        {
            arma::mat chargeOutput = GetChargeScaled(_charge[i]);
            chargeOutput.save("charge" + std::to_string(i + 1) + "_" + std::to_string(_iterationCount), arma::raw_ascii);
        }

        sldOutput.save("sld" + std::to_string(_iterationCount), arma::raw_ascii);
        imageOutput.save("image" + std::to_string(_iterationCount) , arma::raw_ascii);
	}
    if(_iterationCount == _totalIterations)
    {
        arma::mat sldOutput = GetSLDScaled();
	    arma::mat imageOutput = GetReflectivityScaled();

        for(int i = 0; i < _numSubstances; i++)
        {
            arma::mat chargeOutput = GetChargeScaled(_charge[i]);
            chargeOutput.save("charge" + std::to_string(i + 1) + "Recon", arma::raw_ascii);
        }

        sldOutput.save("sldRecon", arma::raw_ascii);
        imageOutput.save("imageRecon", arma::raw_ascii);
    }

    if(_redChi <= _redChiMin)
    {
        arma::mat sldOutput = GetSLDScaled();
	    arma::mat imageOutput = GetReflectivityScaled();

        for(int i = 0; i < _numSubstances; i++)
        {
            arma::mat chargeOutput = GetChargeScaled(_charge[i]);
            chargeOutput.save("charge" + std::to_string(i + 1) + "Chi", arma::raw_ascii);
        }

        sldOutput.save("sldChi", arma::raw_ascii);
        imageOutput.save("imageChi", arma::raw_ascii);

        _redChiMin = _redChi;
    }

    if(_test <= _testMin)
    {
        arma::mat sldOutput = GetSLDScaled();
	    arma::mat imageOutput = GetReflectivityScaled();

        for(int i = 0; i < _numSubstances; i++)
        {
            arma::mat chargeOutput = GetChargeScaled(_charge[i]);
            chargeOutput.save("charge" + std::to_string(i + 1) + "Test", arma::raw_ascii);
        }

        sldOutput.save("sldTest", arma::raw_ascii);
        imageOutput.save("imageTest", arma::raw_ascii);

        _testMin = _test;
    }
}

/**
 * @brief Private member function to output redChi, minRedChi, TEST and minTest to 
 * a datafile to be subsequently averaged over for parametric testing 
 * 
 */
void MaxEnt::_StoreConvData()
{
    std::ofstream myfile;
    myfile.open ("convData", std::ios_base::app);
    myfile << _redChi << " " << _redChiMin << " " << _test << " " << _testMin << std::endl;
    myfile.close();
}

/**
 * @brief Employs Cambridge Algortihm to reconstruct the maximum _entropy SLD profile consistent with the 
 * input data - this is designed to be implemented for real data
 * 
 */
void MaxEnt::Solve()
{
    //if it is specified that a toy model is under study use the toy model solver
    //otherwise use the general solver
    if(_toyModel == true)
    {
        _SolveToy();
    }
    else
    {
        _SolveMain();
    }

    _StoreConvData();

    _DeleteDFTPlans();
}

/**
 * @brief Employs Cambridge Algortihm to reconstruct the maximum _entropy SLD profile consistent with the 
 * input data - this is designed to be implemented for real data
 * 
 */
void MaxEnt::_SolveMain()
{
    std::cout << "------------------- Max Ent solver - beginning simulation -------------------" << std::endl;

    //Initialise system so that the Cambridge algortihm can be employed
    _Init();

    //main body of Cambridge algorithm
    //at each overall step the Cambridge algorithm for that step is first employed
    //Next the relevant data to check simulation progression is outpute
    //Then the image and charge are stored for certain iteration counts
    //This occurs until the iteration counter maxima as specified in the settings file is reached
    while(_iterationCount <= _totalIterations)
	{
        for(_currentCharge = 0; _currentCharge < _numSubstances; _currentCharge++)
        {
            _Step(_charge[_currentCharge], _total[_currentCharge], _def[_currentCharge], _sldVal[_currentCharge], _l0Squared[_currentCharge]);
            _Print();
            _Store();
        }

        if(_iterationCount == 0)
        {
            _redChiMin = _redChi;
            _testMin = _test;
        }

        _iterationCount += 1;
	}

    std::cout << "------ Minimum Reduced ChiSquared = " << _redChiMin << " ---------------------" << std::endl;
    std::cout << "-------------------- Minimum Test = " << _testMin << " ---------------------" << std::endl;
    std::cout << "-------------------- Max Ent solver - ending simulation ---------------------" << std::endl;
}

/**
 * @brief Employs Cambridge Algortihm to reconstruct the maximum _entropy SLD profile consistent with the 
 * input data - this is for the proof of concept 'toy' model which is a limited single component system only
 * 
 */
void MaxEnt::_SolveToy()
{
    if(_numSubstances != 1 || _volumetricNormalisation != false || _sldVal[0] != 1.0)
    {
        std::cout << "This is not a toy model system" << std::endl;
        std::cout << "Subsequent simulations will only use one substance, an SLD value = 1.0, and no volumetric normalisation" << std::endl;
        std::cout << "Press X to quit or any other button to continue with the simulation" << std::endl;
        char answer;
        std::cin >> answer;
        if(answer == 'X' || answer == 'x')
        {
            exit(EXIT_SUCCESS);
        }
    }

    std::cout << "------------------- Max Ent solver - beginning simulation -------------------" << std::endl;

    //Initialise system so that the Cambridge algortihm can be employed
    _InitToy();

    //main body of Cambridge algorithm
    //at each overall step the Cambridge algorithm for that step is first employed
    //Next the relevant data to check simulation progression is outpute
    //Then the image and charge are stored for certain iteration counts
    //This occurs until the iteration counter maxima as specified in the settings file is reached
    while(_iterationCount <= _totalIterations)
	{

        _Step(_charge[0], _total[0], _def[0], _sldVal[0], _l0Squared[0]);
        _Print();
        _Store();

        if(_iterationCount == 0)
        {
            _redChiMin = _redChi;
            _testMin = _test;
        }

        _iterationCount += 1;
	}

    std::cout << "------ Minimum Reduced ChiSquared = " << _redChiMin << " ---------------------" << std::endl;
    std::cout << "-------------------- Minimum Test = " << _testMin << " ---------------------" << std::endl;
    std::cout << "-------------------- Max Ent solver - ending simulation ---------------------" << std::endl;
}

/**
 * @brief Get the scaled SLD 
 * 
 * @return arma::mat @param sldScaled the scaled SLD
 */
arma::mat MaxEnt::GetSLDScaled()const
{
    //Matrix to store SLD data in 
    arma::mat sldScaled;
    sldScaled.set_size(_depth,2);

    //temporary sld for manipulation
    arma::vec sldTemp = _sld;

    //scale the SLD    
    for(int i = 0; i < _depth; i++)
    {
        sldScaled(i,0) = _sldScale[i];
        sldScaled(i,1) = sldTemp[i];
    }

    return sldScaled;
}

/**
 * @brief Get the scaled charge
 * 
 * @param charge The charge of one of the components of the system
 * @return arma::mat @param chargeScaled the scaled charge
 */
arma::mat MaxEnt::GetChargeScaled(arma::vec& charge)const
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
 * @return arma::mat @param reflectivityScaled the scaled Reflectivity
 */
arma::mat MaxEnt::GetReflectivityScaled()const
{
    //Matrix to store reflectivity data in
    arma::mat reflectivityScaled;
    reflectivityScaled.set_size(_depth,2);
    double normalisation;

    reflectivityScaled(_qOffset,1) = _reflectivityNorm;
    normalisation = _sldImage[_qOffset]/pow(_qOffset*_deltaQ,4.0);
    
    //Apply scaling and add 1/Q^4 back into system
    for(int i = 0; i < _depth; i++)
    {
        reflectivityScaled(i,0) = _reflectivityScale[i];
        
        if(i < _qOffset)
        {
            //set the first value
            reflectivityScaled(i,1) = 0.0;
        }
        else if(i > _qOffset && i <= _qCutOff)
        {
            reflectivityScaled(i,1) = (_sldImage[i]*_reflectivityNorm)/(normalisation*pow(i*_deltaQ,4.0));
        }
        else if(i > _qCutOff)
        {
            reflectivityScaled(i,1) = reflectivityScaled(_qCutOff,1);
        }
        
       //reflectivityScaled(i,1) = _sldImage[i];
    }
    return reflectivityScaled;
}

/**
 * @brief Class member function used to initialise relevant statistical quantities to be used throughout MaxEnt algorithm
 * 
 */
void MaxEnt::_InitStatQuant()
{
    //system normalisations
    _cAim = 0;
    _norm = accu(_dataFit);

    //(Skilling 1986) Eqn. 31
    for(int i = 0; i < _numSubstances; i++)
    {
        _l0Squared[i] = 0.2*_total[i]; //input data should be normalised but may need to check this
    }

    if(_error == true)
    {
        //assign matrix for inverse variance and calculate each value
        //(Elliott 1999) Eqn. 6
        for(int i = 0; i < _depth; i++)
        {
            if (_inverseVar[i] < _minVar)
            {
                _inverseVar[i] = _minVar;   
            }
        }
        _inverseVar = 1.0/_inverseVar;
    }
    else
    {
        //assign matrix for inverse variance and calculate each value
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

    //check stat quant values, data size and range
    #ifdef DEBUG	
    std::cout << "_cAim: " << _cAim << std::endl;
    std::cout << "Normalisation: " << _norm << std::endl;

    for(int i = 0; i < _numSubstances; i++)
    {
        std::cout << "_l0Squared" + std::to_string(i+1) +": " << _l0Squared[i] << std::endl;
    }

    std::cout << "Data Height: " << _dataFit.n_cols << " Data Width: " << _dataFit.n_rows << std::endl; 

    std::cout << "Data Max: " << _dataFit.max() << " Data Min: " << _dataFit.min() << std::endl; 
    assert(_dataFit.max() <= _dataScale && _dataFit.min() >= 0.0);
    #endif			
}

/**
 * @brief Get the maximum total charge of the system and its index
 * 
 */
void MaxEnt::_GetMaxCharge()
{
    arma::vec totalCharge;
    totalCharge.set_size(_depth);
    totalCharge.zeros();

    for(int i = 0; i < _numSubstances; i++)
    {
        totalCharge += _charge[i];
    }

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
 * @brief Class member function used to calculate quantities relevant to the entropy of the system
 * 
 * @param charge The charge of one of the components of the system
 * @param def A parameter used to define the contrained entropy such that the total charge can be kept constant without need for an additional lagrange multiplier
 */
void MaxEnt::_Entropy(arma::vec& charge, double& def)
{
    //calculate _entropy
    //(Elliott 1999) Eqn. 4
    //(Weinberger 2021) Eqn. 90
    arma::vec entropyMat;
    entropyMat.set_size(_depth);
    entropyMat.zeros();

    for(int i = 0; i < _numSubstances; i++)
    {
        entropyMat += _charge[i] % (log(_charge[i]/_def[i]) - 1);
    }

    _entropy = - arma::accu(entropyMat);
    
    //calculate _entropy gradient
    //(Weinberger 2021) Eqn. 101
    _gradEntropy = log(def) - log(charge);

    if(_useEdgeConstraints == true)
    {
        _Constraints(_gradEntropy);
    }
}

/**
 * @brief Class member function used to calculate the conjugate charge for the system
 * 
 */
void MaxEnt::_ConjSLD()
{
  _sldTransformConj = conj(_sldTransform);
}

/**
 * @brief Class member function used to calculate quantities relevant to the chisquared of the system
 * 
 * @param sld The sld value of the substance 
 */
void MaxEnt::_ChiSquared(double& sld)
{
    //calculate chi-Squared
    //(Skilling 1986) Eqn. 2
    arma::vec chiSquaredMat = pow(_sldImage - _dataFit,2)%_inverseVar;

    _chiSquared = accu(chiSquaredMat);
    
    _redChi = _chiSquared/((double)(_depth) - (_qOffset + (_depth -_qCutOff)));

    //calculate gradient of Chi-Squared 
    _temp1 = 2.0*(_sldTransformConj%(_sldImage - _dataFit))%_inverseVar;

    for(int i = 0; i < _qOffset; i++)
    {
        _temp1[i] = 0.0;
    }
    for(int i = _qCutOff; i < _depth; i++)
    {
        _temp1[i] = 0.0;
    }

    //(Elliott 1999) Eqn. 7
    //(Weinberger 2021) Eqn. 75/98
    _GradChiSquared();

    //calculate double derivated of chi Squared
    _temp1 = 2.0*pow(_sldTransformConj,2.0)%_inverseVar;

    for(int i = 0; i < _qOffset; i++)
    {
        _temp1[i] = 0.0;
    }
    for(int i = _qCutOff; i < _depth; i++)
    {
        _temp1[i] = 0.0;
    }

    //(Elliott 1999) Eqn. 7
    //(Weinberger 2021) Eqn. 82/99                            
    _GGChiSquared(sld);

    //scale gradients by linear weighting
    _gradChiSquared = sld*_chiSquaredScale*_gradChiSquared;

    if(_volumetricNormalisation == true || _boundSLD == true)
    {
        _gradChiSquared *= _regularScaling;
        _gradChiSquared[_indexMax] *= _maxIndexScaling/(_regularScaling*sld);
    }


    _gradChiSquared /= (double)_depth;
    
    _ggChiSquared = _chiSquaredScale*_ggChiSquared;

    if(_useEdgeConstraints == true)
    {
        _Constraints(_gradChiSquared);
        _Constraints(_ggChiSquared);
    }
}

/**
 * @brief Class member function to calculate the DEF parameter for the system
 * (Elliot 1999) Eqn. 5
 * 
 * @param charge The charge of one of the components of the system
 * @param def A parameter used to define the contrained entropy such that the total charge can be kept constant without need for an additional lagrange multiplier
 */
void MaxEnt::_DEF(arma::vec& charge, double& def)
{
    arma::vec chargeLogCharge = charge % log(charge);

    double totalCharge = accu(charge);

    double totalChargeLogCharge = arma::accu(chargeLogCharge);

    def = exp(totalChargeLogCharge/totalCharge);

    //check value of def is sensible
    #ifdef DEBUG
    std::cout << "DEF: " << def << std::endl;
    #endif
}

/**
 * @brief Class member function to renormalise the charge or the charge image so that the total charge of the system remains constant at the initial total intensity of the diffraction pattern
 * 
 * @param charge The charge of one of the components of the system
 * @param norm The normalisation of the component of the system
 */
void MaxEnt::_InitCharge(arma::vec& charge, double& norm)
{
    //randomly seed the arma random number generator 
    arma::arma_rng::set_seed_random();
    
    //initialise charge distribution and fill with random numbers chosen as uniform dist [0,1]
    charge.randu();

    //use this to read in constant initial charge for validation
    //arma::mat chargeTemp;
    //chargeTemp.load("chargeInit1", arma::raw_ascii);
    //charge = chargeTemp.col(1);
    //charge.load("chargeInit", arma::raw_ascii);

    //Potential constraints on the system, constraints are used depending on their settings in the settings.cfg file

    if(_spikeCharge == true)
    {
        _SpikeCharge(charge);
    }

    //fixed edge constraints settings
    if(_useEdgeConstraints == true)
    {
        _Constraints(charge);
    }

    //Set any values below the threshold zero level to the threshold level
    _SetZero(charge);


    //renormalise the charge 
    _Renormalise(charge, norm);
}

/**
 * @brief Class member funciton used to generate the SLD profile from the component parts
 * 
 */
void MaxEnt::_SLDGenerate()
{
    _sld.zeros();
    for(int i = 0; i < _depth; i++)
    {
        for(int j = 0; j < _numSubstances; j++)
        {
            _sld[i] += _charge[j][i]*_sldVal[j];
        }
    }

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
        //temp vectors to store propogationa and substrate SLDs to allow for substraction
        arma::vec propagationSLD(_lengthPropagation);
        arma::vec substrateSLD(_lengthSubstrate);

        propagationSLD.fill(_propagationSLD);
        substrateSLD.fill(_substrateSLD);
        
        _sld.head(_lengthPropagation) = propagationSLD;
        _sld.tail(_lengthSubstrate) = substrateSLD;
    }


}

/**
 * @brief Function that 'spikes' the charge in close to the substrate region, this should help the profile 
 * build up from close to the substrate which is physically what is expected (albeit by no means definite)
 * 
 * @param charge The charge of one of the components of the system
 */
void MaxEnt::_SpikeCharge(arma::vec& charge)
{
    int lengthSpike = int(round(_depth*_spikePortion));

    charge.tail(lengthSpike) += _spikeAmount;
}

/**
 * @brief Make all the DFT plans to be used in the simulations
 * 
 */
void MaxEnt::_MakeDFTPlans()
{
    _sldPlan = fftw_plan_dft_1d(_depth, (double(*)[2])&_sldComplex(0), (double(*)[2])&_sldTransform(0), FFTW_FORWARD, FFTW_MEASURE);

    _temp1Plan = fftw_plan_dft_1d(_depth, (double(*)[2])&_temp1(0), (double(*)[2])&_temp1(0), FFTW_FORWARD, FFTW_MEASURE);

    _temp2Plan = fftw_plan_dft_1d(_depth, (double(*)[2])&_temp2(0), (double(*)[2])&_temp2(0), -2, FFTW_MEASURE);

    for(int i = 0; i < _numBasisVectors; i++)
    {
        _eVecPlan[i] = fftw_plan_dft_1d(_depth, (double(*)[2])&_eVecComplex[i](0), (double(*)[2])&_eVecTempA[i](0), FFTW_FORWARD, FFTW_MEASURE);
    }
}

/**
 * @brief Delete all the DFT plans for system cleanup
 * 
 */
void MaxEnt::_DeleteDFTPlans()
{
    fftw_destroy_plan(_sldPlan);

    fftw_destroy_plan(_temp1Plan);

    fftw_destroy_plan(_temp2Plan);

    for(int i = 0; i < _numBasisVectors; i++)
    {
        fftw_destroy_plan(_eVecPlan[i]);
    }
}

/**
 * @brief Make a real vector into a complex vector for FT purposes
 * 
 * @param in The input real vector
 * @param out The output complex vector
 */
void MaxEnt::_MakeComplex(arma::vec& in, arma::cx_vec& out)
{
    arma::vec zerosFill;
    zerosFill.copy_size(in);
    zerosFill.zeros();
    out = arma::cx_vec(in, zerosFill);
}


/**
 * @brief Class member function  to renormalise the charge or the charge image so that the total charge of the system remains constant at the initial total intensity of reflectivity spectrum
 * 
 * @param unnormalisedVector The unormalised input to be renormalised
 * @param norm The normalisation of the vector
 */
void MaxEnt::_Renormalise(arma::vec& unnormalisedVector, double& norm)
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
 * @brief Class member function  to renormalise the charge or the charge image so that the total charge of the system remains constant at the initial total intensity of reflectivity spectrum
 * This overloaded function if for use on Complex vectors
 * There should potentially be a factor of a square root in here *************
 * *************************************************************************
 * 
 * @param unnormalisedVector The unormalised input to be renormalised
 * @param norm The normalisation of the vector
 */
void MaxEnt::_Renormalise(arma::cx_vec& unnormalisedVector, double& norm)
{
  //renormalise the charge so that total charge = matrix size
  double totalCharge = fabs(accu(unnormalisedVector%conj(unnormalisedVector)));

  unnormalisedVector = norm*unnormalisedVector/(totalCharge);//total charge normalised to size of matrix

  //check renormalisation is working as expected
  #ifdef DEBUGPLUS
  std::cout << "Normalisation: " << norm << " Total Charge " << accu(abs(unnormalisedVector)) << std::endl;
  #endif
  
  #ifdef DEBUG
  assert(norm <= accu(abs(unnormalisedVector)) + 0.1 && norm >= accu(abs(unnormalisedVector)) - 0.1);
  #endif 
}


/**
 * @brief ComplexFT method that uses the FFTW3 library to perform a fourier transform on a complex input matrix and output and complex matrix containing the fourier transform. This function allows for in place transforms. Normalisation is defined so that there is no normalisation on forwards transforms and a 1/N factor on backwards transforms
 * 
 * @param in Input matrix containing complex values
 * @param out Output matrix containing complex values
 * @param direction Direction of the transform which corresponds to the sign in the exponent, can take values -1, +1, -2, +2
 */
void MaxEnt::_ComplexFT(arma::cx_vec& in, arma::cx_vec& out, const int& direction)
{
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

        //out /= out.n_elem;

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

        out /= out.n_elem;

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

        //out /= out.n_elem;

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

        out /= out.n_elem;

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
void MaxEnt::_RealFT(const arma::vec& in, arma::cx_vec& out, const int& direction)
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

        //out /= out.n_elem;

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

        out /= out.n_elem;

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
 * @brief Class member function to generate the search vectors to be used in the MaxEnt method
 * (Skilling 1986) Eqn. 22 - 27
 * 
 * @param charge The charge of one of the components of the system
 */
void MaxEnt::_BasisFunctions(arma::vec& charge)
{
    //calculate basis vectors
    arma::vec basisOperator = charge % _ggChiSquared;
    for(int i = 0; i < _numBasisVectors; i++)
	{
        if(i == 0)
		{
            _eVec[i] = charge % _gradEntropy;
            }
        else if (i == 1)
		{
            _eVec[i] = charge % _gradChiSquared;
		}
        else
		{
            _eVec[i] = basisOperator % _eVec[i-2];
		}
	}  

}

/**
 * @brief Class member function to calculate the 'length' (modulus squared) of the entropy and chi squared gradients. These quantities are required for normalising the gradients when the TEST parameter is calculated to determine convergence
 * (Skilling 1986) Eqn. 28
 * (Elliott 1999) Eqn. 10
 * 
 */
void MaxEnt::_DistCalc()
{
    //calculate _entropy and chi squared distances
    arma::vec entDistTemp = _eVec[0] % _gradEntropy;
    arma::vec chiDistTemp = _eVec[1] % _gradChiSquared;

    //if edge constaints are being used, don't include the head and tail of the vector for the calculation as these should be 0
    if(_useEdgeConstraints == true)
    {
        _entDist = sqrt(accu(entDistTemp.subvec(_lengthPropagation,_depth - (_lengthSubstrate + 1))));
        _chiDist = sqrt(accu(chiDistTemp.subvec(_lengthPropagation,_depth - (_lengthSubstrate + 1))));
    }
    else
    {
        _entDist = sqrt(accu(entDistTemp));
        _chiDist = sqrt(accu(chiDistTemp));
    }  
}

/**
 * @brief Class member function used to calculate the TEST parameter which is a measure of convergence of the system. Should be able to obtain TEST < 0.01 fairly easily
 * (Elliott 1999) Eqn. 10
 * 
 */ 
void MaxEnt::_TESTCalc()
{
    //calculate TEST parameter
    //divide by zero catch means that if either _entDist or _chiDist are 0 then _test is defined to be 0, not sure if this is reasonable or not
    if(_entDist !=0 && _chiDist != 0)
    {
        arma::vec testMat = pow(_gradEntropy/_entDist - _gradChiSquared/_chiDist,2);

        //if edge constaints are being used, don't include the head and tail of the vector for the calculation as these should be 0
        if(_useEdgeConstraints == true)
        {
            _test = accu(testMat.subvec(_lengthPropagation,_depth - (_lengthSubstrate + 1)));
        }
        else
        {
            _test = accu(testMat);
        }  
        _test /= 2.0;
    }
    else
    {
        _test = 0.0;
    }
}

/**
 * @brief Class member function to transform the basis vectors into the metric of the entropy (g) and diagonalise the system
 * 
 */
void MaxEnt::_NormBasisVec()
{
  //normalise basis vectors
  for(int i = 0; i < _numBasisVectors; i++)
	{
        double vecnorm = accu(_eVec[i]%_eVec[i]);
        vecnorm = sqrt(vecnorm);
        if(vecnorm != 0.0)
        {
            _eVec[i] = _eVec[i]/vecnorm;
        }
	}
}

/**
 * @brief Class member function to transform the basis vectors into the metric of the entropy (g) and diagonalise the system
 * (Skilling 1986) Eqn. 13
 * (Weinberger 2021) Eqn. 102
 * 
 * @param charge The charge of one of the components of the system
 */
void MaxEnt::_DiagG(arma::vec& charge)
{
    std::vector<arma::vec> temp(_numBasisVectors);
    std::vector<arma::vec> eVecTemp(_numBasisVectors);

    arma::vec eigval;
    arma::mat eigvec;

    _g.zeros();

    //fill temp vector
    for(int i = 0; i < _numBasisVectors; i++)
	{
	    temp[i] = _eVec[i]/(charge);
	}  
  
    //compute values of G
    for(int i = 0; i < _numBasisVectors; i++)
	{
	    for(int j = 0; j <= i; j++)
		{
		    _g(j,i) = accu(_eVec[j]%temp[i]);
		    if(i != j)
			{
			    _g(i,j) = _g(j,i);
			}
		}
	}

    try
	{
	    //calculate eigenvalues and eigenvectors
	    eig_sym(eigval, eigvec, _g);
	}
    catch(...)
	{
	    std::cerr << "Issue diagonalising _g" << std::endl;
	    exit (EXIT_FAILURE);
	}

    //sort eigenvalues and vectors in descending order
    /****************************************/
    /*check if necessary*******************/
    /***************************************/
    //Eigensort(eigval, eigvec);

    //transform basis vectors e to g basis
    _TransformBasis(eigvec, eVecTemp);

    //diagonalise g
    _g.zeros();

    //transform to diagonal _g basis
    for(int i = 0; i < _numBasisVectors; i++)
	{
        double normFactor = sqrt(fabs(eigval[i]));

        if(normFactor != 0)
        {
            _g(i,i) = 1.0/normFactor;
        }
        else
        {
            _g(i,i) = 0.0;
        }

        eVecTemp[i] *= _g(i,i);
	}

    _eVec = eVecTemp;

    //check if G 
    #ifdef DEBUGPLUS
    std::cout << "G: " << std::endl;
    _g.print();

    //this is a check to see if g is now the identity
    
    //fill temp vector
    for(int i = 0; i < _numBasisVectors; i++)
	{
        temp[i] = _eVec[i]/(charge);
	}
  
  
    //compute values of G
    for(int i = 0; i < _numBasisVectors; i++)
	{
	    for(int j = 0; j <= i; j++)
		{
		    _g(j,i) = accu(_eVec[j]%temp[i]);
		    if(i != j)
			{
			    _g(i,j) = _g(j,i);
			}
		}
	}

    std::cout << "G Test: " << std::endl;
    _g.print();

    std::cout << "G Eigenvectors: " << std::endl;
    eigvec.print();

    std::cout << "G Eigenvalues: " << std::endl;
    eigval.print();
    #endif 
}

/**
 * @brief Class member function to transform the basis vectors into the metric of chi squared (h) and diagonalise the system
 * (Skilling 1986) Eqn. 14
 * (Elliott 1999) Eqn. 15-16
 * (Weinberger 2021) Eqn. 83-87
 * 
 * @param sld the sld value of the component
 */
void MaxEnt::_DiagH(double& sld)
{
    std::vector<arma::vec> temp(_numBasisVectors);
    std::vector<arma::vec> eVecTemp(_numBasisVectors);
    std::vector<arma::vec> eVecData(_numBasisVectors);

    arma::vec eigval;
    arma::mat eigvec;

    //transform basis vectors to reciprocal space
    for(int i = 0; i < _numBasisVectors; i++)
	{

        _eVec[i] *= sld;

        if(_volumetricNormalisation == true || _boundSLD == true)
        {
            _eVec[i] *= _regularScaling;
           _eVec[i][_indexMax] *= _maxIndexScaling/(_regularScaling*sld);
        }

        _eVec[i] /= (double)_depth;

        _MakeComplex(_eVec[i],_eVecComplex[i]);
        fftw_execute_dft(_eVecPlan[i], (double(*)[2])&_eVecComplex[i](0), (double(*)[2])&_eVecTempA[i](0));

        //_eVecTempA[i] = fft(_eVec[i]);
        _eVecTempA[i] = _eVecTempA[i]%_sldTransformConj;
        _eVecTempA[i] = _eVecTempA[i]*(1-cexp(-i*_Complex_I*(2*M_PI/(double)_depth)));
        _eVecTempA[i] += _sldTransformConj%(_eVec[_numBasisVectors-1] - _eVec[0]);
        

        eVecData[i] = 2*arma::real(_eVecTempA[i]);
        for(int j = 0; j < _qOffset; j++)
        {
            eVecData[i][j] = 0.0;
        }

        for(int j = _qCutOff; j < _depth; j++)
        {
            eVecData[i][j] = 0.0;
        }
        temp[i] = 2*eVecData[i]%_inverseVar;
	}
  
    //compute values of G
    for(int i = 0; i < _numBasisVectors; i++)
	{
        for(int j = 0; j <= i; j++)
		{
            _h(j,i) = accu(eVecData[j]%temp[i]);
            if(i != j)
			{
                _h(i,j) = _h(j,i);
			}
		}
	}

    _h = _chiSquaredScale*_h;

    try
	{
        //calculate eigenvalues and eigenvectors
        eig_sym(eigval, eigvec, _h);
	}
    catch(...)
	{
        std::cerr << "Issue diagonalising _h" << std::endl;
        exit (EXIT_FAILURE);
	}

    //_h = sld*sld*_h; 

    //sort eigenvalues and vectors in descending order
    _Eigensort(eigval, eigvec);

    //transform basis vectors e to h basis
    _TransformBasis(eigvec, eVecTemp);

    _eVec = eVecTemp;

    //diagonalise h
    _h.zeros();

    //transform to diagonal _g basis
    for(int i = 0; i < _numBasisVectors; i++)
	{
        _h(i,i) = eigval[i];
        _gamma[i] = eigval[i];
	}

    //check diagonalisation
    #ifdef DEBUGPLUS
    std::cout << "H: " << std::endl;
    _h.print();

    std::cout << "H Eigenvectors: " << std::endl;
    eigvec.print();

    std::cout << "H Eigenvalues: " << std::endl;
    eigval.print();
    #endif
}

/**
 * @brief Class member function to transform basis of the search vectors
 * 
 * @param eigvec Matrix containing eigenvectors of the metric to which the vectors are being transformed
 * @param eVecTemp A vector to which the transformed basis vectors are to be stored
 */
void MaxEnt::_TransformBasis(const arma::mat& eigvec, std::vector<arma::vec>& eVecTemp)
{
    //transform to diagonal  basis
    for(int i = 0; i < _numBasisVectors; i++)
	{
        //zero temp vectors
        eVecTemp[i].set_size(_depth);
        eVecTemp[i].zeros();
        for(int j = 0; j < _numBasisVectors; j++)
		{
            eVecTemp[i] += eigvec(j,i)*_eVec[j];
		}
	}
}

/**
 * @brief Class member function to generate the subspace of entropy that is to be searched
 * (Skilling 1986) Eqn. 7,9,11,13
 * 
 */
void MaxEnt::_SubspaceS()
{
    for(int i = 0; i < _numBasisVectors; i++)
	{
	    _s[i] = accu(_eVec[i]%_gradEntropy);
	}
}

/**
 * @brief Class member functionto generate the subspace of chisquared that is to be searched
 * (Skilling 1986) Eqn. 8,10,12,14
 * 
 */
void MaxEnt::_SubspaceC()
{
    for(int i = 0; i < _numBasisVectors; i++)
	{
	    _c[i] = accu(_eVec[i]%_gradChiSquared);
	}
}

/**
 * @brief Class member function to iterate over values of the Lagrange multipliers \a A and \a B in order to find all potential values
 * (Skilling 1986) Section 3 - Control
 * 
 * @param l0Squared Confidence range for quadratic approximation
 */
void MaxEnt::_LagrangeSearch(double& l0Squared)
{
    double b = 0.1; //starting value of b to iterate through
    double a, aPlus, aMinus;
    _currentSearchIter = 0; //search counter for while loop

    bool sharedStop = false;
    int threadSearchIter = 0;
    double bThread;

    arma::vec deltaThread = _delta;

    #pragma omp parallel private(threadSearchIter, deltaThread, aPlus, aMinus, bThread, a) \
    shared(_bVector, _aVector, _currentSearchIter, _gamma, _s, _c, sharedStop)
    {
        while (!sharedStop)
        {
            #pragma omp critical
            {
                b+=0.1; //step b, note this is arbitrary and a smaller step will provide a more precise search
                bThread = b;
            }

            bool bReject = false;
            bool aPlusAccept = false;
            bool aMinusAccept = false;
            bool bAccept = false;

            //generate corresponding values of a
            //(Skilling 1986) Eqn. 37
            deltaThread = -_gamma - bThread;
            double sSum = accu(pow(_s,2)/pow(deltaThread,2));
            double cSum = accu(pow(_c,2)/pow(deltaThread,2));
            double scSum = accu((_c%_s)/pow(deltaThread,2));

            //(Skilling 1986) Eqn. 37
            double dSquared = l0Squared*sSum - sSum*cSum + scSum*scSum;

            //if d squared is less than 0 reject trial b as no points will be inside the trust region
            //(Skilling 1986) Eqn. 37
            if(dSquared <= 0.0)
            {
                bReject = true;
            }

            //find roots of a and b

            if(!bReject)
            {
                //(Skilling 1986) Eqn. 36
                aPlus = (scSum + sqrt(dSquared))/sSum;
                aMinus = (scSum - sqrt(dSquared))/sSum;

                //check allowed values of a
                //(Skilling 1986) Eqn. 38
                if(aMinus > 0.0 && bThread > aMinus)
                {
                    aMinusAccept = true;
                }
                if(aPlus > 0.0 && bThread > aPlus)
                {
                    aPlusAccept = true;
                }
                if(bThread > aMinus && aPlus > bThread)
                {
                    bAccept = true;
                }
            
                if(!(aMinusAccept || aPlusAccept || bAccept))
                {
                    bReject = true;
                }
            
                //check whether each extrema provides true maximum of _s within both possible regimes - distance limited or not distance limited
                //(Skilling 1986) Eqn. 33
                if(bAccept) //no distance limiting, l<r and a=b>0
                {
                    _NoDistanceLimit(a, bThread, bReject, threadSearchIter, sharedStop, deltaThread);
                }
                else if (aMinusAccept || aPlusAccept) //apply distance limiting l=r, 0<a<b
                {
                    _DistanceLimit(a, bThread, aPlus, aMinus, bReject, aPlusAccept, aMinusAccept, threadSearchIter, sharedStop, deltaThread);
                }   
            }
        }
    }
}

/**
 * @brief Class member function to check if a potential combination of Lagrange multipliers is valid when searching within the distance limited region
 * 
 * @param a A Lagrange multiplier
 * @param b A Lagrange multiplier
 * @param bReject Boolean variable determining whether a given value of b is rejected or accepted
 * @param threadSearchIter The thread based search iteration counter
 * @param sharedStop Boolean variable determining whether searching must stop
 * @param deltaThead delta vector for the thread
 */
void MaxEnt::_NoDistanceLimit(double& a, double& b, bool& bReject, int& threadSearchIter, bool& sharedStop, arma::vec& deltaThread)
{
    //(Skilling 1986) Eqn. 33
    a = b;

    //(Skilling 1986) Eqn. 34
    arma::vec x = (_c - a*_s)/deltaThread;

    //(Skilling 1986) Eqn. 39-43
    if(deltaThread[0] <= 0.0)
	{
	    bReject = false; //accept a and b
	}
    if(deltaThread[1] >= 0.0)
	{
	    bReject = true; //reject a and b	  
	}
    if(deltaThread[1] < 0.0 && deltaThread[0] > 0.0)
	{
        double fZero = 0.0; //check sign of F(0)

        //(Skilling 1986) Eqn. 47
        double deltaProduct = prod(deltaThread);

        //(Skilling 1986) Eqn. 48
        fZero = accu(deltaProduct*(pow(_s-x,2)/deltaThread));

        //(Skilling 1986) Eqn. 49
        if(fZero >= 0.0)
		{
            bReject = false;
		}
	    else
		{
		    bReject = true;
		}
	}


    if(!bReject)
	{
        #pragma omp critical
        {
            threadSearchIter = _currentSearchIter;
            if(threadSearchIter < _maximumSearchIter)
            {
                _aVector[threadSearchIter] = a;
                _bVector[threadSearchIter] = b;
            }
            _currentSearchIter++;
            threadSearchIter = _currentSearchIter;
        }

        if(_currentSearchIter >= _maximumSearchIter)
        {
            sharedStop = true;
            //#pragma omp barrier
            #pragma omp flush(sharedStop)
        }  
	}
}

/**
 * @brief Class member function to check if a potential combination of Lagrange multipliers is valid when searching outside the distance limited region
 * 
 * @param a A Lagrange multiplier
 * @param b A Lagrange multiplier
 * @param aPlus Positive root of \a A Lagrange multiplier
 * @param aMinus Negative root of \a A Lagrange multiplier
 * @param bReject Boolean variable determining whether a given value of b is rejected or accepted
 * @param aPlusAccept Boolean variable determining whether a given value of \a aPlus is rejected or accepted
 * @param aMinusAccept Boolean variable determining whether a given value of \a aMinus is rejected or accepted
 * @param threadSearchIter The thread based search iteration counter
 * @param sharedStop Boolean variable determining whether searching must stop
 * @param deltaThead delta vector for the thread
 */
void MaxEnt::_DistanceLimit(double& a, double& b, const double& aPlus, const double& aMinus, bool& bReject, bool& aPlusAccept, bool& aMinusAccept, int& threadSearchIter, bool& sharedStop, arma::vec& deltaThread)
{

    //(Skilling 1986) Eqn. 50 - 57
    while(aPlusAccept || aMinusAccept)
	{
	    if(aPlusAccept)
		{
            a = aPlus;
            aPlusAccept = false;
		}
	    if (aMinusAccept)
		{
            a = aMinus;
            aMinusAccept = false;
		}

        //(Skilling 1986) Eqn. 34
        arma::vec x = (_c - a*_s)/deltaThread;

        if(deltaThread[0] <= 0.0)
		{
		    bReject = false; //accept a and b
		}
	    if(deltaThread[1] <= 0.0 && deltaThread[0] >= 0.0)
		{
            //(Skilling 1986) Eqn. 58
            double phiZero = 0.0; //check sign of phi(0)

            //(Skilling 1986) Eqn. 47
            double deltaProduct = prod(deltaThread);

		    for(int i = 0; i < _numBasisVectors; i++)
			{
			    for(int j = 0; j < _numBasisVectors; j++)
				{
				    if(i != j)
					{
                        phiZero += pow(_s[i]*x[j] - x[i]*_s[j],2)/(deltaThread[i]*deltaThread[j]);
					}
				}
			}

            //(Skilling 1986) Eqn. 59-60
            phiZero = 0.5*deltaProduct*phiZero;

            if(phiZero <= 0.0)
			{
                bReject = true;
			}
		    else
			{
                bReject = false;
			}
		}
	    if(deltaThread[2] <= 0.0 && deltaThread[1] >= 0.0) //check signs of G(0), H(0) and Phi(0)
		{

            //(Skilling 1986) Eqn. 61
            double gZero = 0.0;
            double hZero = 0.0;
            double phiZero = 0.0;
            double deltaProduct = prod(deltaThread);

            gZero = - accu(pow(x,2)/deltaThread);
            hZero = - accu(pow(_s,2)/deltaThread);
            
            for(int i = 0; i < _numBasisVectors; i++)
			{
			    for(int j = 0; j < _numBasisVectors; j++)
				{
				    if(i != j)
					{
					    phiZero += pow(_s[i]*x[j] - x[i]*_s[j],2)/(deltaThread[i]*deltaThread[j]);
					}
				}
			}

            //(Skilling 1986) Eqn. 61
            gZero = -gZero*deltaProduct;
            hZero = -hZero*deltaProduct;
            phiZero = 0.5*deltaProduct*phiZero;

            if(gZero > 0.0 || hZero > 0.0)
			{
			    bReject = true;
			}
		    if(gZero < 0.0 && hZero < 0.0 && phiZero < 0.0)
			{
			    bReject = true;
			}
		    else
			{
			    bReject = false;
			}
		}
	    if(deltaThread[3] >= 0.0)
		{
		    bReject = true;
		}
	}
    if(!bReject)
	{
        #pragma omp critical
        {
            threadSearchIter = _currentSearchIter;
            if(threadSearchIter < _maximumSearchIter)
            {
                _aVector[threadSearchIter] = a;
                _bVector[threadSearchIter] = b;
            }
            _currentSearchIter++;
            threadSearchIter = _currentSearchIter;
        }

        if(_currentSearchIter >= _maximumSearchIter)
        {
            sharedStop = true;
            //#pragma omp barrier
            #pragma omp flush(sharedStop)
        }
	}
}

/**
 * @brief Class member function to choose which combination of \a A and \a B Lagrange multipliers produces the best choice for convergence
 * (Skilling 1986) Section 4 Final Selection of Image Increment
 * 
 */
void MaxEnt::_ChooseAB()
{

    //intialise vectors used for storing values of S and ChiSquared to find maximum a,b combo
    arma::vec sVector(_maximumSearchIter);
    sVector.zeros();
    arma::vec cVector(_maximumSearchIter);
    cVector.zeros();

    for(int i = 0; i < _maximumSearchIter; i++)
	{
        for(int j = 0; j < _numBasisVectors; j++)
		{
            _delta[j] = -_gamma[j] - _bVector[i];
            _x[j] = (_c[j]- _aVector[i]*_s[j])/_delta[j];
            sVector[i] += (_s[j]*_x[j]) - (pow(_x[j], 2)/2.0);
            cVector[i] += (_c[j]*_x[j]) + ((_gamma[j]*pow(_x[j],2))/2.0);
		}
        sVector[i] += _entropy;
        cVector[i] += _chiSquared;
	} 
  

    //find maximum values of S and min values of _c
    double cMin = cVector.min();


    double cTarget = _chiSquared - (_chiSquared - cMin)*0.8; //find where chi is 4/5 towards cMin and determine index of this

    //find index of value closest to cTarget 
    int cTargetIndex = (int)arma::index_min(abs(cVector - cTarget));

    #ifdef DEBUG 
    //define max value of S
    double sMax = sVector.max();
    std::cout << "C Target: " << cTarget << std::endl;
    std::cout << "C Target Index: " << cTargetIndex << std::endl;

    std::cout << "S Max: " << sMax << std::endl;
    std::cout << "C Min: " << cMin << std::endl;
    #endif
    
    //calculate step length
    _delta = -_gamma - _bVector[cTargetIndex];
    _x = (_c - _aVector[cTargetIndex]*_s)/_delta;
}

/**
 * @brief Class member function that sets values of array below a threshold to the threshold value
 * 
 */
void MaxEnt::_SetZero(arma::vec& charge)
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
 * @brief Class member function that increments the charge by the search vectors found using the lagrange multipliers to find an updated charge to be used in the next step (iteratively)
 * 
 * @param charge the component of one of the system species
 * @param norm The total amount of that species in this system
 */
void MaxEnt::_CalcNewCharge(arma::vec& charge, double& norm)
{

    //this constraint determines whether or not limitations should be put on the charge incrementation which smooths the profile to make it more physical
    if(_smoothIncrement == true && _redChi)
    {
        _SmoothIncrement(charge);
    }
    else
    { 
        _RegularIncrement(charge);
    }

    /**
    double maxCharge = charge.max();
    for(int i = 0; i < _depth; i++)
    {
        if(charge[i] <=  0.05*maxCharge)
        {
            charge[i] = 0;
        }
    }
    **/


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
 * @brief Regular incrementation of the charge as defined by the Cambridge Algorithm by Skilling and Gull
 * 
 * @param charge the component of one of the system species
 */
void MaxEnt::_RegularIncrement(arma::vec& charge)
{
    for(int i = 0; i < _numBasisVectors; i++)
	{
	    charge += _x[i]*_eVec[i];
	}
}

/**
 * @brief Charge incrementation with smoothness constraints that produce a more physical SLD profile 
 * (Weinberger 2021) Section 6.3.1
 * 
 * @param charge the component of one of the system species
 */
void MaxEnt::_SmoothIncrement(arma::vec& charge)
{
    for(int i = 0; i < _numBasisVectors; i++)
	{
		for(int j = 0; j < _depth; j++)
		{
			if(j!=0 && j!=_depth-1 && charge[j] < charge[j-1] && charge[j] < charge[j+1] && _x[i]*_eVec[i][j] <= 0)
			{

			}
			else if( j!=0 && j!=_depth-1 && charge[j] > charge[j-1] && charge[j] > charge[j+1] && _x[i]*_eVec[i][j] >= 0)
			{

			}
			else
			{
				charge[j] += _x[i]*_eVec[i][j];
			}
		}
	}
}

/**
 * @brief Constraints on starting (air region) SLD value and final (substrate SLD) values
 * (Weinberger 2021) Section 6.3.1
 * 
 * @param charge the component of one of the system species
 */
void MaxEnt::_Constraints(arma::vec& charge)
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
 * (Weinberger 2021) Eqn. 11/67
 * 
 */
void MaxEnt::_Reflectivity()
{
    //_sldPlan = fftw_plan_dft_1d(_depth, (double(*)[2])&_sldComplex(0), (double(*)[2])&_sldTransform(0), FFTW_FORWARD, FFTW_MEASURE);
    _MakeComplex(_sld, _sldComplex);
    fftw_execute_dft(_sldPlan,(double(*)[2])&_sldComplex(0), (double(*)[2])&_sldTransform(0));

    //Factor for DFT derivative rule
    arma::cx_vec dftDeriv;

    dftDeriv.set_size(_depth);

    for(int i = 0; i < _depth; i++)
    {
        dftDeriv[i] = 1-cexp(-i*_Complex_I*(2*M_PI/(double)_depth));
    }
    
    //(Weinberger 2021) 68/94
    _sldTransform = (_sldTransform%dftDeriv + _sld[_depth-1] - _sld[0]);


    _sldTransform /= (double)_depth;

    for(int i = 0; i < _qOffset; i++)
    {
        _sldTransform[i] = 0.0;
    }
    for(int i = _qCutOff; i < _depth; i++)
    {
        _sldTransform[i] = 0.0;
    }


    arma::vec reflectivityTempInt = abs(_sldTransform);

    for(int i = 0; i < _depth; i++)
    {
        _sldImage[i] = pow(reflectivityTempInt[i], 2.0);
    }

    _Renormalise(_sldImage, _norm);

    for(int i = 0; i < _qOffset; i++)
    {
        _sldImage[i] = 0.0;
    }
    for(int i = _qCutOff; i < _depth; i++)
    {
        _sldImage[i] = 0.0;
    }
}

/**
 * @brief Class member function to sort eigenvectors and eigevalues in descending order according to the eigenvalues of the system
 * 
 * @param eigval A vector containing the eigenvalues of the system
 * @param eigvec A matrix containing the eigenvectors
 */
void MaxEnt::_Eigensort(arma::vec& eigval, arma::mat& eigvec)
{
    arma::uvec sortedIndeces = sort_index(eigval, "descend"); //sort in descending order
    eigval = sort(eigval, "descend"); //sort eigval vector

    arma::mat eigvecTemp; //temporary matrix to store sorted eigenvectors in
    eigvecTemp.set_size(_numBasisVectors,_numBasisVectors); 

    //sort matrix according to eigenvalues
    for(int i = 0; i < _numBasisVectors; i++)
	{
	    eigvecTemp.col(i) = eigvec.col(sortedIndeces(i));
	}

    eigvec = eigvecTemp;
}

/**
 * @brief Class member function to sort one vector according to the order of another
 * 
 * @param vec1 A vector to be sorted
 * @param vec2 A vector to be sorted according to vec1
 */
void MaxEnt::_Eigensort(arma::vec& vec1, arma::vec& vec2)
{
    arma::uvec sortedIndeces = sort_index(vec1, "ascend"); //sort in descending order
    vec1 = sort(vec1, "ascend"); //sort eigval vector

    arma::vec vec2Temp; //temporary matrix to store sorted eigenvectors in
    vec2Temp.set_size(_maximumSearchIter); 

    //sort matrix according to eigenvalues
    for(int i = 0; i < _maximumSearchIter; i++)
	{
	    vec2Temp[i] = vec2[sortedIndeces(i)];
	}

    vec2 = vec2Temp;
}
  
/**
 * @brief Class member function used to calculate the chisquared gradient of the system
 * (Elliott 1999) Eqn. 7
 * (Weinberger 2021) Eqn. 75/98
 * 
 */
void MaxEnt::_GradChiSquared()
{
    //sum temp value as required for the delta function components in the gradient of the chisquared
	arma::cx_double temp1Sum = accu(_temp1);

    //half of the gradient of the chi squared, the other half is just the complex conjugate of this part
	arma::cx_vec gradChiSquaredPart;

    //fourier transform temp
	fftw_execute_dft(_temp1Plan, (double(*)[2])&_temp1(0), (double(*)[2])&_temp1(0));

	gradChiSquaredPart.set_size(_depth);

    //apply chi squared gradient equation
	for(int i = 0; i < _depth ; i++)
	{
		gradChiSquaredPart[i] = _temp1[i] - _temp1[(i+1)%_depth];
	}

	gradChiSquaredPart[0] -= temp1Sum; 
	gradChiSquaredPart[_depth-1] += temp1Sum; 

	_gradChiSquared = 2*arma::real(gradChiSquaredPart);
}

/**
 * @brief Class member function used to calculate the second derivative of the chisquared of the system
 * (Elliott 1999) Eqn. 7
 * (Weinberger 2021) Eqn. 82/99  
 * 
 * @param sld The sld value of the substance
 */
void MaxEnt::_GGChiSquared( double& sld)
{

    //sum temp value as required for the delta function components in the gg of the chisquared
	arma::cx_double temp1Sum = accu(_temp1);

    //create a temporary store for temp1 as temp1 is transformed in two different ways
	_temp2 = _temp1; 

    //half of the imaginary part of gg of the chi squared, the other half is just the complex conjugate of this part
	arma::cx_vec ggChiSquaredPart;

	ggChiSquaredPart.set_size(_depth);

    //fourier transform and second order fourier transform 
	fftw_execute_dft(_temp1Plan, (double(*)[2])&_temp1(0), (double(*)[2])&_temp1(0));
	fftw_execute_dft(_temp2Plan, (double(*)[2])&_temp2(0), (double(*)[2])&_temp2(0));

    //apply ggChiSquared equation
	for(int i = 0; i < _depth; i++)
	{
		ggChiSquaredPart[i] = _temp2[i] - 2.0*_temp1[(2*i+1)%_depth] + _temp2[(i+1)%_depth] + 2.0*_temp1[_depth - 1] - 4.0*_temp1[0] + 2.0*_temp1[1];	
	}

	ggChiSquaredPart[0] -= temp1Sum; 
	ggChiSquaredPart[_depth-1] += temp1Sum;


	_ggChiSquared = 2.0*sld*sld*arma::real(ggChiSquaredPart);

    //final corrections to ggChiSquared
    arma::vec factorK;
    factorK.set_size(_depth);
    arma::vec temp3;

    temp3 = 4.0*(_sldImage)%_inverseVar;

    for(int i = 0; i < _qOffset; i++)
    {
        temp3[i] = 0.0;
    }
    for(int i = _qCutOff; i < _depth; i++)
    {
        temp3[i] = 0.0;
    }

    factorK[0] = accu(temp3);
    factorK[_depth - 1] = 5*accu(temp3);

    double generalOffset = 0;
    double finalOffset = 0;

    for(int i = 0; i < _depth; i++)
    {
        generalOffset += temp3[i]*4.0*(pow(sin((M_PI*i)/(double)_depth),2.0));
    }

    for(int i = 0; i < _depth; i++)
    {
        factorK[i] = generalOffset;
    }

    for(int i = 0; i < _depth; i++)
    {
        finalOffset += temp3[i]*4.0*cos((2.0*M_PI*i)/(double)_depth);
    }

    factorK[_depth - 1] += finalOffset;

    _ggChiSquared += sld*sld*factorK;

    if(_volumetricNormalisation == true || _boundSLD == true)
    {
        _ggChiSquared *= (_regularScaling*_regularScaling);
        _ggChiSquared[_indexMax] *= pow(_maxIndexScaling/(_regularScaling*sld),2.0);
    }

    _ggChiSquared /= ((double)_depth*_depth);


    _ggChiSquared += accu(4.0*(_sldImage - _dataFit)%_inverseVar);
}

/**
 * @brief Scale charge to physical SLD parameters so that value constraints can be imposed on the system
 * (Weinberger 2021) Section 6.3.1
 * 
 */
void MaxEnt::_BoundSLD()
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
 * @brief Smoothes the charge profile according to N_i = 1/4 N_i-1 + 1/2 N_i + 1/4 N_i+1 every smoothInterval steps
 * (Weinberger 2021) Section 7.3.3 Constraints 
 * 
 * @param charge the component of one of the system species
 */
void MaxEnt::_SmoothProfile(arma::vec& charge)
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
 * (Weinberger 2021) Section 7.3.3 Constraints 
 * 
 * @param charge the component of one of the system species
 */
void MaxEnt::_ForceZero(arma::vec& charge)
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





 
  
	  
	  
		 
		  

  
  
  

  



  
  
	   
  
  
  
  

  
  
  

  
  
	

  
  
  

  
	