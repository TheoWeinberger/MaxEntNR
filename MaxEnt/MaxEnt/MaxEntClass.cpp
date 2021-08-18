/**
 * @file MaxEntClass.cpp
 * @author Theo Weinberger
 * @brief Class Function that employs the principle of Maximum Entropy via the Cambridge algorithm
 * to invert an SLD profile from its reflectivity spectrum. This asssumes that the SLD and reflecitivity are related by the transform R[Q] = \frac{1}{Q^4} * |FFT(SLD)[Q]*[1-exp{-Q*I*(2*PI/totalDepth)}] +
 * SLD[totalDepth - 1] - SLD[0]|^2
 * @version 3.0
 * @date 2021-05-01
 * 
 * @copyright Copyright (_c) 2021
 * 
 */

#include <iostream>
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
 */
MaxEnt::MaxEnt(const std::string& dataFile)
{

    //load in file containing reflectivity data
    _dataFit.load(dataFile, arma::raw_ascii);

    //define system _depth which is equal to the number of elements in the array
    _depth = _dataFit.n_elem;

    //remove Q^4 dependence from data, this simplifies the tranform and also creates larger
    //data perturbations which should produce better fitting
    for(int i = 0; i < _depth; i++)
    {
        if(i != 0)
        {
            _dataFit[i] = _dataFit[i]*i*i*i*i;
        }
    }

    //renormalise input data so max value of intensity is 1
    double maxData =  _dataFit.max();
    _dataFit /= maxData/_dataScale; 

    //output data without Q^4 dependence to a file called dataInit so that fitting data can be checked for consistency
    _dataFit.save("dataInit", arma::raw_ascii);

    //set data vector sizes
    _inverseVar.set_size(_depth);
    _charge.set_size(_depth);
    _chargeTransform.set_size(_depth);
    _chargeTransformConj.set_size(_depth);
    _chargeImage.set_size(_depth);
    _gradChiSquared.set_size(_depth);
    _ggChiSquared.set_size(_depth);
    _gradEntropy.set_size(_depth);
}

/**
 * @brief Construct a new Max Ent object which reads in data from a string which specifies the file in 
 * which the data is stored. The data file should contain the reflectivity data in a one dimensionol column without
 * scaled axes (the algorirthm uses unscaled data as any scaling is normalised out). The second string refers to the 
 * settings file, which is a .cfg file containing the simulation specifics such as run number that are used in
 * the Cambridge algortihm
 * 
 */
MaxEnt::MaxEnt(const std::string& dataFile, const std::string& configFile)
{
    //read in data for diffraction grating to be built from
    ReadFile(configFile, _totalIterations, _maximumSearchIter, _numBasisVectors, _zeroLevel, _minVar, _dataScale, _sldMaxBound, _sldMinBound, _propagationSLD, _substrateSLD, _lengthPropagation, _lengthSubstrate, _smoothProfile, _sldScaling, _useEdgeConstraints, _chiSquaredScale, _spikeCharge, _spikePortion, _spikeAmount);


    //load in file containing reflectivity data
    _dataFit.load(dataFile, arma::raw_ascii);

    //define system _depth which is equal to the number of elements in the array
    _depth = _dataFit.n_elem;

    //remove Q^4 dependence from data, this simplifies the tranform and also creates larger
    //data perturbations which should produce better fitting
    for(int i = 0; i < _depth; i++)
    {
        if(i != 0)
        {
            _dataFit[i] = _dataFit[i]*i*i*i*i;
        }
    }

    //renormalise input data so max value of intensity is 1
    double maxData =  _dataFit.max();
    _dataFit /= maxData/_dataScale;

    //Matrix to output scaled data to 
    arma::mat dataFitScaled;
    dataFitScaled.set_size(_depth,2);
    
    for(int i = 0; i < _depth; i++)
    {
        dataFitScaled(i,0) = i;
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
    _charge.set_size(_depth);
    _chargeTransform.set_size(_depth);
    _chargeTransformConj.set_size(_depth);
    _chargeImage.set_size(_depth);
    _gradChiSquared.set_size(_depth);
    _ggChiSquared.set_size(_depth);
    _gradEntropy.set_size(_depth);
}

/**
 * @brief Class member fucntion to initialise the data for the MaxEnt algorithm
 * 
 */
void MaxEnt::_Init()
{
    //Initialise statistical quantities for the simulation 
    _InitStatQuant();

    _InitCharge();

    //output initial charge disitribution to file called chargeInit
    arma::mat sldOutput = GetSLDScaled();

    sldOutput.save("sldInit", arma::raw_ascii);

    //calculate the DEF parameter for the system - the weighted average of the cells
    _DEF();  
}

/**
 * @brief Class member function to apply one step of the MaxEnt algorithm
 * 
 */
void MaxEnt::_Step()
{

    #ifdef BENCHMARKFULL
    start = std::chrono::high_resolution_clock::now();
    #endif

    //calculate starting _entropy and chisquared
    _Entropy();

    #ifdef BENCHMARKFULL
    finish = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "Time taken by function Entropy: " << runtime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARK
    auto beginning = std::chrono::high_resolution_clock::now();
    #endif

    //calculate current charge FTs and image
    _Reflectivity();

    #ifdef BENCHMARK
    auto end = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    auto executiontime = std::chrono::duration_cast<std::chrono::microseconds>(end - beginning);
    std::cout << "Time taken by function Reflectivity: " << executiontime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARKFULL
    start = std::chrono::high_resolution_clock::now();
    #endif

    _ConjCharge();

    #ifdef BENCHMARKFULL
    finish = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    runtime = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
    std::cout << "Time taken by function ConjSLD: " << runtime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARK
    beginning = std::chrono::high_resolution_clock::now();
    #endif

    //get ChiSquared related quantities
    _ChiSquared();

    #ifdef BENCHMARK
    end = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    executiontime = std::chrono::duration_cast<std::chrono::microseconds>(end - beginning);
    std::cout << "Time taken by function ChiSquared: " << executiontime.count() << " microseconds" << std::endl;
    #endif

    //_Renormalise(charge); //is this required/does this help --> few questions about normalisation

    //calculate basis set and calculate search directions including transformation to diagonalised basis sets 
    #ifdef BENCHMARK
    beginning = std::chrono::high_resolution_clock::now();
    #endif

    _BasisFunctions();

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

    _DiagG();

    #ifdef BENCHMARK
    end = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    executiontime = std::chrono::duration_cast<std::chrono::microseconds>(end - beginning);
    std::cout << "Time taken by function DiagG: " << executiontime.count() << " microseconds" << std::endl;
    #endif

    #ifdef BENCHMARK
    beginning = std::chrono::high_resolution_clock::now();
    #endif

    _DiagH();

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

    _LagrangeSearch();

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

    _CalcNewCharge();

    #ifdef BENCHMARK
    end = std::chrono::high_resolution_clock::now();
    // Get runtime of function
    executiontime = std::chrono::duration_cast<std::chrono::microseconds>(end - beginning);
    std::cout << "Time taken by function CalcNewCharge: " << executiontime.count() << " microseconds" << std::endl;
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
        sldOutput.save("sld" + std::to_string(_iterationCount), arma::raw_ascii);
        imageOutput.save("image" + std::to_string(_iterationCount) , arma::raw_ascii);
	}
    if(_iterationCount == _totalIterations)
    {
        arma::mat sldOutput = GetSLDScaled();
	    arma::mat imageOutput = GetReflectivityScaled();
        sldOutput.save("sldRecon", arma::raw_ascii);
        imageOutput.save("imageRecon", arma::raw_ascii);
    }

    if(_redChi <= _redChiMin)
    {
        arma::mat sldOutput = GetSLDScaled();
	    arma::mat imageOutput = GetReflectivityScaled();
        sldOutput.save("sldChi", arma::raw_ascii);
        imageOutput.save("imageChi", arma::raw_ascii);

        _redChiMin = _redChi;
    }

    if(_test <= _testMin)
    {
        arma::mat sldOutput = GetSLDScaled();
	    arma::mat imageOutput = GetReflectivityScaled();
        sldOutput.save("sldTest", arma::raw_ascii);
        imageOutput.save("imageTest", arma::raw_ascii);

        _testMin = _test;
    }
}

/**
 * @brief Employs Cambridge Algortihm to reconstruct the maximum _entropy SLD profile consistent with the 
 * input data
 * 
 */
void MaxEnt::Solve()
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

        _Step();
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
 * @return arma::mat the scaled SLD
 */
arma::mat MaxEnt::GetSLDScaled()const
{
    //Matrix to store SLD data in 
    arma::mat sldScaled;
    sldScaled.set_size(_depth,2);

    //temporary charge for manipulation
    arma::vec chargeTemp = _charge;

    //scale the charge for output if SLD bounds are provided

    if(_sldScaling == true)
    {
        _ScaleCharge(chargeTemp);
    }

    //scale the SLD    
    for(int i = 0; i < _depth; i++)
    {
        sldScaled(i,0) = i;
        sldScaled(i,1) = chargeTemp[i];
    }

    return sldScaled;
}

/**
 * @brief Get the scaled Reflectivity
 * 
 * @return arma::mat the scaled Reflectivity
 */
arma::mat MaxEnt::GetReflectivityScaled()const
{
    //Matrix to store reflectivity data in
    arma::mat reflectivityScaled;
    reflectivityScaled.set_size(_depth,2);
    double normalisation;
    
    //Apply scaling and add 1/Q^4 back into system
    for(int i = 0; i < _depth; i++)
    {
        reflectivityScaled(i,0) = i;
        if(i == 0)
        {
            reflectivityScaled(i,1) = 1.0;
        }
        else if(i == 1)
        {
            reflectivityScaled(i,1) = 1.0;
            normalisation = _chargeImage[i];
        }
        else 
        {
            reflectivityScaled(i,1) = _chargeImage[i]/(normalisation*i*i*i*i);
        }
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
    _l0Squared = 0.2*_norm; //input data should be normalised but may need to check this


    //assign matrix for inverse variance and calculate each value
    //(Elliott 1999) Eqn. 6
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
    //check stat quant values, data size and range
    #ifdef DEBUG	
    std::cout << "_cAim: " << _cAim << std::endl;
    std::cout << "Normalisation: " << _norm << std::endl;
    std::cout << "_l0Squared: " << _l0Squared << std::endl;

    std::cout << "Data Height: " << _dataFit.n_cols << " Data Width: " << _dataFit.n_rows << std::endl; 

    std::cout << "Data Max: " << _dataFit.max() << " Data Min: " << _dataFit.min() << std::endl; 
    assert(_dataFit.max() <= _dataScale && _dataFit.min() >= 0.0);
    #endif		
}


/**
 * @brief Class member function used to calculate quantities relevant to the entropy of the system
 * 
 */
void MaxEnt::_Entropy()
{
    //calculate _entropy
    //(Elliott 1999) Eqn. 4
    arma::vec entropyMat = _charge % (log(_charge/_def) - 1);

    _entropy = - arma::accu(entropyMat);
    
    //calculate _entropy gradient
    //(Weinberger 2021) Eqn. 101
    _gradEntropy = log(_def) - log(_charge);
}

/**
 * @brief Class member function used to calculate the conjugate charge for the system
 * 
 */
void MaxEnt::_ConjCharge()
{
  _chargeTransformConj = conj(_chargeTransform);
}

/**
 * @brief Class member function used to calculate quantities relevant to the chisquared of the system
 * 
 */
void MaxEnt::_ChiSquared()
{
    //initialise temporary data arrays and set size
    arma::cx_vec temp1, temp2;
    temp1.set_size(_depth);

    //calculate chi-Squared
    //(Skilling 1986) Eqn. 2
    arma::vec chiSquaredMat = pow(_chargeImage - _dataFit,2)%_inverseVar;

    _chiSquared = accu(chiSquaredMat);
    
    _redChi = _chiSquared/((double)(_depth));

    //calculate gradient of Chi-Squared 
    temp1 = 2.0*(_chargeTransformConj%(_chargeImage - _dataFit))%_inverseVar;

    //(Elliott 1999) Eqn. 7
    //(Weinberger 2021) Eqn. 75/98
    _GradChiSquared(temp1);

    //calculate double derivated of chi Squared
    temp1 = 2*pow(_chargeTransformConj,2)%_inverseVar;

    //(Elliott 1999) Eqn. 7
    //(Weinberger 2021) Eqn. 82/99                              
    _GGChiSquared(temp1, temp2);

    //scale gradients by linear weighting
    _gradChiSquared = _chiSquaredScale*_gradChiSquared;
    _ggChiSquared = _chiSquaredScale*_ggChiSquared;
}

/**
 * @brief Class member function to calculate the DEF parameter for the system
 * (Elliot 1999) Eqn. 5
 * 
 */
void MaxEnt::_DEF()
{
    arma::vec chargeLogCharge = _charge % log(_charge);

    double totalCharge = accu(_charge);

    double totalChargeLogCharge = arma::accu(chargeLogCharge);

    _def = exp(totalChargeLogCharge/totalCharge);

    //check value of def is sensible
    #ifdef DEBUG
    std::cout << "DEF: " << _def << std::endl;
    #endif
}

/**
 * @brief Class member function to renormalise the charge or the charge image so that the total charge of the system remains constant at the initial total intensity of the diffraction pattern
 * 
 */
void MaxEnt::_InitCharge()
{
    //randomly seed the arma random number generator 
    arma::arma_rng::set_seed_random();
    
    //initialise charge distribution and fill with random numbers chosen as uniform dist [0,1]
    _charge.randu();
    //_charge.load("chargeInit", arma::raw_ascii);
    //_charge.save("chargeInit", arma::raw_ascii);


    //Potential constraints on the system, constraints are used depending on their settings in the settings.cfg file

    if(_spikeCharge == true)
    {
        _SpikeCharge();
    }

    //this constraint scales the charge to the SLD profile bounds if they are known
    if(_sldScaling == true)
    {
        _ScaleCharge();
    }
    
    //fixed edge constraints settings
    if(_useEdgeConstraints == true)
    {
        _Constraints();
    }

    //Set any values below the threshold zero level to the threshold level
    _SetZero();

    //renormalise the charge 
    _Renormalise(_charge);
}

/**
 * @brief Function that 'spikes' the charge in close to the substrate region, this should help the profile 
 * build up from close to the substrate which is physically what is expected (albeit by no means definite)
 * 
 */
void MaxEnt::_SpikeCharge()
{
    int lengthSpike = int(round(_depth*_spikePortion));

    _charge.tail(lengthSpike) += _spikeAmount;
}



/**
 * @brief Class member function  to renormalise the charge or the charge image so that the total charge of the system remains constant at the initial total intensity of reflectivity spectrum
 * 
 * @param unnormalisedVector The unormalised input to be renormalised
 */
void MaxEnt::_Renormalise(arma::vec& unnormalisedVector)
{
  //renormalise the charge so that total charge = matrix size
  double totalCharge = fabs(accu(unnormalisedVector));

  unnormalisedVector = _norm*unnormalisedVector/(totalCharge);//total charge normalised to size of matrix

  //check renormalisation is working as expected
  #ifdef DEBUGPLUS
  std::cout << "Normalisation: " << _norm << " Total Charge " << accu(unnormalisedVector) << std::endl;
  #endif
  
  #ifdef DEBUG
  assert(_norm <= accu(unnormalisedVector) + 0.1 && _norm >= accu(unnormalisedVector) - 0.1);
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
 */
void MaxEnt::_BasisFunctions()
{
    //calculate basis vectors
    arma::vec basisOperator = _charge % _ggChiSquared;
    for(int i = 0; i < _numBasisVectors; i++)
	{
        if(i == 0)
		{
            _eVec[i] = _charge % _gradEntropy;
            }
        else if (i == 1)
		{
            _eVec[i] = _charge % _gradChiSquared;
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
void MaxEnt::_DiagG()
{
    std::vector<arma::vec> temp(_numBasisVectors);
    std::vector<arma::vec> eVecTemp(_numBasisVectors);

    arma::vec eigval;
    arma::mat eigvec;

    _g.zeros();

    //fill temp vector
    for(int i = 0; i < _numBasisVectors; i++)
	{
	    temp[i] = _eVec[i]/(_charge);
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
        temp[i] = _eVec[i]/(_charge);
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
 */
void MaxEnt::_DiagH()
{
    std::vector<arma::vec> temp(_numBasisVectors);
    std::vector<arma::vec> eVecTemp(_numBasisVectors);
    std::vector<arma::vec> eVecData(_numBasisVectors);
    std::vector<arma::cx_vec> eVecTempA(_numBasisVectors);
    std::vector<arma::cx_vec> eVecTempB(_numBasisVectors);

    arma::vec eigval;
    arma::mat eigvec;

    //transform basis vectors to reciprocal space
    for(int i = 0; i < _numBasisVectors; i++)
	{
        _RealFT(_eVec[i], eVecTempA[i], -1);
        //eVecTempA[i] = fft(_eVec[i]);
        eVecTempA[i] = eVecTempA[i]%_chargeTransformConj;
        eVecTempA[i] = eVecTempA[i]*(1-cexp(-i*_Complex_I*(2*M_PI/(double)_depth)));
        eVecTempA[i] += _chargeTransformConj%(_eVec[_numBasisVectors-1] - _eVec[0]);
        

        eVecData[i] = 2*arma::real(eVecTempA[i]);
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

    _h = _h; 

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
 */
void MaxEnt::_LagrangeSearch()
{
    double b = 0.1; //starting value of b to iterate through
    double a, aPlus, aMinus;
    _currentSearchIter = 0; //search counter for while loop

    while (_currentSearchIter < _maximumSearchIter)
	{
        b += 0.1; //step b, note this is arbitrary and a smaller step will provide a more precise search
        bool bReject = false;
        bool aPlusAccept = false;
        bool aMinusAccept = false;
        bool bAccept = false;

        //generate corresponding values of a
        //(Skilling 1986) Eqn. 37
        _delta = -_gamma - b;
        double sSum = accu(pow(_s,2)/pow(_delta,2));
        double cSum = accu(pow(_c,2)/pow(_delta,2));
        double scSum = accu((_c%_s)/pow(_delta,2));

        //(Skilling 1986) Eqn. 37
        double dSquared = _l0Squared*sSum - sSum*cSum + scSum*scSum;


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
            if(aMinus > 0.0 && b > aMinus)
            {
                aMinusAccept = true;
            }
            if(aPlus > 0.0 && b > aPlus)
            {
                aPlusAccept = true;
            }
            if(b > aMinus && aPlus > b)
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
                _NoDistanceLimit(a, b, bReject);
            }
            else if (aMinusAccept || aPlusAccept) //apply distance limiting l=r, 0<a<b
            {
                _DistanceLimit(a, b, aPlus, aMinus, bReject, aPlusAccept, aMinusAccept);
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
 */
void MaxEnt::_NoDistanceLimit(double& a, double& b, bool& bReject)
{
    //(Skilling 1986) Eqn. 33
    a = b;

    //(Skilling 1986) Eqn. 34
    _x = (_c - a*_s)/_delta;

    //(Skilling 1986) Eqn. 39-43
    if(_delta[0] <= 0.0)
	{
	    bReject = false; //accept a and b
	}
    if(_delta[1] >= 0.0)
	{
	    bReject = true; //reject a and b	  
	}
    if(_delta[1] < 0.0 && _delta[0] > 0.0)
	{
        double fZero = 0.0; //check sign of F(0)

        //(Skilling 1986) Eqn. 47
        double deltaProduct = prod(_delta);

        //(Skilling 1986) Eqn. 48
        fZero = accu(deltaProduct*(pow(_s-_x,2)/_delta));

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
        _aVector[_currentSearchIter] = a;
        _bVector[_currentSearchIter] = b;
        _currentSearchIter += 1;
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
 */
void MaxEnt::_DistanceLimit(double& a, double& b, const double& aPlus, const double& aMinus, bool& bReject, bool& aPlusAccept, bool& aMinusAccept)
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
        _x = (_c - a*_s)/_delta;

        if(_delta[0] <= 0.0)
		{
		    bReject = false; //accept a and b
		}
	    if(_delta[1] <= 0.0 && _delta[0] >= 0.0)
		{
            //(Skilling 1986) Eqn. 58
            double phiZero = 0.0; //check sign of phi(0)

            //(Skilling 1986) Eqn. 47
            double deltaProduct = prod(_delta);

		    for(int i = 0; i < _numBasisVectors; i++)
			{
			    for(int j = 0; j < _numBasisVectors; j++)
				{
				    if(i != j)
					{
                        phiZero += pow(_s[i]*_x[j] - _x[i]*_s[j],2)/(_delta[i]*_delta[j]);
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
	    if(_delta[2] <= 0.0 && _delta[1] >= 0.0) //check signs of G(0), H(0) and Phi(0)
		{

            //(Skilling 1986) Eqn. 61
            double gZero = 0.0;
            double hZero = 0.0;
            double phiZero = 0.0;
            double deltaProduct = prod(_delta);

            gZero = - accu(pow(_x,2)/_delta);
            hZero = - accu(pow(_s,2)/_delta);
            
            for(int i = 0; i < _numBasisVectors; i++)
			{
			    for(int j = 0; j < _numBasisVectors; j++)
				{
				    if(i != j)
					{
					    phiZero += pow(_s[i]*_x[j] - _x[i]*_s[j],2)/(_delta[i]*_delta[j]);
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
	    if(_delta[3] >= 0.0)
		{
		    bReject = true;
		}
	}
    if(!bReject)
	{
        _aVector(_currentSearchIter) = a;
        _bVector(_currentSearchIter) = b;
        _currentSearchIter += 1;
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
  

    //find min values of c
    double cMin = cVector.min();
    //int cTargetIndex = (int)arma::index_min(cVector);// 0;//check if required or if better way using cTargetIndex = 0 seems to give better convergenve that the min index


    double cTarget = _chiSquared - (_chiSquared - cMin)*0.8; //find where chi is 4/5 towards cMin and determine index of this
    //int cTargetIndex;
        
    int cTargetIndex = (int)arma::index_min(abs(cVector - cTarget));


    #ifdef DEBUG 
    //find max value of S
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
void MaxEnt::_SetZero()
{
    for(auto& val : _charge)
	{
	    if(val < _zeroLevel)
		{
		    val = _zeroLevel;
		}
	}

    //check limits 
    #ifdef DEBUG
    assert(_charge.min() >= _zeroLevel);
    #endif

    #ifdef DEBUGPLUS
    std::cout << "Minimum Charge: " << _charge.min() << std::endl;
    #endif 
}

/**
 * @brief Class member function that increments the charge by the search vectors found using the lagrange multipliers to find an updated charge to be used in the next step (iteratively)
 * 
 */
void MaxEnt::_CalcNewCharge()
{

    //this constraint determines whether or not limitations should be put on the charge incrementation which smooths the profile to make it more physical
    if(_smoothProfile == true)
    {
        _SmoothIncrement();
    }
    else
    {
        _RegularIncrement();
    }

    //this constraint scales the charge to the SLD profile bounds if they are known
    if(_sldScaling == true)
    {
        _ScaleCharge();
    }
    //this constrains fixes the edge regions if the propagation region and substrate constraints are known 
    if(_useEdgeConstraints == true)
    {
        _Constraints();
    }

    _SetZero();
    _Renormalise(_charge);
}

/**
 * @brief Regular incrementation of the charge as defined by the Cambridge Algorithm by Skilling and Gull
 * 
 */
void MaxEnt::_RegularIncrement()
{
    for(int i = 0; i < _numBasisVectors; i++)
	{
	    _charge += _x[i]*_eVec[i];
	}
}

/**
 * @brief Charge incrementation with smoothness constraints that produce a more physical SLD profile 
 * (Weinberger 2021) Section 6.3.1
 * 
 */
void MaxEnt::_SmoothIncrement()
{
    for(int i = 0; i < _numBasisVectors; i++)
	{
		for(int j = 0; j < _depth; j++)
		{
			if(j!=0 && j!=_depth-1 && _charge[j] < _charge[j-1] && _charge[j] < _charge[j+1] && _x[i]*_eVec[i][j] <= 0)
			{

			}
			else if( j!=0 && j!=_depth-1 && _charge[j] > _charge[j-1] && _charge[j] > _charge[j+1] && _x[i]*_eVec[i][j] >= 0)
			{

			}
			else
			{
				_charge[j] += _x[i]*_eVec[i][j];
			}
		}
	}
}

/**
 * @brief Constraints on starting (air region) SLD value and final (substrate SLD) values
 * (Weinberger 2021) Section 6.3.1
 * 
 */
void MaxEnt::_Constraints()
{
    //temp vectors to store propogationa and substrate SLDs to allow for substraction
    arma::vec propagationSLD(_lengthPropagation);
    arma::vec substrateSLD(_lengthSubstrate);

    propagationSLD.fill(_propagationSLD);
    substrateSLD.fill(_substrateSLD);
    
    _charge.head(_lengthPropagation) = propagationSLD;
    _charge.tail(_lengthSubstrate) = substrateSLD;
}

/**
 * @brief Scale charge to physical SLD parameters so that value constraints can be imposed on the system
 * (Weinberger 2021) Section 6.3.1
 * 
 */
void MaxEnt::_ScaleCharge()
{
    //rescale the magnituyde of the reconstructed SLD so that the max value is the max bound
    arma::vec chargeTemp;
    chargeTemp.set_size(_depth);

    chargeTemp = _charge;


    //get min and max values of the array
    double maxVal = chargeTemp.max();
    double minVal = chargeTemp.min();

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
            chargeTemp *= minValRatio;
        }
    }

    if(maxValRatio < minValRatio || minValRatio == 0.0)
    {
        if(maxValRatio != 0.0)
        {
            chargeTemp *= maxValRatio;
        }
    }

    //check SLD is still bounded
    #ifdef DEBUG
    std::cout << "Max initial value: " << chargeTemp.max() <<  std::endl;
    assert(chargeTemp.max() <= _sldMaxBound + 0.1);

    std::cout << "Min initial value: " << chargeTemp.min() <<  std::endl;
    assert(chargeTemp.min() >= _sldMinBound - 0.1);
    #endif

    _charge  = chargeTemp;
}

/**
 * @brief Scale charge to physical SLD parameters so that value constraints can be imposed on the system
 * this version is a constant member function used in data output
 * (Weinberger 2021) Section 6.3.1
 * 
 */
void MaxEnt::_ScaleCharge(arma::vec& charge)const
{
    //rescale the magnituyde of the reconstructed SLD so that the max value is the max bound
    arma::vec chargeTemp;
    chargeTemp.set_size(_depth);

    chargeTemp = charge;


    //get min and max values of the array
    double maxVal = chargeTemp.max();
    double minVal = chargeTemp.min();

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
    if(minValRatio != 0.0)
    {
        if(minValRatio > maxValRatio)
        {
            chargeTemp *= minValRatio;
        }
    }
    else if(maxValRatio != 0.0)
    {
        if(maxValRatio > minValRatio)
        {
            chargeTemp *= maxValRatio;
        }
    }

    //check SLD is still bounded
    #ifdef DEBUG
    std::cout << "Max initial value: " << chargeTemp.max() <<  std::endl;
    assert(chargeTemp.max() <= _sldMaxBound + 0.1);

    std::cout << "Min initial value: " << chargeTemp.min() <<  std::endl;
    assert(chargeTemp.min() >= _sldMinBound - 0.1);
    #endif

    charge  = chargeTemp;
}

/**
 * @brief Function to calculate the reflectivity (Image) from a charge distribution (in this case the reflectivity spectrum from the SLD profile)
 * (Weinberger 2021) Eqn. 11/67
 * 
 */
void MaxEnt::_Reflectivity()
{
	_RealFT(_charge, _chargeTransform, -1);

    //normalisation factor such that reflectivity spectrum is normalised to one
    double normalisation;

    //Factor for DFT derivative rule
    arma::cx_vec dftDeriv;

    dftDeriv.set_size(_depth);

    for(int i = 0; i < _depth; i++)
    {
        dftDeriv[i] = 1-cexp(-i*_Complex_I*(2*M_PI/(double)_depth));
    }

    //(Weinberger 2021) 68/94
    _chargeTransform = _chargeTransform%dftDeriv + _charge[_depth-1] - _charge[0];

    arma::vec reflectivityTempInt = abs(_chargeTransform);

    for(int i = 0; i < _depth; i++)
    {
        //Value at Q = 1 is undefined so set equal to 1 such that it is at the maximum value
        if(i == 0)
        {
            _chargeImage[0] = _dataScale;
        }
        //define the first measureable value of reflectivity as being 1
        else if(i == 1)
        {
            _chargeImage[1] = _dataScale;
            //Factor for DFT derivative rule
            normalisation = pow(reflectivityTempInt[1], 2.0)/_dataScale;
        }
        //Calculate remaining values of reflectivity according to reflectivity formula
        else
        {
            _chargeImage[i] = pow(reflectivityTempInt[i], 2.0)/normalisation;
        }
    }

    _Renormalise(_chargeImage);
    //_Renormalise(_sldTransform, _norm);
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
    for(unsigned int i = 0; i < eigval.n_elem; i++)
	{
	    eigvecTemp.col(i) = eigvec.col(sortedIndeces(i));
	}

    eigvec = eigvecTemp;
}
  
/**
 * @brief Class member function used to calculate the chisquared gradient of the system
 * (Elliott 1999) Eqn. 7
 * (Weinberger 2021) Eqn. 75/98
 * 
 * @param temp1 Combined parameters that undergo fourier transform 
 */
void MaxEnt::_GradChiSquared(arma::cx_vec& temp1)
{
    //sum temp value as required for the delta function components in the gradient of the chisquared
	arma::cx_double temp1Sum = accu(temp1);

    //half of the gradient of the chi squared, the other half is just the complex conjugate of this part
	arma::cx_vec gradChiSquaredPart;

    //fourier transform temp
	_ComplexFT(temp1, temp1, -1);

	gradChiSquaredPart.set_size(_depth);

    //apply chi squared gradient equation
	for(int i = 0; i < _depth ; i++)
	{
		gradChiSquaredPart[i] = temp1[i] - temp1[(i+1)%_depth];
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
 * @param temp1 Combined parameters that undergo fourier transform 
 * @param temp2 Combined parameters that undergo fourier transform 
 */
void MaxEnt::_GGChiSquared(arma::cx_vec& temp1, arma::cx_vec& temp2)
{

    //sum temp value as required for the delta function components in the gg of the chisquared
	arma::cx_double temp1Sum = accu(temp1);

    //create a temporary store for temp1 as temp1 is transformed in two different ways
	temp2 = temp1; 

    //half of the imaginary part of gg of the chi squared, the other half is just the complex conjugate of this part
	arma::cx_vec ggChiSquaredPart;

	ggChiSquaredPart.set_size(_depth);

    //fourier transform and second order fourier transform 
	_ComplexFT(temp1, temp1, -1); 
	_ComplexFT(temp2, temp2, -2); 

    //apply ggChiSquared equation
	for(int i = 0; i < _depth; i++)
	{
		ggChiSquaredPart[i] = temp2[i] - 2.0*temp1[(2*i+1)%_depth] + temp2[(i+1)%_depth] + 2.0*temp1[_depth - 1] - 4.0*temp1[0] + 2.0*temp1[1];	
	}

	ggChiSquaredPart[0] -= temp1Sum; 
	ggChiSquaredPart[_depth-1] += temp1Sum;


	_ggChiSquared = 2.0*arma::real(ggChiSquaredPart) + accu(4.0*(_chargeImage - _dataFit)%_inverseVar);
    //_ggChiSquared.print();

    //final corrections to ggChiSquared
    arma::vec factorK;
    factorK.set_size(_depth);
    arma::vec temp3;

    temp3 = 4.0*(_chargeImage)%_inverseVar;

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

    _ggChiSquared += factorK;
}



 
  
	  
	  
		 
		  

  
  
  

  



  
  
	   
  
  
  
  

  
  
  

  
  
	

  
  
  

  
	
	  
 
