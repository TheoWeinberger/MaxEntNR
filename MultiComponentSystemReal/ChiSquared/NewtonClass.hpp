/****************************************************************************************
 * 
 * @file NewtonClass.hpp
 * @author Theo Weinberger 
 * 
 ****************************************************************************************
 *
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

#ifndef NEWTONCLASS_HPP
#define NEWTONCLASS_HPP

/**
 * @brief Class containing functions and parameters used by Newton algorithm
 * 
 */
class Newton{
public:

    /**
     * @brief Construct a Newton object which reads in data from a string which specifies the file in 
     * which the data is stored. The data file should contain the reflectivity data in a one dimensionol column without
     * scaled axes (the algorirthm uses unscaled data as any scaling is normalised out).
     * 
     */
    Newton(const std::string&);

    /**
     * @brief Construct a new Newton object which reads in data from a string which specifies the file in 
     * which the data is stored. The data file should contain the reflectivity data in a one dimensionol column without
     * scaled axes (the algorirthm uses unscaled data as any scaling is normalised out). The second string refers to the 
     * settings file, which is a .cfg file containing the simulation specifics such as run number that are used in
     * the Newton algortihm
     * 
     */
    Newton(const std::string&, const std::string&);

    /**
     * @brief Employs Newton algorirthm to further fit the maximum entropy SLD profile consistent with the 
     * input data that has been produced by the MaxEnt code
     * 
     */
    void Solve();

    /**
     * @brief Get the scaled SLD 
     * 
     * @return arma::mat the scaled SLD
     */
    arma::mat GetSLDScaled()const;

    /**
     * @brief Get the scaled charge
     * 
     * @param charge The charge of one of the components of the system
     * @return arma::mat the scaled charge
     */
    arma::mat GetChargeScaled(arma::vec&)const;

    /**
     * @brief Get the scaled Reflectivity
     * 
     * @return arma::mat the scaled Reflectivity
     */
    arma::mat GetReflectivityScaled()const;

private:

    /**
     * @brief Class member function used to initialise relevant statistical quantities to be used throughout Newton algorithm
     * 
     */
    void _InitStatQuant();

    /**
     * @brief Convert real world value to and index
     * 
     * @param scale The real world scale for this
     * @param value The real world value of the scale
     * @param index the index 
     */
    void _RealToIndex(arma::vec&, double&, int&);

    /**
     * @brief Convert real world value to and index
     * 
     * @param scale The real world scale for this
     * @param value The real world value of the scale
     * @param index the index 
     */
    void _RealToIndexLength(arma::vec&, double&, int&);

    /**
     * @brief Class member function to apply one step of the MaxEnt algorithm
     * 
     * @param charge The charge of one of the components of the system
     * @param norm The normalisation of the component of the system
     * @param sld SLD value of substance
     */
    void _Step(arma::vec&, double&, double&);

    /**
     * @brief Class member function to print out data from MaxEnt algortihm
     * 
     */
    void _Print();

    /**
     * @brief Class member function to save data from MaxEnt algorithm
     * 
     */
    void _Store();

    /**
     * @brief Class member function that sets values of array below a threshold to the threshold value
     * 
     * @param charge The charge of one of the components of the system
     */
    void _SetZero(arma::vec&);

    /**
     * @brief Class member function  to renormalise the charge or the charge image so that the total charge of the system remains constant at the initial total intensity of reflectivity spectrum
     * 
     * @param unnormalisedVector The unormalised input to be renormalised
     * @param norm The normalisation of the vector
     */
    void _Renormalise(arma::vec&, double&);

    /**
     * @brief Class member function used to calculate the conjugate charge for the system
     * 
     */
    void _ConjSLD();

    /**
     * @brief Class member function used to calculate quantities relevant to the chisquared of the system
     * 
     */
    void _ChiSquared();

    /**
     * @brief Class member function used to calculate quantities relevant to the chisquared of the system
     * this overloadewd funciton allows for the potential updated chisquared to be calculated
     * 
     * @param charge The charge of one of the components of the system
     * @param sld the SLD value of one of the system components
     */
    double _ChiSquared(arma::vec&, double&);

    /**
     * @brief Class member function used to calculate the chisquared gradient of the system
     * 
     * @param sld SLD value of substance
     */
    void _GradChiSquared(double&);

    /**
     * @brief Class member function used to calculate the second derivative of the chisquared of the system
     * 
     * @param sld SLD value of substance
     */
    void _Hessian(double&);

    /**
     * @brief ComplexFT method that uses the FFTW3 library to perform a fourier transform on a complex input matrix and output and complex matrix containing the fourier transform. This function allows for in place transforms. Normalisation is defined so that there is no normalisation on forwards transforms and a 1/N factor on backwards transforms
     * 
     * @param in Input matrix containing complex values
     * @param out Output matrix containing complex values
     * @param direction Direction of the transform which corresponds to the sign in the exponent, can take values -1, +1, -2, +2
     */
    void _ComplexFT(arma::cx_vec&, arma::cx_vec&, const int&);

    /**
     * @brief RealFT method that uses the FFTW3 library to perform a fourier transform on a real input matrix and output and complex matrix containing the fourier transform. This function does not allow for in place transforms. Normalisation is defined so that there is no normalisation on forwards transforms and a 1/N factor on backwards transforms
     * 
     * @param in Input matrix containing real values
     * @param out Output matrix containing complex values
     * @param direction Direction of the transform which corresponds to the sign in the exponent, can take values -1, +1, -2, +2
     */
    void _RealFT(const arma::vec&, arma::cx_vec&, const int&);

    /**
     * @brief Function to calculate the reflectivity (Image) from a charge distribution (in this case the reflectivity spectrum from the SLD profile)
     * 
     */
    void _Reflectivity();

    /**
     * @brief Function to calculate the reflectivity (Image) from a charge distribution (in this case the reflectivity spectrum from the SLD profile)
     * This overloaded function is used in chisquared update calculation
     * 
     * @param sldImageTemp Temporary sld image
     * @param sldTemp Temporary sld profile
     */
    void _Reflectivity(arma::vec&, arma::vec&);

    /**
     * @brief Class member funciton used to generate the SLD profile from the component parts
     * 
     */
    void _SLDGenerate();

    /**
     * @brief Class member function used to generate the SLD profile from the component parts
     * 
     * @param sldTemp The temporary SLD profile used for backtracing
     * @param charge The charge of one of the components of the system
     * @param sld the SLD value of the system component
     */
    void _SLDGenerate(arma::vec&, arma::vec&, double&);

    /**
     * @brief Class member function that finds the increment of the charge via the Newton method
     * and implement backtracing
     * 
     * @param charge The charge of one of the components of the system
     * @param sld the SLD value of the system component
     */
    void _CalcNewCharge(arma::vec&, double&);

    /**
     * @brief Class member function that increments the charge by the search vectors found using the lagrange multipliers to find an updated charge to be used in the next step (iteratively)
     * 
     * @param charge The charge of one of the components of the system
     * @param norm The total amount of the charge in the system for normalisation
     * 
     */
    void _StepNewCharge(arma::vec&, double&);

    /**
     * @brief Regular incrementation of the charge as defined by the Cambridge Algorithm by Skilling and Gull
     * 
     * @param charge The charge of one of the components of the system
     */
    void _RegularIncrement(arma::vec&);

    /**
     * @brief Charge incrementation with smoothness constraints that produce a more physical SLD profile 
     * 
     * @param charge The charge of one of the components of the system
     */
    void _SmoothIncrement(arma::vec&);

    /**
     * @brief Constraints on starting (air region) SLD value and final (substrate SLD) values
     * 
     * @param charge The charge of one of the components of the system
     */
    void _Constraints(arma::vec&);

    /**
     * @brief Smoothes the charge profile according to N_i = 1/4 N_i-1 + 1/2 N_i + 1/4 N_i+1 every smoothInterval steps
     * 
     * @param charge The charge of one of the components of the system
     */
    void _SmoothProfile(arma::vec&);

    /**
     * @brief Forces values of charge below a cutoff of the max charge to 0
     * 
     * @param charge The charge of one of the components of the system
     */
    void _ForceZero(arma::vec&);

    /**
     * @brief Implement Newton's minimisationb algorithm
     * 
     */
    void _Newton();

    /**
     * @brief Get the maximum total charge of the system and its index
     * 
     */
    void _GetMaxCharge();

    /**
     * @brief Scale charge to physical SLD parameters so that value constraints can be imposed on the system
     * 
     */
    void _BoundSLD();

    /**
     * @brief Make a real vector into a complex vector for FT purposes
     * 
     * @param in The input real vector
     * @param out The output complex vector
     */
    void _MakeComplex(arma::vec&, arma::cx_vec&);

    /**
     * @brief Make all the DFT plans to be used in the simulations
     * 
     */
    void _MakeDFTPlans();

    /**
     * @brief Delete all the DFT plans for system cleanup
     * 
     */
    void _DeleteDFTPlans();

    //counter variables at default values
    /**
     * @brief The current interation number
     * 
     */
    int  _iterationCount = 0;
    
    /**
     * @brief The total number of iterations used in algorithm
     * 
     */
    int _totalIterations = 1000;

    //Define charge parameters 
    /**
     * @brief Minimum value that the matrix is allowed to contain otherwise it is set to \a zeroLevel
     * 
     */
    double _zeroLevel = 1e-6;

    /**
     * @brief Minimum value of poissonian fluctuations the input data is allowed to have
     * 
     */
    double _minVar = 0.25; 

    /**
     * @brief The SLD of the propogation region of the neutrons (typically air with a value of 0.0)
     * 
     */
    double _propagationSLD = 0.0;

    /**
     * @brief The length (in vector indeces) of the region of air propagation that should be fixed
     * 
     */
    int _lengthPropagation = 5;

    /**
     * @brief Real world length of propogation region
     * 
     */
    double _lengthPropagationReal;

    /**
     * @brief The SLD of the substrate region 
     * 
     */
    double _substrateSLD = 2.5;

    /**
     * @brief The length (in vector indeces) of the region of the substrate that should be fixed
     * 
     */
    int _lengthSubstrate = 5;

    /**
     * @brief The real world substrate length
     * 
     */
    double _lengthSubstrateReal;

    /**
     * @brief Value by which reflectivity data is scaled, have found a value of 10.0 tends to work quite well
     * 
     */
    double _dataScale = 10.0;

    /**
     * @brief Total sum over the whole intensity 
     * 
     */
    double _norm;

    /**
     * @brief The depth of the entire system in element count
     * 
     */
    int _depth;

    /**
     * @brief Number of different substances in this system
     * 
     */
    int _numSubstances = 1; 

    /**
     * @brief Value of the SLD of this substance
     * 
     */
    arma::vec _sldVal;

    /**
     * @brief Total amount of this substance
     * 
     */
    arma::vec _total; 

    //boolean variables determine Newton method

    /**
     * @brief Use the smooth incrementation of the profile generator which can help produce more physical profiles
     * 
     */
    bool _smoothIncrement = true;

    /**
     * @brief Use the constraints on the edges of the SLD profile to constrain the system, this required sldScaling = True
     * 
     */
    bool _useEdgeConstraints = true;

    /**
     * @brief Boolean determinig whether Armijo backtracing should be used
     * 
     */
    bool _useDamping = true;

    /**
     * @brief Matrix containing input data to be fitted to
     * 
     */
    arma::vec _dataFit;

/**
     * @brief The inverse variance for the system, poissonian fluctuations are assumed with a minimum value determined by \a _minvar
     * 
     */
    arma::vec _inverseVar;

    /**
     * @brief The constrained 'charge' of the system, This is effectively the reconstructed SLD profile
     * 
     */
    std::vector<arma::vec> _charge;

    /**
     * @brief Temporary vector containing the potential updated next step for the charge
     * 
     */
    arma::vec _tempCharge;

    /**
     * @brief Shift in charge vector from matrix equation
     * 
     */
    arma::vec _delta;

    /**
     * @brief The SLD profile generated from the charges
     * 
     */
    arma::vec _sld;

    /**
     * @brief The data space transform of the system charge, this is calculated by the relationship FFT(SLD)[Q]*[1-exp{-Q*I*(2*PI/totalDepth)}] +
     * SLD[totalDepth - 1] - SLD[0]
     * 
     */
    arma::cx_vec _sldTransform;

    /**
     * @brief The conjugate of the data space transform of the system charge
     * 
     */
    arma::cx_vec _sldTransformConj;

    /**
     * @brief The reflectivity spectrum produced by the reconstructed SLD profile
     * 
     */
    arma::vec _sldImage;

    /**
     * @brief The total chi squared for the system
     * 
     */
    double _chiSquared;

    /**
     * @brief The reduced chi squared for the system
     * 
     */
    double _redChi;

    /**
     * @brief The gradient of the chi squared
     * 
     */
    arma::vec _gradChiSquared;

    /**
     * @brief The second Hessian of the chi squared
     * 
     */
    arma::mat _hessian;

    /**
     * @brief Step scaling to Newton algorithm
     * 
     */
    double _gammaInit = 1;

    /**
     * @brief Scaling factor for Armijo backtracing
     * 
     */
    double _gammaFactor = 0.5;

    /**
     * @brief Multiplicative factor for Armijo backtracing
     * 
     */
    double _alphaInit = 0.5;

     /**
     * @brief Scaling factor for Armijo backtracing
     * 
     */
    double _alphaFactor = 0.5;

        /**
     * @brief index of the maximum sum of charges
     * 
     */
    int _indexMax;

    /**
     * @brief Maximum sum of charges
     * 
     */
    double _chargeMax;

    /**
     * @brief Current species being studied
     * 
     */
    int _currentCharge = 0;

    /**
     * @brief Scaling factor for most elements apart from j = jmax
     * 
     */
    double _regularScaling;

    /**
     * @brief Scaling factor for the element j = jmax
     * 
     */
    double _maxIndexScaling;

    /**
     * @brief Check whether matrix system is solvable and break if not 
     * 
     */
    bool _solvable;

    /**
     * @brief Interval at which smoothing operator applied
     * 
     */
    int _smoothInterval = 1000; 

    /**
     * @brief Boolean to determine whether profile smoothing should be applied
     * 
     */
    bool _smoothProfile = false;

    /**
     * @brief Boolean to determine whether force zero should be applied
     * 
     */
    bool _forceZero = false;

    /**
     * @brief Interval at which force zero is applied
     * 
     */
    int _forceInterval = 1000;

    /**
     * @brief The min fraction of the max charge of a species below which all charge is zeroed
     * 
     */
    double _fracMax = 0.1;

    /**
     * @brief Whether to use volumetric normalisation for SLD calculation
     * 
     */
    bool _volumetricNormalisation = true;

    /**
     * @brief Whether to use SLD bounds for SLD scaling
     * 
     */
    bool _boundSLD = false;

    /**
     * @brief The maximum bound of the SLD profile
     * 
     */
    double _sldMaxBound;

    /**
     * @brief The minimum bound of the SLD profile
     * 
     */
    double _sldMinBound;

    /**
     * @brief The index of the first data point from which the fitting will work 
     * 
     */
    int _qOffset;

    /**
     * @brief Offset in real world units
     * 
     */
    double _qOffsetReal;

    /**
     * @brief The index of the point in data space where the data is cutoff
     * 
     */
    int _qCutOff;

    /**
     * @brief Cutoff in real world units
     * 
     */
    double _qCutOffReal;

    /**
     * @brief The scale of the reflectivity data
     * 
     */
    arma::vec _reflectivityScale;

    /**
     * @brief The SLD scale
     * 
     */
    arma::vec _sldScale;

    /**
     * @brief The data separation in Q space
     * 
     */
    double _deltaQ;

    /**
     * @brief The first value of the reflectivity data - used for scaling
     * 
     */
    double _reflectivityNorm;

    /**
     * @brief Whether data contains errors
     * 
     */
    bool _error = false;

    /**
     * @brief Plan for in place transforms
     * 
     */
    fftw_plan _inPlacePlan;

    /**
     * @brief Plan for out of place transforms
     * 
     */
    fftw_plan _outOfPlacePlan;

    /**
     * @brief Boolean stating whether real world scaling is being used
     * 
     */
    bool _realWorldScaling = false;

    
};

/**
 * @brief Template to determine the sign of a number
 * 
 * @tparam T 
 * @param x
 * @return int 
 */
template <typename T> int sgn(T x) {
    return (T(0) < x) - (x < T(0));
}

#endif