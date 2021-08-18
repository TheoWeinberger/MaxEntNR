/****************************************************************************************
 * @file MaxEntClass.hpp                                                 
 * @brief Code to reconstruct the SLD profile from the reflectivity spectrum via the MaxEnt method using the 
 * Fourier form of the reflectivity relationship which is defined to be
 * 
 *          R[Q] = \frac{1}{Q^4} * |FFT(SLD)[Q]*[1-exp{-Q*I*(2*PI/totalDepth)}] +
 *                           SLD[totalDepth - 1] - SLD[0]|^2
 * 
 * This method gets rid of the 1/q^4 dependence and scales the charge up to match the system 
 * gradients
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
 * @version 2.0
 * @date 2021-06-08
 * 
 * @copyright Copyright (c) 2021
 *
 ****************************************************************************************
 */

#ifndef MAXENTCLASS_H
#define MAXENTCLASS_H

/**
 * @brief Class object that is used to employ the principle of Maximum Entropy via the Cambridge algorithm
 * to invert an SLD profile from its reflectivity spectrum. This asssumes that the SLD and reflecitivity are related by the transform R[Q] = \frac{1}{Q^4} * |FFT(SLD)[Q]*[1-exp{-Q*I*(2*PI/totalDepth)}] +
 * SLD[totalDepth - 1] - SLD[0]|^2
 * 
 */

class MaxEnt 
{
public:

    /**
     * @brief Construct a new Max Ent object
     * 
     */
    MaxEnt();

    /**
     * @brief Construct a new Max Ent object which reads in data from a string which specifies the file in 
     * which the data is stored. The data file should contain the reflectivity data in a one dimensionol column without
     * scaled axes (the algorirthm uses unscaled data as any scaling is normalised out).
     * 
     * @param dataFile The name of the file containing the reflectivity data
     */
    MaxEnt(const std::string&);

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
    MaxEnt(const std::string&, const std::string&);

    /**
     * @brief Employs Cambridge Algortihm to reconstruct the maximum entropy SLD profile consistent with the 
     * input data - can pick whether this solves for toy or real data
     * 
     */
    void Solve();

    /**
     * @brief Get the scaled SLD 
     * 
     * @return arma::mat @param sldScaled the scaled SLD
     */
    arma::mat GetSLDScaled()const;

    /**
     * @brief Get the scaled charge
     * 
     * @param charge The charge of one of the components of the system
     * @return arma::mat @param chargeScaled the scaled charge
     */
    arma::mat GetChargeScaled(arma::vec&)const;

    /**
     * @brief Get the scaled Reflectivity
     * 
     * @return arma::mat @param reflectivityScaled the scaled Reflectivity
     */
    arma::mat GetReflectivityScaled()const;

private:

    /**
     * @brief Employs Cambridge Algortihm to reconstruct the maximum entropy SLD profile consistent with the 
     * input data - this is designed to be implemented for real data
     * 
     */
    void _SolveMain();

    /**
     * @brief Employs Cambridge Algortihm to reconstruct the maximum entropy SLD profile consistent with the 
     * input data - this is for the proof of concept 'toy' model which is a limited single component system only
     * 
     */
    void _SolveToy();

    /**
     * @brief Class member fucntion to initialise the data for the MaxEnt algorithm
     * 
     */
    void _Init();

    /**
     * @brief Class member fucntion to initialise the data for the MaxEnt algorithm for toy systems
     * these include no normalisation and no physical SLD values are used
     * 
     */
    void _InitToy();

    /**
     * @brief Class member function to apply one step of the MaxEnt algorithm
     * 
     * @param charge The charge of one of the components of the system
     * @param norm The normalisation of the component of the system
     * @param def A parameter used to define the contrained entropy such that the total charge can be kept constant without need for an additional lagrange multiplier
     * @param sld SLD value of substance
     * @param l0Squared Confidence range for quadratic approximation
     */
    void _Step(arma::vec&, double&, double&, double&, double&);

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
     * @brief Private member function to output redChi, minRedChi, TEST and minTest to 
     * a datafile to be subsequently averaged over for parametric testing 
     * 
     */
    void _StoreConvData();

    /**
     * @brief Class member function used to initialise relevant statistical quantities to be used throughout MaxEnt algorithm
     * 
     */
    void _InitStatQuant();

    /**
     * @brief Class member function to calculate the DEF parameter for the system
     * (Elliot 1999) Eqn. 5
     * 
     * @param charge The charge of one of the components of the system
     * @param def A parameter used to define the contrained entropy such that the total charge can be kept constant without need for an additional lagrange multiplier
     */
    void _DEF(arma::vec&, double&);

    /**
     * @brief Class member function to renormalise the charge or the charge image so that the total charge of the system remains constant at the initial total intensity of the diffraction pattern
     * 
     * @param charge The charge of one of the components of the system
     * @param norm The normalisation of the component of the system
     */
    void _InitCharge(arma::vec&, double&);

    /**
     * @brief Class member funciton used to generate the SLD profile from the component parts
     * 
     */
    void _SLDGenerate();

    /**
     * @brief Function that 'spikes' the charge in close to the substrate region, this should help the profile 
     * build up from close to the substrate which is physically what is expected (albeit by no means definite)
     * 
     * @param charge The charge of one of the components of the system
     */

    void _SpikeCharge(arma::vec&);

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
     * @brief Class member function  to renormalise the charge or the charge image so that the total charge of the system remains constant at the initial total intensity of reflectivity spectrum
     * This overloaded function if for use on Complex vectors
     * 
     * @param unnormalisedVector The unormalised input to be renormalised
     * @param norm The normalisation of the vector
     */
    void _Renormalise(arma::cx_vec&, double&);

    /**
     * @brief Get the maximum total charge of the system and its index
     * 
     */
    void _GetMaxCharge();

    /**
     * @brief Class member function used to calculate quantities relevant to the entropy of the system
     * 
     * @param charge The charge of one of the components of the system
     * @param def A parameter used to define the contrained entropy such that the total charge can be kept constant without need for an additional lagrange multiplier
     */
    void _Entropy(arma::vec&, double&);

    /**
     * @brief Class member function used to calculate the conjugate charge for the system
     * 
     */
    void _ConjSLD();

    /**
     * @brief Class member function used to calculate quantities relevant to the chisquared of the system
     * (Elliott 1999) Eqn. 7
     * (Weinberger 2021) Eqn. 75/98
     * 
     * @param sld SLD value of substance
     */
    void _ChiSquared(double&);

    /**
     * @brief Class member function used to calculate the chisquared gradient of the system
     * (Elliott 1999) Eqn. 7
     * (Weinberger 2021) Eqn. 75/98
     * 
     */
    void _GradChiSquared();

    /**
     * @brief Class member function used to calculate the second derivative of the chisquared of the system
     * (Elliott 1999) Eqn. 7
     * (Weinberger 2021) Eqn. 82/99  
     * 
     * @param sld The sld value of the substance
     */
    void _GGChiSquared(double&);

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
     * (Weinberger 2021) Eqn. 11/67
     * 
     */
    void _Reflectivity();

    /**
     * @brief Class member function to generate the search vectors to be used in the MaxEnt method
     * (Skilling 1986) Eqn. 22 - 27
     * 
     * @param charge The charge of one of the components of the system
     */
    void _BasisFunctions(arma::vec&);

    /**
     * @brief Class member function to calculate the 'length' (modulus squared) of the entropy and chi squared gradients. These quantities are required for normalising the gradients when the TEST parameter is calculated to determine convergence
     * (Skilling 1986) Eqn. 28
     * (Elliott 1999) Eqn. 10
     * 
     */
    void _DistCalc();

    /**
     * @brief Class member function used to calculate the TEST parameter which is a measure of convergence of the system. Should be able to obtain TEST < 0.01 fairly easily
     * (Elliott 1999) Eqn. 10
     * 
     */ 
    void _TESTCalc();

    /**
     * @brief Class member function to transform the basis vectors into the metric of the entropy (g) and diagonalise the system
     * 
     */
    void _NormBasisVec();

    /**
     * @brief Class member function to transform the basis vectors into the metric of the entropy (g) and diagonalise the system
     * (Skilling 1986) Eqn. 13
     * (Weinberger 2021) Eqn. 102
     * 
     * @param charge The charge of one of the components of the system
     */
    void _DiagG(arma::vec&);

    /**
     * @brief Class member function to transform the basis vectors into the metric of chi squared (h) and diagonalise the system
     * (Skilling 1986) Eqn. 14
     * (Elliott 1999) Eqn. 15-16
     * (Weinberger 2021) Eqn. 83-87
     * 
     * @param sld the sld value of the component
     */
    void _DiagH(double& );

    /**
     * @brief Class member function to sort eigenvectors and eigevalues in descending order according to the eigenvalues of the system
     * 
     * @param eigval A vector containing the eigenvalues of the system
     * @param eigvec A matrix containing the eigenvectors
     */
    void _Eigensort(arma::vec&, arma::mat&);

    /**
     * @brief Class member function to sort one vector according to the order of another
     * 
     * @param vec1 A vector to be sorted
     * @param vec2 A vector to be sorted according to vec1
     */
    void _Eigensort(arma::vec&, arma::vec&);

    /**
     * @brief Class member function to transform basis of the search vectors
     * 
     * @param eigvec Matrix containing eigenvectors of the metric to which the vectors are being transformed
     * @param eVecTemp A vector to which the transformed basis vectors are to be stored
     */
    void _TransformBasis(const arma::mat&, std::vector<arma::vec>&);

    /**
     * @brief Class member function to generate the subspace of entropy that is to be searched
     * (Skilling 1986) Eqn. 7,9,11,13
     * 
     */
    void _SubspaceS();

    /**
     * @brief Class member functionto generate the subspace of chisquared that is to be searched
     * (Skilling 1986) Eqn. 8,10,12,14
     * 
     */
    void _SubspaceC();

    /**
     * @brief Class member function to iterate over values of the Lagrange multipliers \a A and \a B in order to find all potential values
     * (Skilling 1986) Section 3 - Control
     * 
     * @param l0Squared Confidence range for quadratic approximation
     */
    void _LagrangeSearch(double&);

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
    void _NoDistanceLimit(double&, double&, bool&, int&, bool&, arma::vec&);

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
    void _DistanceLimit(double&, double&, const double&, const double&, bool&, bool&, bool&, int&, bool&, arma::vec&);

    /**
     * @brief Class member function to choose which combination of \a A and \a B Lagrange multipliers produces the best choice for convergence
     * (Skilling 1986) Section 4 Final Selection of Image Increment
     * 
     */
    void _ChooseAB();

    /**
     * @brief Class member function that increments the charge by the search vectors found using the lagrange multipliers to find an updated charge to be used in the next step (iteratively)
     * 
     * @param charge the component of one of the system species
     * @param norm The total amount of that species in this system
     */
    void _CalcNewCharge(arma::vec&, double&);

    /**
     * @brief Regular incrementation of the charge as defined by the Cambridge Algorithm by Skilling and Gull
     * 
     * @param charge the component of one of the system species
     */
    void _RegularIncrement(arma::vec&);

    /**
     * @brief Charge incrementation with smoothness constraints that produce a more physical SLD profile 
     * (Weinberger 2021) Section 6.3.1
     * 
     * @param charge the component of one of the system species
     */
    void _SmoothIncrement(arma::vec&);

    /**
     * @brief Constraints on starting (air region) SLD value and final (substrate SLD) values
     * (Weinberger 2021) Section 6.3.1
     * 
     * @param charge the component of one of the system species
     */
    void _Constraints(arma::vec&);

    /**
     * @brief Scale charge to physical SLD parameters so that value constraints can be imposed on the system
     * (Weinberger 2021) Section 6.3.1
     * 
     */
    void _BoundSLD();

    /**
     * @brief Smoothes the charge profile according to N_i = 1/4 N_i-1 + 1/2 N_i + 1/4 N_i+1 every smoothInterval steps
     * (Weinberger 2021) Section 7.3.3 Constraints 
     * 
     * @param charge the component of one of the system species
     */
    void _SmoothProfile(arma::vec&);

    /**
     * @brief Forces values of charge below a cutoff of the max charge to 0
     * (Weinberger 2021) Section 7.3.3 Constraints 
     * 
     * @param charge the component of one of the system species
     */
    void _ForceZero(arma::vec&);

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

    /**
     * @brief The maximum number of potential Lagrange multipliers to be recorded
     * 
     */
    int _maximumSearchIter = 1000;

    /**
     * @brief The current search iterations for the potential Lagrange multipliers
     * 
     */
    int _currentSearchIter = 0;

    /**
     * @brief The number of basis vectors used for system searching
     * 
     */

    int _numBasisVectors = 8;

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
     * @brief The best guess of the air region within the system given in vector access counters
     * 
     */
    int _lengthAir = 0;

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
     * @brief Value by which reflectivity data is scaled, have found a value of 10.0 tends to work quite well
     * 
     */
    double _dataScale = 10.0;

    /**
     * @brief Linear scaling the chisquared to weight the fitting to change relative weight of the data fitting vs the entropy maximisation in the Langrangian
     * 
     */
    double _chiSquaredScale = 1.0;

    /**
     * @brief The depth of the entire system in element count
     * 
     */
    int _depth;

    /**
     * @brief Portion of charge profile to be spiked (measured from tail)
     * 
     */
    double _spikePortion = 0.25;

    /**
     * @brief Amount by which initial distribution in spiked
     * 
     */
    double _spikeAmount = 1.0;

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

    //boolean variables determine MaxEnt method

    /**
     * @brief Use the smooth incrementation of the profile generator which can help produce more physical profiles
     * 
     */
    bool _smoothIncrement = true;

    /**
     * @brief Use this to recalculate the charges as N_i = 1/4 N_i-1 + 1/2 N_i + 1/4 N_i+1 every smoothInterval steps
     * 
     */
    bool _smoothProfile = false;

    /**
     * @brief The number of steps inbetween smoothing operations
     * 
     */
    int _smoothInterval = 100;

    /**
     * @brief Forces values below a fraction of the max charge to be set to 0
     * 
     */
    bool _forceZero = true;

    /**
     * @brief Interval at which _forceZero is applied
     * 
     */
    int _forceInterval = 100;

    /**
     * @brief The fraction of the maximum charge for a given species that defined the cutoff where below this cutoff the value will be forced to 0
     * 
     */
    double _fracMax = 0.1;


    /**
     * @brief Use the constraints on the edges of the SLD profile to constrain the system, this required sldScaling = True
     * 
     */
    bool _useEdgeConstraints = true;

    /**
     * @brief Spike the initial charge distribution near the substrate to encourage SLD formation in that region
     * 
     */

    bool _spikeCharge = true;

    /**
     * @brief Boolean to determine whether to use volumetric normalisation or not
     * 
     */
    bool _volumetricNormalisation = false; 

    /**
     * @brief Boolean to determine whether the model being studied is a 'toy' model
     * 
     */
    bool _toyModel = false;

    //Algorithm variables

    /**
     * @brief Matrix containing input data to be fitted to
     * 
     */
    arma::vec _dataFit;

    /**
     * @brief The scale of the reflectivity data
     * 
     */
    arma::vec _reflectivityScale;

    /**
     * @brief The first value of Q in the data set
     * 
     */
    double _initQ;

    /**
     * @brief The index of the first data point from which the fitting will work 
     * 
     */
    int _qOffset = 1;

    /**
     * @brief The index of the point in data space where the data is cutoff
     * 
     */
    int _qCutOff; 

    /**
     * @brief The reflectivity value in the region i <= _qOffset, used for normalisation 
     * 
     */
    double _undefR;

    /**
     * @brief The reflectivity value in the region i >- _qCutoff, used for normalisation
     * 
     */
    double _undefR2;

    /**
     * @brief The data separation in Q space
     * 
     */
    double _deltaQ;

    /**
     * @brief Scaling factor to charge total
     * 
     */
    double _chargeScale = 1.0;

    /**
     * @brief The first value of the reflectivity data - used for scaling
     * 
     */
    double _reflectivityNorm;

    /**
     * @brief The normalisation factor used throughout, this is equal to the sum of all the input data
     * 
     */
    double _norm;

    /**
     * @brief The desired final value of chi squared that the system should tend towards
     * 
     */
    double _cAim;

    /**
     * @brief The confidence range for which the quadratic approximation can be considered valid
     * 
     */
    arma::vec _l0Squared;

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
     * @brief The SLD profile generated from the charges
     * 
     */
    arma::vec _sld;

    /**
     * @brief The SLD scale
     * 
     */
    arma::vec _sldScale;

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
     * @brief Minimum value of chisquared achieved by the simulation
     * 
     */

    double _redChiMin;

    /**
     * @brief The gradient of the chi squared
     * 
     */
    arma::vec _gradChiSquared;

    /**
     * @brief The second derivative of the chi squared
     * 
     */
    arma::vec _ggChiSquared;

    /**
     * @brief The length of the chi squared gradient
     * 
     */
    double _chiDist;

    /**
     * @brief The constrained entropy of the system
     * 
     */
    double _entropy;

    /**
     * @brief A parameter used to define the contrained entropy such that the total charge can be kept constant without need for an additional lagrange multiplier
     * 
     */
    arma::vec _def;

    /**
     * @brief The gradient of the entropy of the system
     * 
     */
    arma::vec _gradEntropy;

    /**
     * @brief The length of the entropy gradient
     * 
     */
    double _entDist;

    /**
     * @brief A measure of convergence to a true MaxEnt reconstruction
     * 
     */
    double _test;

    /**
     * @brief Minimum value of test achieved by the simulation
     * 
     */

    double _testMin;

    /**
     * @brief A vector containing all of the basis vectors used in the MaxEnt search
     * 
     */
    std::vector<arma::vec> _eVec = std::vector<arma::vec>(8);

    /**
     * @brief Entropy metric 
     * 
     */
    arma::mat _g = arma::mat(8,8);

    /**
     * @brief Chi squared metric
     * 
     */
    arma::mat _h = arma::mat(8,8);

    /**
     * @brief The subspace of entropy to be searched
     * 
     */
    arma::vec _s = arma::vec(8);

    /**
     * @brief The subspace of chi squared to be searched
     * 
     */
    arma::vec _c = arma::vec(8);

    /**
     * @brief The final search vector directions
     * 
     */
    arma::vec _x = arma::vec(8);

    /**
     * @brief A vector containing the diagonalised chi squared metric (the eigenvalues of h)
     * 
     */
    arma::vec _gamma = arma::vec(8);

    /**
     * @brief The final search vector displacement
     * 
     */
    arma::vec _delta = arma::vec(8);

    /**
     * @brief A vector containing all the \a A Lagrange multiplers
     * 
     */
    arma::vec _aVector = arma::vec(1000);

    /**
     * @brief A vector containing all the \a B lagrange multiplers
     * 
     */
    arma::vec _bVector = arma::vec(1000);

    //Parameters for the normalisation behaviour
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
     * @brief Rescale the SLD profile from the renormalisation process
     * 
     */
    double _reScaling;

    //variables determining how errors are treated
    /**
     * @brief Boolean stating whether or not an error file exists
     * 
     */
    bool _error = false;

    /**
     * @brief String containing name of error file
     * 
     */
    std::string _errorFile;

    /**
    * @brief Avogadros Number
    * 
    */
    double _nA = 6.02214076e+23;  

    /**
     * @brief Boolean stating whether sld should be bound
     * 
     */
    bool _boundSLD;

    /**
     * @brief Minimum bound of profile if known
     * 
     */
    double _sldMinBound;

    /**
     * @brief Maximum bound of profile if known
     * 
     */
    double _sldMaxBound;

    /**
     * @brief Temporary variable used in DFT
     * 
     */
    arma::cx_vec _temp1;

    /**
     * @brief Temporary variable used in DFT
     * 
     */
    arma::cx_vec _temp2;

    /**
     * @brief Complex vector storing DFT values of basis vectors
     * 
     */
    std::vector<arma::cx_vec> _eVecTempA = std::vector<arma::cx_vec>(8);

    /**
     * @brief Complex form of eVec
     * 
     */
    std::vector<arma::cx_vec> _eVecComplex = std::vector<arma::cx_vec>(8);

    /**
     * @brief Complex form of SLD profile
     * 
     */
    arma::cx_vec _sldComplex;

    /**
     * @brief fftw plan for the temp1 variabe
     * 
     */
    fftw_plan _temp1Plan;

    /**
     * @brief fftw plan for the temp2 variabe
     * 
     */
    fftw_plan _temp2Plan;

    /**
     * @brief fftw plan for the sld profile
     * 
     */
    fftw_plan _sldPlan;

    /**
     * @brief Vector containing fftw plans for _eVecComplex
     * 
     */
    std::vector<fftw_plan> _eVecPlan = std::vector<fftw_plan>(8);

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
