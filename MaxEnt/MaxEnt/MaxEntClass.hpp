/**
 * @file MaxEntClass.hpp
 * @author Theo Weinberger
 * @brief Header file for Maximum Entropy class as used to reconstruct SLD profiles from their reflectivity spectra
 * @version 3.0
 * @date 2021-05-01
 * 
 * @copyright Copyright (c) 2021
 * 
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
     */
    MaxEnt(const std::string&);

    /**
     * @brief Construct a new Max Ent object which reads in data from a string which specifies the file in 
     * which the data is stored. The data file should contain the reflectivity data in a one dimensionol column without
     * scaled axes (the algorirthm uses unscaled data as any scaling is normalised out). The second string refers to the 
     * settings file, which is a .cfg file containing the simulation specifics such as run number that are used in
     * the Cambridge algortihm
     * 
     */
    MaxEnt(const std::string&, const std::string&);

    /**
     * @brief Employs Cambridge Algortihm to reconstruct the maximum entropy SLD profile consistent with the 
     * input data
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
     * @brief Get the scaled Reflectivity
     * 
     * @return arma::mat the scaled Reflectivity
     */
    arma::mat GetReflectivityScaled()const;

private:

    /**
     * @brief Class member fucntion to initialise the data for the MaxEnt algorithm
     * 
     */
    void _Init();

    /**
     * @brief Class member function to apply one step of the MaxEnt algorithm
     * 
     */
    void _Step();

    /**
     * @brief Class member function to print out data from MaxEnt algortihm
     * 
     */
    void _Print();

    /**
     * @brief Private member function to output redChi, minRedChi, TEST and minTest to 
     * a datafile to be subsequently averaged over for parametric testing 
     * 
     */
    void _StoreConvData();

    /**
     * @brief Class member function to save data from MaxEnt algorithm
     * 
     */
    void _Store();

    /**
     * @brief Class member function used to initialise relevant statistical quantities to be used throughout MaxEnt algorithm
     * 
     */
    void _InitStatQuant();

    /**
     * @brief Class member function to calculate the DEF parameter for the system
     * (Elliot 1999) Eqn. 5
     * 
     */
    void _DEF();

    /**
     * @brief Class member function to renormalise the charge or the charge image so that the total charge of the system remains constant at the initial total intensity of the diffraction pattern
     * 
     */
    void _InitCharge();

    /**
     * @brief Function that 'spikes' the charge in close to the substrate region, this should help the profile 
     * build up from close to the substrate which is physically what is expected (albeit by no means definite)
     * 
     */

    void _SpikeCharge();

    /**
     * @brief Class member function that enforces a zero region within the charge distribution corresponding to the air region
     * required for the correct resolution of the image
     * 
     */
    void _AirRegion();

    /**
     * @brief Class member function that sets values of array below a threshold to the threshold value
     * 
     */
    void _SetZero();

    /**
     * @brief Class member function  to renormalise the charge or the charge image so that the total charge of the system remains constant at the initial total intensity of reflectivity spectrum
     * 
     * @param unnormalisedVector The unormalised input to be renormalised
     */
    void _Renormalise(arma::vec& unnormalisedVector);

    /**
     * @brief Class member function used to calculate quantities relevant to the entropy of the system
     * 
     */
    void _Entropy();

    /**
     * @brief Class member function used to calculate the conjugate charge for the system
     * 
     */
    void _ConjCharge();

    /**
     * @brief Class member function used to calculate quantities relevant to the chisquared of the system
     * 
     */
    void _ChiSquared();

    /**
     * @brief Class member function used to calculate the chisquared gradient of the system
     * (Elliott 1999) Eqn. 7
     * (Weinberger 2021) Eqn. 75/98
     * 
     * @param temp1 Combined parameters that undergo fourier transform 
     */
    void _GradChiSquared(arma::cx_vec&);

    /**
     * @brief Class member function used to calculate the second derivative of the chisquared of the system
     * (Elliott 1999) Eqn. 7
     * (Weinberger 2021) Eqn. 82/99  
     * 
     * @param temp1 Combined parameters that undergo fourier transform 
     * @param temp2 Combined parameters that undergo fourier transform 
     */
    void _GGChiSquared(arma::cx_vec&, arma::cx_vec&);

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
     */
    void _BasisFunctions();

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
    void _DiagG();

    /**
     * @brief Class member function to transform the basis vectors into the metric of chi squared (h) and diagonalise the system
     * (Skilling 1986) Eqn. 14
     * (Elliott 1999) Eqn. 15-16
     * (Weinberger 2021) Eqn. 83-87
     * 
     */
    void _DiagH();

    /**
     * @brief Class member function to sort eigenvectors and eigevalues in descending order according to the eigenvalues of the system
     * 
     * @param eigval A vector containing the eigenvalues of the system
     * @param eigvec A matrix containing the eigenvectors
     */
    void _Eigensort(arma::vec&, arma::mat&);

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
     */
    void _LagrangeSearch();

    /**
     * @brief Class member function to check if a potential combination of Lagrange multipliers is valid when searching within the distance limited region
     * 
     * @param a A Lagrange multiplier
     * @param b A Lagrange multiplier
     * @param bReject Boolean variable determining whether a given value of b is rejected or accepted
     */
    void _NoDistanceLimit(double&, double&, bool&);

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
    void _DistanceLimit(double& a, double& b, const double& aPlus, const double& aMinus, bool& bReject, bool& aPlusAccept, bool& aMinusAccept);

    /**
     * @brief Class member function to choose which combination of \a A and \a B Lagrange multipliers produces the best choice for convergence
     * (Skilling 1986) Section 4 Final Selection of Image Increment
     * 
     */
    void _ChooseAB();

    /**
     * @brief Class member function that increments the charge by the search vectors found using the lagrange multipliers to find an updated charge to be used in the next step (iteratively)
     * 
     */
    void _CalcNewCharge();

    /**
     * @brief Regular incrementation of the charge as defined by the Cambridge Algorithm by Skilling and Gull
     * 
     */
    void _RegularIncrement();

    /**
     * @brief Charge incrementation with smoothness constraints that produce a more physical SLD profile 
     * (Weinberger 2021) Section 6.3.1
     * 
     */
    void _SmoothIncrement();

    /**
     * @brief Constraints on starting (air region) SLD value and final (substrate SLD) values
     * (Weinberger 2021) Section 6.3.1
     * 
     */
    void _Constraints();

    /**
     * @brief Scale charge to physical SLD parameters so that value constraints can be imposed on the system
     * (Weinberger 2021) Section 6.3.1
     * 
     */
    void _ScaleCharge();

    /**
     * @brief Scale charge to physical SLD parameters so that value constraints can be imposed on the system
     * this version is a constant member function used in data output
     * (Weinberger 2021) Section 6.3.1
     * 
     */
    void _ScaleCharge(arma::vec&)const;


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
     * @brief Maximum bound for the SLD profile if known (only used for scaling)
     * 
     */
    double _sldMaxBound = 5;

    /**
     * @brief Minimum bound of the SLD profile if known (only used for scaling)
     * 
     */
    double _sldMinBound = 0;

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

    //boolean variables determine MaxEnt method

    /**
     * @brief Use the smooth incrementation of the profile generator which can help produce more physical profiles
     * 
     */
    bool _smoothProfile = true;


    /**
     * @brief Scale the SLD profile to the known max and min values
     * 
     */
    bool _sldScaling = true; 

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

    //Algorithm variables

    /**
     * @brief Matrix containing input data to be fitted to
     * 
     */
    arma::vec _dataFit;

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
    double _l0Squared;

    /**
     * @brief The inverse variance for the system, poissonian fluctuations are assumed with a minimum value determined by \a _minvar
     * 
     */
    arma::vec _inverseVar;

    /**
     * @brief The constrained 'charge' of the system, This is effectively the reconstructed SLD profile
     * 
     */
    arma::vec _charge;

    /**
     * @brief The data space transform of the system charge, this is calculated by the relationship FFT(SLD)[Q]*[1-exp{-Q*I*(2*PI/totalDepth)}] +
     * SLD[totalDepth - 1] - SLD[0]
     * 
     */
    arma::cx_vec _chargeTransform;

    /**
     * @brief The conjugate of the data space transform of the system charge
     * 
     */
    arma::cx_vec _chargeTransformConj;

    /**
     * @brief The reflectivity spectrum produced by the reconstructed SLD profile
     * 
     */
    arma::vec _chargeImage;

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
    double _def;

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
