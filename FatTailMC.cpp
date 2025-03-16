#include <iostream>       // Standard I/O stream library
#include <vector>         // For using std::vector container
#include <random>         // For random number generation
#include <cmath>          // For math functions such as std::sqrt and std::log
#include <fstream>        // For file input/output operations
#include <sstream>        // For string stream operations
#include <chrono>         // For time-related functions (used in seeding RNG)
#include <algorithm>      // For algorithms like std::max

// --------------------------------------------------------
// HELPER function calculates the cumulative distribution function for the standard normal distribution.
double normCDF(double x) {
    // erfc computes the complementary error function and we adjust for standard normal properties
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// --------------------------------------------------------
// 1. ParameterManager:
//    Stores simulation parameters & scenario settings, including optional stochastic volatility toggles.
class ParameterManager {
public:
    // Time & simulation parameters
    double T;      // Time to maturity in years
    int nSteps;    // Number of time steps per simulation path
    int nSim;      // Total number of Monte Carlo simulation paths
    double r;      // Risk-free interest rate
    double S0;     // Initial price of the underlying asset

    // Fat-tail jump parameters (for modeling jump risk using a Pareto distribution)
    double alpha;         // Tail index for the Pareto distribution
    double jumpIntensity; // Jump intensity (expected number of jumps per year)
    double jumpScale;     // Scale parameter for the Pareto distribution

    // Volatility model parameters
    enum VolModel { CONST_VOL, HESTON_LIKE };  // Enumeration to choose between constant volatility and Heston-like model
    VolModel volModelType;  // Selected volatility model
    double sigma;           // Baseline constant volatility level

    // Parameters for a minimal Heston-like stochastic volatility model
    double kappa; // Speed of mean reversion of variance
    double theta; // Long-run average variance level
    double xi;    // (vol-of-vol)

    // Stress test toggles and shock parameters
    bool  stressMode;       // Flag to enable/disable stress mode
    double alphaShock;      // Shocked tail index value for stress mode
    double volShockFactor;  // Factor by which volatility is shocked

    // Constructor with default parameter values
    ParameterManager() {
        // Set default baseline simulation parameters
        T       = 1.0;        // Maturity of 1 year
        nSteps  = 252;        // Typical number of trading days in a year
        nSim    = 10000;      // Number of Monte Carlo simulation paths
        r       = 0.05;       // Risk-free rate of 5%
        S0      = 100.0;      // Initial underlying price of 100

        // Set default jump parameters
        alpha         = 3.0;  // Default tail index
        jumpIntensity = 1.0;  // One jump per year on average
        jumpScale     = 1.1;  // Default jump scale

        // Set default volatility model to constant volatility
        volModelType  = CONST_VOL; //HESTON_LIKE; // To Experiment w Heston-like stochastic volatility
        sigma         = 0.2;  // 20% constant volatility

        // Set default Heston-like parameters
        kappa         = 1.5;  // Speed of reversion
        theta         = 0.04; // Long-run variance level
        xi            = 0.3;  // Volatility of volatility

        // Set default stress mode parameters (initially off)
        stressMode    = false;      // Stress mode is disabled by default
        alphaShock    = 2.0;        // Shock value for tail index when stress is on
        volShockFactor= 2.0;        // Volatility shock factor when stress is on
    }

    // Load scenario-specific parameters based on scenarioID
    void loadScenario(int scenarioID) {
        // Scenario 0: Baseline scenario
        if (scenarioID == 0) {
            alpha         = 3.0;
            sigma         = 0.2;
            jumpIntensity = 1.0;
            volModelType  = CONST_VOL;
            stressMode    = false;
        }
        // Scenario 1: Stress on tail index only
        else if (scenarioID == 1) {
            alpha         = alphaShock; // Use shocked tail index value
            sigma         = 0.2;
            jumpIntensity = 1.5;        // Increase jump intensity
            volModelType  = CONST_VOL;
            stressMode    = true;
        }
        // Scenario 2: Stress on volatility only
        else if (scenarioID == 2) {
            alpha         = 3.0;
            sigma         = 0.2 * volShockFactor; // Increase volatility by shock factor
            jumpIntensity = 1.0;
            volModelType  = CONST_VOL; //HESTON_LIKE;
            stressMode    = true;
        }
        // Scenario 3: Stress on both tail index and volatility
        else if (scenarioID == 3) {
            alpha         = alphaShock;             // Shock tail index value
            sigma         = 0.2 * volShockFactor;     // Shock volatility
            jumpIntensity = 1.5;                    // Increase jump intensity
            volModelType  = CONST_VOL;
            stressMode    = true;
        }
        // Additional scenarios could be added here, e.g., for Heston-like stochastic volatility
    }

    // Print all simulation parameters to the console
    void printParameters() const {
        std::cout << "PARAMETERS:\n"
                  << "  T=" << T << ", nSteps=" << nSteps << ", nSim=" << nSim << "\n"
                  << "  r=" << r << ", S0=" << S0 << "\n"
                  << "  alpha=" << alpha << ", jumpIntensity=" << jumpIntensity
                  << ", jumpScale=" << jumpScale << "\n"
                  << "  VolModel=" << (volModelType == HESTON_LIKE ? "HESTON_LIKE" : "CONST_VOL")
                  << ", sigma=" << sigma << "\n"
                  << "  kappa=" << kappa << ", theta=" << theta << ", xi=" << xi << "\n"
                  << "  StressMode=" << (stressMode ? "ON" : "OFF")
                  << ", alphaShock=" << alphaShock
                  << ", volShockFactor=" << volShockFactor
                  << "\n\n";
    }
};

// --------------------------------------------------------
// 2. RandomNumberGenerator: 
//    Provides uniformly distributed, Gaussian, and Pareto random number generation.
class RandomNumberGenerator {
private:
    std::mt19937_64 rng;                                 // Mersenne Twister 64-bit engine for randomness
    std::uniform_real_distribution<double> uniformDist;  // Uniform distribution for [0,1]
    std::normal_distribution<double> normalDist;         // Normal distribution with mean 0 and std 1
public:
    // Constructor: seed the random number generator using current time by default
    RandomNumberGenerator(unsigned seed = std::chrono::system_clock::now().time_since_epoch().count())
        : rng(seed), uniformDist(0.0, 1.0), normalDist(0.0, 1.0) {}

    // Returns a uniformly distributed number in [0,1]
    double uniform01() { return uniformDist(rng); }
    // Returns a normally distributed number (Gaussian)
    double gaussian()  { return normalDist(rng); }
    // Returns a Pareto-distributed random number with tail index alpha and scale parameter scale
    double pareto(double alpha, double scale) {
        double U = uniform01(); // Get a uniform random variable
        // Transform to Pareto distribution using inverse transform sampling
        return scale * std::pow(1.0 - U, -1.0 / alpha);
    }
};

// --------------------------------------------------------
// 3. PathSimulator:
//    Generates simulation paths for the underlying asset price.
//    Handles both constant volatility and minimal Heston-like (stochastic volatility) dynamics.
class PathSimulator {
private:
    ParameterManager* params;     // Pointer to simulation parameters
    RandomNumberGenerator rng;    // Instance of the random number generator
    const double maxJumpFactor = 5.0; // Maximum allowed jump multiplier (to cap extreme jumps)
public:
    // Constructor taking a pointer to ParameterManager to access simulation settings
    PathSimulator(ParameterManager* p) : params(p) {}

    // Generate a single simulation path
    std::vector<double> generatePath() {
        double S = params->S0;                       // Start with the initial asset price
        std::vector<double> path;                    // Container for the path prices
        path.reserve(params->nSteps + 1);            // Reserve space for efficiency
        path.push_back(S);                           // Store the initial price

        double dt = params->T / params->nSteps;      // Compute time increment per step

        // Initialize variance for Heston-like simulation; default is sigma squared
        double v = params->sigma * params->sigma;    

        // Loop over each time step in the path
        for (int i = 0; i < params->nSteps; ++i) {
            double sigma_t = params->sigma;  // Default volatility for constant vol scenario

            // Check if using Heston-like stochastic volatility
            if (params->volModelType == ParameterManager::HESTON_LIKE) {
                double dWv = rng.gaussian() * std::sqrt(dt);  // Random shock for variance
                // Update variance with mean reversion and volatility of volatility
                v = std::max(v + params->kappa*(params->theta - v)*dt 
                             + params->xi*std::sqrt(std::max(v,0.0))*dWv, 0.0);
                sigma_t = std::sqrt(v);  // Update volatility to the square root of variance
            }

            // Determine if a jump occurs at this time step
            double pJump = params->jumpIntensity * dt;         // Probability of a jump
            bool jumpOccurred = (rng.uniform01() < pJump);       // Random check for jump occurrence
            double jumpFactor = 1.0;                             // Default jump factor is 1 (no jump)
            if (jumpOccurred) {
                double jumpVal = rng.pareto(params->alpha, params->jumpScale); // Compute jump size
                // Cap the jump factor at maxJumpFactor to avoid extreme values
                jumpFactor = std::min(jumpVal, maxJumpFactor);
            }

            // Generate a standard normal random shock for asset diffusion
            double dW = rng.gaussian() * std::sqrt(dt);
            // Compute drift component with correction for volatility
            double drift = (params->r - 0.5 * sigma_t * sigma_t) * dt;
            // Compute diffusion component using the random shock
            double diffusion = sigma_t * dW;

            // Update the asset price using the exponential model and apply the jump factor
            S = S * std::exp(drift + diffusion) * jumpFactor;
            path.push_back(S);  // Record the new price in the path
        }
        return path;  // Return the complete simulated path
    }

    // Generate multiple simulation paths
    std::vector<std::vector<double>> generateAllPaths() {
        std::vector<std::vector<double>> allPaths;     // Container for all paths
        allPaths.reserve(params->nSim);                  // Reserve space for efficiency
        // Generate paths one-by-one
        for (int i = 0; i < params->nSim; ++i) {
            allPaths.push_back(generatePath());
        }
        return allPaths;  // Return all simulated paths
    }
};


struct MCResult {
    // Price and std error(mean price) of the option
    double price;      
    double stdErr;     
    // Confidence interval bounds
    double ciLower95;  
    double ciUpper95;  
};

// --------------------------------------------------------
// 4. OptionPricer:
//    Provides pricing functions for European call options using Monte Carlo,
//    Black-Scholes, and a tail pricing method (Taleb's Karamata approach).

class OptionPricer {
private:
    ParameterManager* params;  // pointer to the simulation parameters
public:
    // Constructor taking a pointer to ParameterManager
    OptionPricer(ParameterManager* p) : params(p) {}

    // Compute the Monte Carlo price of a European call option along with its confidence intervals.
    MCResult priceEuropeanCallWithCI(const std::vector<std::vector<double>>& paths, double strike) {
        double payoffSum = 0.0;    // Sum of payoffs across all paths
        double payoffSqSum = 0.0;  // Sum of squared payoffs for variance calculation
        int nPaths = static_cast<int>(paths.size());  // Total number of simulation paths

        // Loop over every simulation path
        for (auto & path : paths) {
            double ST = path.back();                        // Get the terminal asset price
            double payoff = std::max(ST - strike, 0.0);       // Compute call payoff: max(S_T - K, 0)

            // payoff = std::max(strike - ST, 0.0);           // Put option payoff: max(K - S_T, 0)

            // Accumulate payoffs for mean and variance calculations
            payoffSum   += payoff;                        
            payoffSqSum += payoff * payoff;                  
        }
        double meanPayoff = payoffSum / nPaths;              // Calculate mean payoff
        double meanSq     = payoffSqSum / nPaths;            // Calculate mean of squared payoffs
        double varPayoff  = meanSq - meanPayoff * meanPayoff; // Compute variance of the payoff
        double sePayoff   = std::sqrt(varPayoff / nPaths);   // Compute standard error of the mean payoff

        double discount = std::exp(-params->r * params->T);  // Discount factor to present value
        double mcPrice  = discount * meanPayoff;              // Discounted Monte Carlo price
        double mcSE     = discount * sePayoff;                // Discounted standard error

        // Calculate 95% confidence interval using standard normal critical value 1.96
        double ciOffset = 1.96 * mcSE;
        MCResult res;
        res.price     = mcPrice;
        res.stdErr    = mcSE;
        res.ciLower95 = mcPrice - ciOffset;
        res.ciUpper95 = mcPrice + ciOffset;

        return res; // Return the Monte Carlo price and confidence intervals
    }

    // Calculate European call price using the Black–Scholes formula
    double bsPriceCall(double S0, double strike, double r, double sigma, double T) {
        // Compute d1 and d2 used in the Black-Scholes formula
        double d1 = (std::log(S0 / strike) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        double cdf_d1 = normCDF(d1);  // Compute CDF for d1
        double cdf_d2 = normCDF(d2);  // Compute CDF for d2
        // Return the Black–Scholes call option price
        return S0 * cdf_d1 - strike * std::exp(-r * T) * cdf_d2;
    }

    // Taleb's Karamata tail pricing method
    // Uses an anchor option price at a given strike to infer price at other strikes using a power law.
    double talebKaramataPrice(double strike, double anchorStrike, double anchorCallPrice, double alpha) {
        // Scale the anchor price by a power law based on the tail index alpha
        return std::pow(strike / anchorStrike, 1.0 - alpha) * anchorCallPrice;
    }
};

// --------------------------------------------------------
// 5. ScenarioManager:
//    Manages simulation scenarios by generating paths, pricing options using different methods,
//    and writing the results to CSV files.
class ScenarioManager {
private:
    ParameterManager paramMgr;    // Instance to hold parameters
    PathSimulator simulator;      // Instance to generate simulation paths
    OptionPricer pricer;          // Instance to price options
public:
    // Constructor initializes simulator and pricer with the parameter manager
    ScenarioManager() : simulator(&paramMgr), pricer(&paramMgr) {}

    // Run a simulation scenario based on write to a CSV file.
    void runScenario(int scenarioID, const std::string &outputFilename) {
        // Load scenario-specific parameters and print them
        paramMgr.loadScenario(scenarioID);    
        paramMgr.printParameters();        

        std::cout << "Generating Monte Carlo paths...\n";
        auto paths = simulator.generateAllPaths(); // simulation paths w MC

        // Open output CSV file
        std::ofstream outFile(outputFilename);
        if(!outFile.is_open()) {
            std::cerr << "Error: cannot open " << outputFilename << "\n";
            return;
        }

        // Header w extra columns for different pricing methods
        outFile << "Strike,MC_Price,MC_StdErr,MC_95Lower,MC_95Upper,"
                << "BS_Price,TalebPrice,MC/BS\n";

        // Define strike range for the options to be priced
        double startStrike = 0.8 * paramMgr.S0;  // 80% of the initial price
        double endStrike   = 1.2 * paramMgr.S0;    // 120% of the initial price
        int nStrikes = 20;                         // Number of different strikes to evaluate
        double dK = (endStrike - startStrike) / (nStrikes - 1); // Strike increment


        // Use the first strike as the anchor for Taleb's Karamata tail pricing
        double anchorStrike = startStrike;
        auto anchorRes = pricer.priceEuropeanCallWithCI(paths, anchorStrike);
        double anchorMCPrice = anchorRes.price; // Anchor Monte Carlo price

        // Loop over each strike in the defined range
        for (int i = 0; i < nStrikes; i++) {
            double strike = startStrike + i * dK;  // Compute current strike

            // Price using Monte Carlo simulation
            auto mcRes  = pricer.priceEuropeanCallWithCI(paths, strike);
            // Price using the Black–Scholes formula
            double bsPrice = pricer.bsPriceCall(paramMgr.S0, strike, paramMgr.r, paramMgr.sigma, paramMgr.T);
            // Price using Taleb's Karamata tail pricing method
            double talebPrice = pricer.talebKaramataPrice(strike, anchorStrike, anchorMCPrice, paramMgr.alpha);

            // Compute the ratio of Monte Carlo price to Black–Scholes price if BS price is non-zero
            double ratio = (bsPrice > 1e-12) ? (mcRes.price / bsPrice) : 0.0;

            // Write results to CSV file in the defined order
            outFile << strike << ","
                    << mcRes.price << ","
                    << mcRes.stdErr << ","
                    << mcRes.ciLower95 << ","
                    << mcRes.ciUpper95 << ","
                    << bsPrice << ","
                    << talebPrice << ","
                    << ratio << "\n";
        }
        outFile.close(); // Close the file after writing
        std::cout << "Scenario " << scenarioID << " -> wrote " << outputFilename << "\n";
    }
};

// --------------------------------------------------------
// main: Entry point of the program.
int main(){
    std::cout << "Fat-Tail Option Pricing Under Power Laws\n\n"; 

    ScenarioManager manager;  // Create an instance of ScenarioManager to handle scenarios
    // Loop through several scenarios (0 through 3)
    for (int scenarioID = 0; scenarioID < 4; scenarioID++){
        // Create a filename for the output CSV file and run the scenario
        std::stringstream ss;       
        ss << "scenario_" << scenarioID << ".csv"; 
        manager.runScenario(scenarioID, ss.str()); 
    }

    std::cout << "\nAll scenarios completed. Check the 'scenario_*.csv' files.\n";
    return 0; 
}
