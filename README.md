# FatTailMC

FatTailMC is a C++ tool for pricing European call options under fat-tail risk using Monte Carlo simulations. It models jump risk and optional stochastic volatility, then prices options using multiple methods, outputting the results to CSV files.

## Project Structure

- **ParameterManager:** Manages simulation settings such as time, simulation paths, risk-free rate, jump parameters, and volatility mode.
- **RandomNumberGenerator:** Provides uniform, Gaussian, and Pareto random number generation.
- **PathSimulator:** Generates asset price paths, incorporating drift, diffusion, jumps, and optional stochastic volatility.
- **OptionPricer:** Prices European call options using Monte Carlo simulations, the Black–Scholes method, and Taleb's Karamata approach.
- **ScenarioManager:** Runs various simulation scenarios and writes the results to CSV files.

## Compilation & Usage

### Compilation

Use a C++ compiler (e.g., g++) to compile the code:

```bash
g++ -std=c++11 -O2 -o FatTailMC main.cpp
