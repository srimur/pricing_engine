# Multi-Model Volatility Surface Calibration & Exotic Option Pricing Engine

A comprehensive pricing library built in Python that calibrates multiple stochastic models (Black-Scholes, Heston, Variance Gamma) to market options data using FFT-based pricing and numerical optimization, constructs implied volatility surfaces, prices exotic derivatives via Monte Carlo, and models interest rates using short-rate models.

**Built as a capstone project applying concepts from Columbia University's *Computational Methods in Pricing and Model Calibration* course.**

---

## Project Architecture

```
pricing_engine/
│
├── models/
│   ├── black_scholes.py      # BMS model: analytical + FFT pricing + Greeks
│   ├── heston.py             # Heston stochastic volatility model
│   ├── variance_gamma.py     # Variance Gamma (pure jump) model
│   └── base_model.py         # Abstract base class for all models
│
├── calibration/
│   ├── vol_surface.py        # Implied vol surface construction
│   ├── calibrator.py         # Model calibration engine (brute-force, Nelder-Mead, BFGS)
│   └── objective.py          # Objective functions (RMSE, IVRMSE, weighted)
│
├── pricing/
│   ├── fft_pricer.py         # Carr-Madan FFT option pricing
│   ├── monte_carlo.py        # Monte Carlo engine for exotics
│   └── exotic_options.py     # Asian, Barrier, Lookback payoffs
│
├── rates/
│   ├── curve_builder.py      # Yield curve bootstrapping
│   ├── vasicek.py            # Vasicek short-rate model
│   ├── cir.py                # CIR short-rate model
│   └── bond_pricer.py        # Bond and swap pricing
│
├── utils/
│   ├── data_loader.py        # Market data fetching
│   └── visualization.py      # Vol surface plots, calibration dashboards
│
├── notebooks/
│   └── demo_full_pipeline.py # End-to-end demonstration script
│
├── main.py                   # CLI entry point for full pipeline
├── requirements.txt
└── README.md
```

## Key Features

### 1. FFT-Based Option Pricing
- Carr-Madan FFT method for pricing European options across all strikes simultaneously
- Characteristic function implementations for BMS, Heston, and Variance Gamma models
- ~100x faster than individual Black-Scholes evaluations for full option chains

### 2. Multi-Model Calibration
- Calibrates model parameters by minimizing error between model and market prices
- Three optimization approaches: brute-force grid search, Nelder-Mead, BFGS
- Supports price-based and implied-vol-based objective functions
- Produces calibration diagnostics: RMSE, parameter stability, fit visualization

### 3. Implied Volatility Surface
- Constructs 3D vol surfaces from market options data
- Newton-Raphson implied vol solver with bisection fallback
- Visualizes smile/skew across maturities

### 4. Exotic Option Pricing
- Monte Carlo pricing with calibrated Heston parameters
- Payoffs: Asian (arithmetic/geometric), Barrier (up/down, in/out), Lookback
- Variance reduction: antithetic variates, control variates

### 5. Interest Rate Modelling
- Yield curve bootstrapping from swap rates
- Vasicek and CIR model calibration via MLE regression
- Zero-coupon bond pricing under calibrated short-rate models

---

## Quick Start

```bash
pip install numpy scipy matplotlib
python main.py
```

## Models Overview

| Model | Dynamics | Parameters | Captures |
|-------|----------|-----------|----------|
| Black-Scholes | dS = rSdt + σSdW | σ | Flat vol (baseline) |
| Heston | dS = rSdt + √v·S·dW₁, dv = κ(θ-v)dt + ξ√v·dW₂ | v₀, κ, θ, ξ, ρ | Vol smile, mean-reverting vol |
| Variance Gamma | S(t) = S(0)exp((r+ω)t + X_VG(t)) | σ, ν, θ | Skewness, excess kurtosis, jumps |
| Vasicek | dr = κ(θ-r)dt + σdW | κ, θ, σ | Mean-reverting rates |
| CIR | dr = κ(θ-r)dt + σ√r·dW | κ, θ, σ | Non-negative rates |

---

## Technical Highlights

- **Characteristic functions** derived and implemented for each model
- **FFT inversion** via Carr-Madan with Simpson's rule dampening
- **Calibration pipeline** with automatic initial guess via grid search
- **Walk-forward calibration** to test parameter stability over time
- **Interest rate bootstrapping** with cubic spline interpolation

## References

- Carr, P. & Madan, D. (1999). *Option valuation using the fast Fourier transform*
- Heston, S. (1993). *A closed-form solution for options with stochastic volatility*
- Madan, D., Carr, P. & Chang, E. (1998). *The Variance Gamma process and option pricing*
- Hirsa, A. (2013). *Computational Methods in Finance*
- Vasicek, O. (1977). *An equilibrium characterization of the term structure*
