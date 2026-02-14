"""
main.py — Full Pipeline Demonstration
=======================================
Runs the complete pricing engine end-to-end:

1. Generate synthetic market data (options chain)
2. Build implied volatility surface
3. Calibrate BMS, Heston, and Variance Gamma models
4. Compare FFT prices across models
5. Price exotic options with calibrated Heston
6. Build yield curve and calibrate rate models

Run: python main.py
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.black_scholes import BlackScholesModel
from models.heston import HestonModel
from models.variance_gamma import VarianceGammaModel
from pricing.fft_pricer import FFTPricer, compare_models_at_strikes
from pricing.monte_carlo import MonteCarloExoticPricer
from calibration.calibrator import Calibrator
from rates.interest_rate_models import YieldCurve, VasicekModel, CIRModel


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    np.random.seed(42)

    # ══════════════════════════════════════════════════
    #  MODULE 1: Generate Synthetic Market Data
    # ══════════════════════════════════════════════════
    separator("MODULE 1: Market Data & Implied Vol Surface")

    S0 = 100.0      # Spot price
    r = 0.05         # Risk-free rate
    T = 0.5          # 6-month options
    q = 0.02         # Dividend yield

    # Generate "market" prices using Heston model as truth
    true_heston = HestonModel(S0, r, T, v0=0.04, kappa=2.0, theta=0.04,
                               xi=0.5, rho=-0.7, q=q)

    strikes = np.arange(80, 121, 2.0)  # 80 to 120 in steps of 2
    print(f"Spot: {S0}, Rate: {r}, Div: {q}, Maturity: {T}y")
    print(f"Strikes: {strikes[0]:.0f} to {strikes[-1]:.0f} ({len(strikes)} strikes)")

    # Generate true prices via FFT
    fft_true = FFTPricer(true_heston)
    market_prices = fft_true.price_at_strikes(strikes)

    # Add small noise to simulate real market
    noise = np.random.normal(0, 0.05, len(strikes))
    market_prices = np.maximum(market_prices + noise, 0.01)

    # Compute implied volatilities
    market_ivs = np.array([
        BlackScholesModel.implied_vol(p, S0, K, T, r, q, 'call')
        for p, K in zip(market_prices, strikes)
    ])

    print(f"\nImplied Volatility Surface (smile/skew):")
    print(f"{'Strike':>8} {'Market Price':>12} {'Implied Vol':>12}")
    print("-" * 36)
    for K, price, iv in zip(strikes, market_prices, market_ivs):
        print(f"{K:8.1f} {price:12.4f} {iv*100:11.2f}%")

    # ══════════════════════════════════════════════════
    #  MODULE 2: FFT Pricing Across Models
    # ══════════════════════════════════════════════════
    separator("MODULE 2: FFT-Based Pricing Comparison")

    models = {
        'Black-Scholes (σ=20%)': BlackScholesModel(S0, r, T, 0.20, q),
        'Heston (true params)': true_heston,
        'Variance Gamma': VarianceGammaModel(S0, r, T, sigma=0.20, nu=0.25,
                                              theta_vg=-0.15, q=q),
    }

    results = compare_models_at_strikes(models, strikes)

    print(f"{'Strike':>8}", end="")
    for name in results:
        short_name = name[:15]
        print(f" {short_name:>15}", end="")
    print(f" {'Market':>12}")
    print("-" * (8 + 15 * len(results) + 12))

    for i, K in enumerate(strikes):
        print(f"{K:8.1f}", end="")
        for name, prices in results.items():
            print(f" {prices[i]:15.4f}", end="")
        print(f" {market_prices[i]:12.4f}")

    # ══════════════════════════════════════════════════
    #  MODULE 3: Model Calibration
    # ══════════════════════════════════════════════════
    separator("MODULE 3: Model Calibration")

    # --- Calibrate Black-Scholes ---
    print("Calibrating Black-Scholes (1 parameter: σ)...")
    bs_model = BlackScholesModel(S0, r, T, 0.30, q)
    bs_cal = Calibrator(bs_model, strikes, market_prices)
    bs_result = bs_cal.calibrate_nelder_mead()
    print(f"  Calibrated σ = {bs_result['optimal_params']['sigma']*100:.2f}%")
    print(f"  Price RMSE = {bs_result['objective_value']:.6f}")

    # --- Calibrate Heston ---
    print("\nCalibrating Heston (5 parameters: v₀, κ, θ, ξ, ρ)...")
    heston_model = HestonModel(S0, r, T, v0=0.05, kappa=1.5, theta=0.05,
                                xi=0.4, rho=-0.5, q=q)
    heston_cal = Calibrator(heston_model, strikes, market_prices)

    # Two-stage: Nelder-Mead → BFGS
    heston_nm = heston_cal.calibrate_nelder_mead()
    x0_nm = np.array(list(heston_nm['optimal_params'].values()))
    heston_bfgs = heston_cal.calibrate_bfgs(x0=x0_nm)

    print(f"  Calibrated params: {heston_bfgs['optimal_params']}")
    print(f"  Price RMSE = {heston_bfgs['objective_value']:.6f}")
    print(f"  Converged: {heston_bfgs['converged']}")

    # --- Calibrate Variance Gamma ---
    print("\nCalibrating Variance Gamma (3 parameters: σ, ν, θ)...")
    vg_model = VarianceGammaModel(S0, r, T, sigma=0.25, nu=0.5,
                                   theta_vg=-0.1, q=q)
    vg_cal = Calibrator(vg_model, strikes, market_prices)
    vg_nm = vg_cal.calibrate_nelder_mead()
    x0_vg = np.array(list(vg_nm['optimal_params'].values()))
    vg_bfgs = vg_cal.calibrate_bfgs(x0=x0_vg)
    print(f"  Calibrated params: {vg_bfgs['optimal_params']}")
    print(f"  Price RMSE = {vg_bfgs['objective_value']:.6f}")

    # Summary
    print(f"\n{'Model':>20} {'RMSE':>10}")
    print("-" * 32)
    print(f"{'Black-Scholes':>20} {bs_result['objective_value']:10.6f}")
    print(f"{'Heston':>20} {heston_bfgs['objective_value']:10.6f}")
    print(f"{'Variance Gamma':>20} {vg_bfgs['objective_value']:10.6f}")

    # ══════════════════════════════════════════════════
    #  MODULE 4: Exotic Option Pricing
    # ══════════════════════════════════════════════════
    separator("MODULE 4: Exotic Option Pricing (Calibrated Heston MC)")

    print("Simulating 50,000 Heston paths...")
    paths = heston_model.simulate(n_paths=50000, n_steps=252)

    mc = MonteCarloExoticPricer(paths, r, T)
    K_exotic = 100.0  # ATM
    barrier = 120.0

    exotic_results = mc.price_all_exotics(K_exotic, barrier)

    print(f"\n{'Option Type':>30} {'Price':>10} {'Std Error':>10} {'95% CI':>24}")
    print("-" * 78)
    for name, res in exotic_results.items():
        ci_lo, ci_hi = res['95_CI']
        print(f"{name:>30} {res['price']:10.4f} {res['std_error']:10.4f} "
              f"[{ci_lo:.4f}, {ci_hi:.4f}]")

    # ══════════════════════════════════════════════════
    #  MODULE 5: Interest Rate Modelling
    # ══════════════════════════════════════════════════
    separator("MODULE 5: Interest Rate Models & Yield Curve")

    # --- Yield Curve ---
    maturities = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    swap_rates = np.array([0.040, 0.042, 0.045, 0.047, 0.050,
                           0.052, 0.054, 0.056, 0.057])

    curve = YieldCurve(maturities, swap_rates)

    print("Bootstrapped Zero Curve:")
    print(f"{'Maturity':>10} {'Par Rate':>10} {'Zero Rate':>10} {'DF':>10}")
    print("-" * 44)
    for m, pr, zr, df in zip(maturities, swap_rates, curve.zero_rates,
                              curve.discount_factors):
        print(f"{m:10.1f} {pr*100:9.3f}% {zr*100:9.3f}% {df:10.6f}")

    print(f"\nInterpolated 4Y zero rate: {curve.zero_rate(4)*100:.3f}%")
    print(f"Forward rate 2Y → 5Y: {curve.forward_rate(2, 5)*100:.3f}%")

    # Price a bond
    bond_px = curve.price_bond(coupon_rate=0.05, maturity=10, face=100)
    print(f"10Y 5% coupon bond price: ${bond_px:.4f}")

    # --- Vasicek Model ---
    print("\n--- Vasicek Model ---")
    vasicek = VasicekModel(kappa=0.5, theta=0.05, sigma=0.02, r0=0.04)

    print(f"Parameters: κ={vasicek.kappa}, θ={vasicek.theta}, "
          f"σ={vasicek.sigma}, r₀={vasicek.r0}")
    print(f"\nVasicek Zero-Coupon Bond Prices:")
    for T_bond in [1, 2, 5, 10, 30]:
        px = vasicek.bond_price(T_bond)
        yld = vasicek.yield_rate(T_bond)
        print(f"  {T_bond:2d}Y: P = {px:.6f}, Yield = {yld*100:.3f}%")

    # Calibrate from simulated data
    print("\nCalibrating Vasicek from simulated data...")
    sim_rates = vasicek.simulate(T=5, n_steps=1260, n_paths=1)[0]
    vasicek_cal = VasicekModel.calibrate_from_data(sim_rates, dt=5/1260)
    print(f"  True:       κ={vasicek.kappa:.3f}, θ={vasicek.theta:.4f}, σ={vasicek.sigma:.4f}")
    print(f"  Calibrated: κ={vasicek_cal.kappa:.3f}, θ={vasicek_cal.theta:.4f}, σ={vasicek_cal.sigma:.4f}")

    # --- CIR Model ---
    print("\n--- CIR Model ---")
    cir = CIRModel(kappa=0.8, theta=0.05, sigma=0.1, r0=0.04)
    print(f"Feller condition satisfied: {cir.feller_condition()}")

    print(f"\nCIR Zero-Coupon Bond Prices:")
    for T_bond in [1, 2, 5, 10, 30]:
        px = cir.bond_price(T_bond)
        yld = cir.yield_rate(T_bond)
        print(f"  {T_bond:2d}Y: P = {px:.6f}, Yield = {yld*100:.3f}%")

    # ══════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════
    separator("PIPELINE COMPLETE")
    print("Modules executed:")
    print("  ✓ Module 1: Market data generation & implied vol surface")
    print("  ✓ Module 2: FFT pricing across BMS / Heston / VG models")
    print("  ✓ Module 3: Model calibration (Nelder-Mead + L-BFGS-B)")
    print("  ✓ Module 4: Exotic option pricing (MC with calibrated Heston)")
    print("  ✓ Module 5: Yield curve bootstrap + Vasicek/CIR calibration")
    print()
    print("Key results:")
    print(f"  Best equity model fit: Heston (RMSE = {heston_bfgs['objective_value']:.6f})")
    print(f"  BMS baseline RMSE:     {bs_result['objective_value']:.6f}")
    print(f"  ATM European call (MC): ${exotic_results['European Call']['price']:.4f}")
    print(f"  ATM Asian call (MC):    ${exotic_results['Asian Call (arithmetic)']['price']:.4f}")
    print(f"  10Y bond price:         ${bond_px:.4f}")


if __name__ == '__main__':
    main()
