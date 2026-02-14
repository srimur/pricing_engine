"""
Carr-Madan FFT Option Pricer
==============================
Prices European call options across ALL strikes simultaneously using the
Fast Fourier Transform of the characteristic function.

Key idea (Carr & Madan 1999):
    C(K) = exp(-α·ln K)/π · ∫₀^∞ exp(-i·v·ln K) · ψ(v) dv

    where ψ(v) = exp(-rT)·φ(v-(α+1)i) / (α² + α - v² + i(2α+1)v)

    and α > 0 is a dampening factor (typically 1.5) that ensures integrability.

The integral is evaluated via FFT with Simpson's rule weighting.

Why FFT?
    - Black-Scholes prices one strike at a time: O(1) per strike
    - FFT prices N strikes simultaneously: O(N·log N) total
    - For a chain of 100+ strikes, FFT is ~100x faster
"""

import numpy as np
from models.base_model import BaseModel


class FFTPricer:
    """
    Carr-Madan FFT-based European option pricer.

    Works with ANY model that implements the characteristic_function method.
    """

    def __init__(self, model: BaseModel, N: int = 4096, alpha: float = 1.5,
                 eta: float = 0.25):
        """
        Parameters
        ----------
        model : BaseModel – Any model with characteristic_function(u)
        N     : int – FFT size (power of 2 for efficiency)
        alpha : float – Dampening factor (must be > 0; typical: 1.5)
        eta   : float – Grid spacing in frequency domain
        """
        self.model = model
        self.N = N
        self.alpha = alpha
        self.eta = eta

        # Derived quantities
        self.lam = 2 * np.pi / (N * eta)  # Strike grid spacing in log-space
        self.b = N * self.lam / 2          # Upper bound of log-strike grid

    def price_calls(self) -> tuple:
        """
        Price European calls for a grid of strikes.

        Returns
        -------
        strikes : np.ndarray – Strike prices
        prices  : np.ndarray – Call option prices
        """
        N, eta, alpha = self.N, self.eta, self.alpha
        lam, b = self.lam, self.b
        r, T = self.model.r, self.model.T

        # Frequency grid: v_j = j · eta, j = 0, ..., N-1
        v = np.arange(N) * eta

        # Log-strike grid: k_u = -b + λ·u, u = 0, ..., N-1
        k = -b + lam * np.arange(N)

        # ── Build the integrand ──
        # ψ(v) = exp(-rT) · φ(v - (α+1)i) / (α² + α - v² + i(2α+1)v)
        u_shifted = v - (alpha + 1) * 1j
        phi = self.model.characteristic_function(u_shifted)

        denominator = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
        psi = np.exp(-r * T) * phi / denominator

        # ── Simpson's rule weights ──
        simpson = 3 + (-1)**np.arange(N)  # [3,1,3,1,...,3,1] pattern
        simpson[0] = 1
        simpson = simpson / 3

        # ── FFT input ──
        x = np.exp(1j * v * b) * psi * eta * simpson

        # ── Compute FFT ──
        fft_result = np.fft.fft(x)

        # ── Extract call prices ──
        call_prices = np.exp(-alpha * k) / np.pi * np.real(fft_result)

        # Convert log-strikes to strikes
        strikes = np.exp(k)

        return strikes, call_prices

    def price_at_strikes(self, target_strikes: np.ndarray) -> np.ndarray:
        """
        Price calls at specific strikes by interpolating FFT output.

        Parameters
        ----------
        target_strikes : np.ndarray – Desired strike prices

        Returns
        -------
        prices : np.ndarray – Interpolated call prices
        """
        strikes, prices = self.price_calls()

        # Only use the valid range (positive prices, reasonable strikes)
        valid = (prices > 0) & (strikes > 0)
        if valid.sum() < 2:
            return np.full_like(target_strikes, np.nan, dtype=float)

        return np.interp(target_strikes, strikes[valid], prices[valid])

    def price_puts_via_parity(self, target_strikes: np.ndarray) -> np.ndarray:
        """
        Price puts using put-call parity: P = C - S·exp(-qT) + K·exp(-rT)
        """
        call_prices = self.price_at_strikes(target_strikes)
        S, r, q, T = self.model.S, self.model.r, self.model.q, self.model.T
        return call_prices - S * np.exp(-q * T) + target_strikes * np.exp(-r * T)


def compare_models_at_strikes(models: dict, strikes: np.ndarray) -> dict:
    """
    Price European calls under multiple models for comparison.

    Parameters
    ----------
    models  : dict of {name: BaseModel}
    strikes : np.ndarray of strike prices

    Returns
    -------
    results : dict of {name: np.ndarray of prices}
    """
    results = {}
    for name, model in models.items():
        pricer = FFTPricer(model)
        prices = pricer.price_at_strikes(strikes)
        results[name] = prices
    return results
