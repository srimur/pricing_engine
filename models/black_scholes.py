"""
Black-Merton-Scholes Model
==========================
The foundational model. Stock price follows Geometric Brownian Motion:
    dS = (r - q)·S·dt + σ·S·dW

Characteristic function of log-price:
    φ(u) = exp(i·u·(ln S + (r-q-σ²/2)T) - σ²T·u²/2)

This module provides:
- Analytical call/put pricing via the closed-form formula
- Characteristic function for FFT-based pricing
- All five Greeks (analytical)
- Implied volatility solver (Newton-Raphson + bisection fallback)
"""

import numpy as np
from scipy.stats import norm
from .base_model import BaseModel


class BlackScholesModel(BaseModel):
    """Black-Merton-Scholes option pricing model."""

    def __init__(self, S: float, r: float, T: float, sigma: float, q: float = 0.0):
        super().__init__(S, r, T, q)
        self.sigma = sigma

    # ──────────────────────────────────────────────
    #  Characteristic Function (for FFT pricing)
    # ──────────────────────────────────────────────
    def characteristic_function(self, u: np.ndarray) -> np.ndarray:
        """
        φ(u) = exp(i·u·ln(F) - ½·σ²·T·u² + i·u·(−½σ²T))
        where F = S·exp((r-q)T) is the forward price.

        More precisely:
        φ(u) = exp(i·u·(ln S + (r-q-σ²/2)·T) − σ²·T·u²/2)
        """
        x0 = np.log(self.S)
        drift = (self.r - self.q - 0.5 * self.sigma**2) * self.T
        diffusion = -0.5 * self.sigma**2 * self.T * u**2

        return np.exp(1j * u * (x0 + drift) + diffusion)

    # ──────────────────────────────────────────────
    #  Analytical Pricing
    # ──────────────────────────────────────────────
    def _d1_d2(self, K: float):
        d1 = (np.log(self.S / K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) \
             / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def call_price(self, K: float) -> float:
        d1, d2 = self._d1_d2(K)
        return (self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
                - K * np.exp(-self.r * self.T) * norm.cdf(d2))

    def put_price(self, K: float) -> float:
        d1, d2 = self._d1_d2(K)
        return (K * np.exp(-self.r * self.T) * norm.cdf(-d2)
                - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))

    def call_prices_vectorized(self, strikes: np.ndarray) -> np.ndarray:
        """Price calls for an array of strikes at once."""
        K = np.asarray(strikes)
        d1 = (np.log(self.S / K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) \
             / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return (self.S * np.exp(-self.q * self.T) * norm.cdf(d1)
                - K * np.exp(-self.r * self.T) * norm.cdf(d2))

    # ──────────────────────────────────────────────
    #  Greeks (Analytical)
    # ──────────────────────────────────────────────
    def delta(self, K: float, option_type: str = 'call') -> float:
        d1, _ = self._d1_d2(K)
        if option_type == 'call':
            return np.exp(-self.q * self.T) * norm.cdf(d1)
        return np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)

    def gamma(self, K: float) -> float:
        d1, _ = self._d1_d2(K)
        return (np.exp(-self.q * self.T) * norm.pdf(d1)
                / (self.S * self.sigma * np.sqrt(self.T)))

    def vega(self, K: float) -> float:
        """Per 1% move in vol."""
        d1, _ = self._d1_d2(K)
        return self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T) / 100

    def theta(self, K: float, option_type: str = 'call') -> float:
        """Per calendar day."""
        d1, d2 = self._d1_d2(K)
        common = -(self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma) \
                 / (2 * np.sqrt(self.T))
        if option_type == 'call':
            return (common
                    - self.r * K * np.exp(-self.r * self.T) * norm.cdf(d2)
                    + self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1)) / 365
        return (common
                + self.r * K * np.exp(-self.r * self.T) * norm.cdf(-d2)
                - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)) / 365

    def rho(self, K: float, option_type: str = 'call') -> float:
        """Per 1% move in rate."""
        _, d2 = self._d1_d2(K)
        if option_type == 'call':
            return K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        return -K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100

    # ──────────────────────────────────────────────
    #  Implied Volatility Solver
    # ──────────────────────────────────────────────
    @staticmethod
    def implied_vol(market_price: float, S: float, K: float, T: float,
                    r: float, q: float = 0.0, option_type: str = 'call',
                    tol: float = 1e-8, max_iter: int = 100) -> float:
        """
        Newton-Raphson implied vol solver with bisection fallback.

        Uses vega as the derivative: σ_{n+1} = σ_n - (BS(σ_n) - market) / vega(σ_n)
        """
        # Initial guess via Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T) * market_price / S

        # Bounds for bisection fallback
        lo, hi = 0.001, 5.0

        for i in range(max_iter):
            bs = BlackScholesModel(S, r, T, sigma, q)
            price = bs.call_price(K) if option_type == 'call' else bs.put_price(K)
            diff = price - market_price

            if abs(diff) < tol:
                return sigma

            vega_val = bs.vega(K) * 100  # Undo the /100 scaling
            if abs(vega_val) < 1e-15:
                # Fallback to bisection
                break

            sigma_new = sigma - diff / vega_val
            if sigma_new <= 0 or sigma_new > 5:
                break
            sigma = sigma_new

        # Bisection fallback
        for _ in range(200):
            mid = (lo + hi) / 2
            bs = BlackScholesModel(S, r, T, mid, q)
            price = bs.call_price(K) if option_type == 'call' else bs.put_price(K)
            if price > market_price:
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                return mid

        return (lo + hi) / 2

    # ──────────────────────────────────────────────
    #  Parameter Interface
    # ──────────────────────────────────────────────
    def get_params(self) -> dict:
        return {'sigma': self.sigma}

    def set_params(self, **kwargs):
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']

    def param_bounds(self) -> list:
        return [(0.01, 3.0)]  # sigma bounds

    def __repr__(self):
        return f"BlackScholesModel(S={self.S}, σ={self.sigma:.4f}, r={self.r}, T={self.T})"
