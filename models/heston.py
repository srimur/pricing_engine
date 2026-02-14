"""
Heston Stochastic Volatility Model
====================================
Stock price dynamics:
    dS = (r - q)·S·dt + √v·S·dW₁
    dv = κ(θ - v)·dt + ξ√v·dW₂
    Corr(dW₁, dW₂) = ρ

Parameters:
    v0  : initial variance
    kappa (κ) : speed of mean reversion of variance
    theta (θ) : long-run variance level
    xi (ξ)    : volatility of variance ("vol of vol")
    rho (ρ)   : correlation between stock and variance Brownian motions

Feller condition: 2κθ > ξ² ensures variance stays positive.

The characteristic function has a semi-analytical closed form,
enabling FFT-based pricing.
"""

import numpy as np
from .base_model import BaseModel


class HestonModel(BaseModel):
    """Heston (1993) stochastic volatility model."""

    def __init__(self, S: float, r: float, T: float,
                 v0: float, kappa: float, theta: float,
                 xi: float, rho: float, q: float = 0.0):
        super().__init__(S, r, T, q)
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def characteristic_function(self, u: np.ndarray) -> np.ndarray:
        """
        Heston characteristic function (log-stock price under risk-neutral measure).

        Uses the formulation from Albrecher et al. (2007) which avoids
        the branch-cut discontinuity issue in the original Heston (1993) form.

        φ(u) = exp(C(u) + D(u)·v₀ + i·u·ln(S·exp((r-q)T)))

        where:
            d = sqrt((ρξiu - κ)² + ξ²(iu + u²))
            g = (κ - ρξiu - d) / (κ - ρξiu + d)

            C = (r-q)·i·u·T + κθ/ξ² · [(κ - ρξiu - d)T - 2ln((1 - g·exp(-dT))/(1-g))]
            D = (κ - ρξiu - d)/ξ² · (1 - exp(-dT))/(1 - g·exp(-dT))
        """
        S, r, q, T = self.S, self.r, self.q, self.T
        v0, kappa, theta, xi, rho = self.v0, self.kappa, self.theta, self.xi, self.rho

        x0 = np.log(S)

        # Complex-valued intermediate quantities
        d = np.sqrt((rho * xi * 1j * u - kappa)**2 + xi**2 * (1j * u + u**2))

        # Use formulation 2 (Albrecher) for numerical stability
        g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

        # C and D functions
        exp_neg_dT = np.exp(-d * T)

        C = ((r - q) * 1j * u * T
             + kappa * theta / xi**2
             * ((kappa - rho * xi * 1j * u - d) * T
                - 2 * np.log((1 - g * exp_neg_dT) / (1 - g))))

        D = ((kappa - rho * xi * 1j * u - d) / xi**2
             * (1 - exp_neg_dT) / (1 - g * exp_neg_dT))

        return np.exp(C + D * v0 + 1j * u * x0)

    def feller_condition(self) -> bool:
        """Check if 2κθ > ξ² (variance stays positive)."""
        return 2 * self.kappa * self.theta > self.xi**2

    def simulate(self, n_paths: int = 10000, n_steps: int = 252) -> np.ndarray:
        """
        Simulate Heston model paths using Euler discretization.
        Uses full truncation scheme for variance process.

        Returns
        -------
        prices : np.ndarray of shape (n_paths, n_steps + 1)
        """
        dt = self.T / n_steps
        S_paths = np.zeros((n_paths, n_steps + 1))
        v_paths = np.zeros((n_paths, n_steps + 1))
        S_paths[:, 0] = self.S
        v_paths[:, 0] = self.v0

        for t in range(1, n_steps + 1):
            z1 = np.random.standard_normal(n_paths)
            z2 = np.random.standard_normal(n_paths)

            # Correlated Brownian motions
            w1 = z1
            w2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

            # Variance process (full truncation: use max(v, 0))
            v_pos = np.maximum(v_paths[:, t - 1], 0)
            v_paths[:, t] = (v_paths[:, t - 1]
                             + self.kappa * (self.theta - v_pos) * dt
                             + self.xi * np.sqrt(v_pos * dt) * w2)

            # Stock price (log-Euler scheme)
            v_pos_curr = np.maximum(v_paths[:, t - 1], 0)
            S_paths[:, t] = S_paths[:, t - 1] * np.exp(
                (self.r - self.q - 0.5 * v_pos_curr) * dt
                + np.sqrt(v_pos_curr * dt) * w1
            )

        return S_paths

    # ──────────────────────────────────────────────
    #  Parameter Interface
    # ──────────────────────────────────────────────
    def get_params(self) -> dict:
        return {
            'v0': self.v0, 'kappa': self.kappa, 'theta': self.theta,
            'xi': self.xi, 'rho': self.rho
        }

    def set_params(self, **kwargs):
        for k in ['v0', 'kappa', 'theta', 'xi', 'rho']:
            if k in kwargs:
                setattr(self, k, kwargs[k])

    def param_bounds(self) -> list:
        return [
            (0.001, 1.0),    # v0:    initial variance
            (0.1, 10.0),     # kappa: mean reversion speed
            (0.001, 1.0),    # theta: long-run variance
            (0.05, 2.0),     # xi:    vol of vol
            (-0.99, 0.0),    # rho:   correlation (typically negative for equities)
        ]

    def __repr__(self):
        return (f"HestonModel(v0={self.v0:.4f}, κ={self.kappa:.2f}, "
                f"θ={self.theta:.4f}, ξ={self.xi:.2f}, ρ={self.rho:.2f})")
