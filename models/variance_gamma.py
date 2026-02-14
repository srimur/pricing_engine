"""
Variance Gamma Model
=====================
The VG model (Madan, Carr & Chang 1998) captures:
- Skewness in returns (via θ parameter)
- Excess kurtosis / fat tails (via ν parameter)
- Pure jump process (no diffusion component)

Construction:
    X_VG(t) = θ·G(t) + σ·W(G(t))

    where G(t) is a Gamma process with mean rate 1 and variance rate ν.
    This is Brownian motion with drift, time-changed by a Gamma process.

Stock price:
    S(t) = S(0)·exp((r - q + ω)t + X_VG(t))
    where ω = (1/ν)·ln(1 - θν - σ²ν/2) is the convexity correction.

Parameters:
    sigma (σ) : volatility of the Brownian motion component
    nu (ν)    : variance rate of the Gamma time change (controls kurtosis)
    theta_vg  : drift of the Brownian motion (controls skewness)
"""

import numpy as np
from .base_model import BaseModel


class VarianceGammaModel(BaseModel):
    """Variance Gamma (VG) option pricing model."""

    def __init__(self, S: float, r: float, T: float,
                 sigma: float, nu: float, theta_vg: float, q: float = 0.0):
        super().__init__(S, r, T, q)
        self.sigma = sigma
        self.nu = nu
        self.theta_vg = theta_vg

    @property
    def omega(self) -> float:
        """
        Convexity correction (martingale condition):
        ω = (1/ν)·ln(1 - θν - σ²ν/2)
        Ensures E[S(T)] = S(0)·exp((r-q)T)
        """
        return (1 / self.nu) * np.log(1 - self.theta_vg * self.nu
                                       - 0.5 * self.sigma**2 * self.nu)

    def characteristic_function(self, u: np.ndarray) -> np.ndarray:
        """
        VG characteristic function of log-price:

        φ(u) = exp(i·u·(ln S + (r-q+ω)T)) · (1 - i·u·θ·ν + σ²ν·u²/2)^(-T/ν)

        This is derived from the characteristic function of the VG process:
        φ_VG(u; t) = (1 - i·u·θ·ν + σ²νu²/2)^(-t/ν)
        """
        x0 = np.log(self.S)
        sigma, nu, theta = self.sigma, self.nu, self.theta_vg

        # Log-price drift under risk-neutral measure
        drift = (self.r - self.q + self.omega) * self.T

        # VG characteristic exponent
        vg_char = (1 - 1j * u * theta * nu + 0.5 * sigma**2 * nu * u**2) ** (-self.T / nu)

        return np.exp(1j * u * (x0 + drift)) * vg_char

    def simulate(self, n_paths: int = 10000, n_steps: int = 252) -> np.ndarray:
        """
        Simulate VG model using Gamma time-changed Brownian motion.

        Steps:
        1. Generate Gamma increments: ΔG ~ Gamma(dt/ν, ν)
        2. Generate BM increments: ΔX = θ·ΔG + σ·√ΔG·Z
        3. Accumulate: X(t) = Σ ΔX
        4. S(t) = S(0)·exp((r-q+ω)t + X(t))
        """
        dt = self.T / n_steps
        shape = dt / self.nu  # Gamma shape parameter
        scale = self.nu       # Gamma scale parameter

        S_paths = np.zeros((n_paths, n_steps + 1))
        S_paths[:, 0] = self.S
        log_S = np.log(self.S) * np.ones(n_paths)

        for t in range(1, n_steps + 1):
            # Gamma time increments
            dG = np.random.gamma(shape, scale, n_paths)
            # Brownian motion increments in Gamma time
            z = np.random.standard_normal(n_paths)
            dX = self.theta_vg * dG + self.sigma * np.sqrt(dG) * z

            log_S += (self.r - self.q + self.omega) * dt + dX
            S_paths[:, t] = np.exp(log_S)

        return S_paths

    # ──────────────────────────────────────────────
    #  Parameter Interface
    # ──────────────────────────────────────────────
    def get_params(self) -> dict:
        return {'sigma': self.sigma, 'nu': self.nu, 'theta_vg': self.theta_vg}

    def set_params(self, **kwargs):
        for k in ['sigma', 'nu', 'theta_vg']:
            if k in kwargs:
                setattr(self, k, kwargs[k])

    def param_bounds(self) -> list:
        return [
            (0.01, 1.5),     # sigma
            (0.01, 2.0),     # nu (variance rate)
            (-0.5, 0.1),     # theta_vg (usually negative for equities → negative skew)
        ]

    def __repr__(self):
        return (f"VarianceGammaModel(σ={self.sigma:.4f}, ν={self.nu:.4f}, "
                f"θ={self.theta_vg:.4f})")
