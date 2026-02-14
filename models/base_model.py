"""
Base class for all pricing models.
Every model must implement its characteristic function, which is then used
by the FFT pricer to compute option prices across all strikes simultaneously.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Abstract base class for option pricing models."""

    def __init__(self, S: float, r: float, T: float, q: float = 0.0):
        """
        Parameters
        ----------
        S : float – Current spot price
        r : float – Risk-free rate (annualized, continuous compounding)
        T : float – Time to maturity in years
        q : float – Continuous dividend yield
        """
        self.S = S
        self.r = r
        self.T = T
        self.q = q

    @abstractmethod
    def characteristic_function(self, u: np.ndarray) -> np.ndarray:
        """
        Return the characteristic function φ(u) = E[exp(i·u·ln(S_T))]
        evaluated under the risk-neutral measure.

        This is the Fourier transform of the log-price density and is
        the key building block for FFT-based option pricing.

        Parameters
        ----------
        u : np.ndarray – Frequency-domain variable (can be complex)

        Returns
        -------
        np.ndarray – Characteristic function values
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Return current model parameters as a dictionary."""
        pass

    @abstractmethod
    def set_params(self, **kwargs):
        """Set model parameters from keyword arguments."""
        pass

    @abstractmethod
    def param_bounds(self) -> list:
        """Return list of (lower, upper) bounds for each calibration parameter."""
        pass

    def forward_price(self) -> float:
        """Risk-neutral forward price: F = S·exp((r-q)·T)"""
        return self.S * np.exp((self.r - self.q) * self.T)

    def discount_factor(self) -> float:
        """Discount factor: exp(-r·T)"""
        return np.exp(-self.r * self.T)
