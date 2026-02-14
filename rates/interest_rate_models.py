"""
Interest Rate Model
=====================
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline


# ══════════════════════════════════════════════════════════
#  YIELD CURVE BOOTSTRAPPING
# ══════════════════════════════════════════════════════════

class YieldCurve:
    """
    Bootstraps zero rates from par swap rates and provides
    interpolation for any maturity.
    """

    def __init__(self, maturities: np.ndarray, par_rates: np.ndarray,
                 freq: int = 2):
        """
        Parameters
        ----------
        maturities : np.ndarray – Maturities in years (e.g., [0.5, 1, 2, 5, 10])
        par_rates  : np.ndarray – Par swap rates (e.g., [0.04, 0.042, ...])
        freq       : int – Payment frequency per year
        """
        self.maturities = np.asarray(maturities)
        self.par_rates = np.asarray(par_rates)
        self.freq = freq

        self.zero_rates, self.discount_factors = self._bootstrap()
        self.spline = CubicSpline(self.maturities, self.zero_rates)

    def _bootstrap(self):
        """Bootstrap zero rates from par rates."""
        zero_rates = []
        disc_factors = []

        for i, (T, par_rate) in enumerate(zip(self.maturities, self.par_rates)):
            c = par_rate / self.freq

            if i == 0:
                # First maturity: zero rate = par rate (approximately)
                z = par_rate
            else:
                # Sum PV of known coupons using previously bootstrapped DFs
                pv_coupons = 0
                for j in range(i):
                    t_j = self.maturities[j]
                    if t_j < T:
                        pv_coupons += c * disc_factors[j]

                # Solve for df_T: 1 = pv_coupons + (c + 1) × df_T
                df_T = (1 - pv_coupons) / (c + 1)

                if df_T <= 0:
                    z = zero_rates[-1]  # Fallback
                else:
                    z = -np.log(df_T) / T  # Continuous compounding

            zero_rates.append(z)
            disc_factors.append(np.exp(-z * T))

        return np.array(zero_rates), np.array(disc_factors)

    def zero_rate(self, T: float) -> float:
        """Interpolated zero rate at maturity T."""
        return float(self.spline(T))

    def discount(self, T: float) -> float:
        """Discount factor at maturity T."""
        return np.exp(-self.zero_rate(T) * T)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Forward rate between t1 and t2."""
        z1 = self.zero_rate(t1)
        z2 = self.zero_rate(t2)
        return (z2 * t2 - z1 * t1) / (t2 - t1)

    def price_bond(self, coupon_rate: float, maturity: float,
                   face: float = 100, freq: int = 2) -> float:
        """Price a bond using the bootstrapped zero curve."""
        n_periods = int(maturity * freq)
        coupon = face * coupon_rate / freq
        price = 0

        for i in range(1, n_periods + 1):
            t = i / freq
            cf = coupon if i < n_periods else coupon + face
            price += cf * self.discount(t)

        return price


# ══════════════════════════════════════════════════════════
#  VASICEK MODEL
# ══════════════════════════════════════════════════════════

class VasicekModel:
    """
    Vasicek (1977) short-rate model:
        dr = κ(θ - r)dt + σdW

    Properties:
    - Mean-reverting to θ with speed κ
    - Rates can go negative (limitation)
    - Closed-form zero-coupon bond price

    Parameters:
        kappa (κ) : Speed of mean reversion
        theta (θ) : Long-run mean rate
        sigma (σ) : Volatility of rate
        r0        : Current short rate
    """

    def __init__(self, kappa: float, theta: float, sigma: float, r0: float):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0

    def bond_price(self, T: float) -> float:
        """
        Closed-form zero-coupon bond price P(0, T).

        P(0,T) = A(T) · exp(-B(T) · r₀)

        B(T) = (1 - exp(-κT)) / κ
        A(T) = exp((θ - σ²/(2κ²))(B(T) - T) - σ²B(T)²/(4κ))
        """
        kappa, theta, sigma, r0 = self.kappa, self.theta, self.sigma, self.r0

        B = (1 - np.exp(-kappa * T)) / kappa
        A = np.exp(
            (theta - sigma**2 / (2 * kappa**2)) * (B - T)
            - sigma**2 * B**2 / (4 * kappa)
        )
        return A * np.exp(-B * r0)

    def yield_rate(self, T: float) -> float:
        """Zero-coupon yield at maturity T: y(T) = -ln(P(0,T)) / T"""
        return -np.log(self.bond_price(T)) / T

    def simulate(self, T: float, n_steps: int = 1000,
                 n_paths: int = 10000) -> np.ndarray:
        """Simulate short-rate paths using Euler discretization."""
        dt = T / n_steps
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = self.r0

        for t in range(1, n_steps + 1):
            z = np.random.standard_normal(n_paths)
            rates[:, t] = (rates[:, t-1]
                           + self.kappa * (self.theta - rates[:, t-1]) * dt
                           + self.sigma * np.sqrt(dt) * z)
        return rates

    @staticmethod
    def calibrate_from_data(rates: np.ndarray, dt: float = 1/252) -> 'VasicekModel':
        """
        Calibrate Vasicek parameters from historical short-rate data
        using OLS regression on the discretized SDE:

        r_{t+1} - r_t = κ(θ - r_t)dt + σ√dt·ε_t
        Δr = a + b·r_t + ε  where a = κθdt, b = -κdt, σ = std(ε)/√dt
        """
        dr = np.diff(rates)
        r = rates[:-1]

        # OLS: Δr = a + b × r
        n = len(dr)
        r_mean = r.mean()
        dr_mean = dr.mean()

        b = np.sum((r - r_mean) * (dr - dr_mean)) / np.sum((r - r_mean)**2)
        a = dr_mean - b * r_mean

        kappa = -b / dt
        theta = a / (kappa * dt)
        residuals = dr - a - b * r
        sigma = residuals.std() / np.sqrt(dt)

        return VasicekModel(kappa, theta, sigma, rates[-1])


# ══════════════════════════════════════════════════════════
#  CIR MODEL
# ══════════════════════════════════════════════════════════

class CIRModel:
    """
    Cox-Ingersoll-Ross (1985) short-rate model:
        dr = κ(θ - r)dt + σ√r·dW

    Properties:
    - Mean-reverting to θ with speed κ
    - Rates are non-negative if Feller condition holds: 2κθ > σ²
    - Closed-form zero-coupon bond price

    Parameters:
        kappa (κ) : Speed of mean reversion
        theta (θ) : Long-run mean rate
        sigma (σ) : Volatility parameter
        r0        : Current short rate
    """

    def __init__(self, kappa: float, theta: float, sigma: float, r0: float):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0

    def feller_condition(self) -> bool:
        """Check if 2κθ > σ² (ensures non-negative rates)."""
        return 2 * self.kappa * self.theta > self.sigma**2

    def bond_price(self, T: float) -> float:
        """
        Closed-form zero-coupon bond price P(0, T).

        Uses the CIR analytical formula involving hyperbolic functions.
        """
        kappa, theta, sigma, r0 = self.kappa, self.theta, self.sigma, self.r0

        gamma = np.sqrt(kappa**2 + 2 * sigma**2)

        B = (2 * (np.exp(gamma * T) - 1)) / \
            ((gamma + kappa) * (np.exp(gamma * T) - 1) + 2 * gamma)

        A = ((2 * gamma * np.exp((kappa + gamma) * T / 2)) /
             ((gamma + kappa) * (np.exp(gamma * T) - 1) + 2 * gamma)
             ) ** (2 * kappa * theta / sigma**2)

        return A * np.exp(-B * r0)

    def yield_rate(self, T: float) -> float:
        """Zero-coupon yield at maturity T."""
        return -np.log(self.bond_price(T)) / T

    def simulate(self, T: float, n_steps: int = 1000,
                 n_paths: int = 10000) -> np.ndarray:
        """Simulate using full truncation Euler scheme."""
        dt = T / n_steps
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = self.r0

        for t in range(1, n_steps + 1):
            z = np.random.standard_normal(n_paths)
            r_pos = np.maximum(rates[:, t-1], 0)
            rates[:, t] = (rates[:, t-1]
                           + self.kappa * (self.theta - r_pos) * dt
                           + self.sigma * np.sqrt(r_pos * dt) * z)
            rates[:, t] = np.maximum(rates[:, t], 0)  # Ensure non-negative
        return rates

    @staticmethod
    def calibrate_from_data(rates: np.ndarray, dt: float = 1/252) -> 'CIRModel':
        """
        Calibrate CIR parameters from historical data using OLS
        on the transformed SDE:

        Δr/√r = κθ√dt/√r - κ√dt·√r + σ√dt·ε
        """
        dr = np.diff(rates)
        r = rates[:-1]
        r_pos = np.maximum(r, 1e-10)
        sqrt_r = np.sqrt(r_pos)

        # Regression: Δr = a/√r + b·√r + σ√r·ε → OLS on Δr vs (1/√r, √r)
        X = np.column_stack([1/sqrt_r, sqrt_r])
        y = dr / (np.sqrt(dt))

        # OLS: y = X·β
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        kappa_dt = -beta[1]
        kappa = kappa_dt / np.sqrt(dt)
        theta = beta[0] / (kappa * np.sqrt(dt))

        # Estimate sigma from residuals
        y_pred = X @ beta
        residuals = y - y_pred
        sigma = residuals.std()

        return CIRModel(abs(kappa), abs(theta), abs(sigma), rates[-1])
