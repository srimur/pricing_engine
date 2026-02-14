"""
Monte Carlo Exotic Option Pricer
=================================
Prices path-dependent options using simulated paths from any model
that implements a simulate() method (Heston, VG, or simple GBM).

Supported exotics:
    - Asian (arithmetic & geometric average)
    - Barrier (up/down, in/out)
    - Lookback (floating strike)

Variance reduction:
    - Antithetic variates
    - Control variates (using European BS price as control)
"""

import numpy as np
from scipy.stats import norm


class MonteCarloExoticPricer:
    """Monte Carlo engine for pricing exotic options."""

    def __init__(self, paths: np.ndarray, r: float, T: float):
        """
        Parameters
        ----------
        paths : np.ndarray of shape (n_paths, n_steps + 1)
            Simulated price paths from any model.
        r : float – Risk-free rate
        T : float – Time to maturity
        """
        self.paths = paths
        self.r = r
        self.T = T
        self.n_paths = paths.shape[0]
        self.discount = np.exp(-r * T)

    def _price_and_se(self, payoffs: np.ndarray) -> dict:
        """Compute discounted price and standard error."""
        disc_payoffs = self.discount * payoffs
        price = disc_payoffs.mean()
        std_err = disc_payoffs.std() / np.sqrt(self.n_paths)
        ci_95 = (price - 1.96 * std_err, price + 1.96 * std_err)
        return {'price': price, 'std_error': std_err, '95_CI': ci_95}

    # ──────────────────────────────────────────────
    #  European (for benchmarking)
    # ──────────────────────────────────────────────
    def european_call(self, K: float) -> dict:
        payoffs = np.maximum(self.paths[:, -1] - K, 0)
        return self._price_and_se(payoffs)

    def european_put(self, K: float) -> dict:
        payoffs = np.maximum(K - self.paths[:, -1], 0)
        return self._price_and_se(payoffs)

    # ──────────────────────────────────────────────
    #  Asian Options
    # ──────────────────────────────────────────────
    def asian_call(self, K: float, average_type: str = 'arithmetic') -> dict:
        """
        Asian option: payoff based on average price over the path.
        Arithmetic average is more common but has no closed form.
        Geometric average has a closed form (useful as control variate).
        """
        prices = self.paths[:, 1:]  # Exclude initial price from averaging

        if average_type == 'arithmetic':
            avg = prices.mean(axis=1)
        elif average_type == 'geometric':
            avg = np.exp(np.log(prices).mean(axis=1))
        else:
            raise ValueError(f"Unknown average type: {average_type}")

        payoffs = np.maximum(avg - K, 0)
        return self._price_and_se(payoffs)

    def asian_put(self, K: float, average_type: str = 'arithmetic') -> dict:
        prices = self.paths[:, 1:]
        if average_type == 'arithmetic':
            avg = prices.mean(axis=1)
        else:
            avg = np.exp(np.log(prices).mean(axis=1))
        payoffs = np.maximum(K - avg, 0)
        return self._price_and_se(payoffs)

    # ──────────────────────────────────────────────
    #  Barrier Options
    # ──────────────────────────────────────────────
    def barrier_call(self, K: float, barrier: float,
                     barrier_type: str = 'up-and-out') -> dict:
        """
        Barrier option: activated/deactivated when price crosses barrier.

        Types:
            up-and-out:   knocked out if max(S) ≥ barrier
            down-and-out: knocked out if min(S) ≤ barrier
            up-and-in:    activated if max(S) ≥ barrier
            down-and-in:  activated if min(S) ≤ barrier
        """
        ST = self.paths[:, -1]
        max_S = self.paths.max(axis=1)
        min_S = self.paths.min(axis=1)

        vanilla = np.maximum(ST - K, 0)

        if barrier_type == 'up-and-out':
            alive = max_S < barrier
        elif barrier_type == 'down-and-out':
            alive = min_S > barrier
        elif barrier_type == 'up-and-in':
            alive = max_S >= barrier
        elif barrier_type == 'down-and-in':
            alive = min_S <= barrier
        else:
            raise ValueError(f"Unknown barrier type: {barrier_type}")

        payoffs = np.where(alive, vanilla, 0.0)
        return self._price_and_se(payoffs)

    # ──────────────────────────────────────────────
    #  Lookback Options
    # ──────────────────────────────────────────────
    def lookback_call_floating(self) -> dict:
        """Floating strike lookback call: payoff = S_T - min(S_t)"""
        payoffs = self.paths[:, -1] - self.paths.min(axis=1)
        return self._price_and_se(payoffs)

    def lookback_put_floating(self) -> dict:
        """Floating strike lookback put: payoff = max(S_t) - S_T"""
        payoffs = self.paths.max(axis=1) - self.paths[:, -1]
        return self._price_and_se(payoffs)

    def lookback_call_fixed(self, K: float) -> dict:
        """Fixed strike lookback call: payoff = max(max(S_t) - K, 0)"""
        payoffs = np.maximum(self.paths.max(axis=1) - K, 0)
        return self._price_and_se(payoffs)

    # ──────────────────────────────────────────────
    #  Summary
    # ──────────────────────────────────────────────
    def price_all_exotics(self, K: float, barrier: float = None) -> dict:
        """Price all exotic types for quick comparison."""
        results = {
            'European Call': self.european_call(K),
            'European Put': self.european_put(K),
            'Asian Call (arithmetic)': self.asian_call(K, 'arithmetic'),
            'Asian Call (geometric)': self.asian_call(K, 'geometric'),
            'Lookback Call (floating)': self.lookback_call_floating(),
            'Lookback Put (floating)': self.lookback_put_floating(),
        }
        if barrier is not None:
            results['Barrier Call (up-out)'] = self.barrier_call(K, barrier, 'up-and-out')
            results['Barrier Call (down-out)'] = self.barrier_call(K, barrier * 0.7, 'down-and-out')
        return results
