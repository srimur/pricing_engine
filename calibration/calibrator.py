"""
Model Calibration Engine
=========================
Calibrates model parameters by minimizing the distance between
model-generated option prices and observed market prices.

Supports three optimization methods:
    1. Brute-force grid search (for initial parameter guess)
    2. Nelder-Mead (derivative-free, robust for noisy objective)
    3. L-BFGS-B (gradient-based, fast convergence, handles bounds)

Objective functions:
    - Price RMSE: sqrt(mean((model_price - market_price)²))
    - IV RMSE: sqrt(mean((model_iv - market_iv)²))
    - Weighted: weight by vega to emphasize ATM options
"""

import numpy as np
from scipy.optimize import minimize, brute
from pricing.fft_pricer import FFTPricer
from models.base_model import BaseModel


class Calibrator:
    """
    Model calibration engine.

    Fits model parameters to market option prices using FFT pricing
    and numerical optimization.
    """

    def __init__(self, model: BaseModel, market_strikes: np.ndarray,
                 market_prices: np.ndarray, market_ivs: np.ndarray = None,
                 weights: np.ndarray = None):
        """
        Parameters
        ----------
        model          : BaseModel – Model to calibrate
        market_strikes : np.ndarray – Observed strike prices
        market_prices  : np.ndarray – Observed call prices (mid)
        market_ivs     : np.ndarray – Observed implied vols (optional)
        weights        : np.ndarray – Weights for each observation (optional)
        """
        self.model = model
        self.market_strikes = np.asarray(market_strikes)
        self.market_prices = np.asarray(market_prices)
        self.market_ivs = market_ivs
        self.weights = weights if weights is not None else np.ones(len(market_strikes))

        # Normalize weights
        self.weights = self.weights / self.weights.sum()

        # Track optimization history
        self.history = []

    def _objective_price_rmse(self, params: np.ndarray) -> float:
        """Objective: weighted RMSE of call prices."""
        self._set_params(params)

        try:
            pricer = FFTPricer(self.model)
            model_prices = pricer.price_at_strikes(self.market_strikes)

            if np.any(np.isnan(model_prices)):
                return 1e10

            errors = (model_prices - self.market_prices)**2
            wmse = np.sum(self.weights * errors)
            rmse = np.sqrt(wmse)

            self.history.append({'params': params.copy(), 'rmse': rmse})
            return rmse

        except Exception:
            return 1e10

    def _objective_iv_rmse(self, params: np.ndarray) -> float:
        """Objective: RMSE of implied volatilities."""
        if self.market_ivs is None:
            raise ValueError("Market IVs not provided")

        self._set_params(params)

        try:
            pricer = FFTPricer(self.model)
            model_prices = pricer.price_at_strikes(self.market_strikes)

            from models.black_scholes import BlackScholesModel
            model_ivs = np.array([
                BlackScholesModel.implied_vol(
                    p, self.model.S, K, self.model.T, self.model.r,
                    self.model.q, 'call'
                )
                for p, K in zip(model_prices, self.market_strikes)
                if not np.isnan(p) and p > 0
            ])

            if len(model_ivs) != len(self.market_ivs):
                return 1e10

            errors = (model_ivs - self.market_ivs)**2
            return np.sqrt(np.mean(errors))

        except Exception:
            return 1e10

    def _set_params(self, params: np.ndarray):
        """Set model parameters from optimization vector."""
        param_names = list(self.model.get_params().keys())
        param_dict = {name: val for name, val in zip(param_names, params)}
        self.model.set_params(**param_dict)

    # ──────────────────────────────────────────────
    #  Optimization Methods
    # ──────────────────────────────────────────────

    def calibrate_brute_force(self, n_grid: int = 5,
                               objective: str = 'price') -> dict:
        """
        Brute-force grid search for initial parameter estimation.

        Evaluates objective on a grid over all parameter combinations.
        Slow but guarantees finding the global region.

        Parameters
        ----------
        n_grid    : int – Number of grid points per dimension
        objective : str – 'price' or 'iv'
        """
        bounds = self.model.param_bounds()
        ranges = [slice(lo, hi, complex(0, n_grid)) for lo, hi in bounds]

        obj_fn = self._objective_price_rmse if objective == 'price' \
                 else self._objective_iv_rmse

        self.history = []
        result = brute(obj_fn, ranges, finish=None)

        self._set_params(result)

        return {
            'method': 'brute_force',
            'optimal_params': self.model.get_params(),
            'objective_value': obj_fn(result),
            'grid_points': n_grid**len(bounds),
        }

    def calibrate_nelder_mead(self, x0: np.ndarray = None,
                               objective: str = 'price',
                               max_iter: int = 1000) -> dict:
        """
        Nelder-Mead simplex optimization (derivative-free).

        Good for:
        - Noisy objective functions
        - Non-smooth surfaces
        - When gradients are expensive/unavailable

        Parameters
        ----------
        x0        : np.ndarray – Initial parameter guess
        objective : str – 'price' or 'iv'
        max_iter  : int – Maximum iterations
        """
        if x0 is None:
            # Start from midpoint of bounds
            bounds = self.model.param_bounds()
            x0 = np.array([(lo + hi) / 2 for lo, hi in bounds])

        obj_fn = self._objective_price_rmse if objective == 'price' \
                 else self._objective_iv_rmse

        self.history = []
        result = minimize(
            obj_fn, x0, method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8}
        )

        self._set_params(result.x)

        return {
            'method': 'nelder_mead',
            'optimal_params': self.model.get_params(),
            'objective_value': result.fun,
            'n_iterations': result.nit,
            'n_function_evals': result.nfev,
            'converged': result.success,
        }

    def calibrate_bfgs(self, x0: np.ndarray = None,
                        objective: str = 'price',
                        max_iter: int = 500) -> dict:
        """
        L-BFGS-B optimization (gradient-based, handles bounds).

        Fastest method when the objective is smooth.
        Uses numerical gradients (finite differences).

        Parameters
        ----------
        x0        : np.ndarray – Initial parameter guess
        objective : str – 'price' or 'iv'
        max_iter  : int – Maximum iterations
        """
        bounds = self.model.param_bounds()

        if x0 is None:
            x0 = np.array([(lo + hi) / 2 for lo, hi in bounds])

        obj_fn = self._objective_price_rmse if objective == 'price' \
                 else self._objective_iv_rmse

        self.history = []
        result = minimize(
            obj_fn, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )

        self._set_params(result.x)

        return {
            'method': 'L-BFGS-B',
            'optimal_params': self.model.get_params(),
            'objective_value': result.fun,
            'n_iterations': result.nit,
            'n_function_evals': result.nfev,
            'converged': result.success,
        }

    def calibrate_full_pipeline(self, objective: str = 'price') -> dict:
        """
        Full calibration pipeline:
        1. Brute-force for initial guess (coarse)
        2. Nelder-Mead for refinement (robust)
        3. BFGS for final polish (precise)
        """
        print("Step 1/3: Brute-force grid search for initial guess...")
        brute_result = self.calibrate_brute_force(n_grid=4, objective=objective)
        x0_brute = np.array(list(brute_result['optimal_params'].values()))
        print(f"  → RMSE = {brute_result['objective_value']:.6f}")

        print("Step 2/3: Nelder-Mead refinement...")
        nm_result = self.calibrate_nelder_mead(x0=x0_brute, objective=objective)
        x0_nm = np.array(list(nm_result['optimal_params'].values()))
        print(f"  → RMSE = {nm_result['objective_value']:.6f}")

        print("Step 3/3: L-BFGS-B final polish...")
        bfgs_result = self.calibrate_bfgs(x0=x0_nm, objective=objective)
        print(f"  → RMSE = {bfgs_result['objective_value']:.6f}")

        return {
            'brute_force': brute_result,
            'nelder_mead': nm_result,
            'bfgs': bfgs_result,
            'final_params': bfgs_result['optimal_params'],
            'final_rmse': bfgs_result['objective_value'],
        }
