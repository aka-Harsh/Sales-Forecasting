"""
SARIMAX (Seasonal ARIMA with eXogenous variables) model implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools
import pickle
import os

warnings.filterwarnings('ignore')


class SARIMAXModel:
    """SARIMAX model for time series forecasting"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.fitted_model = None
        self.config = config or {}
        
        # Default parameters
        self.order = self.config.get('order', (1, 1, 1))
        self.seasonal_order = self.config.get('seasonal_order', (1, 1, 1, 12))
        self.auto_tune = self.config.get('auto_tune', True)
        
        # Store best parameters found during auto-tuning
        self.best_params = None
        self.best_aic = float('inf')
        
    def auto_tune_parameters(self, data: pd.DataFrame, 
                           max_order: Tuple[int, int, int] = (3, 2, 3),
                           max_seasonal_order: Tuple[int, int, int] = (2, 1, 2)) -> Dict[str, Any]:
        """
        Automatically tune SARIMAX parameters using grid search
        
        Args:
            data: Time series data with 'sales' column
            max_order: Maximum (p, d, q) values to try
            max_seasonal_order: Maximum seasonal (P, D, Q) values to try
            
        Returns:
            Dictionary with best parameters and AIC score
        """
        try:
            self.logger.info("Starting SARIMAX parameter auto-tuning...")
            
            # Define parameter ranges
            p_values = range(0, max_order[0] + 1)
            d_values = range(0, max_order[1] + 1)
            q_values = range(0, max_order[2] + 1)
            
            P_values = range(0, max_seasonal_order[0] + 1)
            D_values = range(0, max_seasonal_order[1] + 1)
            Q_values = range(0, max_seasonal_order[2] + 1)
            
            seasonal_period = self.seasonal_order[3]
            
            best_aic = float('inf')
            best_params = None
            results = []
            
            # Grid search
            total_combinations = len(p_values) * len(d_values) * len(q_values) * len(P_values) * len(D_values) * len(Q_values)
            current_combination = 0
            
            for p, d, q in itertools.product(p_values, d_values, q_values):
                for P, D, Q in itertools.product(P_values, D_values, Q_values):
                    current_combination += 1
                    
                    try:
                        # Skip if seasonal_period is 1 and seasonal parameters are non-zero
                        if seasonal_period == 1 and (P > 0 or D > 0 or Q > 0):
                            continue
                        
                        order = (p, d, q)
                        seasonal_order = (P, D, Q, seasonal_period)
                        
                        # Fit model
                        model = SARIMAX(
                            data['sales'],
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        
                        fitted_model = model.fit(disp=False, maxiter=200)
                        
                        # Store results
                        result = {
                            'order': order,
                            'seasonal_order': seasonal_order,
                            'aic': fitted_model.aic,
                            'bic': fitted_model.bic,
                            'hqic': fitted_model.hqic
                        }
                        results.append(result)
                        
                        # Check if this is the best model so far
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = {
                                'order': order,
                                'seasonal_order': seasonal_order,
                                'aic': fitted_model.aic,
                                'bic': fitted_model.bic,
                                'hqic': fitted_model.hqic
                            }
                        
                        # Log progress every 10 combinations
                        if current_combination % 10 == 0:
                            progress = (current_combination / total_combinations) * 100
                            self.logger.info(f"Auto-tuning progress: {progress:.1f}% "
                                           f"(Best AIC so far: {best_aic:.2f})")
                    
                    except Exception as e:
                        # Skip problematic parameter combinations
                        continue
            
            if best_params:
                self.best_params = best_params
                self.best_aic = best_aic
                self.order = best_params['order']
                self.seasonal_order = best_params['seasonal_order']
                
                self.logger.info(f"Auto-tuning completed. Best parameters: "
                               f"ARIMA{self.order} x {self.seasonal_order} "
                               f"with AIC: {best_aic:.2f}")
            else:
                self.logger.warning("Auto-tuning failed to find suitable parameters. Using defaults.")
                
            return {
                'best_params': best_params,
                'all_results': results,
                'total_combinations_tried': len(results)
            }
            
        except Exception as e:
            self.logger.error(f"Error during auto-tuning: {str(e)}")
            return {'best_params': None, 'all_results': [], 'total_combinations_tried': 0}
    
    def fit(self, data: pd.DataFrame, exog: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Fit SARIMAX model to data
        
        Args:
            data: Time series data with datetime index and 'sales' column
            exog: Exogenous variables (optional)
            
        Returns:
            Dictionary with fit results
        """
        try:
            self.logger.info("Fitting SARIMAX model...")
            
            # Auto-tune parameters if enabled
            if self.auto_tune and self.best_params is None:
                tuning_results = self.auto_tune_parameters(data)
            
            # Prepare data
            y = data['sales']
            
            # Create and fit model
            self.model = SARIMAX(
                y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.fitted_model = self.model.fit(disp=False, maxiter=500)
            
            # Calculate residuals
            residuals = self.fitted_model.resid
            
            # Model diagnostics
            diagnostics = self._calculate_diagnostics(residuals)
            
            fit_results = {
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                'hqic': float(self.fitted_model.hqic),
                'llf': float(self.fitted_model.llf),
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'diagnostics': diagnostics,
                'summary': str(self.fitted_model.summary())
            }
            
            self.logger.info(f"SARIMAX model fitted successfully. AIC: {fit_results['aic']:.2f}")
            return fit_results
            
        except Exception as e:
            self.logger.error(f"Error fitting SARIMAX model: {str(e)}")
            raise
    
    def predict(self, steps: int, exog: pd.DataFrame = None, 
                return_conf_int: bool = True, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Make predictions using fitted SARIMAX model
        
        Args:
            steps: Number of steps to forecast
            exog: Future exogenous variables
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        try:
            if self.fitted_model is None:
                raise ValueError("Model must be fitted before making predictions")
            
            # Make forecast
            forecast_result = self.fitted_model.get_forecast(
                steps=steps,
                exog=exog,
                alpha=alpha
            )
            
            predictions = forecast_result.predicted_mean
            
            result = {
                'predictions': predictions.values.tolist(),
                'dates': predictions.index.strftime('%Y-%m-%d').tolist()
            }
            
            if return_conf_int:
                conf_int = forecast_result.conf_int()
                result['confidence_intervals'] = {
                    'lower': conf_int.iloc[:, 0].values.tolist(),
                    'upper': conf_int.iloc[:, 1].values.tolist()
                }
            
            # Calculate prediction intervals (wider than confidence intervals)
            pred_int = forecast_result.summary_frame(alpha=alpha)
            result['prediction_intervals'] = {
                'lower': pred_int['mean_ci_lower'].values.tolist(),
                'upper': pred_int['mean_ci_upper'].values.tolist()
            }
            
            self.logger.info(f"Generated {steps} predictions")
            return result
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def decompose_series(self, data: pd.DataFrame, model: str = 'additive') -> Dict[str, Any]:
        """
        Perform seasonal decomposition of the time series
        
        Args:
            data: Time series data
            model: Decomposition model ('additive' or 'multiplicative')
            
        Returns:
            Dictionary with decomposition components
        """
        try:
            # Perform decomposition
            decomposition = seasonal_decompose(
                data['sales'], 
                model=model, 
                period=self.seasonal_order[3] if len(data) >= 2 * self.seasonal_order[3] else None
            )
            
            result = {
                'trend': decomposition.trend.dropna().values.tolist(),
                'seasonal': decomposition.seasonal.dropna().values.tolist(),
                'residual': decomposition.resid.dropna().values.tolist(),
                'observed': decomposition.observed.dropna().values.tolist(),
                'dates': decomposition.trend.dropna().index.strftime('%Y-%m-%d').tolist()
            }
            
            self.logger.info("Seasonal decomposition completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in seasonal decomposition: {str(e)}")
            raise
    
    def _calculate_diagnostics(self, residuals: pd.Series) -> Dict[str, Any]:
        """
        Calculate model diagnostics
        
        Args:
            residuals: Model residuals
            
        Returns:
            Dictionary with diagnostic statistics
        """
        try:
            diagnostics = {}
            
            # Ljung-Box test for autocorrelation
            lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=10, return_df=False)
            diagnostics['ljung_box'] = {
                'statistic': float(lb_stat.iloc[-1]) if hasattr(lb_stat, 'iloc') else float(lb_stat),
                'p_value': float(lb_pvalue.iloc[-1]) if hasattr(lb_pvalue, 'iloc') else float(lb_pvalue),
                'is_white_noise': float(lb_pvalue.iloc[-1] if hasattr(lb_pvalue, 'iloc') else lb_pvalue) > 0.05
            }
            
            # Residual statistics
            diagnostics['residuals'] = {
                'mean': float(residuals.mean()),
                'std': float(residuals.std()),
                'min': float(residuals.min()),
                'max': float(residuals.max()),
                'skewness': float(residuals.skew()),
                'kurtosis': float(residuals.kurtosis())
            }
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Error calculating diagnostics: {str(e)}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """
        Save fitted model to file
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.fitted_model is None:
                raise ValueError("No fitted model to save")
            
            model_data = {
                'fitted_model': self.fitted_model,
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'config': self.config,
                'best_params': self.best_params
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load fitted model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.fitted_model = model_data['fitted_model']
            self.order = model_data['order']
            self.seasonal_order = model_data['seasonal_order']
            self.config = model_data.get('config', {})
            self.best_params = model_data.get('best_params')
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model
        
        Returns:
            Dictionary with model information
        """
        if self.fitted_model is None:
            return {'status': 'not_fitted'}
        
        info = {
            'status': 'fitted',
            'model_type': 'SARIMAX',
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': float(self.fitted_model.aic),
            'bic': float(self.fitted_model.bic),
            'parameters': self.fitted_model.params.to_dict(),
            'auto_tuned': self.best_params is not None
        }
        
        if self.best_params:
            info['tuning_results'] = self.best_params
        
        return info