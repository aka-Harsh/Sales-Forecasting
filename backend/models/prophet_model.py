"""
Facebook Prophet model implementation for sales forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pickle
import os

warnings.filterwarnings('ignore')


class ProphetModel:
    """Facebook Prophet model for time series forecasting"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.config = config or {}
        
        # Default parameters
        self.growth = self.config.get('growth', 'linear')
        self.seasonality_mode = self.config.get('seasonality_mode', 'additive')
        self.yearly_seasonality = self.config.get('yearly_seasonality', True)
        self.weekly_seasonality = self.config.get('weekly_seasonality', False)
        self.daily_seasonality = self.config.get('daily_seasonality', False)
        
        # Custom parameters
        self.changepoint_prior_scale = self.config.get('changepoint_prior_scale', 0.05)
        self.seasonality_prior_scale = self.config.get('seasonality_prior_scale', 10.0)
        self.holidays_prior_scale = self.config.get('holidays_prior_scale', 10.0)
        
        # Store training data for diagnostics
        self.training_data = None
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet model
        
        Args:
            data: Time series data with datetime index and 'sales' column
            
        Returns:
            DataFrame in Prophet format (ds, y columns)
        """
        try:
            # Prophet requires 'ds' (datestamp) and 'y' (value) columns
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data['sales'].values
            })
            
            # Remove any missing values
            prophet_data = prophet_data.dropna()
            
            self.logger.info(f"Prepared data for Prophet: {len(prophet_data)} records")
            return prophet_data
            
        except Exception as e:
            self.logger.error(f"Error preparing data for Prophet: {str(e)}")
            raise
    
    def add_holidays(self, country: str = 'US') -> pd.DataFrame:
        """
        Add holiday effects for specified country
        
        Args:
            country: Country code for holidays
            
        Returns:
            DataFrame with holidays
        """
        try:
            from prophet.make_holidays import make_holidays_df
            
            # Get holidays for the specified country
            holidays = make_holidays_df(
                year_list=range(2010, 2030),  # Extended range
                country=country
            )
            
            self.logger.info(f"Added {len(holidays)} holidays for country: {country}")
            return holidays
            
        except Exception as e:
            self.logger.warning(f"Could not add holidays for {country}: {str(e)}")
            return pd.DataFrame()
    
    def add_custom_seasonality(self, name: str, period: float, fourier_order: int = 3):
        """
        Add custom seasonality to the model
        
        Args:
            name: Name of the seasonality
            period: Period of the seasonality in days
            fourier_order: Number of Fourier terms to use
        """
        try:
            if self.model is not None:
                self.model.add_seasonality(
                    name=name,
                    period=period,
                    fourier_order=fourier_order
                )
                self.logger.info(f"Added custom seasonality: {name} (period: {period} days)")
            else:
                self.logger.warning("Model not initialized. Cannot add seasonality.")
                
        except Exception as e:
            self.logger.error(f"Error adding custom seasonality: {str(e)}")
    
    def fit(self, data: pd.DataFrame, holidays: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Fit Prophet model to data
        
        Args:
            data: Time series data with datetime index and 'sales' column
            holidays: Optional holidays DataFrame
            
        Returns:
            Dictionary with fit results
        """
        try:
            self.logger.info("Fitting Prophet model...")
            
            # Prepare data
            prophet_data = self.prepare_data(data)
            self.training_data = prophet_data.copy()
            
            # Initialize Prophet model
            self.model = Prophet(
                growth=self.growth,
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                holidays=holidays
            )
            
            # Add custom seasonalities for monthly data
            if len(prophet_data) >= 24:  # Need at least 2 years of monthly data
                self.add_custom_seasonality('monthly', 30.5, 5)
                self.add_custom_seasonality('quarterly', 91.25, 4)
            
            # Fit the model
            self.model.fit(prophet_data)
            
            # Calculate fit statistics
            fit_results = self._calculate_fit_statistics(prophet_data)
            
            self.logger.info("Prophet model fitted successfully")
            return fit_results
            
        except Exception as e:
            self.logger.error(f"Error fitting Prophet model: {str(e)}")
            raise
    
    def predict(self, steps: int, freq: str = 'M') -> Dict[str, Any]:
        """
        Make predictions using fitted Prophet model
        
        Args:
            steps: Number of steps to forecast
            freq: Frequency of predictions ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            Dictionary with predictions and components
        """
        try:
            if self.model is None:
                raise ValueError("Model must be fitted before making predictions")
            
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=steps, freq=freq)
            
            # Make forecast
            forecast = self.model.predict(future)
            
            # Extract predictions (only future values)
            n_historical = len(self.training_data)
            future_forecast = forecast.iloc[n_historical:]
            
            result = {
                'predictions': future_forecast['yhat'].values.tolist(),
                'dates': future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'confidence_intervals': {
                    'lower': future_forecast['yhat_lower'].values.tolist(),
                    'upper': future_forecast['yhat_upper'].values.tolist()
                },
                'components': {
                    'trend': future_forecast['trend'].values.tolist(),
                    'seasonal': (future_forecast['yearly'] + 
                               future_forecast.get('monthly', 0) +
                               future_forecast.get('quarterly', 0)).values.tolist()
                }
            }
            
            # Add weekly component if it exists
            if 'weekly' in future_forecast.columns:
                weekly_component = future_forecast['weekly'].values.tolist()
                result['components']['weekly'] = weekly_component
            
            # Add holiday component if it exists
            if 'holidays' in future_forecast.columns:
                holiday_component = future_forecast['holidays'].values.tolist()
                result['components']['holidays'] = holiday_component
            
            self.logger.info(f"Generated {steps} predictions")
            return result
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def cross_validate_model(self, initial: str = '730 days', period: str = '180 days', 
                           horizon: str = '365 days') -> Dict[str, Any]:
        """
        Perform cross-validation on the Prophet model
        
        Args:
            initial: Initial training period
            period: Period between cutoff dates
            horizon: Forecast horizon
            
        Returns:
            Dictionary with cross-validation results
        """
        try:
            if self.model is None or self.training_data is None:
                raise ValueError("Model must be fitted before cross-validation")
            
            # Perform cross-validation
            cv_results = cross_validation(
                self.model,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel="processes"
            )
            
            # Calculate performance metrics
            metrics = performance_metrics(cv_results)
            
            result = {
                'cv_results': {
                    'mae': cv_results.groupby('cutoff')['mae'].mean().tolist(),
                    'rmse': cv_results.groupby('cutoff')['rmse'].mean().tolist(),
                    'mape': cv_results.groupby('cutoff')['mape'].mean().tolist(),
                    'cutoffs': cv_results['cutoff'].unique().astype(str).tolist()
                },
                'average_metrics': {
                    'mae': float(metrics['mae'].mean()),
                    'rmse': float(metrics['rmse'].mean()),
                    'mape': float(metrics['mape'].mean()),
                    'smape': float(metrics['smape'].mean())
                }
            }
            
            self.logger.info("Cross-validation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            return {'cv_results': {}, 'average_metrics': {}}
    
    def detect_changepoints(self) -> Dict[str, Any]:
        """
        Detect and analyze changepoints in the time series
        
        Returns:
            Dictionary with changepoint information
        """
        try:
            if self.model is None:
                raise ValueError("Model must be fitted before detecting changepoints")
            
            # Get changepoints
            changepoints = self.model.changepoints
            
            # Get changepoint deltas (magnitude of changes)
            changepoint_deltas = self.model.params['delta'].values
            
            # Identify significant changepoints (above threshold)
            threshold = np.std(changepoint_deltas) * 1.5
            significant_changepoints = changepoints[np.abs(changepoint_deltas) > threshold]
            
            result = {
                'all_changepoints': changepoints.strftime('%Y-%m-%d').tolist(),
                'significant_changepoints': significant_changepoints.strftime('%Y-%m-%d').tolist(),
                'changepoint_deltas': changepoint_deltas.tolist(),
                'n_changepoints': len(changepoints),
                'n_significant': len(significant_changepoints)
            }
            
            self.logger.info(f"Detected {len(significant_changepoints)} significant changepoints")
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting changepoints: {str(e)}")
            return {}
    
    def get_component_importance(self) -> Dict[str, float]:
        """
        Calculate the relative importance of different components
        
        Returns:
            Dictionary with component importance scores
        """
        try:
            if self.model is None or self.training_data is None:
                raise ValueError("Model must be fitted to calculate component importance")
            
            # Make predictions on training data
            forecast = self.model.predict(self.training_data)
            
            # Calculate variance of each component
            components = ['trend', 'yearly']
            if 'weekly' in forecast.columns:
                components.append('weekly')
            if 'monthly' in forecast.columns:
                components.append('monthly')
            if 'quarterly' in forecast.columns:
                components.append('quarterly')
            if 'holidays' in forecast.columns:
                components.append('holidays')
            
            importance = {}
            total_variance = 0
            
            for component in components:
                if component in forecast.columns:
                    variance = forecast[component].var()
                    importance[component] = variance
                    total_variance += variance
            
            # Normalize to percentages
            if total_variance > 0:
                for component in importance:
                    importance[component] = (importance[component] / total_variance) * 100
            
            self.logger.info("Calculated component importance")
            return importance
            
        except Exception as e:
            self.logger.error(f"Error calculating component importance: {str(e)}")
            return {}
    
    def _calculate_fit_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate fit statistics for the model
        
        Args:
            data: Training data
            
        Returns:
            Dictionary with fit statistics
        """
        try:
            # Make predictions on training data
            forecast = self.model.predict(data)
            
            # Calculate residuals
            residuals = data['y'].values - forecast['yhat'].values
            
            # Calculate statistics
            stats = {
                'mean_residual': float(np.mean(residuals)),
                'std_residual': float(np.std(residuals)),
                'mae': float(np.mean(np.abs(residuals))),
                'rmse': float(np.sqrt(np.mean(residuals**2))),
                'mape': float(np.mean(np.abs(residuals / data['y'].values)) * 100),
                'r2': float(1 - np.sum(residuals**2) / np.sum((data['y'] - np.mean(data['y']))**2))
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating fit statistics: {str(e)}")
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
            if self.model is None:
                raise ValueError("No fitted model to save")
            
            model_data = {
                'model': self.model,
                'config': self.config,
                'training_data': self.training_data
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Prophet model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving Prophet model: {str(e)}")
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
            
            self.model = model_data['model']
            self.config = model_data.get('config', {})
            self.training_data = model_data.get('training_data')
            
            self.logger.info(f"Prophet model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading Prophet model: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {'status': 'not_fitted'}
        
        info = {
            'status': 'fitted',
            'model_type': 'Prophet',
            'growth': self.growth,
            'seasonality_mode': self.seasonality_mode,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'n_changepoints': len(self.model.changepoints) if hasattr(self.model, 'changepoints') else 0
        }
        
        # Add seasonality information
        if hasattr(self.model, 'seasonalities'):
            info['custom_seasonalities'] = list(self.model.seasonalities.keys())
        
        return info