"""
Ensemble model combining multiple forecasting approaches
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
import pickle
import os

# Import individual models
from .sarimax_model import SARIMAXModel
from .prophet_model import ProphetModel
from .lstm_model import LSTMModel
from .linear_model import LinearModel

warnings.filterwarnings('ignore')


class EnsembleModel:
    """Ensemble model combining SARIMAX, Prophet, LSTM, and Linear Regression"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Individual models
        self.models = {}
        self.fitted_models = {}
        
        # Ensemble configuration
        self.method = self.config.get('method', 'weighted_average')  # 'weighted_average', 'simple_average', 'stacking'
        self.weights = self.config.get('weights', 'auto')  # 'auto' or list of weights
        
        # Model weights (will be calculated or set)
        self.model_weights = {}
        
        # Performance tracking
        self.individual_performance = {}
        self.ensemble_performance = {}
        
        # Training data
        self.training_data = None
        self.is_fitted = False
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize individual models with their configurations
        
        Returns:
            Dictionary with initialization results
        """
        try:
            model_configs = self.config.get('model_configs', {})
            
            # Initialize SARIMAX
            sarimax_config = model_configs.get('sarimax', {})
            self.models['sarimax'] = SARIMAXModel(sarimax_config)
            
            # Initialize Prophet
            prophet_config = model_configs.get('prophet', {})
            self.models['prophet'] = ProphetModel(prophet_config)
            
            # Initialize LSTM
            lstm_config = model_configs.get('lstm', {})
            self.models['lstm'] = LSTMModel(lstm_config)
            
            # Initialize Linear Regression
            linear_config = model_configs.get('linear', {})
            self.models['linear'] = LinearModel(linear_config)
            
            self.logger.info("All individual models initialized successfully")
            return {'status': 'success', 'models_initialized': list(self.models.keys())}
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def fit_individual_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit all individual models
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with fitting results for each model
        """
        try:
            self.training_data = data.copy()
            fit_results = {}
            
            # Fit each model
            for model_name, model in self.models.items():
                try:
                    self.logger.info(f"Fitting {model_name} model...")
                    
                    # Fit model
                    model_result = model.fit(data)
                    self.fitted_models[model_name] = model
                    fit_results[model_name] = {
                        'status': 'success',
                        'results': model_result
                    }
                    
                    # Calculate individual model performance for weight calculation
                    self._evaluate_individual_model(model_name, model, data)
                    
                except Exception as model_error:
                    self.logger.error(f"Error fitting {model_name}: {str(model_error)}")
                    fit_results[model_name] = {
                        'status': 'failed',
                        'error': str(model_error)
                    }
            
            # Calculate ensemble weights
            if len(self.fitted_models) > 0:
                self._calculate_weights()
                self.is_fitted = True
            
            self.logger.info(f"Ensemble model fitted with {len(self.fitted_models)} individual models")
            return fit_results
            
        except Exception as e:
            self.logger.error(f"Error fitting ensemble model: {str(e)}")
            raise
    
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit the ensemble model
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with ensemble fit results
        """
        try:
            # Initialize models if not already done
            if not self.models:
                self.initialize_models()
            
            # Fit individual models
            individual_results = self.fit_individual_models(data)
            
            # Prepare ensemble results
            ensemble_results = {
                'individual_models': individual_results,
                'ensemble_weights': self.model_weights,
                'ensemble_method': self.method,
                'n_successful_models': len(self.fitted_models),
                'model_performance': self.individual_performance
            }
            
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"Error fitting ensemble: {str(e)}")
            raise
    
    def predict(self, steps: int) -> Dict[str, Any]:
        """
        Make ensemble predictions
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Dictionary with ensemble predictions
        """
        try:
            if not self.is_fitted:
                raise ValueError("Ensemble model must be fitted before making predictions")
            
            individual_predictions = {}
            individual_dates = {}
            individual_intervals = {}
            
            # Get predictions from each fitted model
            for model_name, model in self.fitted_models.items():
                try:
                    pred_result = model.predict(steps)
                    individual_predictions[model_name] = pred_result['predictions']
                    individual_dates[model_name] = pred_result['dates']
                    
                    # Store confidence intervals if available
                    if 'confidence_intervals' in pred_result:
                        individual_intervals[model_name] = pred_result['confidence_intervals']
                    
                except Exception as model_error:
                    self.logger.warning(f"Error getting predictions from {model_name}: {str(model_error)}")
                    continue
            
            if not individual_predictions:
                raise ValueError("No individual models produced valid predictions")
            
            # Combine predictions using ensemble method
            if self.method == 'simple_average':
                ensemble_predictions = self._simple_average(individual_predictions)
            elif self.method == 'weighted_average':
                ensemble_predictions = self._weighted_average(individual_predictions)
            else:
                raise ValueError(f"Unsupported ensemble method: {self.method}")
            
            # Use dates from the first successful model
            first_model = list(individual_dates.keys())[0]
            ensemble_dates = individual_dates[first_model]
            
            # Calculate ensemble confidence intervals
            ensemble_intervals = self._calculate_ensemble_intervals(
                individual_predictions, individual_intervals
            )
            
            result = {
                'predictions': ensemble_predictions,
                'dates': ensemble_dates,
                'confidence_intervals': ensemble_intervals,
                'individual_predictions': individual_predictions,
                'model_weights': self.model_weights,
                'contributing_models': list(individual_predictions.keys())
            }
            
            self.logger.info(f"Generated ensemble predictions using {len(individual_predictions)} models")
            return result
            
        except Exception as e:
            self.logger.error(f"Error making ensemble predictions: {str(e)}")
            raise
    
    def _evaluate_individual_model(self, model_name: str, model: Any, data: pd.DataFrame):
        """
        Evaluate individual model performance for weight calculation
        
        Args:
            model_name: Name of the model
            model: Fitted model instance
            data: Training data
        """
        try:
            # Simple evaluation using a portion of training data
            split_point = int(len(data) * 0.8)
            train_data = data.iloc[:split_point]
            test_data = data.iloc[split_point:]
            
            if len(test_data) < 3:  # Need at least 3 points for evaluation
                # Use simple metrics if not enough test data
                self.individual_performance[model_name] = {'mae': 1.0, 'rmse': 1.0, 'mape': 10.0}
                return
            
            # Make predictions on test data
            test_steps = len(test_data)
            
            # Re-fit model on train data for fair evaluation
            temp_model = type(model)(model.config if hasattr(model, 'config') else {})
            temp_model.fit(train_data)
            pred_result = temp_model.predict(test_steps)
            
            # Calculate metrics
            actual = test_data['sales'].values
            predicted = np.array(pred_result['predictions'])
            
            # Ensure same length
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]
            
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            self.individual_performance[model_name] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape)
            }
            
        except Exception as e:
            self.logger.warning(f"Could not evaluate {model_name}: {str(e)}")
            # Default performance values
            self.individual_performance[model_name] = {'mae': 1.0, 'rmse': 1.0, 'mape': 10.0}
    
    def _calculate_weights(self):
        """
        Calculate weights for ensemble based on individual model performance
        """
        try:
            if self.weights != 'auto':
                # Use provided weights
                if isinstance(self.weights, dict):
                    self.model_weights = self.weights
                elif isinstance(self.weights, list):
                    model_names = list(self.fitted_models.keys())
                    self.model_weights = dict(zip(model_names, self.weights))
                else:
                    # Equal weights
                    n_models = len(self.fitted_models)
                    weight = 1.0 / n_models
                    self.model_weights = {name: weight for name in self.fitted_models.keys()}
                return
            
            # Calculate weights based on inverse of error (lower error = higher weight)
            weights = {}
            
            for model_name in self.fitted_models.keys():
                if model_name in self.individual_performance:
                    # Use MAPE as the primary metric for weighting
                    mape = self.individual_performance[model_name]['mape']
                    # Inverse weight (add small constant to avoid division by zero)
                    weights[model_name] = 1.0 / (mape + 1.0)
                else:
                    weights[model_name] = 1.0  # Default weight
            
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                self.model_weights = {name: weight / total_weight for name, weight in weights.items()}
            else:
                # Equal weights as fallback
                n_models = len(self.fitted_models)
                weight = 1.0 / n_models
                self.model_weights = {name: weight for name in self.fitted_models.keys()}
            
            self.logger.info(f"Calculated ensemble weights: {self.model_weights}")
            
        except Exception as e:
            self.logger.error(f"Error calculating weights: {str(e)}")
            # Use equal weights as fallback
            n_models = len(self.fitted_models)
            weight = 1.0 / n_models
            self.model_weights = {name: weight for name in self.fitted_models.keys()}
    
    def _simple_average(self, predictions: Dict[str, List[float]]) -> List[float]:
        """
        Calculate simple average of predictions
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            List of averaged predictions
        """
        prediction_arrays = [np.array(pred) for pred in predictions.values()]
        
        # Ensure all predictions have the same length
        min_length = min(len(pred) for pred in prediction_arrays)
        prediction_arrays = [pred[:min_length] for pred in prediction_arrays]
        
        # Calculate simple average
        ensemble_pred = np.mean(prediction_arrays, axis=0)
        return ensemble_pred.tolist()
    
    def _weighted_average(self, predictions: Dict[str, List[float]]) -> List[float]:
        """
        Calculate weighted average of predictions
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            List of weighted averaged predictions
        """
        prediction_arrays = []
        weights = []
        
        for model_name, pred in predictions.items():
            if model_name in self.model_weights:
                prediction_arrays.append(np.array(pred))
                weights.append(self.model_weights[model_name])
        
        if not prediction_arrays:
            raise ValueError("No valid predictions for weighted average")
        
        # Ensure all predictions have the same length
        min_length = min(len(pred) for pred in prediction_arrays)
        prediction_arrays = [pred[:min_length] for pred in prediction_arrays]
        weights = np.array(weights)
        
        # Calculate weighted average
        weighted_sum = np.sum([w * pred for w, pred in zip(weights, prediction_arrays)], axis=0)
        ensemble_pred = weighted_sum / np.sum(weights)
        
        return ensemble_pred.tolist()
    
    def _calculate_ensemble_intervals(self, predictions: Dict[str, List[float]], 
                                    intervals: Dict[str, Dict[str, List[float]]]) -> Dict[str, List[float]]:
        """
        Calculate ensemble confidence intervals
        
        Args:
            predictions: Individual model predictions
            intervals: Individual model confidence intervals
            
        Returns:
            Dictionary with ensemble confidence intervals
        """
        try:
            if not intervals:
                # Fallback: use prediction variance as uncertainty measure
                pred_arrays = [np.array(pred) for pred in predictions.values()]
                min_length = min(len(pred) for pred in pred_arrays)
                pred_arrays = [pred[:min_length] for pred in pred_arrays]
                
                pred_std = np.std(pred_arrays, axis=0)
                ensemble_pred = np.mean(pred_arrays, axis=0)
                
                return {
                    'lower': (ensemble_pred - 1.96 * pred_std).tolist(),
                    'upper': (ensemble_pred + 1.96 * pred_std).tolist()
                }
            
            # Combine confidence intervals using weights
            lower_bounds = []
            upper_bounds = []
            weights = []
            
            for model_name, interval in intervals.items():
                if model_name in self.model_weights and 'lower' in interval and 'upper' in interval:
                    lower_bounds.append(np.array(interval['lower']))
                    upper_bounds.append(np.array(interval['upper']))
                    weights.append(self.model_weights[model_name])
            
            if not lower_bounds:
                # Fallback to prediction variance
                return self._calculate_ensemble_intervals(predictions, {})
            
            # Ensure same length
            min_length = min(len(bound) for bound in lower_bounds)
            lower_bounds = [bound[:min_length] for bound in lower_bounds]
            upper_bounds = [bound[:min_length] for bound in upper_bounds]
            weights = np.array(weights)
            
            # Weighted average of bounds
            ensemble_lower = np.sum([w * bound for w, bound in zip(weights, lower_bounds)], axis=0) / np.sum(weights)
            ensemble_upper = np.sum([w * bound for w, bound in zip(weights, upper_bounds)], axis=0) / np.sum(weights)
            
            return {
                'lower': ensemble_lower.tolist(),
                'upper': ensemble_upper.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble intervals: {str(e)}")
            # Return simple ±20% intervals as fallback
            ensemble_pred = self._weighted_average(predictions)
            ensemble_array = np.array(ensemble_pred)
            return {
                'lower': (ensemble_array * 0.8).tolist(),
                'upper': (ensemble_array * 1.2).tolist()
            }
    
    def get_model_contributions(self) -> Dict[str, Any]:
        """
        Get information about individual model contributions to the ensemble
        
        Returns:
            Dictionary with model contribution information
        """
        return {
            'model_weights': self.model_weights,
            'individual_performance': self.individual_performance,
            'fitted_models': list(self.fitted_models.keys()),
            'ensemble_method': self.method,
            'total_models': len(self.models),
            'successful_models': len(self.fitted_models)
        }
    
    def save_model(self, filepath: str) -> bool:
        """
        Save ensemble model to file
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_fitted:
                raise ValueError("No fitted ensemble model to save")
            
            # Save individual models separately
            model_dir = os.path.dirname(filepath)
            individual_model_paths = {}
            
            for model_name, model in self.fitted_models.items():
                model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
                if model.save_model(model_path):
                    individual_model_paths[model_name] = model_path
            
            # Save ensemble metadata
            ensemble_data = {
                'config': self.config,
                'model_weights': self.model_weights,
                'individual_performance': self.individual_performance,
                'method': self.method,
                'individual_model_paths': individual_model_paths,
                'is_fitted': self.is_fitted,
                'training_data': self.training_data
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(ensemble_data, f)
            
            self.logger.info(f"Ensemble model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving ensemble model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load ensemble model from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load ensemble metadata
            with open(filepath, 'rb') as f:
                ensemble_data = pickle.load(f)
            
            self.config = ensemble_data['config']
            self.model_weights = ensemble_data['model_weights']
            self.individual_performance = ensemble_data['individual_performance']
            self.method = ensemble_data['method']
            self.is_fitted = ensemble_data['is_fitted']
            self.training_data = ensemble_data.get('training_data')
            
            # Load individual models
            individual_model_paths = ensemble_data['individual_model_paths']
            self.initialize_models()
            
            for model_name, model_path in individual_model_paths.items():
                if model_name in self.models:
                    if self.models[model_name].load_model(model_path):
                        self.fitted_models[model_name] = self.models[model_name]
            
            self.logger.info(f"Ensemble model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading ensemble model: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the ensemble model
        
        Returns:
            Dictionary with ensemble model information
        """
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        individual_info = {}
        for model_name, model in self.fitted_models.items():
            individual_info[model_name] = model.get_model_info()
        
        return {
            'status': 'fitted',
            'model_type': 'Ensemble',
            'ensemble_method': self.method,
            'n_individual_models': len(self.fitted_models),
            'model_weights': self.model_weights,
            'individual_performance': self.individual_performance,
            'individual_models': individual_info
        }