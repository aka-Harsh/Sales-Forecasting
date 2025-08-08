"""
Model evaluation utilities for sales forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Model evaluation and performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic forecasting metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with basic metrics
        """
        try:
            # Ensure arrays are the same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Remove any NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            if len(y_true) == 0:
                raise ValueError("No valid data points for evaluation")
            
            metrics = {}
            
            # Mean Absolute Error
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            
            # Root Mean Squared Error
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            
            # Mean Squared Error
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            
            # R-squared
            metrics['r2'] = float(r2_score(y_true, y_pred))
            
            # Mean Absolute Percentage Error
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                metrics['mape'] = float(mape) if not np.isnan(mape) else 100.0
            
            # Symmetric Mean Absolute Percentage Error
            with np.errstate(divide='ignore', invalid='ignore'):
                smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
                metrics['smape'] = float(smape) if not np.isnan(smape) else 100.0
            
            # Mean Error (Bias)
            metrics['me'] = float(np.mean(y_pred - y_true))
            
            # Mean Percentage Error
            with np.errstate(divide='ignore', invalid='ignore'):
                mpe = np.mean((y_pred - y_true) / y_true) * 100
                metrics['mpe'] = float(mpe) if not np.isnan(mpe) else 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {str(e)}")
            raise
    
    def calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate directional accuracy metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with directional accuracy metrics
        """
        try:
            # Ensure arrays are the same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            if len(y_true) < 2:
                return {'directional_accuracy': 0.0, 'trend_accuracy': 0.0}
            
            # Calculate actual and predicted changes
            actual_changes = np.diff(y_true)
            predicted_changes = np.diff(y_pred)
            
            # Directional accuracy (same sign)
            same_direction = np.sign(actual_changes) == np.sign(predicted_changes)
            directional_accuracy = np.mean(same_direction) * 100
            
            # Trend accuracy (considering magnitude)
            trend_accuracy = 0.0
            if len(actual_changes) > 0:
                # Correlation between actual and predicted changes
                if np.std(actual_changes) > 0 and np.std(predicted_changes) > 0:
                    correlation = np.corrcoef(actual_changes, predicted_changes)[0, 1]
                    trend_accuracy = max(0, correlation) * 100
            
            return {
                'directional_accuracy': float(directional_accuracy),
                'trend_accuracy': float(trend_accuracy)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating directional accuracy: {str(e)}")
            return {'directional_accuracy': 0.0, 'trend_accuracy': 0.0}
    
    def calculate_confidence_intervals(self, y_pred: np.ndarray, residuals: np.ndarray,
                                     confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Calculate confidence intervals for predictions
        
        Args:
            y_pred: Predicted values
            residuals: Model residuals
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with confidence intervals
        """
        try:
            # Calculate standard error of residuals
            std_error = np.std(residuals)
            
            # Calculate t-critical value
            alpha = 1 - confidence_level
            degrees_freedom = len(residuals) - 1
            t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom)
            
            # Calculate margin of error
            margin_error = t_critical * std_error
            
            # Calculate confidence intervals
            lower_bound = y_pred - margin_error
            upper_bound = y_pred + margin_error
            
            return {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'margin_error': np.full_like(y_pred, margin_error)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence intervals: {str(e)}")
            return {
                'lower_bound': y_pred * 0.9,
                'upper_bound': y_pred * 1.1,
                'margin_error': y_pred * 0.1
            }
    
    def calculate_seasonal_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 seasonal_periods: int = 12) -> Dict[str, Any]:
        """
        Calculate seasonal performance metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            seasonal_periods: Number of seasonal periods
            
        Returns:
            Dictionary with seasonal metrics
        """
        try:
            # Ensure arrays are the same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            seasonal_metrics = {}
            
            # Calculate metrics for each season
            for season in range(seasonal_periods):
                # Get indices for this season
                season_indices = np.arange(season, len(y_true), seasonal_periods)
                
                if len(season_indices) > 0:
                    season_true = y_true[season_indices]
                    season_pred = y_pred[season_indices]
                    
                    # Calculate basic metrics for this season
                    season_metrics = self.calculate_basic_metrics(season_true, season_pred)
                    seasonal_metrics[f'season_{season + 1}'] = season_metrics
            
            # Calculate overall seasonal performance
            if seasonal_metrics:
                # Average metrics across seasons
                avg_metrics = {}
                metric_names = ['mae', 'rmse', 'mape', 'r2']
                
                for metric in metric_names:
                    values = [seasonal_metrics[season][metric] 
                             for season in seasonal_metrics.keys() 
                             if metric in seasonal_metrics[season]]
                    if values:
                        avg_metrics[f'avg_seasonal_{metric}'] = np.mean(values)
                        avg_metrics[f'std_seasonal_{metric}'] = np.std(values)
                
                seasonal_metrics['summary'] = avg_metrics
            
            return seasonal_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating seasonal metrics: {str(e)}")
            return {}
    
    def calculate_forecast_bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate forecast bias metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with bias metrics
        """
        try:
            # Ensure arrays are the same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            if len(y_true) == 0:
                return {'bias': 0.0, 'bias_percentage': 0.0, 'forecast_tendency': 'neutral'}
            
            # Calculate bias
            bias = np.mean(y_pred - y_true)
            bias_percentage = (bias / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else 0
            
            # Determine forecast tendency
            if bias > 0:
                tendency = 'overforecasting'
            elif bias < 0:
                tendency = 'underforecasting'
            else:
                tendency = 'neutral'
            
            return {
                'bias': float(bias),
                'bias_percentage': float(bias_percentage),
                'forecast_tendency': tendency
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating forecast bias: {str(e)}")
            return {'bias': 0.0, 'bias_percentage': 0.0, 'forecast_tendency': 'neutral'}
    
    def cross_validate_time_series(self, model, data: pd.DataFrame, 
                                  initial_train_size: int = None,
                                  horizon: int = 1, step: int = 1) -> Dict[str, Any]:
        """
        Perform time series cross-validation
        
        Args:
            model: Model object with fit and predict methods
            data: Time series data
            initial_train_size: Initial training size
            horizon: Forecast horizon
            step: Step size for rolling window
            
        Returns:
            Dictionary with cross-validation results
        """
        try:
            if initial_train_size is None:
                initial_train_size = len(data) // 2
            
            cv_results = {
                'fold_metrics': [],
                'predictions': [],
                'actuals': [],
                'dates': []
            }
            
            # Rolling window cross-validation
            for i in range(initial_train_size, len(data) - horizon + 1, step):
                # Split data
                train_data = data.iloc[:i]
                test_data = data.iloc[i:i + horizon]
                
                try:
                    # Fit model
                    model.fit(train_data)
                    
                    # Make predictions
                    predictions = model.predict(len(test_data))
                    
                    # Calculate metrics for this fold
                    fold_metrics = self.calculate_basic_metrics(
                        test_data['sales'].values, predictions
                    )
                    
                    cv_results['fold_metrics'].append(fold_metrics)
                    cv_results['predictions'].extend(predictions.tolist())
                    cv_results['actuals'].extend(test_data['sales'].tolist())
                    cv_results['dates'].extend(test_data.index.strftime('%Y-%m-%d').tolist())
                    
                except Exception as fold_error:
                    self.logger.warning(f"Error in CV fold {i}: {str(fold_error)}")
                    continue
            
            # Calculate average metrics across folds
            if cv_results['fold_metrics']:
                avg_metrics = {}
                for metric in cv_results['fold_metrics'][0].keys():
                    values = [fold[metric] for fold in cv_results['fold_metrics']]
                    avg_metrics[f'avg_{metric}'] = np.mean(values)
                    avg_metrics[f'std_{metric}'] = np.std(values)
                
                cv_results['average_metrics'] = avg_metrics
                cv_results['n_folds'] = len(cv_results['fold_metrics'])
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            return {'fold_metrics': [], 'average_metrics': {}}
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare performance of multiple models
        
        Args:
            model_results: Dictionary with model names and their results
            
        Returns:
            Dictionary with model comparison results
        """
        try:
            comparison = {
                'model_rankings': {},
                'best_model': {},
                'metrics_comparison': {},
                'summary': {}
            }
            
            # Extract metrics for comparison
            metrics_data = {}
            for model_name, results in model_results.items():
                if 'metrics' in results:
                    metrics_data[model_name] = results['metrics']
            
            if not metrics_data:
                return comparison
            
            # Key metrics for ranking (lower is better except for R²)
            ranking_metrics = ['mae', 'rmse', 'mape']
            
            # Calculate rankings for each metric
            for metric in ranking_metrics:
                metric_values = {}
                for model_name, metrics in metrics_data.items():
                    if metric in metrics:
                        metric_values[model_name] = metrics[metric]
                
                if metric_values:
                    # Sort models by metric (ascending for error metrics)
                    sorted_models = sorted(metric_values.items(), key=lambda x: x[1])
                    comparison['model_rankings'][metric] = [
                        {'model': model, 'value': value, 'rank': rank + 1}
                        for rank, (model, value) in enumerate(sorted_models)
                    ]
            
            # Calculate overall ranking (average rank across metrics)
            if comparison['model_rankings']:
                model_avg_ranks = {}
                for model_name in metrics_data.keys():
                    ranks = []
                    for metric_ranking in comparison['model_rankings'].values():
                        for item in metric_ranking:
                            if item['model'] == model_name:
                                ranks.append(item['rank'])
                                break
                    
                    if ranks:
                        model_avg_ranks[model_name] = np.mean(ranks)
                
                # Find best model (lowest average rank)
                if model_avg_ranks:
                    best_model_name = min(model_avg_ranks.items(), key=lambda x: x[1])[0]
                    comparison['best_model'] = {
                        'name': best_model_name,
                        'average_rank': model_avg_ranks[best_model_name],
                        'metrics': metrics_data[best_model_name]
                    }
            
            # Create metrics comparison table
            comparison_table = {}
            for metric in ['mae', 'rmse', 'mape', 'r2']:
                comparison_table[metric] = {}
                for model_name, metrics in metrics_data.items():
                    if metric in metrics:
                        comparison_table[metric][model_name] = metrics[metric]
            
            comparison['metrics_comparison'] = comparison_table
            
            # Generate summary
            comparison['summary'] = {
                'total_models': len(metrics_data),
                'metrics_evaluated': list(ranking_metrics),
                'best_model': comparison['best_model'].get('name', 'N/A')
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            return {'model_rankings': {}, 'best_model': {}}
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = "Model", 
                                 residuals: np.ndarray = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            residuals: Model residuals (optional)
            
        Returns:
            Comprehensive evaluation report
        """
        try:
            report = {
                'model_name': model_name,
                'data_info': {
                    'n_samples': len(y_true),
                    'date_range': 'N/A'
                },
                'basic_metrics': {},
                'directional_metrics': {},
                'bias_metrics': {},
                'confidence_intervals': {},
                'residual_analysis': {}
            }
            
            # Calculate basic metrics
            report['basic_metrics'] = self.calculate_basic_metrics(y_true, y_pred)
            
            # Calculate directional accuracy
            report['directional_metrics'] = self.calculate_directional_accuracy(y_true, y_pred)
            
            # Calculate bias metrics
            report['bias_metrics'] = self.calculate_forecast_bias(y_true, y_pred)
            
            # Calculate confidence intervals if residuals provided
            if residuals is not None:
                report['confidence_intervals'] = self.calculate_confidence_intervals(
                    y_pred, residuals
                )
                
                # Residual analysis
                report['residual_analysis'] = {
                    'mean_residual': float(np.mean(residuals)),
                    'std_residual': float(np.std(residuals)),
                    'residual_autocorr': float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
                        if len(residuals) > 1 else 0.0
                }
            
            # Overall performance assessment
            mae = report['basic_metrics']['mae']
            mape = report['basic_metrics']['mape']
            r2 = report['basic_metrics']['r2']
            
            if mape < 10 and r2 > 0.8:
                performance = 'Excellent'
            elif mape < 20 and r2 > 0.6:
                performance = 'Good'
            elif mape < 30 and r2 > 0.4:
                performance = 'Fair'
            else:
                performance = 'Poor'
            
            report['overall_assessment'] = {
                'performance_level': performance,
                'key_strengths': [],
                'areas_for_improvement': []
            }
            
            # Add specific insights
            if r2 > 0.8:
                report['overall_assessment']['key_strengths'].append("High explanatory power")
            if mape < 15:
                report['overall_assessment']['key_strengths'].append("Low prediction error")
            if report['directional_metrics']['directional_accuracy'] > 70:
                report['overall_assessment']['key_strengths'].append("Good directional accuracy")
            
            if r2 < 0.5:
                report['overall_assessment']['areas_for_improvement'].append("Low explanatory power")
            if mape > 25:
                report['overall_assessment']['areas_for_improvement'].append("High prediction error")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating evaluation report: {str(e)}")
            return {'model_name': model_name, 'error': str(e)}