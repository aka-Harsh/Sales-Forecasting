"""
Forecasting API routes
"""

from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
import traceback
import json

from backend.models.sarimax_model import SARIMAXModel
from backend.models.prophet_model import ProphetModel
from backend.models.lstm_model import LSTMModel
from backend.models.linear_model import LinearModel
from backend.models.ensemble_model import EnsembleModel
from backend.utils.data_processor import DataProcessor
from backend.utils.model_evaluator import ModelEvaluator

forecast_bp = Blueprint('forecast', __name__)
logger = logging.getLogger(__name__)


@forecast_bp.route('/models', methods=['GET'])
def get_available_models():
    """Get list of available forecasting models"""
    try:
        models = {
            'sarimax': {
                'name': 'SARIMAX',
                'description': 'Seasonal ARIMA with eXogenous variables',
                'type': 'statistical',
                'supports_seasonality': True,
                'supports_exogenous': True
            },
            'prophet': {
                'name': 'Prophet',
                'description': 'Facebook Prophet for time series forecasting',
                'type': 'statistical',
                'supports_seasonality': True,
                'supports_holidays': True
            },
            'lstm': {
                'name': 'LSTM',
                'description': 'Long Short-Term Memory neural network',
                'type': 'deep_learning',
                'supports_multivariate': True,
                'requires_scaling': True
            },
            'linear': {
                'name': 'Linear Regression',
                'description': 'Linear regression with feature engineering',
                'type': 'machine_learning',
                'supports_feature_engineering': True,
                'interpretable': True
            },
            'ensemble': {
                'name': 'Ensemble',
                'description': 'Combination of multiple models',
                'type': 'ensemble',
                'combines_all': True,
                'typically_best_performance': True
            }
        }
        
        return jsonify({
            'success': True,
            'models': models
        })
        
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@forecast_bp.route('/train', methods=['POST'])
def train_model():
    """Train a forecasting model"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        model_type = data.get('model_type', 'ensemble')
        model_config = data.get('config', {})
        data_source = data.get('data_source', 'uploaded')  # 'uploaded' or 'sample'
        
        # Load data
        processor = DataProcessor()
        
        if data_source == 'sample':
            # Use sample data
            sample_path = 'data/sample/sample_data.csv'
            df = processor.load_data(sample_path)
        else:
            # Use uploaded data (should be handled by data routes first)
            return jsonify({
                'success': False,
                'error': 'Data upload functionality should be used first'
            }), 400
        
        # Clean and prepare data
        df_clean, cleaning_report = processor.clean_data(df)
        
        if len(df_clean) < 24:  # Need at least 2 years of monthly data
            return jsonify({
                'success': False,
                'error': 'Insufficient data for training. Need at least 24 data points.'
            }), 400
        
        # Initialize and train model
        if model_type == 'sarimax':
            model = SARIMAXModel(model_config)
        elif model_type == 'prophet':
            model = ProphetModel(model_config)
        elif model_type == 'lstm':
            model = LSTMModel(model_config)
        elif model_type == 'linear':
            model = LinearModel(model_config)
        elif model_type == 'ensemble':
            model = EnsembleModel(model_config)
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported model type: {model_type}'
            }), 400
        
        # Train model
        logger.info(f"Training {model_type} model...")
        fit_results = model.fit(df_clean)
        
        # Save model
        model_path = f'models/trained/{model_type}_model.pkl'
        save_success = model.save_model(model_path)
        
        response = {
            'success': True,
            'model_type': model_type,
            'fit_results': fit_results,
            'data_info': {
                'original_shape': df.shape,
                'cleaned_shape': df_clean.shape,
                'cleaning_report': cleaning_report,
                'date_range': {
                    'start': str(df_clean.index.min()),
                    'end': str(df_clean.index.max())
                }
            },
            'model_saved': save_success
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@forecast_bp.route('/predict', methods=['POST'])
def make_forecast():
    """Make predictions using trained model"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        model_type = data.get('model_type', 'ensemble')
        steps = data.get('steps', 12)  # Default to 12 months
        
        # Load trained model
        model_path = f'models/trained/{model_type}_model.pkl'
        
        # Initialize model
        if model_type == 'sarimax':
            model = SARIMAXModel()
        elif model_type == 'prophet':
            model = ProphetModel()
        elif model_type == 'lstm':
            model = LSTMModel()
        elif model_type == 'linear':
            model = LinearModel()
        elif model_type == 'ensemble':
            model = EnsembleModel()
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported model type: {model_type}'
            }), 400
        
        # Load model
        if not model.load_model(model_path):
            return jsonify({
                'success': False,
                'error': f'Could not load trained {model_type} model'
            }), 404
        
        # Make predictions
        logger.info(f"Making {steps} predictions with {model_type} model...")
        predictions = model.predict(steps)
        
        # Get model info
        model_info = model.get_model_info()
        
        response = {
            'success': True,
            'model_type': model_type,
            'predictions': predictions['predictions'],
            'dates': predictions['dates'],
            'confidence_intervals': predictions.get('confidence_intervals', {}),
            'model_info': model_info,
            'forecast_horizon': steps
        }
        
        # Add model-specific information
        if model_type == 'ensemble' and 'individual_predictions' in predictions:
            response['individual_predictions'] = predictions['individual_predictions']
            response['model_weights'] = predictions.get('model_weights', {})
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error making forecast: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@forecast_bp.route('/compare', methods=['POST'])
def compare_models():
    """Compare predictions from multiple models"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        model_types = data.get('models', ['sarimax', 'prophet', 'lstm', 'linear'])
        steps = data.get('steps', 12)
        
        comparison_results = {}
        
        # Get predictions from each model
        for model_type in model_types:
            try:
                model_path = f'models/trained/{model_type}_model.pkl'
                
                # Initialize model
                if model_type == 'sarimax':
                    model = SARIMAXModel()
                elif model_type == 'prophet':
                    model = ProphetModel()
                elif model_type == 'lstm':
                    model = LSTMModel()
                elif model_type == 'linear':
                    model = LinearModel()
                else:
                    continue
                
                # Load and predict
                if model.load_model(model_path):
                    predictions = model.predict(steps)
                    model_info = model.get_model_info()
                    
                    comparison_results[model_type] = {
                        'predictions': predictions['predictions'],
                        'dates': predictions['dates'],
                        'confidence_intervals': predictions.get('confidence_intervals', {}),
                        'model_info': model_info
                    }
                
            except Exception as e:
                logger.warning(f"Could not get predictions from {model_type}: {str(e)}")
                continue
        
        if not comparison_results:
            return jsonify({
                'success': False,
                'error': 'No models could generate predictions'
            }), 404
        
        # Calculate ensemble prediction as simple average
        if len(comparison_results) > 1:
            all_predictions = [results['predictions'] for results in comparison_results.values()]
            ensemble_pred = np.mean(all_predictions, axis=0).tolist()
            
            comparison_results['ensemble_average'] = {
                'predictions': ensemble_pred,
                'dates': list(comparison_results.values())[0]['dates'],
                'model_info': {'status': 'computed', 'model_type': 'Simple Average Ensemble'}
            }
        
        response = {
            'success': True,
            'model_comparison': comparison_results,
            'forecast_horizon': steps,
            'models_compared': list(comparison_results.keys())
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@forecast_bp.route('/evaluate', methods=['POST'])
def evaluate_model():
    """Evaluate model performance using cross-validation"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        model_type = data.get('model_type', 'ensemble')
        cv_folds = data.get('cv_folds', 5)
        
        # Load training data
        processor = DataProcessor()
        sample_path = 'data/sample/sample_data.csv'
        df = processor.load_data(sample_path)
        df_clean, _ = processor.clean_data(df)
        
        # Initialize model
        if model_type == 'sarimax':
            model = SARIMAXModel()
        elif model_type == 'prophet':
            model = ProphetModel()
        elif model_type == 'lstm':
            model = LSTMModel()
        elif model_type == 'linear':
            model = LinearModel()
        elif model_type == 'ensemble':
            model = EnsembleModel()
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported model type: {model_type}'
            }), 400
        
        # Perform cross-validation
        evaluator = ModelEvaluator()
        cv_results = evaluator.cross_validate_time_series(
            model, df_clean, 
            initial_train_size=len(df_clean) // 2,
            horizon=6,  # 6-month forecast horizon
            step=3      # Move forward 3 months each time
        )
        
        response = {
            'success': True,
            'model_type': model_type,
            'cv_results': cv_results,
            'evaluation_setup': {
                'cv_folds': len(cv_results.get('fold_metrics', [])),
                'forecast_horizon': 6,
                'step_size': 3
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@forecast_bp.route('/decompose', methods=['POST'])
def decompose_series():
    """Perform seasonal decomposition of time series"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        decomposition_type = data.get('type', 'additive')  # 'additive' or 'multiplicative'
        
        # Load data
        processor = DataProcessor()
        sample_path = 'data/sample/sample_data.csv'
        df = processor.load_data(sample_path)
        df_clean, _ = processor.clean_data(df)
        
        # Perform decomposition using SARIMAX model (which has decomposition capability)
        sarimax_model = SARIMAXModel()
        decomposition = sarimax_model.decompose_series(df_clean, decomposition_type)
        
        response = {
            'success': True,
            'decomposition_type': decomposition_type,
            'components': decomposition,
            'data_info': {
                'total_observations': len(df_clean),
                'date_range': {
                    'start': str(df_clean.index.min()),
                    'end': str(df_clean.index.max())
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error decomposing series: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@forecast_bp.route('/model-info/<model_type>', methods=['GET'])
def get_model_info(model_type):
    """Get information about a specific trained model"""
    try:
        model_path = f'models/trained/{model_type}_model.pkl'
        
        # Initialize model
        if model_type == 'sarimax':
            model = SARIMAXModel()
        elif model_type == 'prophet':
            model = ProphetModel()
        elif model_type == 'lstm':
            model = LSTMModel()
        elif model_type == 'linear':
            model = LinearModel()
        elif model_type == 'ensemble':
            model = EnsembleModel()
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported model type: {model_type}'
            }), 400
        
        # Load model
        if not model.load_model(model_path):
            return jsonify({
                'success': False,
                'error': f'Could not load trained {model_type} model'
            }), 404
        
        # Get model info
        model_info = model.get_model_info()
        
        response = {
            'success': True,
            'model_type': model_type,
            'model_info': model_info
        }
        
        # Add model-specific detailed information
        if model_type == 'ensemble':
            response['model_contributions'] = model.get_model_contributions()
        elif model_type == 'linear':
            response['feature_importance'] = model.get_feature_importance()
        elif model_type == 'prophet':
            try:
                response['changepoints'] = model.detect_changepoints()
                response['component_importance'] = model.get_component_importance()
            except:
                pass
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@forecast_bp.route('/retrain', methods=['POST'])
def retrain_model():
    """Retrain a model with new parameters"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        model_type = data.get('model_type', 'ensemble')
        new_config = data.get('config', {})
        
        # Load data
        processor = DataProcessor()
        sample_path = 'data/sample/sample_data.csv'
        df = processor.load_data(sample_path)
        df_clean, cleaning_report = processor.clean_data(df)
        
        # Initialize model with new configuration
        if model_type == 'sarimax':
            model = SARIMAXModel(new_config)
        elif model_type == 'prophet':
            model = ProphetModel(new_config)
        elif model_type == 'lstm':
            model = LSTMModel(new_config)
        elif model_type == 'linear':
            model = LinearModel(new_config)
        elif model_type == 'ensemble':
            model = EnsembleModel(new_config)
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported model type: {model_type}'
            }), 400
        
        # Train model
        logger.info(f"Retraining {model_type} model with new configuration...")
        fit_results = model.fit(df_clean)
        
        # Save model
        model_path = f'models/trained/{model_type}_model.pkl'
        save_success = model.save_model(model_path)
        
        response = {
            'success': True,
            'model_type': model_type,
            'new_config': new_config,
            'fit_results': fit_results,
            'model_saved': save_success
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500