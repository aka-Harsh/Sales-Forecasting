"""
LSTM (Long Short-Term Memory) neural network model for sales forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import json

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


class LSTMModel:
    """LSTM neural network model for time series forecasting"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = MinMaxScaler()
        self.config = config or {}
        
        # Default parameters
        self.lookback_window = self.config.get('lookback_window', 12)
        self.units = self.config.get('units', 50)
        self.dropout = self.config.get('dropout', 0.2)
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        
        # Model architecture parameters
        self.n_layers = self.config.get('n_layers', 2)
        self.dense_units = self.config.get('dense_units', 25)
        
        # Training history
        self.history = None
        self.is_fitted = False
        
        # Store original data for inverse scaling
        self.original_data = None
    
    def prepare_sequences(self, data: pd.DataFrame, features: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        
        Args:
            data: Time series data
            features: List of feature columns to use
            
        Returns:
            Tuple of (X, y) arrays for LSTM training
        """
        try:
            if features is None:
                features = ['sales']
            
            # Select and scale features
            feature_data = data[features].values
            scaled_data = self.scaler.fit_transform(feature_data)
            
            X, y = [], []
            
            # Create sequences
            for i in range(self.lookback_window, len(scaled_data)):
                X.append(scaled_data[i-self.lookback_window:i])
                y.append(scaled_data[i, 0])  # Assuming 'sales' is the first feature
            
            X = np.array(X)
            y = np.array(y)
            
            self.logger.info(f"Prepared LSTM sequences: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing sequences: {str(e)}")
            raise
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled LSTM model
        """
        try:
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(
                units=self.units,
                return_sequences=True if self.n_layers > 1 else False,
                input_shape=input_shape
            ))
            model.add(Dropout(self.dropout))
            
            # Additional LSTM layers
            for i in range(1, self.n_layers):
                return_sequences = i < self.n_layers - 1
                model.add(LSTM(
                    units=self.units // (2 ** i),  # Reduce units in each layer
                    return_sequences=return_sequences
                ))
                model.add(Dropout(self.dropout))
            
            # Dense layers
            model.add(Dense(units=self.dense_units, activation='relu'))
            model.add(Dropout(self.dropout / 2))
            model.add(Dense(units=1))  # Output layer
            
            # Compile model
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            self.logger.info(f"Built LSTM model with {model.count_params()} parameters")
            return model
            
        except Exception as e:
            self.logger.error(f"Error building LSTM model: {str(e)}")
            raise
    
    def fit(self, data: pd.DataFrame, features: List[str] = None, 
            validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Fit LSTM model to data
        
        Args:
            data: Time series data
            features: List of feature columns to use
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with fit results
        """
        try:
            self.logger.info("Fitting LSTM model...")
            
            # Store original data
            self.original_data = data.copy()
            
            # Prepare sequences
            X, y = self.prepare_sequences(data, features)
            
            if len(X) == 0:
                raise ValueError("Not enough data to create sequences")
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build model
            self.model = self.build_model((X.shape[1], X.shape[2]))
            
            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=0
                )
            ]
            
            # Train model
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_fitted = True
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_train, verbose=0)
            val_predictions = self.model.predict(X_val, verbose=0)
            
            # Inverse transform predictions
            train_pred_orig = self.scaler.inverse_transform(
                np.column_stack([train_predictions.flatten(), 
                               np.zeros((len(train_predictions), self.scaler.n_features_in_ - 1))])
            )[:, 0]
            
            val_pred_orig = self.scaler.inverse_transform(
                np.column_stack([val_predictions.flatten(), 
                               np.zeros((len(val_predictions), self.scaler.n_features_in_ - 1))])
            )[:, 0]
            
            # Inverse transform actual values
            y_train_orig = self.scaler.inverse_transform(
                np.column_stack([y_train, 
                               np.zeros((len(y_train), self.scaler.n_features_in_ - 1))])
            )[:, 0]
            
            y_val_orig = self.scaler.inverse_transform(
                np.column_stack([y_val, 
                               np.zeros((len(y_val), self.scaler.n_features_in_ - 1))])
            )[:, 0]
            
            # Calculate metrics
            fit_results = {
                'training_metrics': self._calculate_metrics(y_train_orig, train_pred_orig),
                'validation_metrics': self._calculate_metrics(y_val_orig, val_pred_orig),
                'training_history': {
                    'loss': [float(x) for x in self.history.history['loss']],
                    'val_loss': [float(x) for x in self.history.history['val_loss']],
                    'mae': [float(x) for x in self.history.history['mae']],
                    'val_mae': [float(x) for x in self.history.history['val_mae']]
                },
                'model_summary': self._get_model_summary(),
                'epochs_trained': len(self.history.history['loss'])
            }
            
            self.logger.info(f"LSTM model fitted successfully in {fit_results['epochs_trained']} epochs")
            return fit_results
            
        except Exception as e:
            self.logger.error(f"Error fitting LSTM model: {str(e)}")
            raise
    
    def predict(self, steps: int, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Make predictions using fitted LSTM model
        
        Args:
            steps: Number of steps to forecast
            data: Optional data to base predictions on (uses training data if None)
            
        Returns:
            Dictionary with predictions
        """
        try:
            if not self.is_fitted or self.model is None:
                raise ValueError("Model must be fitted before making predictions")
            
            # Use training data if no data provided
            if data is None:
                data = self.original_data
            
            # Prepare last sequence from data
            features = ['sales'] if 'sales' in data.columns else [data.columns[0]]
            scaled_data = self.scaler.transform(data[features].values)
            
            # Get last sequence
            last_sequence = scaled_data[-self.lookback_window:].reshape(1, self.lookback_window, len(features))
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            # Generate predictions iteratively
            for _ in range(steps):
                # Predict next value
                next_pred = self.model.predict(current_sequence, verbose=0)
                predictions.append(next_pred[0, 0])
                
                # Update sequence for next prediction
                # Create new input with prediction
                if len(features) == 1:
                    new_input = next_pred.reshape(1, 1, 1)
                else:
                    # For multiple features, use prediction for sales and repeat last values for others
                    new_features = np.zeros((1, 1, len(features)))
                    new_features[0, 0, 0] = next_pred[0, 0]
                    for i in range(1, len(features)):
                        new_features[0, 0, i] = current_sequence[0, -1, i]
                    new_input = new_features
                
                # Shift sequence and add new prediction
                current_sequence = np.concatenate([current_sequence[:, 1:, :], new_input], axis=1)
            
            # Inverse transform predictions
            predictions_array = np.array(predictions).reshape(-1, 1)
            if len(features) > 1:
                # Pad with zeros for other features
                predictions_padded = np.column_stack([
                    predictions_array.flatten(),
                    np.zeros((len(predictions), len(features) - 1))
                ])
            else:
                predictions_padded = predictions_array
            
            predictions_orig = self.scaler.inverse_transform(predictions_padded)[:, 0]
            
            # Generate future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1), 
                periods=steps, 
                freq='M'
            )
            
            result = {
                'predictions': predictions_orig.tolist(),
                'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'confidence_intervals': self._calculate_prediction_intervals(predictions_orig)
            }
            
            self.logger.info(f"Generated {steps} LSTM predictions")
            return result
            
        except Exception as e:
            self.logger.error(f"Error making LSTM predictions: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        try:
            # Remove any NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            if len(y_true) == 0:
                return {'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf')}
            
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            # Calculate MAPE with handling for zero values
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mape = mape if not np.isnan(mape) else 100.0
            
            return {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf')}
    
    def _calculate_prediction_intervals(self, predictions: np.ndarray, 
                                     confidence_level: float = 0.95) -> Dict[str, List[float]]:
        """
        Calculate prediction intervals for LSTM forecasts
        
        Args:
            predictions: Array of predictions
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with confidence intervals
        """
        try:
            # Simple approach: use training error to estimate uncertainty
            if self.history is not None:
                # Get final validation loss as proxy for prediction uncertainty
                val_losses = self.history.history['val_loss']
                final_val_loss = val_losses[-1]
                std_error = np.sqrt(final_val_loss)
            else:
                # Fallback: use 10% of mean prediction as uncertainty
                std_error = np.mean(predictions) * 0.1
            
            # Calculate confidence intervals
            z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
            margin_error = z_score * std_error
            
            lower_bound = predictions - margin_error
            upper_bound = predictions + margin_error
            
            return {
                'lower': lower_bound.tolist(),
                'upper': upper_bound.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction intervals: {str(e)}")
            # Return simple ±20% intervals as fallback
            return {
                'lower': (predictions * 0.8).tolist(),
                'upper': (predictions * 1.2).tolist()
            }
    
    def _get_model_summary(self) -> Dict[str, Any]:
        """
        Get model architecture summary
        
        Returns:
            Dictionary with model summary information
        """
        try:
            if self.model is None:
                return {}
            
            return {
                'total_params': int(self.model.count_params()),
                'trainable_params': int(sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])),
                'layers': len(self.model.layers),
                'optimizer': self.model.optimizer.get_config()['name'],
                'loss_function': self.model.loss
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model summary: {str(e)}")
            return {}
    
    def plot_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history for plotting
        
        Returns:
            Dictionary with training history
        """
        if self.history is None:
            return {}
        
        return {
            'epochs': list(range(1, len(self.history.history['loss']) + 1)),
            'training_loss': [float(x) for x in self.history.history['loss']],
            'validation_loss': [float(x) for x in self.history.history['val_loss']],
            'training_mae': [float(x) for x in self.history.history['mae']],
            'validation_mae': [float(x) for x in self.history.history['val_mae']]
        }
    
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
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save Keras model
            model_path = filepath.replace('.pkl', '_keras.h5')
            self.model.save(model_path)
            
            # Save other components
            model_data = {
                'config': self.config,
                'scaler': self.scaler,
                'history': self.history.history if self.history else None,
                'is_fitted': self.is_fitted,
                'lookback_window': self.lookback_window,
                'original_data': self.original_data
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"LSTM model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving LSTM model: {str(e)}")
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
            # Load Keras model
            model_path = filepath.replace('.pkl', '_keras.h5')
            self.model = tf.keras.models.load_model(model_path)
            
            # Load other components
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.config = model_data['config']
            self.scaler = model_data['scaler']
            self.is_fitted = model_data['is_fitted']
            self.lookback_window = model_data['lookback_window']
            self.original_data = model_data.get('original_data')
            
            # Reconstruct history if available
            if model_data.get('history'):
                self.history = type('History', (), {'history': model_data['history']})()
            
            self.logger.info(f"LSTM model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted or self.model is None:
            return {'status': 'not_fitted'}
        
        info = {
            'status': 'fitted',
            'model_type': 'LSTM',
            'lookback_window': self.lookback_window,
            'units': self.units,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
            'total_parameters': int(self.model.count_params()),
            'epochs_trained': len(self.history.history['loss']) if self.history else 0
        }
        
        # Add training metrics if available
        if self.history:
            final_loss = self.history.history['val_loss'][-1]
            final_mae = self.history.history['val_mae'][-1]
            info['final_validation_loss'] = float(final_loss)
            info['final_validation_mae'] = float(final_mae)
        
        return info