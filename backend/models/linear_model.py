"""
Linear Regression model with feature engineering for sales forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

warnings.filterwarnings('ignore')


class LinearModel:
    """Linear Regression model with advanced feature engineering"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.poly_features = None
        self.config = config or {}
        
        # Model configuration
        self.model_type = self.config.get('model_type', 'linear')  # 'linear', 'ridge', 'lasso'
        self.alpha = self.config.get('alpha', 1.0)  # Regularization parameter
        self.polynomial_degree = self.config.get('polynomial_degree', 1)
        self.feature_selection = self.config.get('feature_selection', True)
        self.n_features = self.config.get('n_features', 15)
        
        # Feature engineering parameters
        self.n_lags = self.config.get('n_lags', 12)
        self.rolling_windows = self.config.get('rolling_windows', [3, 6, 12])
        
        # Store training data and features
        self.training_data = None
        self.feature_names = []
        self.is_fitted = False
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for linear regression
        
        Args:
            data: Time series data with datetime index and 'sales' column
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df_features = data.copy()
            
            # Lag features
            for lag in range(1, self.n_lags + 1):
                df_features[f'sales_lag_{lag}'] = df_features['sales'].shift(lag)
            
            # Rolling statistics
            for window in self.rolling_windows:
                df_features[f'rolling_mean_{window}'] = df_features['sales'].rolling(window).mean()
                df_features[f'rolling_std_{window}'] = df_features['sales'].rolling(window).std()
                df_features[f'rolling_min_{window}'] = df_features['sales'].rolling(window).min()
                df_features[f'rolling_max_{window}'] = df_features['sales'].rolling(window).max()
            
            # Time-based features
            df_features['year'] = df_features.index.year
            df_features['month'] = df_features.index.month
            df_features['quarter'] = df_features.index.quarter
            df_features['day_of_year'] = df_features.index.dayofyear
            
            # Cyclical encoding
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
            df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
            df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
            
            # Trend features
            df_features['linear_trend'] = range(len(df_features))
            df_features['quadratic_trend'] = df_features['linear_trend'] ** 2
            
            # Growth rate and momentum
            df_features['growth_rate'] = df_features['sales'].pct_change()
            df_features['momentum'] = df_features['sales'].diff()
            
            # Moving averages and their ratios
            ma_short = df_features['sales'].rolling(3).mean()
            ma_long = df_features['sales'].rolling(12).mean()
            df_features['ma_ratio'] = ma_short / ma_long
            df_features['ma_diff'] = ma_short - ma_long
            
            # Volatility measures
            df_features['volatility'] = df_features['sales'].rolling(12).std() / df_features['sales'].rolling(12).mean()
            
            # Seasonal indicators
            df_features['is_q4'] = (df_features['quarter'] == 4).astype(int)
            df_features['is_holiday_season'] = ((df_features['month'] == 12) | (df_features['month'] == 1)).astype(int)
            
            # Interaction features (simple ones)
            df_features['trend_month'] = df_features['linear_trend'] * df_features['month']
            df_features['growth_season'] = df_features['growth_rate'] * df_features['month_sin']
            
            # Remove rows with NaN values
            df_features = df_features.dropna()
            
            self.logger.info(f"Created {df_features.shape[1] - 1} features from time series")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error creating features: {str(e)}")
            raise
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'f_regression') -> pd.DataFrame:
        """
        Select best features using statistical methods
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method
            
        Returns:
            DataFrame with selected features
        """
        try:
            if not self.feature_selection:
                return X
            
            # Initialize feature selector
            if method == 'f_regression':
                self.feature_selector = SelectKBest(score_func=f_regression, k=min(self.n_features, X.shape[1]))
            else:
                raise ValueError(f"Unsupported feature selection method: {method}")
            
            # Fit and transform features
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            self.feature_names = selected_features
            
            X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            self.logger.info(f"Selected {len(selected_features)} features using {method}")
            return X_selected_df
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            return X
    
    def create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with polynomial features
        """
        try:
            if self.polynomial_degree <= 1:
                return X
            
            # Initialize polynomial feature transformer
            self.poly_features = PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=False,
                interaction_only=True  # Only interaction terms, no pure polynomials
            )
            
            # Transform features
            X_poly = self.poly_features.fit_transform(X)
            
            # Create feature names
            poly_feature_names = self.poly_features.get_feature_names_out(X.columns)
            
            X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
            
            self.logger.info(f"Created {X_poly_df.shape[1]} polynomial features")
            return X_poly_df
            
        except Exception as e:
            self.logger.error(f"Error creating polynomial features: {str(e)}")
            return X
    
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit linear regression model to data
        
        Args:
            data: Time series data with datetime index and 'sales' column
            
        Returns:
            Dictionary with fit results
        """
        try:
            self.logger.info(f"Fitting {self.model_type} regression model...")
            
            # Store training data
            self.training_data = data.copy()
            
            # Create features
            df_features = self.create_features(data)
            
            # Prepare X and y
            y = df_features['sales']
            X = df_features.drop('sales', axis=1)
            
            # Feature selection
            X_selected = self.select_features(X, y)
            
            # Create polynomial features
            X_poly = self.create_polynomial_features(X_selected)
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_poly),
                columns=X_poly.columns,
                index=X_poly.index
            )
            
            # Initialize model based on type
            if self.model_type == 'linear':
                self.model = LinearRegression()
            elif self.model_type == 'ridge':
                self.model = Ridge(alpha=self.alpha)
            elif self.model_type == 'lasso':
                self.model = Lasso(alpha=self.alpha, max_iter=2000)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Fit model
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            
            # Calculate fit statistics
            y_pred = self.model.predict(X_scaled)
            
            fit_results = {
                'r2_score': float(r2_score(y, y_pred)),
                'mae': float(mean_absolute_error(y, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                'n_features': X_scaled.shape[1],
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }
            
            # Add model-specific information
            if hasattr(self.model, 'coef_'):
                fit_results['n_nonzero_coef'] = int(np.sum(np.abs(self.model.coef_) > 1e-6))
                
                # Feature importance (absolute coefficients)
                feature_importance = dict(zip(X_scaled.columns, np.abs(self.model.coef_)))
                sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                fit_results['top_features'] = sorted_importance[:10]
            
            self.logger.info(f"{self.model_type} model fitted successfully. R²: {fit_results['r2_score']:.4f}")
            return fit_results
            
        except Exception as e:
            self.logger.error(f"Error fitting {self.model_type} model: {str(e)}")
            raise
    
    def predict(self, steps: int, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Make predictions using fitted linear regression model
        
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
                data = self.training_data
            
            predictions = []
            current_data = data.copy()
            
            # Generate predictions iteratively
            for step in range(steps):
                # Create features for current data
                df_features = self.create_features(current_data)
                
                if len(df_features) == 0:
                    break
                
                # Get last row features (exclude target)
                X_last = df_features.drop('sales', axis=1).iloc[-1:].copy()
                
                # Apply feature selection if used
                if self.feature_selector is not None:
                    # Ensure we have all the features that were selected during training
                    missing_features = set(self.feature_names) - set(X_last.columns)
                    if missing_features:
                        for feature in missing_features:
                            X_last[feature] = 0  # Fill with default value
                    
                    X_last = X_last[self.feature_names]
                
                # Apply polynomial features if used
                if self.poly_features is not None:
                    X_last_poly = pd.DataFrame(
                        self.poly_features.transform(X_last),
                        columns=self.poly_features.get_feature_names_out(X_last.columns),
                        index=X_last.index
                    )
                else:
                    X_last_poly = X_last
                
                # Scale features
                X_scaled = self.scaler.transform(X_last_poly)
                
                # Make prediction
                pred = self.model.predict(X_scaled)[0]
                predictions.append(pred)
                
                # Add prediction to data for next iteration
                next_date = current_data.index[-1] + pd.DateOffset(months=1)
                new_row = pd.DataFrame({'sales': [pred]}, index=[next_date])
                current_data = pd.concat([current_data, new_row])
            
            # Generate future dates
            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=len(predictions),
                freq='M'
            )
            
            result = {
                'predictions': predictions,
                'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'confidence_intervals': self._calculate_confidence_intervals(predictions)
            }
            
            self.logger.info(f"Generated {len(predictions)} linear regression predictions")
            return result
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def _calculate_confidence_intervals(self, predictions: List[float], 
                                     confidence_level: float = 0.95) -> Dict[str, List[float]]:
        """
        Calculate confidence intervals for predictions
        
        Args:
            predictions: List of predictions
            confidence_level: Confidence level
            
        Returns:
            Dictionary with confidence intervals
        """
        try:
            # Simple approach: use training residuals to estimate uncertainty
            if self.training_data is not None and self.is_fitted:
                # Recreate training predictions to calculate residuals
                df_features = self.create_features(self.training_data)
                y_true = df_features['sales']
                X = df_features.drop('sales', axis=1)
                
                # Apply same transformations as during training
                if self.feature_selector is not None:
                    X = X[self.feature_names]
                
                if self.poly_features is not None:
                    X = pd.DataFrame(
                        self.poly_features.transform(X),
                        columns=self.poly_features.get_feature_names_out(X.columns),
                        index=X.index
                    )
                
                X_scaled = self.scaler.transform(X)
                y_pred = self.model.predict(X_scaled)
                
                residuals = y_true - y_pred
                std_error = np.std(residuals)
            else:
                # Fallback: use 15% of mean prediction as uncertainty
                std_error = np.mean(predictions) * 0.15
            
            # Calculate confidence intervals
            z_score = 1.96 if confidence_level == 0.95 else 2.58
            margin_error = z_score * std_error
            
            predictions_array = np.array(predictions)
            lower_bound = predictions_array - margin_error
            upper_bound = predictions_array + margin_error
            
            return {
                'lower': lower_bound.tolist(),
                'upper': upper_bound.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence intervals: {str(e)}")
            # Return simple ±20% intervals as fallback
            predictions_array = np.array(predictions)
            return {
                'lower': (predictions_array * 0.8).tolist(),
                'upper': (predictions_array * 1.2).tolist()
            }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance based on model coefficients
        
        Returns:
            Dictionary with feature importance scores
        """
        try:
            if not self.is_fitted or not hasattr(self.model, 'coef_'):
                return {}
            
            # Get feature names (after all transformations)
            if self.poly_features is not None:
                feature_names = self.poly_features.get_feature_names_out(self.feature_names)
            else:
                feature_names = self.feature_names
            
            # Calculate importance as absolute coefficients
            importance = dict(zip(feature_names, np.abs(self.model.coef_)))
            
            # Sort by importance
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
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
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'poly_features': self.poly_features,
                'config': self.config,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted,
                'training_data': self.training_data
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Linear model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving linear model: {str(e)}")
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
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.poly_features = model_data['poly_features']
            self.config = model_data.get('config', {})
            self.feature_names = model_data.get('feature_names', [])
            self.is_fitted = model_data.get('is_fitted', False)
            self.training_data = model_data.get('training_data')
            
            self.logger.info(f"Linear model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading linear model: {str(e)}")
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
            'model_type': f'Linear Regression ({self.model_type})',
            'n_features': len(self.feature_names),
            'polynomial_degree': self.polynomial_degree,
            'feature_selection': self.feature_selection,
            'regularization': self.alpha if self.model_type in ['ridge', 'lasso'] else None
        }
        
        # Add model coefficients info
        if hasattr(self.model, 'coef_'):
            info['n_nonzero_coefficients'] = int(np.sum(np.abs(self.model.coef_) > 1e-6))
            info['intercept'] = float(self.model.intercept_)
        
        return info