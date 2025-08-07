"""
Feature engineering utilities for sales forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureEngineer:
    """Feature engineering for time series forecasting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
    
    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """
        Create lag features for time series
        
        Args:
            df: DataFrame with datetime index and 'sales' column
            lags: List of lag periods to create
            
        Returns:
            DataFrame with lag features
        """
        if lags is None:
            lags = [1, 2, 3, 6, 12]  # Default lags
        
        try:
            df_features = df.copy()
            
            for lag in lags:
                df_features[f'sales_lag_{lag}'] = df_features['sales'].shift(lag)
            
            self.logger.info(f"Created lag features for periods: {lags}")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error creating lag features: {str(e)}")
            raise
    
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: DataFrame with datetime index and 'sales' column
            windows: List of window sizes for rolling statistics
            
        Returns:
            DataFrame with rolling features
        """
        if windows is None:
            windows = [3, 6, 12]  # Default windows
        
        try:
            df_features = df.copy()
            
            for window in windows:
                # Rolling mean
                df_features[f'sales_rolling_mean_{window}'] = (
                    df_features['sales'].rolling(window=window).mean()
                )
                
                # Rolling standard deviation
                df_features[f'sales_rolling_std_{window}'] = (
                    df_features['sales'].rolling(window=window).std()
                )
                
                # Rolling minimum
                df_features[f'sales_rolling_min_{window}'] = (
                    df_features['sales'].rolling(window=window).min()
                )
                
                # Rolling maximum
                df_features[f'sales_rolling_max_{window}'] = (
                    df_features['sales'].rolling(window=window).max()
                )
            
            self.logger.info(f"Created rolling features for windows: {windows}")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error creating rolling features: {str(e)}")
            raise
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create seasonal features
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with seasonal features
        """
        try:
            df_features = df.copy()
            
            # Extract date components
            df_features['year'] = df_features.index.year
            df_features['month'] = df_features.index.month
            df_features['quarter'] = df_features.index.quarter
            df_features['day_of_year'] = df_features.index.dayofyear
            df_features['week_of_year'] = df_features.index.isocalendar().week
            
            # Cyclical encoding for seasonal patterns
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
            
            df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
            df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
            
            # Holiday indicators (basic - can be extended)
            df_features['is_holiday_season'] = (
                (df_features['month'] == 12) | (df_features['month'] == 1)
            ).astype(int)
            
            df_features['is_summer'] = (
                (df_features['month'] >= 6) & (df_features['month'] <= 8)
            ).astype(int)
            
            self.logger.info("Created seasonal features")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error creating seasonal features: {str(e)}")
            raise
    
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend features
        
        Args:
            df: DataFrame with datetime index and 'sales' column
            
        Returns:
            DataFrame with trend features
        """
        try:
            df_features = df.copy()
            
            # Linear trend
            df_features['linear_trend'] = range(len(df_features))
            
            # Growth rate
            df_features['growth_rate'] = df_features['sales'].pct_change()
            
            # Moving averages for trend
            df_features['ma_short'] = df_features['sales'].rolling(window=3).mean()
            df_features['ma_long'] = df_features['sales'].rolling(window=12).mean()
            
            # Trend direction
            df_features['trend_direction'] = np.where(
                df_features['ma_short'] > df_features['ma_long'], 1, 0
            )
            
            # Acceleration (second derivative)
            df_features['acceleration'] = df_features['growth_rate'].diff()
            
            self.logger.info("Created trend features")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error creating trend features: {str(e)}")
            raise
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features
        
        Args:
            df: DataFrame with datetime index and 'sales' column
            
        Returns:
            DataFrame with statistical features
        """
        try:
            df_features = df.copy()
            
            # Z-score (standardized values)
            df_features['sales_zscore'] = stats.zscore(df_features['sales'])
            
            # Percentile rank
            df_features['sales_percentile'] = df_features['sales'].rank(pct=True)
            
            # Distance from mean
            mean_sales = df_features['sales'].mean()
            df_features['distance_from_mean'] = abs(df_features['sales'] - mean_sales)
            
            # Volatility (rolling standard deviation normalized by rolling mean)
            window = 12
            rolling_mean = df_features['sales'].rolling(window=window).mean()
            rolling_std = df_features['sales'].rolling(window=window).std()
            df_features['volatility'] = rolling_std / rolling_mean
            
            # Coefficient of variation
            df_features['cv'] = rolling_std / rolling_mean
            
            self.logger.info("Created statistical features")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error creating statistical features: {str(e)}")
            raise
    
    def create_all_features(self, df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Create all engineered features
        
        Args:
            df: DataFrame with datetime index and 'sales' column
            config: Configuration for feature engineering
            
        Returns:
            DataFrame with all engineered features
        """
        if config is None:
            config = {
                'lags': [1, 2, 3, 6, 12],
                'rolling_windows': [3, 6, 12],
                'include_seasonal': True,
                'include_trend': True,
                'include_statistical': True
            }
        
        try:
            df_features = df.copy()
            
            # Create lag features
            if config.get('lags'):
                df_features = self.create_lag_features(df_features, config['lags'])
            
            # Create rolling features
            if config.get('rolling_windows'):
                df_features = self.create_rolling_features(df_features, config['rolling_windows'])
            
            # Create seasonal features
            if config.get('include_seasonal', True):
                df_features = self.create_seasonal_features(df_features)
            
            # Create trend features
            if config.get('include_trend', True):
                df_features = self.create_trend_features(df_features)
            
            # Create statistical features
            if config.get('include_statistical', True):
                df_features = self.create_statistical_features(df_features)
            
            # Remove rows with NaN values created by lag/rolling features
            initial_rows = len(df_features)
            df_features = df_features.dropna()
            removed_rows = initial_rows - len(df_features)
            
            if removed_rows > 0:
                self.logger.info(f"Removed {removed_rows} rows with NaN values after feature engineering")
            
            self.logger.info(f"Feature engineering completed. Final shape: {df_features.shape}")
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', 
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale features for machine learning models
        
        Args:
            df: DataFrame with features
            method: Scaling method ('standard' or 'minmax')
            fit: Whether to fit the scaler or use existing one
            
        Returns:
            DataFrame with scaled features
        """
        try:
            # Identify numeric columns (exclude target variable)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'sales' in numeric_cols:
                numeric_cols.remove('sales')
            
            df_scaled = df.copy()
            
            if not numeric_cols:
                self.logger.warning("No numeric features found for scaling")
                return df_scaled
            
            # Initialize scaler
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unsupported scaling method: {method}")
            
            # Fit and transform or just transform
            if fit:
                scaled_values = scaler.fit_transform(df_scaled[numeric_cols])
                self.scalers[method] = scaler
            else:
                if method not in self.scalers:
                    raise ValueError(f"Scaler for method '{method}' not fitted yet")
                scaled_values = self.scalers[method].transform(df_scaled[numeric_cols])
            
            # Replace values in DataFrame
            df_scaled[numeric_cols] = scaled_values
            
            self.logger.info(f"Features scaled using {method} scaling")
            return df_scaled
            
        except Exception as e:
            self.logger.error(f"Error scaling features: {str(e)}")
            raise
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'sales') -> Dict[str, float]:
        """
        Calculate feature importance using correlation
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            
        Returns:
            Dictionary with feature importance scores
        """
        try:
            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col not in numeric_cols:
                raise ValueError(f"Target column '{target_col}' not found in numeric columns")
            
            # Remove target column from features
            feature_cols = [col for col in numeric_cols if col != target_col]
            
            # Calculate correlations
            correlations = df[feature_cols + [target_col]].corr()[target_col].abs()
            
            # Sort by importance
            importance = correlations[feature_cols].sort_values(ascending=False)
            
            self.logger.info(f"Calculated feature importance for {len(feature_cols)} features")
            return importance.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            raise
    
    def select_features(self, df: pd.DataFrame, target_col: str = 'sales', 
                       top_k: int = 20) -> List[str]:
        """
        Select top-k features based on importance
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            top_k: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        try:
            importance = self.get_feature_importance(df, target_col)
            
            # Select top-k features
            selected_features = list(importance.keys())[:top_k]
            
            self.logger.info(f"Selected top {len(selected_features)} features")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            raise
    
    def prepare_sequences(self, df: pd.DataFrame, lookback: int = 12, 
                         target_col: str = 'sales') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM model
        
        Args:
            df: DataFrame with features
            lookback: Number of time steps to look back
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) arrays for LSTM training
        """
        try:
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            data = df[numeric_cols].values
            
            X, y = [], []
            target_idx = df.columns.get_loc(target_col)
            
            for i in range(lookback, len(data)):
                X.append(data[i-lookback:i])  # Features from lookback periods
                y.append(data[i, target_idx])  # Target value at current period
            
            X = np.array(X)
            y = np.array(y)
            
            self.logger.info(f"Prepared sequences: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing sequences: {str(e)}")
            raise