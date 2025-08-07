"""
Anomaly detection utilities for sales forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """Anomaly detection for time series data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
    
    def detect_statistical_outliers(self, df: pd.DataFrame, column: str = 'sales', 
                                   method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using statistical methods
        
        Args:
            df: DataFrame with time series data
            column: Column to analyze for outliers
            method: Method to use ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier indicators
        """
        try:
            df_outliers = df.copy()
            
            if method == 'iqr':
                # Interquartile Range method
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                df_outliers['is_outlier_iqr'] = (
                    (df[column] < lower_bound) | (df[column] > upper_bound)
                )
                df_outliers['outlier_score_iqr'] = np.where(
                    df[column] < lower_bound, 
                    (lower_bound - df[column]) / IQR,
                    np.where(df[column] > upper_bound, 
                            (df[column] - upper_bound) / IQR, 0)
                )
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs(stats.zscore(df[column]))
                df_outliers['is_outlier_zscore'] = z_scores > threshold
                df_outliers['outlier_score_zscore'] = z_scores
                
            elif method == 'modified_zscore':
                # Modified Z-score using median
                median = df[column].median()
                mad = np.median(np.abs(df[column] - median))
                modified_z_scores = 0.6745 * (df[column] - median) / mad
                
                df_outliers['is_outlier_modified_zscore'] = np.abs(modified_z_scores) > threshold
                df_outliers['outlier_score_modified_zscore'] = np.abs(modified_z_scores)
                
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            outlier_count = df_outliers[f'is_outlier_{method}'].sum()
            self.logger.info(f"Detected {outlier_count} outliers using {method} method")
            
            return df_outliers
            
        except Exception as e:
            self.logger.error(f"Error detecting statistical outliers: {str(e)}")
            raise
    
    def detect_isolation_forest_outliers(self, df: pd.DataFrame, 
                                       contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect outliers using Isolation Forest
        
        Args:
            df: DataFrame with features
            contamination: Expected proportion of outliers
            
        Returns:
            DataFrame with outlier indicators
        """
        try:
            df_outliers = df.copy()
            
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                raise ValueError("No numeric columns found for anomaly detection")
            
            # Prepare data
            X = df[numeric_cols].values
            
            # Handle missing values
            if np.isnan(X).any():
                X = pd.DataFrame(X).fillna(method='ffill').fillna(method='bfill').values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            outlier_labels = iso_forest.fit_predict(X_scaled)
            outlier_scores = iso_forest.score_samples(X_scaled)
            
            # Convert to boolean (Isolation Forest returns -1 for outliers, 1 for inliers)
            df_outliers['is_outlier_isolation'] = outlier_labels == -1
            df_outliers['outlier_score_isolation'] = -outlier_scores  # Convert to positive scores
            
            # Store model for future use
            self.models['isolation_forest'] = {
                'model': iso_forest,
                'scaler': scaler,
                'features': numeric_cols
            }
            
            outlier_count = df_outliers['is_outlier_isolation'].sum()
            self.logger.info(f"Detected {outlier_count} outliers using Isolation Forest")
            
            return df_outliers
            
        except Exception as e:
            self.logger.error(f"Error detecting Isolation Forest outliers: {str(e)}")
            raise
    
    def detect_seasonal_outliers(self, df: pd.DataFrame, column: str = 'sales',
                               seasonal_period: int = 12) -> pd.DataFrame:
        """
        Detect seasonal outliers
        
        Args:
            df: DataFrame with time series data
            column: Column to analyze
            seasonal_period: Seasonal period (e.g., 12 for monthly data)
            
        Returns:
            DataFrame with seasonal outlier indicators
        """
        try:
            df_outliers = df.copy()
            
            # Calculate seasonal decomposition manually
            seasonal_means = {}
            seasonal_stds = {}
            
            # Group by seasonal period
            df_outliers['seasonal_index'] = df_outliers.index.month
            
            for season in range(1, seasonal_period + 1):
                seasonal_data = df_outliers[df_outliers['seasonal_index'] == season][column]
                if len(seasonal_data) > 1:
                    seasonal_means[season] = seasonal_data.mean()
                    seasonal_stds[season] = seasonal_data.std()
                else:
                    seasonal_means[season] = df_outliers[column].mean()
                    seasonal_stds[season] = df_outliers[column].std()
            
            # Calculate seasonal z-scores
            seasonal_z_scores = []
            for idx, row in df_outliers.iterrows():
                season = row['seasonal_index']
                if seasonal_stds[season] > 0:
                    z_score = abs((row[column] - seasonal_means[season]) / seasonal_stds[season])
                else:
                    z_score = 0
                seasonal_z_scores.append(z_score)
            
            df_outliers['seasonal_zscore'] = seasonal_z_scores
            df_outliers['is_outlier_seasonal'] = df_outliers['seasonal_zscore'] > 2.5
            
            # Clean up
            df_outliers.drop('seasonal_index', axis=1, inplace=True)
            
            outlier_count = df_outliers['is_outlier_seasonal'].sum()
            self.logger.info(f"Detected {outlier_count} seasonal outliers")
            
            return df_outliers
            
        except Exception as e:
            self.logger.error(f"Error detecting seasonal outliers: {str(e)}")
            raise
    
    def detect_time_series_outliers(self, df: pd.DataFrame, column: str = 'sales',
                                  window: int = 5, threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect time series outliers using rolling statistics
        
        Args:
            df: DataFrame with time series data
            column: Column to analyze
            window: Rolling window size
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with time series outlier indicators
        """
        try:
            df_outliers = df.copy()
            
            # Calculate rolling statistics
            rolling_mean = df[column].rolling(window=window, center=True).mean()
            rolling_std = df[column].rolling(window=window, center=True).std()
            
            # Calculate z-scores using rolling statistics
            rolling_z_scores = np.abs((df[column] - rolling_mean) / rolling_std)
            
            df_outliers['rolling_zscore'] = rolling_z_scores
            df_outliers['is_outlier_rolling'] = rolling_z_scores > threshold
            
            # Handle NaN values at the edges
            df_outliers['is_outlier_rolling'] = df_outliers['is_outlier_rolling'].fillna(False)
            df_outliers['rolling_zscore'] = df_outliers['rolling_zscore'].fillna(0)
            
            outlier_count = df_outliers['is_outlier_rolling'].sum()
            self.logger.info(f"Detected {outlier_count} time series outliers")
            
            return df_outliers
            
        except Exception as e:
            self.logger.error(f"Error detecting time series outliers: {str(e)}")
            raise
    
    def comprehensive_outlier_detection(self, df: pd.DataFrame, 
                                      column: str = 'sales') -> pd.DataFrame:
        """
        Perform comprehensive outlier detection using multiple methods
        
        Args:
            df: DataFrame with time series data
            column: Column to analyze
            
        Returns:
            DataFrame with all outlier detection results
        """
        try:
            df_outliers = df.copy()
            
            # Apply all detection methods
            df_outliers = self.detect_statistical_outliers(df_outliers, column, 'iqr')
            df_outliers = self.detect_statistical_outliers(df_outliers, column, 'zscore', 2.5)
            df_outliers = self.detect_isolation_forest_outliers(df_outliers)
            df_outliers = self.detect_seasonal_outliers(df_outliers, column)
            df_outliers = self.detect_time_series_outliers(df_outliers, column)
            
            # Create consensus outlier indicator
            outlier_columns = [col for col in df_outliers.columns if col.startswith('is_outlier_')]
            df_outliers['outlier_votes'] = df_outliers[outlier_columns].sum(axis=1)
            df_outliers['is_outlier_consensus'] = df_outliers['outlier_votes'] >= 2
            
            # Calculate overall outlier score
            score_columns = [col for col in df_outliers.columns if col.startswith('outlier_score_')]
            if score_columns:
                df_outliers['outlier_score_combined'] = df_outliers[score_columns].mean(axis=1)
            
            total_outliers = df_outliers['is_outlier_consensus'].sum()
            self.logger.info(f"Comprehensive outlier detection completed. "
                           f"Found {total_outliers} consensus outliers")
            
            return df_outliers
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive outlier detection: {str(e)}")
            raise
    
    def get_outlier_summary(self, df_outliers: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of outlier detection results
        
        Args:
            df_outliers: DataFrame with outlier detection results
            
        Returns:
            Dictionary with outlier summary
        """
        try:
            summary = {
                'total_records': len(df_outliers),
                'outlier_methods': {},
                'outlier_dates': {},
                'severity_distribution': {}
            }
            
            # Count outliers by method
            outlier_columns = [col for col in df_outliers.columns if col.startswith('is_outlier_')]
            for col in outlier_columns:
                method_name = col.replace('is_outlier_', '')
                count = df_outliers[col].sum()
                percentage = (count / len(df_outliers)) * 100
                summary['outlier_methods'][method_name] = {
                    'count': int(count),
                    'percentage': round(percentage, 2)
                }
            
            # Get outlier dates for consensus outliers
            if 'is_outlier_consensus' in df_outliers.columns:
                outlier_dates = df_outliers[df_outliers['is_outlier_consensus']].index
                summary['outlier_dates']['consensus'] = [str(date) for date in outlier_dates]
            
            # Severity distribution
            if 'outlier_votes' in df_outliers.columns:
                vote_counts = df_outliers['outlier_votes'].value_counts().sort_index()
                summary['severity_distribution'] = {
                    int(votes): int(count) for votes, count in vote_counts.items()
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating outlier summary: {str(e)}")
            raise
    
    def clean_outliers(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Clean outliers from the dataset
        
        Args:
            df: DataFrame with outlier detection results
            method: Cleaning method ('remove', 'interpolate', 'winsorize')
            
        Returns:
            Cleaned DataFrame
        """
        try:
            if 'is_outlier_consensus' not in df.columns:
                self.logger.warning("No consensus outliers found. Returning original data.")
                return df.copy()
            
            df_clean = df.copy()
            outlier_mask = df_clean['is_outlier_consensus']
            
            if method == 'remove':
                # Remove outlier rows
                df_clean = df_clean[~outlier_mask]
                self.logger.info(f"Removed {outlier_mask.sum()} outlier records")
                
            elif method == 'interpolate':
                # Interpolate outlier values
                df_clean.loc[outlier_mask, 'sales'] = np.nan
                df_clean['sales'] = df_clean['sales'].interpolate(method='time')
                self.logger.info(f"Interpolated {outlier_mask.sum()} outlier values")
                
            elif method == 'winsorize':
                # Winsorize outliers to 5th and 95th percentiles
                p05 = df_clean['sales'].quantile(0.05)
                p95 = df_clean['sales'].quantile(0.95)
                
                df_clean.loc[df_clean['sales'] < p05, 'sales'] = p05
                df_clean.loc[df_clean['sales'] > p95, 'sales'] = p95
                self.logger.info(f"Winsorized outliers to 5th-95th percentile range")
                
            else:
                raise ValueError(f"Unsupported cleaning method: {method}")
            
            # Remove outlier detection columns
            outlier_cols = [col for col in df_clean.columns if 'outlier' in col.lower()]
            df_clean = df_clean.drop(columns=outlier_cols)
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error cleaning outliers: {str(e)}")
            raise