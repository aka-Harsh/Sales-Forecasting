"""
Data processing utilities for sales forecasting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from datetime import datetime
import io
import openpyxl


class DataProcessor:
    """Data processing and validation utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path: str = None, file_content: bytes = None, 
                  file_type: str = 'csv') -> pd.DataFrame:
        """
        Load data from file or content
        
        Args:
            file_path: Path to the file
            file_content: File content as bytes
            file_type: Type of file ('csv', 'xlsx', 'xls', 'json')
            
        Returns:
            DataFrame with loaded data
        """
        try:
            if file_content:
                if file_type == 'csv':
                    df = pd.read_csv(io.BytesIO(file_content))
                elif file_type in ['xlsx', 'xls']:
                    df = pd.read_excel(io.BytesIO(file_content))
                elif file_type == 'json':
                    df = pd.read_json(io.BytesIO(file_content))
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")
            elif file_path:
                if file_type == 'csv':
                    df = pd.read_csv(file_path)
                elif file_type in ['xlsx', 'xls']:
                    df = pd.read_excel(file_path)
                elif file_type == 'json':
                    df = pd.read_json(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")
            else:
                raise ValueError("Either file_path or file_content must be provided")
            
            self.logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data structure and quality
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # Check if DataFrame is empty
            if df.empty:
                validation_results['is_valid'] = False
                validation_results['errors'].append("DataFrame is empty")
                return validation_results
            
            # Check for required columns - case insensitive
            date_keywords = ['date', 'month', 'year_month', 'time', 'period']
            sales_keywords = ['sales', 'revenue', 'value', 'amount']
            
            date_columns = [col for col in df.columns 
                          if any(keyword in col.lower() for keyword in date_keywords)]
            
            sales_columns = [col for col in df.columns 
                           if any(keyword in col.lower() for keyword in sales_keywords)]
            
            if not date_columns:
                validation_results['errors'].append("No date column found. Expected columns containing: date, month, year_month, time, period")
            
            if not sales_columns:
                validation_results['errors'].append("No sales/value column found. Expected columns containing: sales, revenue, value, amount")
            
            if validation_results['errors']:
                validation_results['is_valid'] = False
                return validation_results
            
            # Auto-detect columns
            date_col = date_columns[0]
            sales_col = sales_columns[0]
            
            validation_results['info']['date_column'] = date_col
            validation_results['info']['sales_column'] = sales_col
            
            # Check data types and missing values
            missing_dates = df[date_col].isnull().sum()
            missing_sales = df[sales_col].isnull().sum()
            
            if missing_dates > 0:
                validation_results['warnings'].append(f"Found {missing_dates} missing date values")
            
            if missing_sales > 0:
                validation_results['warnings'].append(f"Found {missing_sales} missing sales values")
            
            # Check for duplicates
            duplicate_dates = df[date_col].duplicated().sum()
            if duplicate_dates > 0:
                validation_results['warnings'].append(f"Found {duplicate_dates} duplicate date entries")
            
            # Basic statistics
            validation_results['info']['total_records'] = len(df)
            validation_results['info']['date_range'] = {
                'start': str(df[date_col].min()),
                'end': str(df[date_col].max())
            }
            validation_results['info']['sales_stats'] = {
                'mean': float(df[sales_col].mean()),
                'std': float(df[sales_col].std()),
                'min': float(df[sales_col].min()),
                'max': float(df[sales_col].max())
            }
            
            self.logger.info(f"Data validation completed. Valid: {validation_results['is_valid']}")
            return validation_results
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
            self.logger.error(f"Error during validation: {str(e)}")
            return validation_results
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean and prepare data for modeling
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning report)
        """
        cleaning_report = {
            'original_shape': df.shape,
            'actions_taken': [],
            'final_shape': None
        }
        
        try:
            # Make a copy to avoid modifying original
            df_clean = df.copy()
            
            # Auto-detect columns using case-insensitive search
            date_keywords = ['date', 'month', 'year_month', 'time', 'period']
            sales_keywords = ['sales', 'revenue', 'value', 'amount']
            
            date_col = None
            sales_col = None
            
            # Find date column
            for col in df_clean.columns:
                if any(keyword in col.lower() for keyword in date_keywords):
                    date_col = col
                    break
            
            # Find sales column
            for col in df_clean.columns:
                if any(keyword in col.lower() for keyword in sales_keywords):
                    sales_col = col
                    break
            
            if not date_col or not sales_col:
                raise ValueError("Could not identify date and sales columns")
            
            self.logger.info(f"Detected columns - Date: {date_col}, Sales: {sales_col}")
            
            # Standardize column names
            df_clean = df_clean.rename(columns={
                date_col: 'date',
                sales_col: 'sales'
            })
            
            # Convert date column
            df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
            
            # Remove rows with invalid dates
            invalid_dates = df_clean['date'].isnull().sum()
            if invalid_dates > 0:
                df_clean = df_clean.dropna(subset=['date'])
                cleaning_report['actions_taken'].append(f"Removed {invalid_dates} rows with invalid dates")
            
            # Handle missing sales values
            missing_sales = df_clean['sales'].isnull().sum()
            if missing_sales > 0:
                # Forward fill then backward fill
                df_clean['sales'] = df_clean['sales'].fillna(method='ffill').fillna(method='bfill')
                cleaning_report['actions_taken'].append(f"Filled {missing_sales} missing sales values")
            
            # Remove duplicates (keep last)
            duplicates = df_clean.duplicated(subset=['date']).sum()
            if duplicates > 0:
                df_clean = df_clean.drop_duplicates(subset=['date'], keep='last')
                cleaning_report['actions_taken'].append(f"Removed {duplicates} duplicate date entries")
            
            # Sort by date
            df_clean = df_clean.sort_values('date').reset_index(drop=True)
            
            # Convert sales to numeric
            df_clean['sales'] = pd.to_numeric(df_clean['sales'], errors='coerce')
            
            # Remove rows with non-numeric sales
            invalid_sales = df_clean['sales'].isnull().sum()
            if invalid_sales > 0:
                df_clean = df_clean.dropna(subset=['sales'])
                cleaning_report['actions_taken'].append(f"Removed {invalid_sales} rows with non-numeric sales")
            
            # Set date as index
            df_clean.set_index('date', inplace=True)
            
            cleaning_report['final_shape'] = df_clean.shape
            
            self.logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
            return df_clean, cleaning_report
            
        except Exception as e:
            self.logger.error(f"Error during data cleaning: {str(e)}")
            raise
    
    def detect_frequency(self, df: pd.DataFrame) -> str:
        """
        Detect the frequency of the time series
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            Frequency string ('D', 'W', 'M', 'Q', 'Y')
        """
        try:
            # Calculate time differences
            time_diffs = df.index.to_series().diff().dropna()
            
            # Get most common difference
            most_common_diff = time_diffs.mode().iloc[0]
            
            if most_common_diff.days <= 1:
                return 'D'  # Daily
            elif most_common_diff.days <= 7:
                return 'W'  # Weekly
            elif most_common_diff.days <= 31:
                return 'M'  # Monthly
            elif most_common_diff.days <= 93:
                return 'Q'  # Quarterly
            else:
                return 'Y'  # Yearly
                
        except Exception as e:
            self.logger.warning(f"Could not detect frequency: {str(e)}")
            return 'M'  # Default to monthly
    
    def resample_data(self, df: pd.DataFrame, frequency: str = 'M') -> pd.DataFrame:
        """
        Resample data to specified frequency
        
        Args:
            df: DataFrame with datetime index
            frequency: Target frequency ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            Resampled DataFrame
        """
        try:
            # Resample and aggregate
            if frequency == 'D':
                df_resampled = df.resample('D').sum()
            elif frequency == 'W':
                df_resampled = df.resample('W').sum()
            elif frequency == 'M':
                df_resampled = df.resample('M').sum()
            elif frequency == 'Q':
                df_resampled = df.resample('Q').sum()
            elif frequency == 'Y':
                df_resampled = df.resample('Y').sum()
            else:
                raise ValueError(f"Unsupported frequency: {frequency}")
            
            # Forward fill missing values
            df_resampled = df_resampled.fillna(method='ffill')
            
            self.logger.info(f"Data resampled to {frequency} frequency. Shape: {df_resampled.shape}")
            return df_resampled
            
        except Exception as e:
            self.logger.error(f"Error during resampling: {str(e)}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive data summary
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data summary
        """
        try:
            summary = {
                'shape': df.shape,
                'date_range': {
                    'start': str(df.index.min()),
                    'end': str(df.index.max()),
                    'total_periods': len(df)
                },
                'sales_statistics': {
                    'mean': float(df['sales'].mean()),
                    'median': float(df['sales'].median()),
                    'std': float(df['sales'].std()),
                    'min': float(df['sales'].min()),
                    'max': float(df['sales'].max()),
                    'q25': float(df['sales'].quantile(0.25)),
                    'q75': float(df['sales'].quantile(0.75))
                },
                'frequency': self.detect_frequency(df),
                'missing_values': int(df['sales'].isnull().sum()),
                'zero_values': int((df['sales'] == 0).sum())
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating data summary: {str(e)}")
            raise