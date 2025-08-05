"""
Data management API routes
"""

from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import os
import json
from typing import Dict, Any
import logging
import traceback
from datetime import datetime

from backend.utils.data_processor import DataProcessor
from backend.utils.anomaly_detector import AnomalyDetector

data_bp = Blueprint('data', __name__)
logger = logging.getLogger(__name__)


def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed_extensions = current_app.config.get('APP_CONFIG').get('allowed_extensions', ['csv', 'xlsx', 'xls', 'json'])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@data_bp.route('/upload', methods=['POST'])
def upload_data():
    """Upload and validate data file"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'File type not allowed'
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        
        upload_folder = current_app.config.get('UPLOAD_FOLDER')
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        # Process and validate data
        processor = DataProcessor()
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        # Load data
        df = processor.load_data(filepath, file_type=file_extension)
        
        # Validate data
        validation_results = processor.validate_data(df)
        
        # Get data summary
        if validation_results['is_valid']:
            # Clean data for summary
            df_clean, cleaning_report = processor.clean_data(df)
            data_summary = processor.get_data_summary(df_clean)
        else:
            data_summary = {}
            cleaning_report = {}
        
        response = {
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'validation': validation_results,
            'data_summary': data_summary,
            'cleaning_report': cleaning_report,
            'upload_timestamp': timestamp
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/validate', methods=['POST'])
def validate_data():
    """Validate uploaded data"""
    try:
        data = request.get_json()
        
        if not data or 'filepath' not in data:
            return jsonify({
                'success': False,
                'error': 'No filepath provided'
            }), 400
        
        filepath = data['filepath']
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        # Load and validate data
        processor = DataProcessor()
        file_extension = filepath.rsplit('.', 1)[1].lower()
        df = processor.load_data(filepath, file_type=file_extension)
        
        validation_results = processor.validate_data(df)
        
        response = {
            'success': True,
            'validation': validation_results
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/preview', methods=['POST'])
def preview_data():
    """Preview uploaded data"""
    try:
        data = request.get_json()
        
        if not data or 'filepath' not in data:
            return jsonify({
                'success': False,
                'error': 'No filepath provided'
            }), 400
        
        filepath = data['filepath']
        rows = data.get('rows', 50)  # Default to 50 rows
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        # Load data
        processor = DataProcessor()
        file_extension = filepath.rsplit('.', 1)[1].lower()
        df = processor.load_data(filepath, file_type=file_extension)
        
        # Get preview data
        preview_df = df.head(rows)
        
        # Convert to JSON-serializable format
        preview_data = {
            'columns': df.columns.tolist(),
            'data': preview_df.to_dict('records'),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': df.shape,
            'index_info': {
                'name': df.index.name,
                'type': str(type(df.index).__name__)
            }
        }
        
        # Add date range if datetime index
        if hasattr(df.index, 'min'):
            try:
                preview_data['date_range'] = {
                    'start': str(df.index.min()),
                    'end': str(df.index.max())
                }
            except:
                pass
        
        response = {
            'success': True,
            'preview': preview_data
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error previewing data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/clean', methods=['POST'])
def clean_data():
    """Clean and process data"""
    try:
        data = request.get_json()
        
        if not data or 'filepath' not in data:
            return jsonify({
                'success': False,
                'error': 'No filepath provided'
            }), 400
        
        filepath = data['filepath']
        cleaning_options = data.get('options', {})
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        # Load data
        processor = DataProcessor()
        file_extension = filepath.rsplit('.', 1)[1].lower()
        df = processor.load_data(filepath, file_type=file_extension)
        
        # Clean data
        df_clean, cleaning_report = processor.clean_data(df)
        
        # Save cleaned data
        clean_filename = filepath.replace('.', '_cleaned.')
        df_clean.to_csv(clean_filename)
        
        # Get summary of cleaned data
        data_summary = processor.get_data_summary(df_clean)
        
        response = {
            'success': True,
            'cleaned_filepath': clean_filename,
            'cleaning_report': cleaning_report,
            'data_summary': data_summary,
            'original_shape': df.shape,
            'cleaned_shape': df_clean.shape
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/detect-outliers', methods=['POST'])
def detect_outliers():
    """Detect outliers in the data"""
    try:
        data = request.get_json()
        
        if not data or 'filepath' not in data:
            return jsonify({
                'success': False,
                'error': 'No filepath provided'
            }), 400
        
        filepath = data['filepath']
        detection_methods = data.get('methods', ['comprehensive'])
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        # Load and clean data
        processor = DataProcessor()
        file_extension = filepath.rsplit('.', 1)[1].lower()
        df = processor.load_data(filepath, file_type=file_extension)
        df_clean, _ = processor.clean_data(df)
        
        # Detect outliers
        detector = AnomalyDetector()
        
        if 'comprehensive' in detection_methods:
            df_outliers = detector.comprehensive_outlier_detection(df_clean)
        else:
            df_outliers = df_clean.copy()
            for method in detection_methods:
                if method == 'iqr':
                    df_outliers = detector.detect_statistical_outliers(df_outliers, method='iqr')
                elif method == 'zscore':
                    df_outliers = detector.detect_statistical_outliers(df_outliers, method='zscore')
                elif method == 'isolation_forest':
                    df_outliers = detector.detect_isolation_forest_outliers(df_outliers)
                elif method == 'seasonal':
                    df_outliers = detector.detect_seasonal_outliers(df_outliers)
        
        # Get outlier summary
        outlier_summary = detector.get_outlier_summary(df_outliers)
        
        # Get outlier data for visualization
        outlier_data = []
        if 'is_outlier_consensus' in df_outliers.columns:
            outlier_dates = df_outliers[df_outliers['is_outlier_consensus']].index
            outlier_values = df_outliers[df_outliers['is_outlier_consensus']]['sales'].values
            
            for date, value in zip(outlier_dates, outlier_values):
                outlier_data.append({
                    'date': str(date),
                    'value': float(value),
                    'is_outlier': True
                })
        
        response = {
            'success': True,
            'outlier_summary': outlier_summary,
            'outlier_data': outlier_data,
            'detection_methods': detection_methods
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/statistics', methods=['POST'])
def get_data_statistics():
    """Get comprehensive data statistics"""
    try:
        data = request.get_json()
        
        if not data or 'filepath' not in data:
            return jsonify({
                'success': False,
                'error': 'No filepath provided'
            }), 400
        
        filepath = data['filepath']
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        # Load and clean data
        processor = DataProcessor()
        file_extension = filepath.rsplit('.', 1)[1].lower()
        df = processor.load_data(filepath, file_type=file_extension)
        df_clean, _ = processor.clean_data(df)
        
        # Calculate comprehensive statistics
        stats = {
            'basic_stats': {
                'count': int(len(df_clean)),
                'mean': float(df_clean['sales'].mean()),
                'median': float(df_clean['sales'].median()),
                'std': float(df_clean['sales'].std()),
                'min': float(df_clean['sales'].min()),
                'max': float(df_clean['sales'].max()),
                'q25': float(df_clean['sales'].quantile(0.25)),
                'q75': float(df_clean['sales'].quantile(0.75))
            },
            'distribution_stats': {
                'skewness': float(df_clean['sales'].skew()),
                'kurtosis': float(df_clean['sales'].kurtosis()),
                'variance': float(df_clean['sales'].var())
            },
            'time_series_stats': {
                'frequency': processor.detect_frequency(df_clean),
                'date_range': {
                    'start': str(df_clean.index.min()),
                    'end': str(df_clean.index.max()),
                    'total_periods': len(df_clean)
                }
            }
        }
        
        # Calculate year-over-year growth if enough data
        if len(df_clean) >= 12:
            try:
                yoy_growth = df_clean['sales'].pct_change(periods=12).dropna()
                stats['growth_stats'] = {
                    'mean_yoy_growth': float(yoy_growth.mean()),
                    'std_yoy_growth': float(yoy_growth.std()),
                    'latest_yoy_growth': float(yoy_growth.iloc[-1]) if len(yoy_growth) > 0 else None
                }
            except:
                stats['growth_stats'] = {}
        
        # Monthly/seasonal statistics
        try:
            df_clean['month'] = df_clean.index.month
            monthly_stats = df_clean.groupby('month')['sales'].agg(['mean', 'std']).to_dict()
            stats['seasonal_stats'] = {
                'monthly_means': {str(k): float(v) for k, v in monthly_stats['mean'].items()},
                'monthly_stds': {str(k): float(v) for k, v in monthly_stats['std'].items()}
            }
        except:
            stats['seasonal_stats'] = {}
        
        response = {
            'success': True,
            'statistics': stats
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting data statistics: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/export', methods=['POST'])
def export_data():
    """Export processed data"""
    try:
        data = request.get_json()
        
        if not data or 'filepath' not in data:
            return jsonify({
                'success': False,
                'error': 'No filepath provided'
            }), 400
        
        filepath = data['filepath']
        export_format = data.get('format', 'csv')  # csv, excel, json
        include_forecasts = data.get('include_forecasts', False)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        # Load and clean data
        processor = DataProcessor()
        file_extension = filepath.rsplit('.', 1)[1].lower()
        df = processor.load_data(filepath, file_type=file_extension)
        df_clean, _ = processor.clean_data(df)
        
        # Add forecasts if requested
        if include_forecasts:
            # This would require loading a trained model and making predictions
            # For now, we'll just export the cleaned data
            pass
        
        # Generate export filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"sales_data_export_{timestamp}"
        
        export_path = os.path.join(current_app.config.get('UPLOAD_FOLDER'), export_filename)
        
        # Export data
        if export_format == 'csv':
            export_path += '.csv'
            df_clean.to_csv(export_path)
        elif export_format == 'excel':
            export_path += '.xlsx'
            df_clean.to_excel(export_path)
        elif export_format == 'json':
            export_path += '.json'
            df_clean.to_json(export_path, orient='records', date_format='iso')
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported export format: {export_format}'
            }), 400
        
        response = {
            'success': True,
            'export_path': export_path,
            'export_format': export_format,
            'records_exported': len(df_clean),
            'export_timestamp': timestamp
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/sample', methods=['GET'])
def get_sample_data():
    """Get sample data for demonstration"""
    try:
        # Load sample data
        processor = DataProcessor()
        sample_path = 'data/sample/sample_data.csv'
        
        if not os.path.exists(sample_path):
            # Create sample data if it doesn't exist
            dates = pd.date_range('1964-01-01', '2023-12-01', freq='M')
            # Generate realistic sales data with trend and seasonality
            trend = np.linspace(2000, 5000, len(dates))
            seasonal = 500 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
            noise = np.random.normal(0, 200, len(dates))
            sales = trend + seasonal + noise
            
            sample_df = pd.DataFrame({
                'Year_Month': dates.strftime('%Y-%m'),
                'Sales': sales.astype(int)
            })
            
            os.makedirs('data/sample', exist_ok=True)
            sample_df.to_csv(sample_path, index=False)
        
        df = processor.load_data(sample_path)
        
        # Get data summary
        validation_results = processor.validate_data(df)
        
        if validation_results['is_valid']:
            df_clean, cleaning_report = processor.clean_data(df)
            data_summary = processor.get_data_summary(df_clean)
            
            # Get recent data for preview
            recent_data = df_clean.tail(50).reset_index()
            recent_data['date'] = recent_data['date'].dt.strftime('%Y-%m-%d')
            
            response = {
                'success': True,
                'sample_data': recent_data.to_dict('records'),
                'data_summary': data_summary,
                'validation': validation_results,
                'cleaning_report': cleaning_report
            }
        else:
            response = {
                'success': False,
                'validation': validation_results,
                'error': 'Sample data validation failed'
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting sample data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/list-uploads', methods=['GET'])
def list_uploaded_files():
    """List all uploaded files"""
    try:
        upload_folder = current_app.config.get('UPLOAD_FOLDER')
        
        if not os.path.exists(upload_folder):
            return jsonify({
                'success': True,
                'files': []
            })
        
        files = []
        for filename in os.listdir(upload_folder):
            if allowed_file(filename):
                filepath = os.path.join(upload_folder, filename)
                file_stats = os.stat(filepath)
                
                files.append({
                    'filename': filename,
                    'filepath': filepath,
                    'size': file_stats.st_size,
                    'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    'extension': filename.rsplit('.', 1)[1].lower()
                })
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        response = {
            'success': True,
            'files': files,
            'total_files': len(files)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error listing uploaded files: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_bp.route('/delete', methods=['DELETE'])
def delete_file():
    """Delete an uploaded file"""
    try:
        data = request.get_json()
        
        if not data or 'filepath' not in data:
            return jsonify({
                'success': False,
                'error': 'No filepath provided'
            }), 400
        
        filepath = data['filepath']
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        # Security check: ensure file is in upload folder
        upload_folder = current_app.config.get('UPLOAD_FOLDER')
        if not filepath.startswith(upload_folder):
            return jsonify({
                'success': False,
                'error': 'Invalid filepath'
            }), 400
        
        # Delete file
        os.remove(filepath)
        
        response = {
            'success': True,
            'message': f'File deleted successfully'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500