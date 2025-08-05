"""
Analytics API routes for advanced data analysis
"""

from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
import traceback
from datetime import datetime, timedelta

from backend.utils.data_processor import DataProcessor
from backend.utils.anomaly_detector import AnomalyDetector
from backend.utils.feature_engineer import FeatureEngineer
from backend.utils.model_evaluator import ModelEvaluator

analytics_bp = Blueprint('analytics', __name__)
logger = logging.getLogger(__name__)


@analytics_bp.route('/kpis', methods=['GET'])
def get_key_performance_indicators():
    """Get key performance indicators for the dashboard"""
    try:
        # Load sample data (in production, this would come from user's data)
        processor = DataProcessor()
        sample_path = 'data/sample/sample_data.csv'
        df = processor.load_data(sample_path)
        df_clean, _ = processor.clean_data(df)
        
        # Calculate KPIs
        current_month_sales = float(df_clean['sales'].iloc[-1])
        previous_month_sales = float(df_clean['sales'].iloc[-2]) if len(df_clean) > 1 else current_month_sales
        
        # Year-over-year comparison
        if len(df_clean) >= 12:
            yoy_sales = float(df_clean['sales'].iloc[-13])  # 12 months ago
            yoy_growth = ((current_month_sales - yoy_sales) / yoy_sales) * 100
        else:
            yoy_growth = 0.0
        
        # Month-over-month growth
        mom_growth = ((current_month_sales - previous_month_sales) / previous_month_sales) * 100
        
        # Average monthly sales
        avg_monthly_sales = float(df_clean['sales'].mean())
        
        # Trend calculation (last 6 months)
        recent_data = df_clean.tail(6)
        if len(recent_data) >= 2:
            trend_slope = np.polyfit(range(len(recent_data)), recent_data['sales'], 1)[0]
            trend_direction = 'increasing' if trend_slope > 0 else 'decreasing' if trend_slope < 0 else 'stable'
        else:
            trend_direction = 'stable'
        
        # Volatility (coefficient of variation)
        volatility = float(df_clean['sales'].std() / df_clean['sales'].mean() * 100)
        
        # Forecast accuracy (simulate based on recent data)
        forecast_accuracy = max(70, 100 - abs(volatility) / 2)  # Simple simulation
        
        kpis = {
            'current_month_sales': {
                'value': current_month_sales,
                'formatted': f"${current_month_sales:,.0f}",
                'unit': 'USD'
            },
            'mom_growth': {
                'value': mom_growth,
                'formatted': f"{mom_growth:+.1f}%",
                'trend': 'up' if mom_growth > 0 else 'down' if mom_growth < 0 else 'neutral'
            },
            'yoy_growth': {
                'value': yoy_growth,
                'formatted': f"{yoy_growth:+.1f}%",
                'trend': 'up' if yoy_growth > 0 else 'down' if yoy_growth < 0 else 'neutral'
            },
            'avg_monthly_sales': {
                'value': avg_monthly_sales,
                'formatted': f"${avg_monthly_sales:,.0f}",
                'unit': 'USD'
            },
            'trend_direction': {
                'value': trend_direction,
                'formatted': trend_direction.title(),
                'icon': '↗️' if trend_direction == 'increasing' else '↘️' if trend_direction == 'decreasing' else '→'
            },
            'volatility': {
                'value': volatility,
                'formatted': f"{volatility:.1f}%",
                'level': 'high' if volatility > 20 else 'medium' if volatility > 10 else 'low'
            },
            'forecast_accuracy': {
                'value': forecast_accuracy,
                'formatted': f"{forecast_accuracy:.1f}%",
                'level': 'excellent' if forecast_accuracy > 90 else 'good' if forecast_accuracy > 80 else 'fair'
            }
        }
        
        return jsonify({
            'success': True,
            'kpis': kpis,
            'data_period': {
                'start': str(df_clean.index.min()),
                'end': str(df_clean.index.max()),
                'total_months': len(df_clean)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting KPIs: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/trends', methods=['POST'])
def analyze_trends():
    """Analyze sales trends and patterns"""
    try:
        data = request.get_json()
        analysis_type = data.get('type', 'comprehensive')  # 'comprehensive', 'seasonal', 'trend'
        
        # Load data
        processor = DataProcessor()
        sample_path = 'data/sample/sample_data.csv'
        df = processor.load_data(sample_path)
        df_clean, _ = processor.clean_data(df)
        
        # Feature engineering for trend analysis
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.create_seasonal_features(df_clean)
        df_features = feature_engineer.create_trend_features(df_features)
        
        analysis_results = {}
        
        if analysis_type in ['comprehensive', 'trend']:
            # Trend analysis
            # Linear trend
            x = np.arange(len(df_clean))
            trend_coef = np.polyfit(x, df_clean['sales'], 1)
            trend_line = np.poly1d(trend_coef)(x)
            
            # Growth rate analysis
            growth_rates = df_clean['sales'].pct_change().dropna()
            
            analysis_results['trend_analysis'] = {
                'linear_trend': {
                    'slope': float(trend_coef[0]),
                    'intercept': float(trend_coef[1]),
                    'trend_line': trend_line.tolist(),
                    'interpretation': 'increasing' if trend_coef[0] > 0 else 'decreasing' if trend_coef[0] < 0 else 'stable'
                },
                'growth_rates': {
                    'mean': float(growth_rates.mean()),
                    'std': float(growth_rates.std()),
                    'recent_3m': float(growth_rates.tail(3).mean()),
                    'recent_12m': float(growth_rates.tail(12).mean()) if len(growth_rates) >= 12 else None
                }
            }
        
        if analysis_type in ['comprehensive', 'seasonal']:
            # Seasonal analysis
            df_clean['month'] = df_clean.index.month
            df_clean['quarter'] = df_clean.index.quarter
            df_clean['year'] = df_clean.index.year
            
            # Monthly patterns
            monthly_avg = df_clean.groupby('month')['sales'].agg(['mean', 'std', 'count']).to_dict()
            quarterly_avg = df_clean.groupby('quarter')['sales'].agg(['mean', 'std', 'count']).to_dict()
            
            # Peak and trough months
            monthly_means = df_clean.groupby('month')['sales'].mean()
            peak_month = int(monthly_means.idxmax())
            trough_month = int(monthly_means.idxmin())
            
            analysis_results['seasonal_analysis'] = {
                'monthly_patterns': {
                    'averages': {str(k): float(v) for k, v in monthly_avg['mean'].items()},
                    'std_devs': {str(k): float(v) for k, v in monthly_avg['std'].items()},
                    'counts': {str(k): int(v) for k, v in monthly_avg['count'].items()}
                },
                'quarterly_patterns': {
                    'averages': {str(k): float(v) for k, v in quarterly_avg['mean'].items()},
                    'std_devs': {str(k): float(v) for k, v in quarterly_avg['std'].items()}
                },
                'peak_month': peak_month,
                'trough_month': trough_month,
                'seasonality_strength': float(monthly_means.std() / monthly_means.mean())
            }
        
        if analysis_type == 'comprehensive':
            # Cyclical patterns (longer-term cycles)
            yearly_avg = df_clean.groupby('year')['sales'].mean()
            if len(yearly_avg) >= 3:
                year_growth = yearly_avg.pct_change().dropna()
                analysis_results['cyclical_analysis'] = {
                    'yearly_averages': yearly_avg.to_dict(),
                    'year_over_year_growth': year_growth.to_dict(),
                    'cycle_length_estimate': 'Insufficient data' if len(yearly_avg) < 5 else 'Analysis needed'
                }
        
        return jsonify({
            'success': True,
            'analysis_type': analysis_type,
            'results': analysis_results,
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'data_points_analyzed': len(df_clean),
                'date_range': {
                    'start': str(df_clean.index.min()),
                    'end': str(df_clean.index.max())
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error analyzing trends: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/anomalies', methods=['POST'])
def detect_anomalies():
    """Detect and analyze anomalies in sales data"""
    try:
        data = request.get_json()
        detection_method = data.get('method', 'comprehensive')
        sensitivity = data.get('sensitivity', 'medium')  # 'low', 'medium', 'high'
        
        # Load data
        processor = DataProcessor()
        sample_path = 'data/sample/sample_data.csv'
        df = processor.load_data(sample_path)
        df_clean, _ = processor.clean_data(df)
        
        # Initialize anomaly detector
        detector = AnomalyDetector()
        
        # Adjust sensitivity
        if sensitivity == 'high':
            contamination = 0.15
            threshold = 1.5
        elif sensitivity == 'low':
            contamination = 0.05
            threshold = 2.5
        else:  # medium
            contamination = 0.1
            threshold = 2.0
        
        # Detect anomalies based on method
        if detection_method == 'comprehensive':
            df_anomalies = detector.comprehensive_outlier_detection(df_clean)
        elif detection_method == 'isolation_forest':
            df_anomalies = detector.detect_isolation_forest_outliers(df_clean, contamination=contamination)
        elif detection_method == 'statistical':
            df_anomalies = detector.detect_statistical_outliers(df_clean, threshold=threshold)
        elif detection_method == 'seasonal':
            df_anomalies = detector.detect_seasonal_outliers(df_clean)
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported detection method: {detection_method}'
            }), 400
        
        # Get anomaly summary
        anomaly_summary = detector.get_outlier_summary(df_anomalies)
        
        # Prepare anomaly data for visualization
        anomaly_data = []
        normal_data = []
        
        for idx, row in df_anomalies.iterrows():
            data_point = {
                'date': str(idx),
                'sales': float(row['sales']),
                'is_anomaly': False
            }
            
            # Check if it's an anomaly using consensus or main method
            if 'is_outlier_consensus' in df_anomalies.columns:
                data_point['is_anomaly'] = bool(row['is_outlier_consensus'])
            elif f'is_outlier_{detection_method}' in df_anomalies.columns:
                data_point['is_anomaly'] = bool(row[f'is_outlier_{detection_method}'])
            
            if data_point['is_anomaly']:
                # Add anomaly score if available
                if 'outlier_score_combined' in df_anomalies.columns:
                    data_point['anomaly_score'] = float(row['outlier_score_combined'])
                anomaly_data.append(data_point)
            else:
                normal_data.append(data_point)
        
        # Analyze anomaly patterns
        anomaly_patterns = {}
        if anomaly_data:
            anomaly_df = pd.DataFrame(anomaly_data)
            anomaly_df['date'] = pd.to_datetime(anomaly_df['date'])
            anomaly_df['month'] = anomaly_df['date'].dt.month
            anomaly_df['year'] = anomaly_df['date'].dt.year
            
            anomaly_patterns = {
                'monthly_distribution': anomaly_df['month'].value_counts().to_dict(),
                'yearly_distribution': anomaly_df['year'].value_counts().to_dict(),
                'average_anomaly_value': float(anomaly_df['sales'].mean()),
                'anomaly_magnitude': {
                    'min': float(anomaly_df['sales'].min()),
                    'max': float(anomaly_df['sales'].max()),
                    'std': float(anomaly_df['sales'].std())
                }
            }
        
        return jsonify({
            'success': True,
            'detection_method': detection_method,
            'sensitivity': sensitivity,
            'anomaly_summary': anomaly_summary,
            'anomaly_data': anomaly_data,
            'normal_data': normal_data[:100],  # Limit normal data for performance
            'anomaly_patterns': anomaly_patterns,
            'total_anomalies': len(anomaly_data),
            'total_normal': len(normal_data)
        })
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/correlations', methods=['POST'])
def analyze_correlations():
    """Analyze correlations between different features and sales"""
    try:
        data = request.get_json()
        include_external = data.get('include_external', False)  # Future: external data sources
        
        # Load data
        processor = DataProcessor()
        sample_path = 'data/sample/sample_data.csv'
        df = processor.load_data(sample_path)
        df_clean, _ = processor.clean_data(df)
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.create_all_features(df_clean)
        
        # Calculate correlations with sales
        correlations = df_features.corr()['sales'].abs().sort_values(ascending=False)
        
        # Remove self-correlation
        correlations = correlations.drop('sales')
        
        # Get top correlations
        top_correlations = correlations.head(15).to_dict()
        
        # Categorize features
        feature_categories = {
            'lag_features': [k for k in top_correlations.keys() if 'lag' in k.lower()],
            'rolling_features': [k for k in top_correlations.keys() if 'rolling' in k.lower()],
            'seasonal_features': [k for k in top_correlations.keys() if any(x in k.lower() for x in ['month', 'quarter', 'seasonal'])],
            'trend_features': [k for k in top_correlations.keys() if 'trend' in k.lower()],
            'statistical_features': [k for k in top_correlations.keys() if any(x in k.lower() for x in ['growth', 'volatility', 'ma_'])]
        }
        
        # Create correlation matrix for heatmap (top features only)
        top_feature_names = list(correlations.head(10).index) + ['sales']
        correlation_matrix = df_features[top_feature_names].corr().to_dict()
        
        return jsonify({
            'success': True,
            'top_correlations': top_correlations,
            'feature_categories': feature_categories,
            'correlation_matrix': correlation_matrix,
            'insights': {
                'strongest_predictor': correlations.index[0] if len(correlations) > 0 else None,
                'strongest_correlation': float(correlations.iloc[0]) if len(correlations) > 0 else 0,
                'weak_predictors': [k for k, v in top_correlations.items() if v < 0.3],
                'strong_predictors': [k for k, v in top_correlations.items() if v > 0.7]
            }
        })
        
    except Exception as e:
        logger.error(f"Error analyzing correlations: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/forecasting-insights', methods=['GET'])
def get_forecasting_insights():
    """Get insights about forecasting performance and recommendations"""
    try:
        # This would typically analyze multiple model performances
        # For now, we'll simulate insights based on typical scenarios
        
        insights = {
            'model_performance': {
                'best_performing_model': 'Ensemble',
                'accuracy_metrics': {
                    'ensemble': {'mape': 8.5, 'mae': 245.3, 'rmse': 312.1},
                    'prophet': {'mape': 12.3, 'mae': 298.7, 'rmse': 385.4},
                    'sarimax': {'mape': 15.2, 'mae': 367.8, 'rmse': 445.2},
                    'lstm': {'mape': 11.8, 'mae': 289.4, 'rmse': 378.9},
                    'linear': {'mape': 18.7, 'mae': 423.5, 'rmse': 501.3}
                },
                'recommendations': [
                    'Ensemble model shows best overall performance',
                    'Prophet handles seasonality well for this dataset',
                    'Consider retraining LSTM with more features',
                    'Linear model may benefit from feature selection'
                ]
            },
            'data_quality': {
                'completeness_score': 95.8,
                'consistency_score': 92.3,
                'outlier_percentage': 3.2,
                'seasonality_strength': 'Strong',
                'trend_clarity': 'Clear upward trend',
                'recommendations': [
                    'Data quality is good for forecasting',
                    'Consider outlier treatment for better accuracy',
                    'Strong seasonality suggests seasonal models will perform well'
                ]
            },
            'forecast_reliability': {
                'confidence_level': 'High',
                'prediction_intervals': 'Reliable',
                'seasonal_accuracy': 'Excellent',
                'trend_accuracy': 'Good',
                'factors_affecting_accuracy': [
                    'Strong seasonal patterns improve predictability',
                    'Clear trend makes medium-term forecasts reliable',
                    'Some volatility in recent periods'
                ]
            },
            'business_insights': {
                'peak_sales_months': ['November', 'December'],
                'growth_opportunity_months': ['February', 'March'],
                'volatility_periods': ['Q1', 'Mid-summer'],
                'forecast_horizon_recommendations': {
                    '1-3_months': 'Very reliable',
                    '4-6_months': 'Reliable',
                    '7-12_months': 'Moderate reliability',
                    'beyond_12_months': 'Use with caution'
                }
            }
        }
        
        return jsonify({
            'success': True,
            'insights': insights,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting forecasting insights: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/scenario-analysis', methods=['POST'])
def scenario_analysis():
    """Perform what-if scenario analysis"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No scenario data provided'
            }), 400
        
        scenario_type = data.get('type', 'growth_rate')  # 'growth_rate', 'seasonal_adjustment', 'trend_change'
        parameters = data.get('parameters', {})
        
        # Load base forecast data
        processor = DataProcessor()
        sample_path = 'data/sample/sample_data.csv'
        df = processor.load_data(sample_path)
        df_clean, _ = processor.clean_data(df)
        
        # Get base forecast (simplified - would use actual trained model)
        last_value = df_clean['sales'].iloc[-1]
        base_forecast = []
        
        # Simple trend continuation for base scenario
        trend_slope = np.polyfit(range(len(df_clean.tail(12))), df_clean.tail(12)['sales'], 1)[0]
        
        for i in range(12):  # 12-month forecast
            # Add trend and seasonal component
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * (i + df_clean.index[-1].month) / 12)
            forecasted_value = last_value + (i + 1) * trend_slope * seasonal_factor
            base_forecast.append(forecasted_value)
        
        # Apply scenario adjustments
        scenario_forecast = base_forecast.copy()
        
        if scenario_type == 'growth_rate':
            growth_adjustment = parameters.get('growth_rate_change', 0) / 100  # Convert percentage
            for i in range(len(scenario_forecast)):
                scenario_forecast[i] = scenario_forecast[i] * (1 + growth_adjustment * (i + 1) / 12)
        
        elif scenario_type == 'seasonal_adjustment':
            seasonal_boost = parameters.get('seasonal_boost', 0) / 100
            boost_months = parameters.get('boost_months', [11, 12])  # Nov, Dec
            
            for i in range(len(scenario_forecast)):
                month = (df_clean.index[-1].month + i) % 12 + 1
                if month in boost_months:
                    scenario_forecast[i] = scenario_forecast[i] * (1 + seasonal_boost)
        
        elif scenario_type == 'trend_change':
            new_trend = parameters.get('new_trend_slope', trend_slope)
            for i in range(len(scenario_forecast)):
                scenario_forecast[i] = last_value + (i + 1) * new_trend
        
        # Calculate impact
        total_base = sum(base_forecast)
        total_scenario = sum(scenario_forecast)
        impact_percentage = ((total_scenario - total_base) / total_base) * 100
        
        # Generate future dates
        future_dates = pd.date_range(
            start=df_clean.index[-1] + pd.DateOffset(months=1),
            periods=12,
            freq='M'
        ).strftime('%Y-%m-%d').tolist()
        
        return jsonify({
            'success': True,
            'scenario_type': scenario_type,
            'parameters': parameters,
            'base_forecast': base_forecast,
            'scenario_forecast': scenario_forecast,
            'dates': future_dates,
            'impact_analysis': {
                'total_base_forecast': total_base,
                'total_scenario_forecast': total_scenario,
                'absolute_impact': total_scenario - total_base,
                'percentage_impact': impact_percentage,
                'monthly_differences': [s - b for s, b in zip(scenario_forecast, base_forecast)]
            },
            'summary': {
                'scenario_name': f"{scenario_type.replace('_', ' ').title()} Analysis",
                'impact_description': f"{'Positive' if impact_percentage > 0 else 'Negative'} impact of {abs(impact_percentage):.1f}%",
                'recommendation': 'Favorable scenario' if impact_percentage > 5 else 'Neutral impact' if abs(impact_percentage) <= 5 else 'Challenging scenario'
            }
        })
        
    except Exception as e:
        logger.error(f"Error in scenario analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500