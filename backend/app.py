"""
Main Flask application factory for Advanced Sales Forecasting Platform
"""

import os
import logging
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from backend.config import get_config
from backend.api.forecast_routes import forecast_bp
from backend.api.analytics_routes import analytics_bp
from backend.api.data_routes import data_bp


def create_app(config_name: str = None) -> Flask:
    """Create and configure Flask application"""
    
    app = Flask(
        __name__,
        instance_relative_config=True,
        static_folder='../frontend/static',
        template_folder='../frontend/templates'
    )
    
    # Load configuration
    config = get_config(config_name)
    
    # Configure Flask app
    app.config['SECRET_KEY'] = config.get('secret_key')
    app.config['DEBUG'] = config.get('debug', False)
    app.config['TESTING'] = config.get('testing', False)
    app.config['UPLOAD_FOLDER'] = config.get('upload_folder')
    app.config['MAX_CONTENT_LENGTH'] = config.get('max_content_length')
    
    # Store config object for access in routes
    app.config['APP_CONFIG'] = config
    
    # Enable CORS
    CORS(app)
    
    # Initialize SocketIO for real-time updates
    socketio = SocketIO(app, cors_allowed_origins="*")
    app.socketio = socketio
    
    # Create upload directories
    os.makedirs(config.get('upload_folder'), exist_ok=True)
    os.makedirs('models/trained', exist_ok=True)
    
    # Configure logging
    if not app.config['TESTING']:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(name)s %(message)s'
        )
    
    # Register blueprints
    app.register_blueprint(forecast_bp, url_prefix='/api/forecast')
    app.register_blueprint(analytics_bp, url_prefix='/api/analytics')
    app.register_blueprint(data_bp, url_prefix='/api/data')
    
    # Main routes
    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('dashboard.html')
    
    @app.route('/forecast')
    def forecast_page():
        """Forecasting page"""
        return render_template('forecast.html')
    
    @app.route('/analytics')
    def analytics_page():
        """Analytics page"""
        return render_template('analytics.html')
    
    @app.route('/data-management')
    def data_management():
        """Data management page"""
        return render_template('data-management.html')
    
    @app.route('/api/health')
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'environment': os.getenv('FLASK_ENV', 'development'),
            'version': '1.0.0'
        })
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request', 'message': str(error)}), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found', 'message': str(error)}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error', 'message': str(error)}), 500
    
    # SocketIO events for real-time updates
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        app.logger.info('Client connected')
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        app.logger.info('Client disconnected')
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)