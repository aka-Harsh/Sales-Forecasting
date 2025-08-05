#!/usr/bin/env python3
"""
Advanced Sales Forecasting Platform
Entry point for the application
"""

import os
from backend.app import create_app

if __name__ == "__main__":
    # Set environment if not already set
    if not os.getenv('FLASK_ENV'):
        os.environ['FLASK_ENV'] = 'development'
    
    app = create_app()
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=app.config.get('DEBUG', True)
    )