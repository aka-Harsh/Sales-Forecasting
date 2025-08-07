"""
Configuration settings for the Advanced Sales Forecasting Platform
"""

import os
import yaml
from typing import Dict, Any


class Config:
    """Base configuration class"""
    
    def __init__(self):
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yml')
        
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        # Get environment
        env = os.getenv('FLASK_ENV', 'development')
        self.config = config_data.get(env, config_data['development'])
        self.models_config = config_data.get('models', {})
        
        # Override with environment variables if they exist
        self._override_with_env_vars()
    
    def _override_with_env_vars(self):
        """Override config with environment variables"""
        env_mappings = {
            'SECRET_KEY': 'secret_key',
            'DATABASE_URL': 'database_url',
            'REDIS_URL': 'redis_url',
            'UPLOAD_FOLDER': 'upload_folder'
        }
        
        for env_var, config_key in env_mappings.items():
            if os.getenv(env_var):
                self.config[config_key] = os.getenv(env_var)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return self.models_config.get(model_name, {})


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = False
    TESTING = True


# Configuration factory
config_factory = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}


def get_config(env: str = None) -> Config:
    """Get configuration based on environment"""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    
    return config_factory.get(env, DevelopmentConfig)()