import os
import yaml
import logging.config
from typing import Dict, Any, Optional

class ConfigurationManager:
    """Centralized configuration manager for the application."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        
        self.config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
        self.configs = {}
        self._load_all_configs()
        self._setup_logging()
        self._initialized = True
    
    def _load_all_configs(self):
        """Load all configuration files from the config directory."""
        config_files = ['cities.yaml', 'model.yaml', 'pipeline.yaml', 'logging.yaml']
        
        for config_file in config_files:
            config_path = os.path.join(self.config_dir, config_file)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_name = os.path.splitext(config_file)[0]
                    self.configs[config_name] = yaml.safe_load(f)
            else:
                print(f"Warning: Configuration file {config_file} not found")
    
    def _setup_logging(self):
        """Set up logging based on the loaded configuration."""
        if 'logging' in self.configs:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            try:
                logging.config.dictConfig(self.configs['logging'])
            except Exception as e:
                print(f"Error setting up logging: {e}")
                # Fall back to basic logging
                logging.basicConfig(level=logging.INFO)
    
    def get_cities(self):
        """Get the list of configured cities."""
        return self.configs.get('cities', {}).get('cities', [])
    
    def get_model_config(self):
        """Get the model configuration."""
        return self.configs.get('model', {}).get('model', {})
    
    def get_pipeline_config(self):
        """Get the pipeline configuration."""
        return self.configs.get('pipeline', {}).get('pipeline', {})
    
    def get_config(self, config_name: str, section: Optional[str] = None) -> Dict[str, Any]:
        """Get a specific configuration or section."""
        if config_name not in self.configs:
            return {}
            
        if section is not None:
            return self.configs[config_name][config_name].get(section, {})
            
        return self.configs[config_name][config_name]
