from .config_loader import ConfigurationManager

# Create a singleton instance
config_manager = ConfigurationManager()

# Convenience functions
def get_cities():
    return config_manager.get_cities()

def get_model_config():
    return config_manager.get_model_config()

def get_pipeline_config():
    return config_manager.get_pipeline_config()

def get_config(config_name, section=None):
    return config_manager.get_config(config_name, section)
