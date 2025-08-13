import subprocess
from pathlib import Path
from src.logging import get_logger

logger = get_logger(__name__)

def version_trained_model(city_name, version_tag=None):
    """
    Version a trained model with DVC.
    
    Args:
        city_name: Name of the city whose model is being versioned
        version_tag: Optional Git tag to apply to this version
    """
    try:
        # Ensure model exists
        model_path = Path(f"models/trained/{city_name}_model.h5")
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Re-add models directory to DVC
        logger.info("Updating DVC tracking for models...")
        subprocess.run(["dvc", "add", "models/trained"], check=True)
        
        # Git add the DVC file
        subprocess.run(["git", "add", "models/trained.dvc"], check=True)
        
        # Commit the change
        commit_message = f"Update model for {city_name}"
        if version_tag:
            commit_message += f" - version {version_tag}"
        
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # Add tag if provided
        if version_tag:
            tag_name = f"{city_name}_model_{version_tag}"
            subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Model version {version_tag} for {city_name}"], check=True)
        
        logger.info(f"Successfully versioned model for {city_name}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error versioning model: {e}")
        logger.error(f"Command output: {e.stdout}\n{e.stderr}")
        raise
