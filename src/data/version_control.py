# src/data/version_control.py
import subprocess
from src.logging import get_logger
import os
from pathlib import Path

logger = get_logger(__name__)

def track_new_data():
    """Track new or modified data files with DVC."""
    try:
        # Check DVC status
        logger.info("Checking DVC status...")
        result = subprocess.run(
            ["dvc", "status"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        logger.info(f"DVC status: {result.stdout}")
        DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))

        
        # Re-add data directories if changes detected
        if "data/raw" in result.stdout:
            logger.info("Changes detected in raw data, updating DVC tracking...")
            subprocess.run(["dvc", "add", f"{DATA_ROOT}/raw"], check=True)
        
        if "data/processed" in result.stdout:
            logger.info("Changes detected in processed data, updating DVC tracking...")
            subprocess.run(["dvc", "add", f"{DATA_ROOT}/processed"], check=True)
        
        if "data/predictions" in result.stdout:
            logger.info("Changes detected in predictions, updating DVC tracking...")
            subprocess.run(["dvc", "add", f"{DATA_ROOT}/predictions"], check=True)
        
        logger.info("DVC tracking updated successfully")
        
        # Remind to commit DVC files to Git
        logger.info("Remember to commit the updated .dvc files to Git")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error updating DVC tracking: {e}")
        logger.error(f"Command output: {e.stdout}\n{e.stderr}")
        raise

if __name__ == "__main__":
    track_new_data()
