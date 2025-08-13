# src/data/storage_setup.py
from pathlib import Path
from src.logging import get_logger
import os

logger = get_logger(__name__)

def setup_storage_structure():
    """Create the directory structure for storing weather data."""
    data_root = os.getenv("DATA_ROOT", "data")
    # Create base directories
    base_dirs = [
        Path(data_root) / "raw",
        Path(data_root) / "processed",
        Path(data_root) / "predictions"
    ]
    
    for base_dir in base_dirs:
        base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {base_dir}")

    logger.info("Storage structure setup complete")

if __name__ == "__main__":
    setup_storage_structure()
