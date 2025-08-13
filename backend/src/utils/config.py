import os
import json
import yaml
from pathlib import Path
from typing import Any, Optional

def get_config(file_name: str, key: Optional[str] = None) -> Any:
    """
    Load a configuration value from a YAML or JSON file under /app/config.
    - file_name: base name (without extension) of the config file.
    - key: top‐level key in the config dict to return. If None, return the entire dict.
    """
    config_dir = Path(os.getenv("CONFIG_ROOT", "/app/config"))
    yaml_path = config_dir / f"{file_name}.yaml"
    json_path = config_dir / f"{file_name}.json"

    if yaml_path.exists():
        cfg = yaml.safe_load(yaml_path.read_text())
    elif json_path.exists():
        cfg = json.loads(json_path.read_text())
    else:
        raise FileNotFoundError(f"Config not found: {yaml_path} or {json_path}")

    if key is not None:
        try:
            return cfg[key]
        except KeyError:
            raise KeyError(f"Key '{key}' not found in {file_name}.yaml/json")
    return cfg