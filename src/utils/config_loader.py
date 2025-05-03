# src/utils/config_loader.py
import os
import yaml
from pathlib import Path

def convert_to_float(value):
    """Recursively convert scientific notation strings to floats."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    elif isinstance(value, dict):
        return {k: convert_to_float(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_float(item) for item in value]
    else:
        return value


def resolve_path(path, base_dir):
    """Resolve a path relative to the base directory if it's not absolute."""
    if path and not os.path.isabs(path):
        return os.path.abspath(os.path.join(base_dir, path))
    return path

from pathlib import Path
def load_config(config_path="config.yaml"):
    """Load configuration from a YAML file, convert scientific notation strings to floats,
    and resolve input paths relative to the config directory."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Recursively convert scientific notation strings to floats
    config = convert_to_float(config)
    
    # Get the directory of the config file
    # config_dir = os.path.dirname(config_path)
    config_dir = Path(config_path).parent.parent
    
    # Resolve only input paths in 'paths' section
    input_paths = ['data_csv']
    if "paths" in config:
        for key in input_paths:
            if key in config["paths"]:
                config["paths"][key] = resolve_path(config["paths"][key], config_dir)
    
    return config

# ------------------------------------------------------------

# import os
# import yaml
# from pathlib import Path

# def convert_to_float(value):
#     """Recursively convert scientific notation strings to floats."""
#     if isinstance(value, str):
#         try:
#             return float(value)
#         except ValueError:
#             return value
#     elif isinstance(value, dict):
#         return {k: convert_to_float(v) for k, v in value.items()}
#     elif isinstance(value, list):
#         return [convert_to_float(item) for item in value]
#     else:
#         return value

# def resolve_path(path, base_dir):
#     """Resolve a path relative to the base directory if it's not absolute."""
#     if path and not os.path.isabs(path):
#         return os.path.abspath(os.path.join(base_dir, path))
#     return path

# def load_config(config_path="config.yaml"):
#     """Load configuration from a YAML file, convert scientific notation strings to floats, and resolve relative paths."""
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file not found at {config_path}")
    
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
    
#     # Recursively convert scientific notation strings to floats
#     config = convert_to_float(config)
    
#     # Get the directory of the config file
#     config_dir = os.path.dirname(config_path)
    
#     # Resolve paths in 'paths' section
#     if "paths" in config:
#         for key in config["paths"]:
#             config["paths"][key] = resolve_path(config["paths"][key], config_dir)
    
#     return config