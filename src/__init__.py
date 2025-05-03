# src/__init__.py

# Import all submodules
from . import utils
from . import data
from . import models
from . import training
from . import ssl

# Define __all__ to control what gets imported with "from src import *"
__all__ = [
    "utils",
    "data",
    "models",
    "training",
    "ssl",
    # Add any other submodules you want to include
]