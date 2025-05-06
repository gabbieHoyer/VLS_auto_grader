# src/models/__init__.py

# Re-export other modules
# from . import models
# from . import metrics

# Re-export specific classes and functions from models
from .model import get_model

from .model2 import get_model2

from .metrics import compute_metrics, flatten_metrics

# Define __all__ for controlled imports
__all__ = [
    # Modules
    # "models",
    # "metrics",

    # Classes

    # Functions
    "get_model",
    "get_model2",
    "compute_metrics",
    "flatten_metrics"
]
