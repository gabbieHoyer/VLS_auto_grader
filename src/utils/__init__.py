# src/utils/__init__.py
# Makes utils a Python package and re-exports modules and functions for convenient access

# Import the gpu_setup module and GPUSetup class
from . import gpu_setup
from .gpu_setup import GPUSetup

# Re-export GPUSetup methods directly for convenience
is_distributed = GPUSetup.is_distributed
get_world_size = GPUSetup.get_world_size
get_rank = GPUSetup.get_rank
get_local_rank = GPUSetup.get_local_rank
is_main_process = GPUSetup.is_main_process
reduce_tensor = GPUSetup.reduce_tensor
save_on_master = GPUSetup.save_on_master

# Re-export other modules
from . import logger
# from . import experiment_setup
from . import config_loader
from . import augmentations
from . import checkpointing
from . import visualization
# from . import ssl_losses

# Re-export specific functions from logger
from .logger import main_process_only, log_info, wandb_log  #setup_logger,

# Re-export specific functions from experiment_setup
# from .experiment_setup import get_project_root, determine_run_directory, generate_group_name

# Re-export specific functions from config_loader
from .config_loader import load_config

# Re-export specific functions from augmentations
from .augmentations import get_transforms, get_ssl_transforms

# Re-export specific functions from checkpointing
from .checkpointing import log_and_checkpoint, final_checkpoint_conversion

# Re-export specific functions from visualization
from .visualization import plot_confusion_matrix, plot_loss_curves, save_attention_maps

# Re-export specific functions from ssl_losses
# from .ssl_losses import compute_ssl_loss

# Define __all__ for controlled imports
__all__ = [
    # Modules
    "gpu_setup",
    "logger",
    # "experiment_setup",
    "config_loader",
    "augmentations",
    "checkpointing",
    "visualization",
    "ssl_losses",
    # Classes
    "GPUSetup",
    # Functions
    "is_distributed",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "is_main_process",
    "reduce_tensor",
    "save_on_master",
    # "setup_logger",
    "main_process_only",
    "log_info",
    "wandb_log",
    # "get_project_root",
    # "determine_run_directory",
    # "generate_group_name",
    "load_config",
    "get_transforms",
    "get_ssl_transforms",
    "log_and_checkpoint",
    "final_checkpoint_conversion",
    "plot_confusion_matrix",
    "plot_loss_curves",
    "save_attention_maps",
    # "compute_ssl_loss",
]



# # src/utils/__init__.py
# # Makes utils a Python package and re-exports modules and functions for convenient access

# # Re-export entire modules
# from . import gpu_setup
# from . import logger
# from . import experiment_setup
# from . import config_loader
# from . import augmentations
# from . import checkpointing
# from . import visualization
# from . import ssl_losses

# # Re-export GPUSetup class only
# from .gpu_setup import GPUSetup

# # Re-export specific functions from logger
# from .logger import setup_logger, main_process_only, log_info, wandb_log

# # Re-export specific functions from experiment_setup
# from .experiment_setup import get_project_root, determine_run_directory, generate_group_name

# # Re-export specific functions from config_loader
# from .config_loader import load_config

# # Re-export specific functions from augmentations
# from .augmentations import get_transforms, get_ssl_transforms

# # Re-export specific functions from checkpointing
# from .checkpointing import log_and_checkpoint, final_checkpoint_conversion

# # Re-export specific functions from visualization
# from .visualization import plot_confusion_matrix, plot_loss_curves, save_attention_maps

# # Re-export specific functions from ssl_losses
# from .ssl_losses import compute_ssl_loss

# # Define __all__ for controlled imports
# __all__ = [
#     # Modules
#     "gpu_setup",
#     "logger",
#     "experiment_setup",
#     "config_loader",
#     "augmentations",
#     "checkpointing",
#     "visualization",
#     "ssl_losses",
#     # Classes
#     "GPUSetup",
#     # Functions
#     "setup_logger",
#     "main_process_only",
#     "log_info",
#     "wandb_log",
#     "get_project_root",
#     "determine_run_directory",
#     "generate_group_name",
#     "load_config",
#     "get_transforms",
#     "get_ssl_transforms",
#     "log_and_checkpoint",
#     "final_checkpoint_conversion",
#     "plot_confusion_matrix",
#     "plot_loss_curves",
#     "save_attention_maps",
#     "compute_ssl_loss",
# ]




# # src/utils/__init__.py
# # Makes utils a Python package and re-exports modules and functions for convenient access

# # Re-export entire modules
# from . import gpu_setup
# from . import logger
# from . import experiment_setup
# from . import config_loader
# from . import augmentations
# from . import checkpointing
# from . import visualization
# from . import ssl_losses

# # Re-export specific functions from gpu_setup
# from .gpu_setup import is_distributed, get_world_size, is_main_process, reduce_tensor, save_on_master

# # Re-export specific functions from logger
# # from .logger import setup_logger, main_process_only, log_info, wandb_log


# from .logger import setup_logger, setup_logging, main_process_only, log_info, wandb_log
# # __all__ += ["setup_logging"]

# # Re-export specific functions from experiment_setup
# from .experiment_setup import get_project_root, determine_run_directory, generate_group_name

# # Re-export specific functions from config_loader
# from .config_loader import load_config

# # Re-export specific functions from augmentations
# from .augmentations import get_transforms, get_ssl_transforms

# # Re-export specific functions from checkpointing
# from .checkpointing import log_and_checkpoint, final_checkpoint_conversion

# # Re-export specific functions from visualization
# from .visualization import plot_confusion_matrix, plot_loss_curves, save_attention_maps

# # Re-export specific functions from ssl_losses
# from .ssl_losses import compute_ssl_loss

# __all__ = [
#     # Modules
#     "gpu_setup",
#     "logger",
#     "experiment_setup",
#     "config_loader",
#     "augmentations",
#     "checkpointing",
#     "visualization",
#     "ssl_losses",
#     # Functions
#     "is_distributed",
#     "get_world_size",
#     "is_main_process",
#     "reduce_tensor",
#     "save_on_master",
#     "setup_logger",
#     "setup_logging",
#     "main_process_only",
#     "log_info",
#     "wandb_log",
#     "get_project_root",
#     "determine_run_directory",
#     "generate_group_name",
#     "load_config",
#     "get_transforms",
#     "get_ssl_transforms",
#     "log_and_checkpoint",
#     "final_checkpoint_conversion",
#     "plot_confusion_matrix",
#     "plot_loss_curves",
#     "save_attention_maps",
#     "compute_ssl_loss",
# ]




# # # Empty file to make utils a package# Re-export the entire gpu_setup module and some specific functions from it
# # from . import gpu_setup
# # from .gpu_setup import is_distributed, get_world_size, is_main_process

# # # Re-export logging functions
# # from .logger import main_process_only, log_info, wandb_log

# # # Re-export checkpointing functions
# # from .checkpointing import log_and_checkpoint, final_checkpoint_conversion

# # # Re-export other utility functions
# # # from .utils import reduce_tensor

# # __all__ = [
# #     "gpu_setup",
# # ]