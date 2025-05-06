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
from . import project
from . import paths
from . import checkpointing
from . import visualization
from . import wandb_utils

# Re-export specific functions from logger
from .logger import main_process_only, log_info, wandb_log, setup_logging #setup_logger,

# Re-export specific functions from experiment_setup
# from .experiment_setup import get_project_root, determine_run_directory, generate_group_name

from .project          import get_project_root
from .paths            import initialize_experiment

# Re-export specific functions from config_loader
from .config_loader import load_config

# Re-export specific functions from checkpointing
from .checkpointing import log_and_checkpoint, final_checkpoint_conversion

# Re-export specific functions from visualization
from .visualization import plot_confusion_matrix, plot_loss_curves, save_attention_maps, plot_augmentations, plot_pretrain_augmentations, plot_qc_predictions

from .wandb_utils import init_wandb_run

# Define __all__ for controlled imports
__all__ = [
    # Modules
    "gpu_setup",
    "logger",
    "config_loader",
    "project",
    "paths",
    "checkpointing",
    "visualization",
    "wandb_utils",
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
    "setup_logging",
    "get_project_root",
    "initialize_experiment",

    "load_config",
    "log_and_checkpoint",
    "final_checkpoint_conversion",
    "plot_confusion_matrix",
    "plot_loss_curves",
    "save_attention_maps",
    "plot_augmentations",
    "plot_pretrain_augmentations",
    "plot_qc_predictions",
    "init_wandb_run",
]

