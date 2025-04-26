import logging
import os

import logging
from functools import wraps

from . import gpu_setup as GPUSetup

logger = logging.getLogger(__name__)

def setup_logger(name, log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger



# -------- DECORATOR FOR MAIN PROCESS ONLY FUNCTIONALITY -------- #
def main_process_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if GPUSetup.is_main_process():
            return func(*args, **kwargs)
    return wrapper

@main_process_only
def log_info(message):
    logger.info(message, stacklevel=2)

@main_process_only
def wandb_log(data):
    import wandb  
    wandb.log(data)